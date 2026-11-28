//! a small OpenGL rasterizer for vertex-colored, textured triangles running the
//! texture environment's combiners in one uber-shader

use glow::HasContext;
use std::collections::HashMap;

use super::{ColorTarget, DepthMap, DepthStencil, Screen};
use crate::{format::morton_offset, GpuMemory};

const VERTEX_SHADER: &str = r#"#version 430 core
layout(location = 0) in vec4 packed_a; // position over w, inverse w, z
layout(location = 1) in vec4 packed_b; // color
layout(location = 2) in vec4 packed_c; // texture coords of sets 0 and 1
layout(location = 3) in vec4 packed_d; // texture coord set 2, set 0's w
layout(location = 4) in vec4 packed_e; // the surface quaternion over w
layout(location = 5) in vec4 packed_f; // the view vector over w
out vec4 vertex_color;
 noperspective out float pica_depth;
 noperspective out float pica_inverse_w;
out vec2 uv0;
out vec2 uv1;
out vec2 uv2;
out float q0;
out vec4 quaternion;
out vec3 view;
void main() {
    gl_Position = vec4(packed_a.xy, 0.0, 1.0 / packed_a.z);
    vertex_color = packed_b;
    pica_depth = packed_a.w;
    pica_inverse_w = packed_a.z;
    uv0 = packed_c.xy;
    uv1 = packed_c.zw;
    uv2 = packed_d.xy;
    q0 = packed_d.z;
    quaternion = packed_e;
    view = packed_f.xyz;
}
"#;

/// one fragment shader for every draw the OpenGL path takes: the texture
/// environment's six combiner stages and the three texture units run from
/// uniforms, so no draw ever waits on a shader compile. The combiner math and
/// the texel wrapping follow the software rasterizer's exactly — the wrap modes
/// wrap texel indices, not GL's normalized coordinates, and the level of
/// detail comes from the rate the coordinates change across the screen like
/// `BoundTexture::sample_at` works it
const FRAGMENT_SHADER: &str = r#"#version 430 core
in vec4 vertex_color;
 noperspective in float pica_depth;
 noperspective in float pica_inverse_w;
in vec2 uv0;
in vec2 uv1;
in vec2 uv2;
in float q0;
in vec4 quaternion;
in vec3 view;
uniform float depth_scale;
uniform float depth_offset;
uniform bool w_buffer;
uniform bool alpha_test;
uniform int alpha_function;
uniform int alpha_reference;
// the texture units: samplers are named one by one (the LUT rides unit 3), the
// rest of each unit's configuration is an array
uniform int tex_enabled[3];
uniform sampler2D tex_image0;
uniform sampler2D tex_image1;
uniform sampler2D tex_image2;
uniform sampler2D lut_image;
uniform vec2 tex_size[3];
// x: the level of detail bias in levels, y: the smallest level, z: the largest,
// w: whether the two levels about the level blend
uniform vec4 tex_lod[3];
// 0 clamp to edge, 1 clamp to border, 2 repeat, 3 mirrored repeat
uniform int tex_wrap_s[3];
uniform int tex_wrap_t[3];
uniform vec4 tex_border[3];
// x: bilinear when magnifying, y: bilinear when minifying
uniform vec2 tex_filter[3];
uniform int tex0_projected;
// the texture environment
uniform int tev_active;
uniform int tev_color_sources[18];
uniform int tev_alpha_sources[18];
uniform int tev_color_operands[18];
uniform int tev_alpha_operands[18];
uniform int tev_color_ops[6];
uniform int tev_alpha_ops[6];
uniform vec4 tev_constants[6];
uniform float tev_color_scales[6];
uniform float tev_alpha_scales[6];
uniform int tev_passthrough[6];
uniform int tev_update_mask;
uniform vec4 tev_buffer_color;
// fragment lighting, the words `Lighting::pack` lays out: the config, the count
// and the bump and shadow words, the fresnel and highlight flags, the seven
// table lookups (read, its input, abs, scale bits), the global ambient, then
// per light eight vec4s (its colors, position, direction, spotlight and the
// distance bias and scale) and two words of flags and number
uniform int lighting_active;
uniform uvec4 light_head;
uniform uvec4 light_flags;
uniform uvec4 light_lookups[7];
uniform vec4 light_ambient;
uniform vec4 light_data[64];
uniform uint light_meta[16];
// the fog, by the fragment's depth thru the LUT texture's last row
uniform int fog_active;
uniform int fog_flip;
uniform vec3 fog_color;
out vec4 color;
// rows 0 to 23 are the lighting tables, the fog's 128 entries share the last
const uint FOG_ROW = 24u;
const uint DISTRIBUTION0 = 0u;
const uint DISTRIBUTION1 = 1u;
const uint FRESNEL = 3u;
const uint REFLECT_BLUE = 4u;
const uint REFLECT_GREEN = 5u;
const uint REFLECT_RED = 6u;
const uint SPOTLIGHT = 8u;
const uint DISTANCE = 16u;
/// a lighting table's value at an entry plus the step toward the next one,
/// taken out of the shared LUT texture
float lut_lookup(uint table, uint entry, float delta) {
    vec2 value = texelFetch(lut_image, ivec2(int(entry), int(table)), 0).rg;
    precise float found = value.x + value.y * delta;
    return found;
}
vec3 rotate(vec4 q, vec3 v) {
    vec3 inner = cross(q.xyz, v);
    precise vec3 turned = v + 2.0 * cross(q.xyz, inner + v * q.w);
    return turned;
}
float quantize(float c) {
    precise float scaled = floor(clamp(c, 0.0, 1.0) * 255.0);
    return scaled / 255.0;
}
/// one light's packed vec4, its colors, position, direction, spotlight and
/// distance, seven to a light in `light_data`
vec3 light_rgb(uint light, uint which) {
    return light_data[light * 8u + which].xyz;
}
/// what fragment lighting leaves in the primary and secondary colors, the same
/// math `Lighting::shade` runs: the surface quaternion and view vector the
/// vertices interpolated, the textures what the units sampled there
void shade(vec4 textures[4], out vec4 diffuse_out, out vec4 specular_out) {
    uint config = light_head.x;
    float length_q = length(quaternion);
    vec4 q = length_q > 0.0 ? quaternion / length_q : vec4(0.0, 0.0, 0.0, 1.0);
    uint shadow_word = light_head.w;
    vec4 shadow = vec4(1.0);
    if ((shadow_word & 1u) != 0u) {
        shadow = textures[(shadow_word >> 4) & 3u];
        if ((shadow_word & 0x100u) != 0u) {
            shadow = vec4(1.0) - shadow;
        }
    }
    vec3 surface_normal = vec3(0.0, 0.0, 1.0);
    vec3 surface_tangent = vec3(1.0, 0.0, 0.0);
    uint bump = light_head.z;
    uint bump_mode = bump & 3u;
    if (bump_mode == 1u) {
        surface_normal = textures[(bump >> 4) & 3u].xyz * 2.0 - 1.0;
        if ((bump & 0x100u) != 0u) {
            surface_normal.z = sqrt(max(1.0 - dot(surface_normal.xy, surface_normal.xy), 0.0));
        }
    } else if (bump_mode == 2u) {
        surface_tangent = textures[(bump >> 4) & 3u].xyz * 2.0 - 1.0;
    }
    vec3 normal = rotate(q, surface_normal);
    // only the last configuration has a table that reads the tangent
    vec3 tangent = config == 8u ? rotate(q, surface_tangent) : surface_tangent;
    bool needs_half = (light_flags.w & 1u) != 0u;
    bool needs_view = (light_flags.w & 2u) != 0u;
    vec3 norm_view = needs_view ? normalize(view) : vec3(0.0);
    precise vec4 diffuse_sum = vec4(0.0, 0.0, 0.0, 1.0);
    precise vec4 specular_sum = vec4(0.0, 0.0, 0.0, 1.0);
    uint count = light_head.y;
    for (uint light = 0u; light < count; light++) {
        uint flags = light_meta[light * 2u];
        uint number = light_meta[light * 2u + 1u];
        bool two_sided = (flags & 2u) != 0u;
        vec3 position = light_rgb(light, 4u);
        vec3 light_vector = (flags & 1u) != 0u ? light_rgb(light, 5u) : normalize(position + view);
        vec3 half_vector = vec3(0.0);
        vec3 half_unit = vec3(0.0);
        if (needs_half) {
            half_vector = norm_view + light_vector;
            half_unit = normalize(half_vector);
        }
        float distance = 1.0;
        if ((flags & 16u) != 0u) {
            vec3 offset = -view - position;
            vec2 bias_scale = light_data[light * 8u + 7u].xy;
            float place = clamp(bias_scale.y * length(offset) + bias_scale.x, 0.0, 1.0);
            float entry = clamp(floor(place * 256.0), 0.0, 255.0);
            distance = lut_lookup(DISTANCE + number, uint(entry), place * 256.0 - entry);
        }
        float values[7];
        for (uint i = 0u; i < 7u; i++) {
            values[i] = 1.0;
            uvec4 lookup = light_lookups[i];
            if (lookup.x == 0u || (i == 6u && (flags & 32u) == 0u)) {
                continue;
            }
            float result = 0.0;
            switch (lookup.y) {
                case 0u: result = dot(normal, half_unit); break;
                case 1u: result = dot(norm_view, half_unit); break;
                case 2u: result = dot(normal, norm_view); break;
                case 3u: result = dot(light_vector, normal); break;
                case 4u: result = dot(light_vector, light_rgb(light, 6u)); break;
                case 5u:
                    if (config == 8u) {
                        float along = dot(normal, half_unit);
                        result = dot(half_unit - normal * along, tangent);
                    }
                    break;
                default: break;
            }
            uint entry;
            float delta;
            if (lookup.z != 0u) {
                result = two_sided ? abs(result) : max(result, 0.0);
                float found = clamp(floor(result * 256.0), 0.0, 255.0);
                entry = uint(found);
                delta = result * 256.0 - found;
            } else {
                // the signed index wraps into the upper half of the table
                float found = clamp(floor(result * 128.0), -128.0, 127.0);
                entry = uint(int(found)) & 0xFFu;
                delta = result * 128.0 - found;
            }
            uint table = i == 0u ? DISTRIBUTION0
                : i == 1u ? DISTRIBUTION1
                : i == 2u ? FRESNEL
                : i == 3u ? REFLECT_RED
                : i == 4u ? REFLECT_GREEN
                : i == 5u ? REFLECT_BLUE
                : SPOTLIGHT + number;
            values[i] = uintBitsToFloat(lookup.w) * lut_lookup(table, entry, delta);
        }
        float spot = values[6];
        float red = values[3];
        vec3 reflect_color = vec3(red,
            light_lookups[4].x != 0u ? values[4] : red,
            light_lookups[5].x != 0u ? values[5] : red);
        vec3 specular0 = light_rgb(light, 0u) * values[0];
        vec3 specular1 = values[1] * reflect_color * light_rgb(light, 1u);
        // only the last light applies fresnel
        if (light == count - 1u && light_lookups[2].x != 0u) {
            if (light_flags.x != 0u) {
                diffuse_sum.a = values[2];
            }
            if (light_flags.y != 0u) {
                specular_sum.a = values[2];
            }
        }
        float facing = dot(light_vector, normal);
        facing = two_sided ? abs(facing) : max(facing, 0.0);
        float highlights = light_flags.z != 0u && facing == 0.0 ? 0.0 : 1.0;
        if ((flags & 12u) != 0u) {
            float squared = dot(half_vector, half_vector);
            float factor = squared == 0.0 ? 0.0 : min(facing / squared, 1.0);
            if ((flags & 4u) != 0u) {
                specular0 *= factor;
            }
            if ((flags & 8u) != 0u) {
                specular1 *= factor;
            }
        }
        bool shadowed = (flags & 64u) != 0u && (shadow_word & 1u) != 0u;
        vec3 shadow_primary = shadowed && (shadow_word & 0x200u) != 0u ? shadow.rgb : vec3(1.0);
        vec3 shadow_secondary = shadowed && (shadow_word & 0x400u) != 0u ? shadow.rgb : vec3(1.0);
        vec3 diffuse = (light_rgb(light, 2u) * facing + light_rgb(light, 3u)) * distance * spot;
        vec3 specular = (specular0 + specular1) * highlights * distance * spot;
        diffuse_sum.rgb += diffuse * shadow_primary;
        specular_sum.rgb += specular * shadow_secondary;
    }
    if ((shadow_word & 0x800u) != 0u) {
        if (light_flags.x != 0u) {
            diffuse_sum.a *= shadow.a;
        }
        if (light_flags.y != 0u) {
            specular_sum.a *= shadow.a;
        }
    }
    diffuse_sum.rgb += light_ambient.rgb;
    diffuse_out = vec4(quantize(diffuse_sum.r), quantize(diffuse_sum.g), quantize(diffuse_sum.b), quantize(diffuse_sum.a));
    specular_out = vec4(quantize(specular_sum.r), quantize(specular_sum.g), quantize(specular_sum.b), quantize(specular_sum.a));
}
bool alpha_passes(int value, int reference) {
    if (alpha_function == 0) return false;
    if (alpha_function == 1) return true;
    if (alpha_function == 2) return value == reference;
    if (alpha_function == 3) return value != reference;
    if (alpha_function == 4) return value < reference;
    if (alpha_function == 5) return value <= reference;
    if (alpha_function == 6) return value > reference;
    return value >= reference;
}
/// one texel's coordinate wrapped back inside the texture, the way the software
/// rasterizer wraps texel indices
/// GLSL leaves `%` undefined when an operand is negative, and repeat coordinates
/// outside the texture and bilinear taps on its edge go in negative, so the
/// remainder comes from a shifted value that never is
int positive_remainder(int coordinate, int period) {
    return coordinate >= 0 ? coordinate % period : period - 1 - (-(coordinate + 1) % period);
}
int wrap_coordinate(int mode, int coordinate, int size) {
    if (mode == 2) {
        return positive_remainder(coordinate, size);
    }
    if (mode == 3) {
        int period = positive_remainder(coordinate, 2 * size);
        return period < size ? period : 2 * size - 1 - period;
    }
    return clamp(coordinate, 0, size - 1);
}
/// one texel of a size, its coordinates wrapped like the unit wraps them, the
/// border color where a border clamp puts them outside
vec4 texel(sampler2D image, int unit, int x, int y, ivec2 size, int level) {
    if ((tex_wrap_s[unit] == 1 && (x < 0 || x >= size.x))
        || (tex_wrap_t[unit] == 1 && (y < 0 || y >= size.y))) {
        return tex_border[unit];
    }
    ivec2 at = ivec2(wrap_coordinate(tex_wrap_s[unit], x, size.x),
                     wrap_coordinate(tex_wrap_t[unit], y, size.y));
    return texelFetch(image, at, level);
}
/// one of the texture's sizes at (u, v), filtered or not, like `sample_level`
vec4 sample_size(sampler2D image, int unit, vec2 coord, int level, bool linear) {
    ivec2 size = ivec2(tex_size[unit]) >> level;
    // PICA texture coords have v=0 at the bottom row, the decoded rows run from the top
    vec2 at = vec2(coord.x, 1.0 - coord.y) * vec2(size);
    if (!linear) {
        return texel(image, unit, int(floor(at.x)), int(floor(at.y)), size, level);
    }
    // texel centers sit at half integer positions
    vec2 f = at - 0.5;
    int x0 = int(floor(f.x));
    int y0 = int(floor(f.y));
    vec2 t = f - vec2(float(x0), float(y0));
    vec4 a = texel(image, unit, x0, y0, size, level);
    vec4 b = texel(image, unit, x0 + 1, y0, size, level);
    vec4 c = texel(image, unit, x0, y0 + 1, size, level);
    vec4 d = texel(image, unit, x0 + 1, y0 + 1, size, level);
    vec4 top = a + (b - a) * t.x;
    vec4 bottom = c + (d - c) * t.x;
    return top + (bottom - top) * t.y;
}
/// one unit's sample: the level of detail from the rate the coords change
/// across the screen, plus the unit's bias and bounds, like `sample_at`
vec4 sample_unit(sampler2D image, int unit, vec2 coord) {
    // a coord is a numerator over 1/w (or over set 0's w over w when unit 0 is
    // projected), both going linearly across the screen, so it changes by the
    // numerator's change minus the coord times the denominator's, over the
    // denominator, like the software rasterizer differences them
    float denominator = unit == 0 && tex0_projected == 1 ? q0 * pica_inverse_w : pica_inverse_w;
    float den = abs(denominator) < 1e-8 ? 1.0 : denominator;
    vec2 numerator = coord * den;
    vec2 per_step_x = (dFdx(numerator) - coord * dFdx(den)) / den * tex_size[unit];
    vec2 per_step_y = (dFdy(numerator) - coord * dFdy(den)) / den * tex_size[unit];
    vec4 lod = tex_lod[unit];
    float rate = max(dot(per_step_x, per_step_x), dot(per_step_y, per_step_y));
    // log2(0) is undefined: a coordinate flat across the screen lands at the
    // rate's limit, which the bounds clamp to the smallest level like the
    // software's own -inf does, the base level and magnification filter for a
    // texture of one size
    float level = rate > 0.0 ? 0.5 * log2(rate) : -1.0e30;
    level = clamp(level + lod.x, lod.y, lod.z);
    if (!(level > 0.0)) {
        // magnified, so the magnification filter applies
        return sample_size(image, unit, coord, 0, tex_filter[unit].x >= 0.5);
    }
    if (lod.w < 0.5) {
        // the nearer size, the bigger one halfway
        return sample_size(image, unit, coord, int(ceil(level + 0.5) - 1.0), tex_filter[unit].y >= 0.5);
    }
    float larger = floor(level);
    int smaller = int(min(larger + 1.0, lod.z));
    float blend = level - larger;
    return mix(sample_size(image, unit, coord, int(larger), tex_filter[unit].y >= 0.5),
               sample_size(image, unit, coord, smaller, tex_filter[unit].y >= 0.5),
               blend);
}
/// where one of a stage's inputs comes from: 0 the vertex color, 1 to 3 the
/// texture units, 4 texture 3 (procedural textures stay on the CPU), 5 the
/// combiner buffer, 6 the stage constant, 7 the previous stage, 8 and 9 the
/// fragment lighting colors, answered from the vertex color and black when no
/// lighting ran, like `TexEnv::apply` does
vec4 pick(int source, vec4 primary, vec4 fragment_primary, vec4 fragment_secondary, vec4 t0, vec4 t1, vec4 t2, vec4 held, vec4 constant, vec4 previous) {
    if (source == 0) return primary;
    if (source == 1) return t0;
    if (source == 2) return t1;
    if (source == 3) return t2;
    if (source == 4) return vec4(0.0, 0.0, 0.0, 1.0);
    if (source == 5) return held;
    if (source == 6) return constant;
    if (source == 8) return fragment_primary;
    if (source == 9) return fragment_secondary;
    return previous;
}
vec3 color_operand(vec4 v, int op) {
    if (op == 0) return v.rgb;
    if (op == 1) return 1.0 - v.rgb;
    if (op == 2) return vec3(v.a);
    if (op == 3) return vec3(1.0 - v.a);
    if (op == 4) return vec3(v.r);
    if (op == 5) return vec3(1.0 - v.r);
    if (op == 8) return vec3(v.g);
    if (op == 9) return vec3(1.0 - v.g);
    if (op == 12) return vec3(v.b);
    if (op == 13) return vec3(1.0 - v.b);
    return v.rgb;
}
float alpha_operand(vec4 v, int op) {
    if (op == 0) return v.a;
    if (op == 1) return 1.0 - v.a;
    if (op == 2) return v.r;
    if (op == 3) return 1.0 - v.r;
    if (op == 4) return v.g;
    if (op == 5) return 1.0 - v.g;
    if (op == 6) return v.b;
    return 1.0 - v.b;
}
/// the combiners, one operation number per arm, the same math `combine_rgb` and
/// `combine_alpha` run
vec3 combine_rgb(int op, vec3 v0, vec3 v1, vec3 v2) {
    if (op == 0) return v0;                                    // replace
    if (op == 1) return v0 * v1;                               // modulate
    if (op == 2) return min(v0 + v1, 1.0);                     // add
    if (op == 3) return clamp(v0 + v1 - 0.5, 0.0, 1.0);        // add signed
    if (op == 4) return v0 * v2 + v1 * (1.0 - v2);             // lerp
    if (op == 5) return max(v0 - v1, 0.0);                     // subtract
    if (op == 6 || op == 7) {                                  // dot3 rgb, rgba
        float dot = clamp(dot(v0 * 2.0 - 1.0, v1 * 2.0 - 1.0), 0.0, 1.0);
        return vec3(dot);
    }
    if (op == 8) return min(v0 * v1 + v2, 1.0);                // multiply add
    return min(v0 + v1, 1.0) * v2;                             // add multiply
}
float combine_alpha(int op, float v0, float v1, float v2) {
    if (op == 0) return v0;
    if (op == 1) return v0 * v1;
    if (op == 2) return min(v0 + v1, 1.0);
    if (op == 3) return clamp(v0 + v1 - 0.5, 0.0, 1.0);
    if (op == 4) return v0 * v2 + v1 * (1.0 - v2);
    if (op == 5) return max(v0 - v1, 0.0);
    if (op == 6 || op == 7) return v0;
    if (op == 8) return min(v0 * v1 + v2, 1.0);
    return min(v0 + v1, 1.0) * v2;
}
/// runs the six stages over one fragment, the buffer a stage or two behind, like `apply`
vec4 tev_combine(vec4 primary, vec4 fragment_primary, vec4 fragment_secondary, vec4 t0, vec4 t1, vec4 t2) {
    vec4 previous = primary;
    vec4 held = vec4(0.0);
    vec4 next_buffer = tev_buffer_color;
    for (int i = 0; i < 6; i++) {
        if (tev_passthrough[i] == 0) {
            vec4 c0 = pick(tev_color_sources[i * 3 + 0], primary, fragment_primary, fragment_secondary, t0, t1, t2, held, tev_constants[i], previous);
            vec4 c1 = pick(tev_color_sources[i * 3 + 1], primary, fragment_primary, fragment_secondary, t0, t1, t2, held, tev_constants[i], previous);
            vec4 c2 = pick(tev_color_sources[i * 3 + 2], primary, fragment_primary, fragment_secondary, t0, t1, t2, held, tev_constants[i], previous);
            vec4 a0 = pick(tev_alpha_sources[i * 3 + 0], primary, fragment_primary, fragment_secondary, t0, t1, t2, held, tev_constants[i], previous);
            vec4 a1 = pick(tev_alpha_sources[i * 3 + 1], primary, fragment_primary, fragment_secondary, t0, t1, t2, held, tev_constants[i], previous);
            vec4 a2 = pick(tev_alpha_sources[i * 3 + 2], primary, fragment_primary, fragment_secondary, t0, t1, t2, held, tev_constants[i], previous);
            vec3 rgb = combine_rgb(tev_color_ops[i],
                color_operand(c0, tev_color_operands[i * 3 + 0]),
                color_operand(c1, tev_color_operands[i * 3 + 1]),
                color_operand(c2, tev_color_operands[i * 3 + 2]));
            float alpha = tev_color_ops[i] == 7
                ? rgb.r
                : combine_alpha(tev_alpha_ops[i],
                    alpha_operand(a0, tev_alpha_operands[i * 3 + 0]),
                    alpha_operand(a1, tev_alpha_operands[i * 3 + 1]),
                    alpha_operand(a2, tev_alpha_operands[i * 3 + 2]));
            previous = vec4(clamp(rgb * tev_color_scales[i], 0.0, 1.0), clamp(alpha * tev_alpha_scales[i], 0.0, 1.0));
        }
        held = next_buffer;
        if (i < 4) {
            if ((tev_update_mask & (1 << i)) != 0) next_buffer.rgb = previous.rgb;
            if ((tev_update_mask & (1 << (i + 4))) != 0) next_buffer.a = previous.a;
        }
    }
    return previous;
}
void main() {
    float depth = pica_depth * depth_scale + depth_offset;
    if (w_buffer) depth /= max(pica_inverse_w, 1e-20);
    gl_FragDepth = clamp(depth, 0.0, 1.0);
    vec4 primary = clamp(vertex_color, 0.0, 1.0);
    vec4 result = primary;
    vec4 fragment_primary = primary;
    // w/o fragment lighting there's no specular term
    vec4 fragment_secondary = vec4(0.0, 0.0, 0.0, 1.0);
    if (tev_active == 1 || lighting_active == 1) {
        vec4 t0 = vec4(0.0, 0.0, 0.0, 1.0);
        vec4 t1 = t0;
        vec4 t2 = t0;
        if (tex_enabled[0] == 1) {
            // a projected texture divides set 0's coords by its own w
            t0 = sample_unit(tex_image0, 0, tex0_projected == 1 ? uv0 / q0 : uv0);
        }
        if (tex_enabled[1] == 1) t1 = sample_unit(tex_image1, 1, uv1);
        if (tex_enabled[2] == 1) t2 = sample_unit(tex_image2, 2, uv2);
        if (lighting_active == 1) {
            // a lighting shadow or bump map on unit 3 reads the procedural
            // texture's spot, which stays on the CPU, so unit 3 is the default
            // texel here whenever lighting reaches this shader
            vec4 textures[4] = vec4[4](t0, t1, t2, vec4(0.0, 0.0, 0.0, 1.0));
            shade(textures, fragment_primary, fragment_secondary);
        }
        if (tev_active == 1) {
            result = tev_combine(primary, fragment_primary, fragment_secondary, t0, t1, t2);
        }
    }
    // the color leaves as whole bytes, like the software path truncates, so the
    // alpha test judges the same byte a fallback draw would
    vec4 quantized = floor(clamp(result, 0.0, 1.0) * 255.0);
    if (alpha_test && !alpha_passes(int(quantized.a), alpha_reference)) discard;
    // the fog mixes in after the alpha test judged the fragment, by its depth
    // thru the LUT's last row, the table's factor of the color and the rest of
    // the fog's, like `Fog::apply` does
    if (fog_active == 1) {
        precise float index = (fog_flip == 1 ? 1.0 - clamp(depth, 0.0, 1.0) : clamp(depth, 0.0, 1.0)) * 128.0;
        precise float entry = clamp(floor(index), 0.0, 127.0);
        vec2 lookup = texelFetch(lut_image, ivec2(int(entry), int(FOG_ROW)), 0).rg;
        precise float factor = clamp(lookup.x + lookup.y * (index - entry), 0.0, 1.0);
        quantized.rgb = floor(factor * quantized.rgb + (1.0 - factor) * fog_color);
    }
    color = quantized / 255.0;
}
"#;

const SURFACE_CACHE_BUDGET: u64 = 128 * 1024 * 1024;
/// decoded textures stay resident under this, about as much as the decoded
/// texel cache the CPU keeps draws from
const TEXTURE_CACHE_BUDGET: u64 = 64 * 1024 * 1024;
/// the row of the shared LUT texture the fog's 128 entries sit on, under the
/// 24 lighting tables of 256 entries each
const FOG_ROW: i32 = 24;

/// one texture unit of a draw, its decoded texels and the sampling configuration
/// the shader runs. The texels are the CPU decoder's own output — the GL texture
/// cache keys on that allocation, so a texture the CPU decoded again (new
/// allocation, memory changed under it) uploads again, and one the CPU kept
/// (same allocation) stays without a second upload
pub(crate) struct TextureUnit {
    /// every size's texels, biggest first, row by row from the top
    pub texels: std::sync::Arc<[[u8; 4]]>,
    pub width: u32,
    pub height: u32,
    /// how many sizes the texels hold, each half the last
    pub levels: u32,
    /// bilinear instead of point sampling when magnifying, then when minifying
    pub mag_linear: bool,
    pub min_linear: bool,
    /// how coordinates outside the texture wrap, numbered as the shader reads
    /// them: 0 clamp to edge, 1 clamp to border, 2 repeat, 3 mirrored repeat
    pub wrap_s: u32,
    pub wrap_t: u32,
    /// the color a border clamp puts outside the texture
    pub border: [f32; 4],
    /// the level of detail bias in levels, the smallest level, the largest, and
    /// whether the two levels about the level blend
    pub lod: [f32; 4],
}

/// a draw's three texture units and its texture environment, what the shader
/// runs the draw from
pub(crate) struct TextureSetup {
    pub units: [Option<TextureUnit>; 3],
    /// texture unit 0 divides its coordinates by set 0's w
    pub texture0_projected: bool,
    /// texture unit 2 reads coordinate set 1 instead of its own
    pub texture2_uses_coord1: bool,
    pub tev: crate::tev::TevUniforms,
}

/// the fragment stages past the units and combiners: fragment lighting, its
/// lookup tables and the fog. The lighting's words go up only when a draw
/// decodes different ones, and the tables into a shared LUT texture only when
/// an entry changed, both tracked since the last draw.
pub(crate) struct FragmentSetup<'a> {
    /// the lighting decoded from the registers, None when no combiner reads it
    pub lighting: Option<&'a crate::lighting::Lighting>,
    /// the lighting tables, rows of the shared LUT texture
    pub tables: &'a crate::lighting::Tables,
    /// the fog, None when the mode doesn't mix any in
    pub fog: Option<&'a crate::fog::Fog<'a>>,
}

/// a GL texture holding one upload of decoded texels, kept for later draws
struct CachedTexture {
    texture: glow::Texture,
    /// held so the texels' allocation cannot be freed while the entry lives,
    /// which is what keeps the address this is keyed on from being reused
    #[allow(dead_code)]
    owner: std::sync::Arc<[[u8; 4]]>,
    bytes: u64,
    last_used: u64,
}

/// the fragment shader's uniform locations, fetched once per context
/// the vertices a draw hands the vertex stage
    enum VertexSource {
    /// placed vertices, six vec4s a vertex, drawn thru the vertex array
    Placed {
        floats: Vec<f32>,
        uniforms: Uniforms,
    },
    /// the GPU shades: the shading block and the input registers ride the
    /// storage buffers, the translated program runs, and GL clips and culls
    /// the triangles the topology names
    Shaded {
        program: glow::Program,
        uniforms: Uniforms,
        /// the target's own size, which the placed position divides by
        buffer_size: Option<glow::UniformLocation>,
        /// whether the buffer's rows run from their top
        flip: Option<glow::UniformLocation>,
        flipped: bool,
        /// the words the Shading block holds, uniforms first
        words: Vec<u32>,
        /// each input vertex's registers the program reads, four floats each
        inputs: Vec<u8>,
        /// the triangles into the input block, when an index buffer repeats vertices
        indices: Option<(Vec<u8>, u32)>,
        /// the vertices a straight draw runs over
        count: usize,
        /// the winding GL keeps, when the PICA culls anything
        cull: Option<u32>,
    },
}


/// a draw whose vertices the GL shades itself, the PICA vertex shader running
/// translated, GL clipping and culling the triangles it makes
pub(crate) struct ShadedDraw<'a> {
    /// the vertex shader the GL runs
    pub(crate) unit: &'a crate::shader::ShaderUnit,
    /// each vertex's input registers, once each
    pub(crate) inputs: &'a [[crate::shader::Vec4; crate::shader::INPUT_REGISTERS]],
    /// which input each vertex of the draw is, when an index buffer repeats them
    pub(crate) order: Option<&'a [usize]>,
    /// where each varying the fragment stage reads sits in the output registers
    pub(crate) semantics: [u32; crate::shader::glsl::SEMANTICS],
    /// left, bottom, width and height, the viewport the PICA places with
    pub(crate) viewport: (f32, f32, f32, f32),
    /// GPUREG_FACECULLING_CONFIG, 0 keeps everything
    pub(crate) cull: u32,
    /// an upside down buffer's rows run from its top
    pub(crate) flip: bool,
    /// GPUREG_PRIMITIVE_CONFIG bits 9 to 8, a list, a strip or a fan
    pub(crate) topology: u32,
}

#[derive(Clone)]
struct Uniforms {
    depth_scale: Option<glow::UniformLocation>,
    depth_offset: Option<glow::UniformLocation>,
    w_buffer: Option<glow::UniformLocation>,
    alpha_test: Option<glow::UniformLocation>,
    alpha_function: Option<glow::UniformLocation>,
    alpha_reference: Option<glow::UniformLocation>,
    tex_enabled: Option<glow::UniformLocation>,
    /// set once after the link, the samplers riding units 0 to 2
    #[allow(dead_code)]
    tex_image: [Option<glow::UniformLocation>; 3],
    tex_size: [Option<glow::UniformLocation>; 3],
    tex_lod: [Option<glow::UniformLocation>; 3],
    tex_wrap_s: Option<glow::UniformLocation>,
    tex_wrap_t: Option<glow::UniformLocation>,
    tex_border: Option<glow::UniformLocation>,
    tex_filter: Option<glow::UniformLocation>,
    tex0_projected: Option<glow::UniformLocation>,
    tev_active: Option<glow::UniformLocation>,
    tev_color_sources: Option<glow::UniformLocation>,
    tev_alpha_sources: Option<glow::UniformLocation>,
    tev_color_operands: Option<glow::UniformLocation>,
    tev_alpha_operands: Option<glow::UniformLocation>,
    tev_color_ops: Option<glow::UniformLocation>,
    tev_alpha_ops: Option<glow::UniformLocation>,
    tev_constants: Option<glow::UniformLocation>,
    tev_color_scales: Option<glow::UniformLocation>,
    tev_alpha_scales: Option<glow::UniformLocation>,
    tev_passthrough: Option<glow::UniformLocation>,
    tev_update_mask: Option<glow::UniformLocation>,
    tev_buffer_color: Option<glow::UniformLocation>,
    lighting_active: Option<glow::UniformLocation>,
    light_head: Option<glow::UniformLocation>,
    light_flags: Option<glow::UniformLocation>,
    light_lookups: Option<glow::UniformLocation>,
    light_ambient: Option<glow::UniformLocation>,
    light_data: Option<glow::UniformLocation>,
    light_meta: Option<glow::UniformLocation>,
    fog_active: Option<glow::UniformLocation>,
    fog_flip: Option<glow::UniformLocation>,
    fog_color: Option<glow::UniformLocation>,
}

/// what happens to a fragment once the shader is done w/ it: the alpha test, how
/// it merges w/ the buffer and the channels the draw may change. The GL path
/// mirrors the parts of this its own fixed function can express, and a draw
/// whose state it cannot stays on the CPU rasterizer.
pub(crate) struct OutputState {
    /// the alpha test's function and reference, when the test is on, both as the
    /// fragment shader numbers them
    alpha_test: Option<(u32, u32)>,
    /// the blend, when the output merger blends instead of copying
    blend: Option<BlendSetup>,
    /// red, green, blue and alpha writes, one channel each
    write: [bool; 4],
}

impl Default for OutputState {
    /// the plain copy the output merger makes when nothing is set, every channel
    /// the draw may change
    fn default() -> Self {
        OutputState {
            alpha_test: None,
            blend: None,
            write: [true; 4],
        }
    }
}

/// the blend as GL takes it: the color and alpha sides each carrying an equation
/// and its source and destination factors, over the constant the constant factors read
struct BlendSetup {
    equations: [u32; 2],
    functions: [(u32, u32); 2],
    constant: [f32; 4],
}

/// the output state the OpenGL path can take, or None when it cannot mirror what
/// the registers describe and the draw belongs to the CPU rasterizer
pub(crate) fn output_state(
    blend: Option<&crate::blend::Blend>,
    logic_op: Option<crate::blend::LogicOp>,
    alpha_test: Option<(super::Compare, u8)>,
    write: [bool; 4],
) -> Option<OutputState> {
    // a logic op works bits in the buffer, which GL does thru glLogicOp and the
    // GL binding does not wrap, so those draws stay on the CPU for now
    if logic_op.is_some() {
        return None;
    }
    let blend = match blend {
        None => None,
        Some(blend) => {
            let ([color, alpha], constant) = blend.config();
            // GL takes the saturate factor as the color side's source alone; a
            // draw putting it anywhere else stays on the CPU
            if color.2 == 14 || alpha.1 == 14 || alpha.2 == 14 {
                return None;
            }
            Some(BlendSetup {
                equations: [blend_equation(color.0), blend_equation(alpha.0)],
                functions: [
                    (blend_factor(color.1), blend_factor(color.2)),
                    (blend_factor(alpha.1), blend_factor(alpha.2)),
                ],
                constant,
            })
        }
    };
    Some(OutputState {
        alpha_test: alpha_test
            .map(|(function, reference)| (alpha_function(function), u32::from(reference))),
        blend,
        write,
    })
}

/// the PICA's equation numbering onto GL's, which shares the order but not the values
fn blend_equation(raw: u32) -> u32 {
    match raw {
        1 => glow::FUNC_SUBTRACT,
        2 => glow::FUNC_REVERSE_SUBTRACT,
        3 => glow::MIN,
        4 => glow::MAX,
        // 0, plus the undocumented 5 to 7, hardware treats those as add
        _ => glow::FUNC_ADD,
    }
}

/// the PICA's factor numbering onto GL's blend factors
fn blend_factor(raw: u32) -> u32 {
    match raw {
        0 => glow::ZERO,
        1 => glow::ONE,
        2 => glow::SRC_COLOR,
        3 => glow::ONE_MINUS_SRC_COLOR,
        4 => glow::DST_COLOR,
        5 => glow::ONE_MINUS_DST_COLOR,
        6 => glow::SRC_ALPHA,
        7 => glow::ONE_MINUS_SRC_ALPHA,
        8 => glow::DST_ALPHA,
        9 => glow::ONE_MINUS_DST_ALPHA,
        10 => glow::CONSTANT_COLOR,
        11 => glow::ONE_MINUS_CONSTANT_COLOR,
        12 => glow::CONSTANT_ALPHA,
        13 => glow::ONE_MINUS_CONSTANT_ALPHA,
        // 14, the saturate factor, GL takes as the color side's source only,
        // which output_state checks for
        _ => glow::SRC_ALPHA_SATURATE,
    }
}

/// the compare function as the fragment shader numbers it, the PICA's own encoding
fn alpha_function(value: super::Compare) -> u32 {
    match value {
        super::Compare::Never => 0,
        super::Compare::Always => 1,
        super::Compare::Equal => 2,
        super::Compare::NotEqual => 3,
        super::Compare::Less => 4,
        super::Compare::LessOrEqual => 5,
        super::Compare::Greater => 6,
        super::Compare::GreaterOrEqual => 7,
    }
}

/// one oversized triangle covers the target with no vertex buffer, for the
/// transfer and the upright passes
const FULLSCREEN_VERTEX_SHADER: &str = r#"#version 430 core
void main() {
    vec2 position = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}
"#;

/// a display transfer between two surfaces the renderer holds, the same pixels
/// the CPU engine writes: the rows read in the layouts' own orders, the block
/// of input each output pixel covers averaged the way the CPU rounds it, the
/// pixels past the copied rectangle left to what the surface already holds
const TRANSFER_SHADER: &str = r#"#version 430 core
uniform sampler2D source;
// the pixels written, width and height
uniform ivec2 copy_size;
// the output surface's size
uniform ivec2 out_size;
// the input pixels averaged into each output pixel, across and down
uniform ivec2 block;
uniform int flip;
uniform int out_tiled;
uniform int in_tiled;
// the surface row the input starts at, and the input surface's rows
uniform int input_row;
uniform int input_height;
// the output keeps no alpha of its own, the copy's alpha stays at one
uniform int alpha_forced;
out vec4 color;
void main() {
    ivec2 at = ivec2(gl_FragCoord.xy);
    // the guest row this fragment writes, then the copy row it reads
    int out_y = out_tiled == 1 ? out_size.y - 1 - at.y : at.y;
    int y = flip == 1 ? out_size.y - 1 - out_y : out_y;
    if (at.x >= copy_size.x || y >= copy_size.y) {
        discard;
    }
    uvec4 sum = uvec4(0u);
    for (int dy = 0; dy < block.y; dy++) {
        for (int dx = 0; dx < block.x; dx++) {
            int src_x = at.x * block.x + dx;
            int src_y = y * block.y + dy;
            // a tiled surface's rows count from the top of its buffer, the
            // texture's from the bottom, like the draw path's upload lays them
            int row = in_tiled == 1 ? input_height - 1 - (input_row + src_y) : input_row + src_y;
            sum += uvec4(round(texelFetch(source, ivec2(src_x, row), 0) * 255.0));
        }
    }
    uint count = uint(block.x * block.y);
    vec4 pixel = vec4((sum + count / 2u) / count) / 255.0;
    if (alpha_forced == 1) {
        pixel.a = 1.0;
    }
    color = pixel;
}
"#;

/// a screen's picture turned upright for showing: the surface keeps a row per
/// screen column, each running up the screen from its bottom, and the upright
/// picture gets the screen's rows top first
const UPRIGHT_SHADER: &str = r#"#version 430 core
uniform sampler2D source;
// the screen's pixels a column runs, and the buffer row the screen's first column sits in
uniform int column_pixels;
uniform int first_column;
out vec4 color;
void main() {
    ivec2 at = ivec2(gl_FragCoord.xy);
    color = texelFetch(source, ivec2(column_pixels - 1 - at.y, first_column + at.x), 0);
}
"#;

/// a display transfer the OpenGL renderer takes itself, the same words the
/// Vulkan renderer's carries
pub(crate) struct Transfer {
    pub(crate) input: u32,
    pub(crate) output: u32,
    pub(crate) input_width: u32,
    pub(crate) output_width: u32,
    pub(crate) output_height: u32,
    /// the pixels written, width and height
    pub(crate) copy: (u32, u32),
    /// the input pixels averaged into each output pixel, across and down
    pub(crate) scale: (u32, u32),
    pub(crate) flip: bool,
    pub(crate) input_linear: bool,
    pub(crate) output_tiled: bool,
    pub(crate) input_format: crate::format::ColorFormat,
    pub(crate) output_format: crate::format::ColorFormat,
}

/// OpenGL resources are created and used on the thread where the context is current.
impl Uniforms {
    /// where a program's uniforms sit, the uber fragment stage and every
    /// translated vertex stage sharing the same names
    fn for_program(gl: &glow::Context, program: glow::Program) -> Uniforms {
        unsafe {
            Uniforms {

            depth_scale: gl.get_uniform_location(program, "depth_scale"),
            depth_offset: gl.get_uniform_location(program, "depth_offset"),
            w_buffer: gl.get_uniform_location(program, "w_buffer"),
            alpha_test: gl.get_uniform_location(program, "alpha_test"),
            alpha_function: gl.get_uniform_location(program, "alpha_function"),
            alpha_reference: gl.get_uniform_location(program, "alpha_reference"),
            tex_enabled: gl.get_uniform_location(program, "tex_enabled[0]"),
            tex_image: std::array::from_fn(|unit| {
                gl.get_uniform_location(program, &format!("tex_image{unit}"))
            }),
            tex_size: std::array::from_fn(|unit| {
                gl.get_uniform_location(program, &format!("tex_size[{unit}]"))
            }),
            tex_lod: std::array::from_fn(|unit| {
                gl.get_uniform_location(program, &format!("tex_lod[{unit}]"))
            }),
            tex_wrap_s: gl.get_uniform_location(program, "tex_wrap_s[0]"),
            tex_wrap_t: gl.get_uniform_location(program, "tex_wrap_t[0]"),
            tex_border: gl.get_uniform_location(program, "tex_border[0]"),
            tex_filter: gl.get_uniform_location(program, "tex_filter[0]"),
            tex0_projected: gl.get_uniform_location(program, "tex0_projected"),
            tev_active: gl.get_uniform_location(program, "tev_active"),
            tev_color_sources: gl.get_uniform_location(program, "tev_color_sources[0]"),
            tev_alpha_sources: gl.get_uniform_location(program, "tev_alpha_sources[0]"),
            tev_color_operands: gl.get_uniform_location(program, "tev_color_operands[0]"),
            tev_alpha_operands: gl.get_uniform_location(program, "tev_alpha_operands[0]"),
            tev_color_ops: gl.get_uniform_location(program, "tev_color_ops[0]"),
            tev_alpha_ops: gl.get_uniform_location(program, "tev_alpha_ops[0]"),
            tev_constants: gl.get_uniform_location(program, "tev_constants[0]"),
            tev_color_scales: gl.get_uniform_location(program, "tev_color_scales[0]"),
            tev_alpha_scales: gl.get_uniform_location(program, "tev_alpha_scales[0]"),
            tev_passthrough: gl.get_uniform_location(program, "tev_passthrough[0]"),
            tev_update_mask: gl.get_uniform_location(program, "tev_update_mask"),
            tev_buffer_color: gl.get_uniform_location(program, "tev_buffer_color"),
            lighting_active: gl.get_uniform_location(program, "lighting_active"),
            light_head: gl.get_uniform_location(program, "light_head"),
            light_flags: gl.get_uniform_location(program, "light_flags"),
            light_lookups: gl.get_uniform_location(program, "light_lookups[0]"),
            light_ambient: gl.get_uniform_location(program, "light_ambient"),
            light_data: gl.get_uniform_location(program, "light_data[0]"),
            light_meta: gl.get_uniform_location(program, "light_meta[0]"),
            fog_active: gl.get_uniform_location(program, "fog_active"),
            fog_flip: gl.get_uniform_location(program, "fog_flip"),
            fog_color: gl.get_uniform_location(program, "fog_color")
            }
        }
    }
}

pub(crate) struct OpenGlRasterizer {
    gl: std::sync::Arc<glow::Context>,
    program: glow::Program,
    uniforms: Uniforms,
    vertex_array: glow::VertexArray,
    vertices: glow::Buffer,
    /// the program display transfers the renderer takes run on
    transfer_program: glow::Program,
    /// the program a screen's picture turns upright in
    upright_program: glow::Program,
    /// the vertex array the fullscreen passes draw thru, no attributes of its own
    utility_vertex_array: glow::VertexArray,
    /// the framebuffer the upright pictures render into
    utility_framebuffer: glow::Framebuffer,
    surfaces: HashMap<SurfaceKey, Surface>,
    surface_bytes: u64,
    surface_clock: u64,
    /// what evict_to_budget keeps the cache under, the real budget but for the
    /// tests, which cannot draw the thousands of targets 128 MiB needs to turn over
    surface_budget: u64,
    /// decoded textures by the address of the texels they were uploaded from
    textures: HashMap<usize, CachedTexture>,
    texture_bytes: u64,
    texture_budget: u64,
    /// the combiners the last draw set, so draws that keep them skip the arrays
    last_tev: Option<crate::tev::TevUniforms>,
    /// the lighting and fog tables, one shared texture rows wide: 24 lighting
    /// tables of 256 entries, then the fog's 128 on the last row
    lookup_tables: glow::Texture,
    /// the texture bound on a sampler unit the draw has no guest texture for,
    /// so a sampler the shader still declares can never stay bound to the
    /// framebuffer's own color attachment and read what the draw writes
    dummy_texture: glow::Texture,
    /// the lighting tables these rows hold, so an entry rewritten between draws
    /// is the one row that uploads again
    lut_tables_id: u64,
    lut_tables_changed: [u64; crate::lighting::TABLES],
    /// the fog table's generation the last row holds
    lut_fog_generation: u64,
    /// the lighting words the last draw set, draws that keep them skip the arrays
    last_lighting: Option<Vec<u32>>,
    /// the fog the last draw set
    last_fog: Option<([f32; 3], bool)>,
    /// the vertex programs the GL shades draws with, by the program's
    /// fingerprint, its entry point and the semantics it hands the varyings by
    shaded_programs: HashMap<ShadedKey, Result<ShadedProgram, String>>,
    /// the block the vertex stage reads its uniforms and state out of
    shading_buffer: glow::Buffer,
    /// the block each input vertex's registers ride to the vertex stage
    inputs_buffer: glow::Buffer,
    /// the triangles of a shaded draw, into the input block's vertices
    indices_buffer: glow::Buffer,
}

/// what one compiled vertex program goes by
type ShadedKey = (u64, u32, [u32; crate::shader::glsl::SEMANTICS]);

/// a vertex stage for one translated program: the linked program and where its
/// uniforms sit, plus the two per-draw values the fragment stage cannot know
struct ShadedProgram {
    program: glow::Program,
    uniforms: Uniforms,
    /// the target's own size, which the placed position divides by
    buffer_size: Option<glow::UniformLocation>,
    /// whether the buffer's rows run from its top
    flip: Option<glow::UniformLocation>,
}

impl Drop for OpenGlRasterizer {
    /// GL objects die on the thread the rasterizer is dropped on, the same one
    /// that made them; dropping the program releases the driver's (and the
    /// software rasterizer's) compiled artifacts, which pile up otherwise
    fn drop(&mut self) {
        for (_, surface) in self.surfaces.drain() {
            unsafe {
                delete_surface(&self.gl, surface);
            }
        }
        for (_, entry) in self.textures.drain() {
            unsafe {
                self.gl.delete_texture(entry.texture);
            }
        }
        unsafe {
            self.gl.delete_texture(self.lookup_tables);
            self.gl.delete_texture(self.dummy_texture);
            self.gl.delete_vertex_array(self.vertex_array);
            self.gl.delete_buffer(self.vertices);
            self.gl.delete_buffer(self.shading_buffer);
            self.gl.delete_buffer(self.inputs_buffer);
            self.gl.delete_buffer(self.indices_buffer);
            for (_, made) in self.shaded_programs.drain() {
                if let Ok(made) = made {
                    self.gl.delete_program(made.program);
                }
            }
            self.gl.delete_program(self.program);
            self.gl.delete_program(self.transfer_program);
            self.gl.delete_program(self.upright_program);
            self.gl.delete_vertex_array(self.utility_vertex_array);
            self.gl.delete_framebuffer(self.utility_framebuffer);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct SurfaceKey {
    addr: u32,
    format: u32,
    width: u32,
    height: u32,
    depth_addr: u32,
    depth_bytes: u32,
    /// a linear buffer's rows lie in memory order (a screen's picture a
    /// transfer wrote), not in the Morton order draw targets keep
    linear: bool,
}

struct Surface {
    framebuffer: glow::Framebuffer,
    color: glow::Texture,
    depth: Option<glow::Texture>,
    initialized: bool,
    dirty: bool,
    depth_dirty: bool,
    storage_bytes: u64,
    last_used: u64,
    /// how many pictures wrote a linear buffer, the screen ref's batch
    picture: u64,
    /// the picture turned upright for showing, for a presenter sharing the context
    upright: Option<Upright>,
}

/// a linear surface's picture, transposed upright into its own texture at the
/// screen's own size
struct Upright {
    texture: glow::Texture,
    /// the picture that is in it
    picture: u64,
    width: u32,
    height: u32,
}

impl OpenGlRasterizer {
    pub(crate) unsafe fn new(context: crate::SharedOpenGlContext) -> Result<Self, String> {
        let gl = context.0;
        let version = unsafe { gl.get_parameter_string(glow::VERSION) };
        if !version_at_least(&version, 4, 3) {
            return Err(format!(
                "OpenGL 4.3 is required for hardware rasterization, found {version}"
            ));
        }

        unsafe {
            let program = link_program(&gl, VERTEX_SHADER, FRAGMENT_SHADER)?;
            let vertex_array = match gl.create_vertex_array() {
                Ok(vertex_array) => vertex_array,
                Err(error) => {
                    gl.delete_program(program);
                    return Err(error.to_string());
                }
            };
            let vertices = match gl.create_buffer() {
                Ok(vertices) => vertices,
                Err(error) => {
                    gl.delete_vertex_array(vertex_array);
                    gl.delete_program(program);
                    return Err(error.to_string());
                }
            };
            let shading_buffer = match gl.create_buffer() {
                Ok(buffer) => buffer,
                Err(error) => {
                    gl.delete_buffer(vertices);
                    gl.delete_vertex_array(vertex_array);
                    gl.delete_program(program);
                    return Err(error.to_string());
                }
            };
            let inputs_buffer = match gl.create_buffer() {
                Ok(buffer) => buffer,
                Err(error) => {
                    gl.delete_buffer(vertices);
                    gl.delete_buffer(shading_buffer);
                    gl.delete_vertex_array(vertex_array);
                    gl.delete_program(program);
                    return Err(error.to_string());
                }
            };
            let indices_buffer = match gl.create_buffer() {
                Ok(buffer) => buffer,
                Err(error) => {
                    gl.delete_buffer(vertices);
                    gl.delete_buffer(shading_buffer);
                    gl.delete_buffer(inputs_buffer);
                    gl.delete_vertex_array(vertex_array);
                    gl.delete_program(program);
                    return Err(error.to_string());
                }
            };

            // six whole vec4s a vertex: position over w, inverse w, z, the color,
            // then two texture coordinate sets and set 0's w, the surface
            // quaternion over w and the view vector over w
            gl.bind_vertex_array(Some(vertex_array));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vertices));
            let stride = (24 * size_of::<f32>()) as i32;
            for attribute in 0..6 {
                gl.enable_vertex_attrib_array(attribute);
                gl.vertex_attrib_pointer_f32(
                    attribute,
                    4,
                    glow::FLOAT,
                    false,
                    stride,
                    (attribute as i32 * 4) * size_of::<f32>() as i32,
                );
            }
            gl.bind_vertex_array(None);

            // the samplers ride units 0 to 2 and the lighting and fog LUT unit 3,
            // set once after the link
            gl.use_program(Some(program));
            for unit in 0..3 {
                if let Some(location) =
                    gl.get_uniform_location(program, &format!("tex_image{unit}"))
                {
                    gl.uniform_1_i32(Some(&location), unit as i32);
                }
            }
            if let Some(location) = gl.get_uniform_location(program, "lut_image") {
                gl.uniform_1_i32(Some(&location), 3);
            }
            let lookup_tables = match gl.create_texture() {
                Ok(texture) => texture,
                Err(error) => {
                    gl.delete_buffer(vertices);
                    gl.delete_vertex_array(vertex_array);
                    gl.delete_program(program);
                    return Err(error.to_string());
                }
            };
            gl.active_texture(glow::TEXTURE3);
            gl.bind_texture(glow::TEXTURE_2D, Some(lookup_tables));
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::NEAREST as i32,
            );
            // empty is zero, every entry a zero value with a zero step
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RG32F as i32,
                256,
                25,
                0,
                glow::RG,
                glow::FLOAT,
                glow::PixelUnpackData::Slice(None),
            );
            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.active_texture(glow::TEXTURE0);
            let dummy_texture = match gl.create_texture() {
                Ok(texture) => texture,
                Err(error) => {
                    gl.delete_buffer(indices_buffer);
                    gl.delete_buffer(inputs_buffer);
                    gl.delete_buffer(shading_buffer);
                    gl.delete_buffer(vertices);
                    gl.delete_vertex_array(vertex_array);
                    gl.delete_texture(lookup_tables);
                    gl.delete_program(program);
                    return Err(error.to_string());
                }
            };
            gl.bind_texture(glow::TEXTURE_2D, Some(dummy_texture));
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
            // one opaque texel no draw reads, the sampler's placeholder
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA8 as i32,
                1,
                1,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(&[0u8, 0, 0, 255])),
            );
            gl.bind_texture(glow::TEXTURE_2D, None);
            let uniforms = Uniforms::for_program(&gl, program);

            // the transfer and upright programs, and the objects their
            // fullscreen passes share
            let (transfer_program, upright_program, utility_vertex_array, utility_framebuffer) = {
                let result = (|| {
                    let transfer = link_program(&gl, FULLSCREEN_VERTEX_SHADER, TRANSFER_SHADER)?;
                    let upright = match link_program(&gl, FULLSCREEN_VERTEX_SHADER, UPRIGHT_SHADER) {
                        Ok(upright) => upright,
                        Err(error) => {
                            gl.delete_program(transfer);
                            return Err(error);
                        }
                    };
                    let vertex_array = match gl.create_vertex_array() {
                        Ok(vertex_array) => vertex_array,
                        Err(error) => {
                            gl.delete_program(transfer);
                            gl.delete_program(upright);
                            return Err(error);
                        }
                    };
                    let framebuffer = match gl.create_framebuffer() {
                        Ok(framebuffer) => framebuffer,
                        Err(error) => {
                            gl.delete_program(transfer);
                            gl.delete_program(upright);
                            gl.delete_vertex_array(vertex_array);
                            return Err(error);
                        }
                    };
                    Ok((transfer, upright, vertex_array, framebuffer))
                })();
                match result {
                    Ok(made) => made,
                    Err(error) => {
                        gl.delete_buffer(indices_buffer);
                        gl.delete_buffer(inputs_buffer);
                        gl.delete_buffer(shading_buffer);
                        gl.delete_buffer(vertices);
                        gl.delete_vertex_array(vertex_array);
                        gl.delete_texture(lookup_tables);
                        gl.delete_texture(dummy_texture);
                        gl.delete_program(program);
                        return Err(error);
                    }
                }
            };

            Ok(Self {
                gl,
                program,
                uniforms,
                vertex_array,
                vertices,
                transfer_program,
                upright_program,
                utility_vertex_array,
                utility_framebuffer,
                surfaces: HashMap::new(),
                surface_bytes: 0,
                surface_clock: 0,
                surface_budget: SURFACE_CACHE_BUDGET,
                textures: HashMap::new(),
                texture_bytes: 0,
                texture_budget: TEXTURE_CACHE_BUDGET,
                last_tev: None,
                lookup_tables,
                dummy_texture,
                shaded_programs: HashMap::new(),
                shading_buffer,
                inputs_buffer,
                indices_buffer,
                lut_tables_id: u64::MAX,
                lut_tables_changed: [u64::MAX; crate::lighting::TABLES],
                lut_fog_generation: u64::MAX,
                last_lighting: None,
                last_fog: None,
            })
        }
    }

    /// everything a draw runs once its vertices are ready: the surface, the
    /// output merger's state, the textures and uniforms, then the vertices
    /// themselves, placed by the CPU or shaded by the translated program
    fn submit<M: GpuMemory>(
        &mut self,
        memory: &mut M,
        target: &ColorTarget,
        depth: Option<&DepthStencil>,
        depth_map: DepthMap,
        output: &OutputState,
        textures: &TextureSetup,
        fragment: &FragmentSetup,
        vertices: VertexSource,
    ) -> Result<(), String> {
        let (program, uniforms) = match &vertices {
            VertexSource::Placed { uniforms, .. } => (self.program, uniforms.clone()),
            VertexSource::Shaded { program, uniforms, .. } => (*program, uniforms.clone()),
        };

        let (width, height) = (target.buffer_width, target.buffer_height);
        let bytes_per_pixel = target.format.bytes_per_pixel() as u32;
        let pixels = width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(bytes_per_pixel))
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or("OpenGL target dimensions overflow")?;
        let key = SurfaceKey {
            addr: target.addr,
            format: target.format as u32,
            width,
            height,
            depth_addr: depth.map_or(0, |d| d.addr),
            depth_bytes: depth.map_or(0, |d| d.bytes),
            linear: false,
        };
        let cached = self.surfaces.contains_key(&key);
        let storage_bytes = surface_storage_bytes(width, height, key.depth_bytes);
        if cached {
            self.sync_except(memory, target.addr, pixels as u32, true, Some(key));
            if let Some(depth) = depth {
                self.sync_except(
                    memory,
                    depth.addr,
                    width.saturating_mul(height).saturating_mul(depth.bytes),
                    true,
                    Some(key),
                );
            }
        } else {
            self.evict_to_budget(memory, storage_bytes);
            self.sync(memory, target.addr, pixels as u32, true);
            if let Some(depth) = depth {
                self.sync(
                    memory,
                    depth.addr,
                    width.saturating_mul(height).saturating_mul(depth.bytes),
                    true,
                );
            }
        }
        if !self.surfaces.contains_key(&key) {
            let (framebuffer, color, depth_texture) =
                unsafe { create_surface(&self.gl, width, height, key.depth_bytes)? };
            self.surface_clock = self.surface_clock.wrapping_add(1);
            self.surfaces.insert(
                key,
                Surface {
                    framebuffer,
                    color,
                    depth: depth_texture,
                    initialized: false,
                    dirty: false,
                    depth_dirty: false,
                    storage_bytes,
                    last_used: self.surface_clock,
                    picture: 0,
                    upright: None,
                },
            );
            self.surface_bytes = self.surface_bytes.saturating_add(storage_bytes);
        }
        self.surface_clock = self.surface_clock.wrapping_add(1);
        let (framebuffer, color, depth_texture, initialized) = {
            let surface = self.surfaces.get_mut(&key).unwrap();
            surface.last_used = self.surface_clock;
            (
                surface.framebuffer,
                surface.color,
                surface.depth,
                surface.initialized,
            )
        };
        let mut tiled = vec![0; pixels];
        let linear = if initialized {
            Vec::new()
        } else {
            memory.read(target.addr, &mut tiled);
            tiled_to_linear(&tiled, width, height, target.format)
        };
        let depth_linear = if !initialized && key.depth_bytes != 0 {
            let mut depth_tiled = vec![0; (width * height * key.depth_bytes) as usize];
            memory.read(key.depth_addr, &mut depth_tiled);
            Some(depth_to_linear(
                &depth_tiled,
                width,
                height,
                key.depth_bytes,
            ))
        } else {
            None
        };


        // the textures go into GL before the draw, each uploaded once for as
        // long as the CPU decoder hands out the same texels
        let bound = self.bind_textures(textures)?;
        // a shaded draw runs its own program, whose uniforms hold none of what
        // the last draw's program was set to, so its combiners, lighting and
        // fog arrays go over however unchanged the state looks
        let own_program = matches!(vertices, VertexSource::Shaded { .. });
        let tev_changed = own_program || self.last_tev.as_ref() != Some(&textures.tev);
        // the lighting and fog tables into the LUT texture, a changed row only,
        // then the words the lighting state packed into, only when a draw
        // decoded different ones
        self.upload_luts(fragment.lighting, fragment.tables, fragment.fog)?;
        let lighting_words = fragment.lighting.map(|lighting| {
            let mut words = Vec::new();
            lighting.pack(&mut words);
            words
        });
        let lighting_changed = own_program || lighting_words != self.last_lighting;
        let fog_state = fragment.fog.map(|fog| (fog.color, fog.flip));
        let fog_update = if own_program {
            fog_state
        } else {
            fog_state.filter(|fog| self.last_fog.as_ref() != Some(fog))
        };

        let framebuffer_complete = unsafe {
            let gl = &self.gl;
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
            gl.bind_texture(glow::TEXTURE_2D, Some(color));
            if !initialized {
                gl.tex_sub_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    0,
                    0,
                    width as i32,
                    height as i32,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(Some(&linear)),
                );
                if let (Some(depth_texture), Some(depth_data)) =
                    (depth_texture, depth_linear.as_ref())
                {
                    gl.bind_texture(glow::TEXTURE_2D, Some(depth_texture));
                    gl.tex_sub_image_2d(
                        glow::TEXTURE_2D,
                        0,
                        0,
                        0,
                        width as i32,
                        height as i32,
                        if key.depth_bytes == 4 {
                            glow::DEPTH_STENCIL
                        } else {
                            glow::DEPTH_COMPONENT
                        },
                        if key.depth_bytes == 2 {
                            glow::UNSIGNED_SHORT
                        } else if key.depth_bytes == 4 {
                            glow::UNSIGNED_INT_24_8
                        } else {
                            glow::FLOAT
                        },
                        glow::PixelUnpackData::Slice(Some(depth_data)),
                    );
                }
            }

            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(framebuffer));
            if gl.check_framebuffer_status(glow::FRAMEBUFFER) != glow::FRAMEBUFFER_COMPLETE {
                gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                false
            } else {
                gl.viewport(0, 0, width as i32, height as i32);
                gl.enable(glow::SCISSOR_TEST);
                gl.scissor(
                    target.left,
                    target.bottom,
                    target.right.saturating_sub(target.left),
                    target.top.saturating_sub(target.bottom),
                );
                gl.disable(glow::BLEND);
                if let Some(depth_state) = depth {
                    gl.enable(glow::DEPTH_TEST);
                    gl.depth_func(depth_state.test.map_or(glow::ALWAYS, compare));
                    gl.depth_mask(depth_state.write_depth);
                    if let Some(stencil) = depth_state.stencil {
                        gl.enable(glow::STENCIL_TEST);
                        gl.stencil_func(
                            compare(stencil.function),
                            stencil.reference as i32,
                            stencil.compare_mask as u32,
                        );
                        gl.stencil_mask(if depth_state.write_stencil {
                            stencil.write_mask as u32
                        } else {
                            0
                        });
                        gl.stencil_op(
                            stencil_op(stencil.fail),
                            stencil_op(stencil.depth_fail),
                            stencil_op(stencil.pass),
                        );
                    } else {
                        gl.disable(glow::STENCIL_TEST);
                    }
                } else {
                    gl.disable(glow::DEPTH_TEST);
                    gl.disable(glow::STENCIL_TEST);
                }
                gl.disable(glow::CULL_FACE);
                gl.disable(glow::DITHER);
                gl.disable(glow::FRAMEBUFFER_SRGB);
                // a guest format that keeps no alpha of its own has the alpha
                // channel of its RGBA8 copy hold the 255 its decode answers,
                // and no draw may write it: a blend factor reading the
                // destination alpha then sees the 1.0 the software path sees.
                // The 16 bit formats that do keep alpha (RGB5A1 a bit, RGBA4
                // four) write it thru the copy, and the writeback packs it to
                // the guest's own precision like the software's encode does
                let alpha_writable = output.write[3]
                    && !matches!(
                        target.format,
                        crate::format::ColorFormat::Rgb8 | crate::format::ColorFormat::Rgb565
                    );
                gl.color_mask(
                    output.write[0],
                    output.write[1],
                    output.write[2],
                    alpha_writable,
                );
                match &output.blend {
                    Some(blend) => {
                        gl.enable(glow::BLEND);
                        gl.blend_color(
                            blend.constant[0],
                            blend.constant[1],
                            blend.constant[2],
                            blend.constant[3],
                        );
                        gl.blend_equation_separate(blend.equations[0], blend.equations[1]);
                        gl.blend_func_separate(
                            blend.functions[0].0,
                            blend.functions[0].1,
                            blend.functions[1].0,
                            blend.functions[1].1,
                        );
                    }
                    None => {
                        gl.disable(glow::BLEND);
                    }
                }
                // placed vertices come thru the vertex array; the shaded draws
                // read their inputs out of storage buffers, so no attributes
                let vertex_array = match vertices {
                    VertexSource::Placed { .. } => self.vertex_array,
                    VertexSource::Shaded { .. } => self.utility_vertex_array,
                };
                gl.use_program(Some(program));
                gl.bind_vertex_array(Some(vertex_array));
                if let Some(location) = &uniforms.depth_scale {
                    gl.uniform_1_f32(Some(location), depth_map.scale);
                }
                if let Some(location) = &uniforms.depth_offset {
                    gl.uniform_1_f32(Some(location), depth_map.offset);
                }
                if let Some(location) = &uniforms.w_buffer {
                    gl.uniform_1_i32(Some(location), depth_map.w_buffer as i32);
                }
                if let Some(location) = &uniforms.alpha_test {
                    gl.uniform_1_i32(Some(location), output.alpha_test.is_some() as i32);
                }
                if let Some(location) = &uniforms.alpha_function {
                    gl.uniform_1_i32(
                        Some(location),
                        output.alpha_test.map_or(1, |(function, _)| function as i32),
                    );
                }
                if let Some(location) = &uniforms.alpha_reference {
                    gl.uniform_1_i32(
                        Some(location),
                        output
                            .alpha_test
                            .map_or(0, |(_, reference)| reference as i32),
                    );
                }
                self.set_texture_uniforms(&uniforms, textures, tev_changed);
                if let Some(location) = &uniforms.lighting_active {
                    gl.uniform_1_i32(Some(location), fragment.lighting.is_some() as i32);
                }
                if lighting_changed {
                    if let Some(words) = &lighting_words {
                        self.set_lighting_uniforms(&uniforms, words);
                    }
                }
                if let Some(location) = &uniforms.fog_active {
                    gl.uniform_1_i32(Some(location), fragment.fog.is_some() as i32);
                }
                if let Some((color, flip)) = &fog_update {
                    if let Some(location) = &uniforms.fog_flip {
                        gl.uniform_1_i32(Some(location), *flip as i32);
                    }
                    if let Some(location) = &uniforms.fog_color {
                        gl.uniform_3_f32_slice(Some(location), color);
                    }
                }
                for (unit, texture) in bound.iter().enumerate() {
                    gl.active_texture(glow::TEXTURE0 + unit as u32);
                    let (Some(texture), Some(setup)) =
                        (texture, textures.units[unit].as_ref())
                    else {
                        // a sampler the shader still declares must never stay
                        // bound to the framebuffer's own color attachment
                        gl.bind_texture(glow::TEXTURE_2D, Some(self.dummy_texture));
                        continue;
                    };
                    gl.bind_texture(glow::TEXTURE_2D, Some(*texture));
                    gl.tex_parameter_i32(
                        glow::TEXTURE_2D,
                        glow::TEXTURE_MAX_LEVEL,
                        (setup.levels - 1) as i32,
                    );
                    gl.tex_parameter_i32(
                        glow::TEXTURE_2D,
                        glow::TEXTURE_MAG_FILTER,
                        if setup.mag_linear {
                            glow::LINEAR
                        } else {
                            glow::NEAREST
                        } as i32,
                    );
                    gl.tex_parameter_i32(
                        glow::TEXTURE_2D,
                        glow::TEXTURE_MIN_FILTER,
                        // the shader picks levels out of the chain itself, so a
                        // texture of several sizes keeps a mipmapped filter even
                        // when the guest's minification filter is a plain one;
                        // its point, bilinear and mip-blending decisions ride
                        // the shader's uniforms
                        if setup.levels > 1 {
                            glow::NEAREST_MIPMAP_NEAREST
                        } else if setup.min_linear {
                            glow::LINEAR
                        } else {
                            glow::NEAREST
                        } as i32,
                    );
                    gl.tex_parameter_i32(
                        glow::TEXTURE_2D,
                        glow::TEXTURE_WRAP_S,
                        wrap_sampler_mode(setup.wrap_s) as i32,
                    );
                    gl.tex_parameter_i32(
                        glow::TEXTURE_2D,
                        glow::TEXTURE_WRAP_T,
                        wrap_sampler_mode(setup.wrap_t) as i32,
                    );
                    gl.tex_parameter_f32_slice(
                        glow::TEXTURE_2D,
                        glow::TEXTURE_BORDER_COLOR,
                        &setup.border,
                    );
                }
                match vertices {
                    VertexSource::Placed { floats, .. } => {
                        gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vertices));
                        gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytemuck::cast_slice(&floats), glow::STREAM_DRAW);
                        gl.draw_arrays(glow::TRIANGLES, 0, (floats.len() / 24) as i32);
                    }
                    VertexSource::Shaded { program, uniforms, buffer_size, flip, flipped, words, inputs, indices, count, cull } => {
                        let _ = (program, uniforms);
                        if let Some(location) = buffer_size {
                            gl.uniform_2_f32_slice(Some(&location), &[width as f32, height as f32]);
                        }
                        if let Some(location) = flip {
                            gl.uniform_1_i32(Some(&location), flipped as i32);
                        }
                        // the vertex stage reads its uniforms and state out of one
                        // storage block and the input registers out of another
                        gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(self.shading_buffer));
                        gl.buffer_data_u8_slice(
                            glow::SHADER_STORAGE_BUFFER,
                            bytemuck::cast_slice(&words),
                            glow::STREAM_DRAW,
                        );
                        gl.bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 6, Some(self.shading_buffer));
                        gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(self.inputs_buffer));
                        gl.buffer_data_u8_slice(glow::SHADER_STORAGE_BUFFER, &inputs, glow::STREAM_DRAW);
                        gl.bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 7, Some(self.inputs_buffer));
                        gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, None);
                        // the PICA's volume and winding, which GL clips and culls by
                        gl.enable(glow::CLIP_DISTANCE0);
                        if let Some(front_face) = cull {
                            gl.enable(glow::CULL_FACE);
                            gl.front_face(front_face);
                            gl.cull_face(glow::BACK);
                        }
                        match indices {
                            Some((bytes, elements)) => {
                                gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(self.indices_buffer));
                                gl.buffer_data_u8_slice(glow::ELEMENT_ARRAY_BUFFER, &bytes, glow::STREAM_DRAW);
                                gl.draw_elements(glow::TRIANGLES, elements as i32, glow::UNSIGNED_SHORT, 0);
                            }
                            None => {
                                gl.draw_arrays(glow::TRIANGLES, 0, count as i32);
                            }
                        }
                        gl.disable(glow::CLIP_DISTANCE0);
                        gl.disable(glow::CULL_FACE);
                        gl.bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 6, None);
                        gl.bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 7, None);
                    }
                }
                gl.bind_vertex_array(None);
                gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                gl.disable(glow::SCISSOR_TEST);
                gl.disable(glow::DEPTH_TEST);
                gl.disable(glow::STENCIL_TEST);
                gl.disable(glow::BLEND);
                gl.depth_mask(true);
                gl.stencil_mask(0xFF);
                gl.color_mask(true, true, true, true);
                true
            }
        };
        if !framebuffer_complete {
            self.evict_to_budget(memory, 0);
            return Err("OpenGL framebuffer is incomplete".to_owned());
        }
        let surface = self.surfaces.get_mut(&key).unwrap();
        surface.initialized = true;
        surface.dirty = true;
        surface.depth_dirty |= depth.is_some_and(|d| d.write_depth || d.write_stencil);
        self.last_tev = Some(textures.tev);
        self.last_lighting = lighting_words;
        self.last_fog = fog_state;
        memory.guard(target.addr, width * height * bytes_per_pixel);
        memory.guard_writes(target.addr, width * height * bytes_per_pixel);
        if let Some(depth) = depth {
            let bytes = width.saturating_mul(height).saturating_mul(depth.bytes);
            memory.guard(depth.addr, bytes);
            memory.guard_writes(depth.addr, bytes);
        }
        self.evict_to_budget(memory, 0);
        Ok(())
    }


    /// a draw the GL shades itself: the PICA vertex shader runs translated off
    /// the program the title uploaded, its inputs ride a storage buffer, and GL
    /// clips and culls the triangles it makes before the fragment stage runs.
    /// Returns the triangles the draw names
    pub(super) fn draw_shaded<M: GpuMemory>(
        &mut self,
        memory: &mut M,
        target: &ColorTarget,
        depth: Option<&DepthStencil>,
        depth_map: DepthMap,
        output: &OutputState,
        textures: &TextureSetup,
        fragment: &FragmentSetup,
        shaded: &ShadedDraw,
    ) -> Result<u32, String> {
        if shaded.inputs.is_empty() || target.right <= target.left || target.top <= target.bottom {
            return Ok(0);
        }
        // the program, compiled once per fingerprint, entry point and semantics,
        // or the reason it cannot run, which puts the draw back on the CPU
        let key: ShadedKey = (
            shaded.unit.fingerprint(),
            shaded.unit.entry_point,
            shaded.semantics,
        );
        let made = match self.shaded_programs.get(&key) {
            Some(Ok(made)) => made,
            Some(Err(error)) => return Err(error.clone()),
            None => {
                let made_program = unsafe { self.compile_shaded_program(shaded.unit, shaded.semantics) };
                self.shaded_programs.insert(key, made_program);
                match self.shaded_programs.get(&key).expect("just inserted") {
                    Ok(made) => made,
                    Err(error) => return Err(error.clone()),
                }
            }
        };

        // the words the Shading block holds: the uniforms, the semantics, the
        // viewport, and where each input register the program reads finds its
        // attribute, which is its own place in the input block
        let unit = shaded.unit;
        let read = unit.inputs_read();
        let mut reads = [0usize; crate::shader::INPUT_REGISTERS];
        let mut found = 0;
        for register in (0..crate::shader::INPUT_REGISTERS).filter(|r| read & (1 << r) != 0) {
            reads[found] = register;
            found += 1;
        }
        let registers_read = &reads[..found];
        let stride = (registers_read.len() * 16) as u32;
        let mut attributes = [[0u32; 4]; crate::shader::INPUT_REGISTERS];
        for (slot, &register) in registers_read.iter().enumerate() {
            attributes[register] = [0, stride, (slot as u32 * 16) | 3 << 8 | 4 << 16, 0];
        }
        let mut words = Vec::with_capacity(crate::shader::glsl::SHADING_WORDS);
        words.extend(unit.float_uniforms.as_flattened().iter().map(|value| value.to_bits()));
        for [count, start, step, _] in unit.int_uniforms {
            words.extend([count as u32, start as u32, step as i8 as u32, 0]);
        }
        words.extend([unit.bool_uniforms as u32, unit.entry_point, 0, 0]);
        words.extend(shaded.semantics);
        words.extend([depth_map.scale.to_bits(), depth_map.offset.to_bits(), 0, 0]);
        let (x, y, width, height) = shaded.viewport;
        words.extend([x, y, width, height].map(f32::to_bits));
        words.extend_from_slice(attributes.as_flattened());
        // no defaults: the decoded inputs carry every register's value
        words.resize(crate::shader::glsl::SHADING_WORDS, 0);

        // the input registers each vertex carries, four floats a register
        let mut inputs = Vec::with_capacity(shaded.inputs.len() * registers_read.len() * 16);
        for input in shaded.inputs {
            for &register in registers_read {
                for value in input[register] {
                    inputs.extend_from_slice(&value.to_le_bytes());
                }
            }
        }

        // the triangles the topology names, every one wound like the first, so
        // culling judges the winding the PICA does instead of GL's alternation
        let count = shaded.order.map_or(shaded.inputs.len(), <[usize]>::len);
        let mut indices: Vec<u16> = Vec::new();
        for (a, b, c) in super::assemble(shaded.topology, count) {
            let at = |i: usize| shaded.order.map_or(i, |order| order[i]);
            indices.extend([at(a) as u16, at(b) as u16, at(c) as u16]);
        }
        let triangles = (indices.len() / 3) as u32;
        let indices = (!indices.is_empty()).then(|| {
            let bytes = bytemuck::cast_slice::<u16, u8>(&indices).to_vec();
            (bytes, indices.len() as u32)
        });

        // the PICA keeps clockwise triangles (cull 1) or counterclockwise ones
        // (cull 2) seen w/ y pointing up, which GL's own window counts; an upside
        // down buffer's flipped rows turn every winding over
        let cull = match (shaded.cull, shaded.flip) {
            (0, _) | (_, _) if shaded.cull == 0 => None,
            (1, false) | (2, true) => Some(glow::CW),
            (2, false) | (1, true) => Some(glow::CCW),
            _ => None,
        };

        self.submit(
            memory,
            target,
            depth,
            depth_map,
            output,
            textures,
            fragment,
            VertexSource::Shaded {
                program: made.program,
                uniforms: made.uniforms.clone(),
                buffer_size: made.buffer_size,
                flip: made.flip,
                flipped: shaded.flip,
                words,
                inputs,
                indices,
                count,
                cull,
            },
        )?;
        Ok(triangles)
    }

    /// the vertex stage for one program: the title's program translated to GLSL,
    /// linked w/ the fragment stage the draws share, its samplers pointed at the
    /// units and the two per-draw uniforms found
    unsafe fn compile_shaded_program(
        &mut self,
        unit: &crate::shader::ShaderUnit,
        semantics: [u32; crate::shader::glsl::SEMANTICS],
    ) -> Result<ShadedProgram, String> {
        let glsl = unit
            .translate_gl(&semantics)
            .map_err(|error| format!("the vertex shader would not translate, {error}"))?;
        let program = unsafe { link_program(&self.gl, &glsl, FRAGMENT_SHADER)? };
        unsafe {
            let gl = &self.gl;
            gl.use_program(Some(program));
            for texture in 0..3 {
                if let Some(location) = gl.get_uniform_location(program, &format!("tex_image{texture}")) {
                    gl.uniform_1_i32(Some(&location), texture as i32);
                }
            }
            if let Some(location) = gl.get_uniform_location(program, "lut_image") {
                gl.uniform_1_i32(Some(&location), 3);
            }
            Ok(ShadedProgram {
                program,
                uniforms: Uniforms::for_program(gl, program),
                buffer_size: gl.get_uniform_location(program, "buffer_size"),
                flip: gl.get_uniform_location(program, "flip"),
            })
        }
    }

    /// the lighting and fog tables into the shared LUT texture: a table with a
    /// changed entry uploads its row again, the fog when its table changed, and
    /// the texture stays bound to unit 3 for the draw
    /// a draw the CPU already shaded, clipped and placed: the placed triangles
    /// go in as six vec4s a vertex and GL rasterizes them
    pub(super) fn draw<M: GpuMemory>(
        &mut self,
        memory: &mut M,
        target: &ColorTarget,
        depth: Option<&DepthStencil>,
        depth_map: DepthMap,
        output: &OutputState,
        textures: &TextureSetup,
        fragment: &FragmentSetup,
        triangles: &[[Screen; 3]],
    ) -> Result<(), String> {
        if triangles.is_empty() || target.right <= target.left || target.top <= target.bottom {
            return Ok(());
        }
        let (width, height) = (target.buffer_width, target.buffer_height);
        let mut floats = Vec::with_capacity(triangles.len() * 3 * 24);
        let set2 = if textures.texture2_uses_coord1 { 1 } else { 2 };
        for triangle in triangles {
            // q and -q are the same orientation but halfway between them isn't,
            // so the quaternions all go to the first vertex's side, the way the
            // software rasterizer turns them before it interpolates
            let first = triangle[0].quaternion_over_w;
            let hemisphere = |vertex: &Screen| {
                let same = (0..4)
                    .map(|i| vertex.quaternion_over_w[i] * first[i])
                    .sum::<f32>();
                if same < 0.0 {
                    vertex.quaternion_over_w.map(|component| -component)
                } else {
                    vertex.quaternion_over_w
                }
            };
            let quaternions = [first, hemisphere(&triangle[1]), hemisphere(&triangle[2])];
            for (vertex, quaternion_over_w) in triangle.iter().zip(quaternions) {
                let inverse_w = vertex.inv_w;
                floats.extend([
                    (2.0 * vertex.x / width as f32 - 1.0) / inverse_w,
                    (2.0 * vertex.y / height as f32 - 1.0) / inverse_w,
                    inverse_w,
                    vertex.z,
                ]);
                floats.extend(vertex.color_over_w.map(|component| component / inverse_w));
                let [u0, v0] = vertex.texcoords_over_w[0];
                let [u1, v1] = vertex.texcoords_over_w[1];
                let [u2, v2] = vertex.texcoords_over_w[set2];
                floats.extend([
                    u0 / inverse_w,
                    v0 / inverse_w,
                    u1 / inverse_w,
                    v1 / inverse_w,
                    u2 / inverse_w,
                    v2 / inverse_w,
                    vertex.texcoord0_w_over_w / inverse_w,
                    0.0,
                ]);
                floats.extend(quaternion_over_w.map(|component| component / inverse_w));
                floats.extend([
                    vertex.view_over_w[0] / inverse_w,
                    vertex.view_over_w[1] / inverse_w,
                    vertex.view_over_w[2] / inverse_w,
                    0.0,
                ]);
            }
        }
        self.submit(
            memory,
            target,
            depth,
            depth_map,
            output,
            textures,
            fragment,
            VertexSource::Placed {
                floats,
                uniforms: self.uniforms.clone(),
            },
        )
    }
    fn upload_luts(
        &mut self,
        lighting: Option<&crate::lighting::Lighting>,
        tables: &crate::lighting::Tables,
        fog: Option<&crate::fog::Fog>,
    ) -> Result<(), String> {
        unsafe {
            let gl = &self.gl;
            gl.active_texture(glow::TEXTURE3);
            gl.bind_texture(glow::TEXTURE_2D, Some(self.lookup_tables));
            if lighting.is_some() {
                let id = tables.id();
                let changed = tables.changed();
                if id != self.lut_tables_id {
                    for table in 0..crate::lighting::TABLES {
                        self.upload_table_row(table, tables)?;
                    }
                    self.lut_tables_changed = *changed;
                    self.lut_tables_id = id;
                } else {
                    for table in 0..crate::lighting::TABLES {
                        if changed[table] != self.lut_tables_changed[table] {
                            self.upload_table_row(table, tables)?;
                            self.lut_tables_changed[table] = changed[table];
                        }
                    }
                }
            }
            if let Some(fog) = fog {
                let generation = fog.table.generation();
                if generation != self.lut_fog_generation {
                    gl.tex_sub_image_2d(
                        glow::TEXTURE_2D,
                        0,
                        0,
                        FOG_ROW,
                        crate::fog::ENTRIES as i32,
                        1,
                        glow::RG,
                        glow::FLOAT,
                        glow::PixelUnpackData::Slice(Some(bytemuck::cast_slice(
                            fog.table.entries(),
                        ))),
                    );
                    self.lut_fog_generation = generation;
                }
            }
            gl.active_texture(glow::TEXTURE0);
        }
        Ok(())
    }

    /// one lighting table's 256 entries, a value and a step each, into its row
    fn upload_table_row(
        &self,
        table: usize,
        tables: &crate::lighting::Tables,
    ) -> Result<(), String> {
        let entries = tables
            .entries()
            .get(table)
            .ok_or_else(|| "lighting table out of range".to_owned())?;
        unsafe {
            self.gl.tex_sub_image_2d(
                glow::TEXTURE_2D,
                0,
                0,
                table as i32,
                256,
                1,
                glow::RG,
                glow::FLOAT,
                glow::PixelUnpackData::Slice(Some(bytemuck::cast_slice(entries))),
            );
        }
        Ok(())
    }

    /// the lighting's words onto the shader's uniforms, the vec4s through the
    /// bits they pack, the flags and numbers gathered out of each light's block
    fn set_lighting_uniforms(&self, uniforms: &Uniforms, words: &[u32]) {
        unsafe {
            let gl = &self.gl;
            if let Some(location) = &uniforms.light_head {
                gl.uniform_4_u32_slice(Some(location), &words[0..4]);
            }
            if let Some(location) = &uniforms.light_flags {
                gl.uniform_4_u32_slice(Some(location), &words[4..8]);
            }
            if let Some(location) = &uniforms.light_lookups {
                gl.uniform_4_u32_slice(Some(location), &words[8..36]);
            }
            if let Some(location) = &uniforms.light_ambient {
                let ambient: [f32; 4] = words[36..40]
                    .iter()
                    .map(|word| f32::from_bits(*word))
                    .collect::<Vec<_>>()
                    .try_into()
                    .expect("the global ambient is four words");
                gl.uniform_4_f32_slice(Some(location), &ambient);
            }
            if let Some(location) = &uniforms.light_data {
                // each light's slot carries 32 words of float data, then four of
                // integer metadata the flags ride separately, and the shader
                // strides light_data by eight vec4s of the data alone
                let data: Vec<f32> = words[40..]
                    .as_chunks::<36>()
                    .0
                    .iter()
                    .flat_map(|light| light[..32].iter())
                    .map(|word| f32::from_bits(*word))
                    .collect();
                gl.uniform_4_f32_slice(Some(location), &data);
            }
            if let Some(location) = &uniforms.light_meta {
                let mut meta = [0u32; 16];
                for (slot, pair) in meta.chunks_mut(2).enumerate() {
                    let base = 40 + slot * 36;
                    pair[0] = words[base + 32];
                    pair[1] = words[base + 33];
                }
                gl.uniform_1_u32_slice(Some(location), &meta);
            }
        }
    }

    /// every unit's texture uploaded and in the cache, by unit, so the draw
    /// binds them without touching the cache
    fn bind_textures(
        &mut self,
        textures: &TextureSetup,
    ) -> Result<[Option<glow::Texture>; 3], String> {
        let mut bound = [None; 3];
        for (unit, setup) in textures.units.iter().enumerate() {
            let Some(setup) = setup else { continue };
            self.surface_clock = self.surface_clock.wrapping_add(1);
            let (texture, fresh) =
                self.texture_for(&setup.texels, setup.width, setup.height, setup.levels)?;
            if fresh {
                unsafe {
                    self.upload_texture(
                        &setup.texels,
                        setup.width,
                        setup.height,
                        setup.levels,
                        texture,
                    )?
                };
            }
            bound[unit] = Some(texture);
        }
        Ok(bound)
    }

    /// the cache entry for these texels, creating it when they are new, keyed on
    /// the texels' allocation: the CPU decoder hands a changed texture out as a
    /// new allocation, so a changed texture uploads again, and one it kept
    /// (checked, unchanged) rides the GL texture it already has
    fn texture_for(
        &mut self,
        texels: &std::sync::Arc<[[u8; 4]]>,
        width: u32,
        height: u32,
        levels: u32,
    ) -> Result<(glow::Texture, bool), String> {
        let key = std::sync::Arc::as_ptr(texels) as *const [u8; 4] as usize;
        if let Some(entry) = self.textures.get_mut(&key) {
            self.surface_clock = self.surface_clock.wrapping_add(1);
            entry.last_used = self.surface_clock;
            return Ok((entry.texture, false));
        }
        // a size each of the whole and the smaller levels, four bytes a texel
        let bytes: u64 = (0..levels)
            .map(|level| u64::from((width >> level) * (height >> level)) * 4)
            .sum();
        self.evict_textures_to_budget(bytes);
        let texture = unsafe {
            self.gl
                .create_texture()
                .map_err(|error| error.to_string())?
        };
        self.surface_clock = self.surface_clock.wrapping_add(1);
        self.texture_bytes = self.texture_bytes.saturating_add(bytes);
        self.textures.insert(
            key,
            CachedTexture {
                texture,
                owner: texels.clone(),
                bytes,
                last_used: self.surface_clock,
            },
        );
        Ok((texture, true))
    }

    unsafe fn upload_texture(
        &self,
        texels: &[[u8; 4]],
        width: u32,
        height: u32,
        levels: u32,
        texture: glow::Texture,
    ) -> Result<(), String> {
        unsafe {
            let gl = &self.gl;
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAX_LEVEL,
                (levels - 1) as i32,
            );
            // a chain of sizes keeps a mipmapped filter so sampling past the
            // base level stays defined, like the draw-time setup keeps
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                if levels > 1 {
                    glow::NEAREST_MIPMAP_NEAREST
                } else {
                    glow::NEAREST
                } as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::NEAREST as i32,
            );
            let mut offset = 0usize;
            for level in 0..levels {
                let (w, h) = (width >> level, height >> level);
                let count = (w * h) as usize;
                let Some(texels) = texels.get(offset..offset + count) else {
                    return Err("a texture's texels do not cover its sizes".to_owned());
                };
                gl.tex_image_2d(
                    glow::TEXTURE_2D,
                    level as i32,
                    glow::RGBA8 as i32,
                    w as i32,
                    h as i32,
                    0,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(Some(bytemuck::cast_slice(texels))),
                );
                offset += count;
            }
        }
        Ok(())
    }

    fn evict_textures_to_budget(&mut self, incoming_bytes: u64) {
        let limit = self.texture_budget.saturating_sub(incoming_bytes);
        while self.texture_bytes > limit {
            let Some(key) = self
                .textures
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| *key)
            else {
                break;
            };
            if let Some(entry) = self.textures.remove(&key) {
                self.texture_bytes = self.texture_bytes.saturating_sub(entry.bytes);
                unsafe {
                    self.gl.delete_texture(entry.texture);
                }
            }
        }
    }

    /// the texture units' and combiners' uniforms for one draw. The combiner
    /// arrays only go over when they changed, most draws keeping the last
    /// draw's environment
    unsafe fn set_texture_uniforms(&self, uniforms: &Uniforms, textures: &TextureSetup, tev_changed: bool) {
        unsafe {
            let gl = &self.gl;
            let enabled: [i32; 3] =
                std::array::from_fn(|unit| textures.units[unit].is_some() as i32);
            if let Some(location) = &uniforms.tex_enabled {
                gl.uniform_1_i32_slice(Some(location), &enabled);
            }
            for (unit, setup) in textures.units.iter().enumerate() {
                let Some(setup) = setup else { continue };
                if let Some(location) = &uniforms.tex_size[unit] {
                    gl.uniform_2_f32_slice(
                        Some(location),
                        &[setup.width as f32, setup.height as f32],
                    );
                }
                if let Some(location) = &uniforms.tex_lod[unit] {
                    gl.uniform_4_f32_slice(Some(location), &setup.lod);
                }
            }
            let wraps: [i32; 3] = std::array::from_fn(|unit| {
                textures.units[unit].as_ref().map_or(0, |u| u.wrap_s as i32)
            });
            if let Some(location) = &uniforms.tex_wrap_s {
                gl.uniform_1_i32_slice(Some(location), &wraps);
            }
            let wraps: [i32; 3] = std::array::from_fn(|unit| {
                textures.units[unit].as_ref().map_or(0, |u| u.wrap_t as i32)
            });
            if let Some(location) = &uniforms.tex_wrap_t {
                gl.uniform_1_i32_slice(Some(location), &wraps);
            }
            let filters: [f32; 6] = std::array::from_fn(|index| {
                let (unit, side) = (index / 2, index % 2);
                let linear = match side {
                    0 => textures.units[unit]
                        .as_ref()
                        .map_or(false, |u| u.mag_linear),
                    _ => textures.units[unit]
                        .as_ref()
                        .map_or(false, |u| u.min_linear),
                };
                linear as i32 as f32
            });
            if let Some(location) = &uniforms.tex_filter {
                gl.uniform_2_f32_slice(Some(location), &filters);
            }
            let borders: [f32; 12] = std::array::from_fn(|index| {
                textures.units[index / 4]
                    .as_ref()
                    .map_or(0.0, |u| u.border[index % 4])
            });
            if let Some(location) = &uniforms.tex_border {
                gl.uniform_4_f32_slice(Some(location), &borders);
            }
            if let Some(location) = &uniforms.tex0_projected {
                gl.uniform_1_i32(Some(location), textures.texture0_projected as i32);
            }
            if !tev_changed {
                return;
            }
            let tev = &textures.tev;
            if let Some(location) = &uniforms.tev_active {
                gl.uniform_1_i32(Some(location), !tev.all_passthrough as i32);
            }
            let sources: [i32; 18] = tev.color_sources.map(|source| source as i32);
            if let Some(location) = &uniforms.tev_color_sources {
                gl.uniform_1_i32_slice(Some(location), &sources);
            }
            let sources: [i32; 18] = tev.alpha_sources.map(|source| source as i32);
            if let Some(location) = &uniforms.tev_alpha_sources {
                gl.uniform_1_i32_slice(Some(location), &sources);
            }
            let operands: [i32; 18] = tev.color_operands.map(|operand| operand as i32);
            if let Some(location) = &uniforms.tev_color_operands {
                gl.uniform_1_i32_slice(Some(location), &operands);
            }
            let operands: [i32; 18] = tev.alpha_operands.map(|operand| operand as i32);
            if let Some(location) = &uniforms.tev_alpha_operands {
                gl.uniform_1_i32_slice(Some(location), &operands);
            }
            let ops: [i32; 6] = tev.color_ops.map(|op| op as i32);
            if let Some(location) = &uniforms.tev_color_ops {
                gl.uniform_1_i32_slice(Some(location), &ops);
            }
            let ops: [i32; 6] = tev.alpha_ops.map(|op| op as i32);
            if let Some(location) = &uniforms.tev_alpha_ops {
                gl.uniform_1_i32_slice(Some(location), &ops);
            }
            let constants: [f32; 24] =
                std::array::from_fn(|index| tev.constants[index / 4][index % 4]);
            if let Some(location) = &uniforms.tev_constants {
                gl.uniform_4_f32_slice(Some(location), &constants);
            }
            if let Some(location) = &uniforms.tev_color_scales {
                gl.uniform_1_f32_slice(Some(location), &tev.color_scales);
            }
            if let Some(location) = &uniforms.tev_alpha_scales {
                gl.uniform_1_f32_slice(Some(location), &tev.alpha_scales);
            }
            let passthrough: [i32; 6] = tev.passthrough.map(|flag| flag as i32);
            if let Some(location) = &uniforms.tev_passthrough {
                gl.uniform_1_i32_slice(Some(location), &passthrough);
            }
            if let Some(location) = &uniforms.tev_update_mask {
                gl.uniform_1_i32(Some(location), tev.update_mask as i32);
            }
            if let Some(location) = &uniforms.tev_buffer_color {
                gl.uniform_4_f32_slice(Some(location), &tev.buffer_color);
            }
        }
    }

    /// Drop cached targets and uploaded textures without resolving their
    /// contents into guest memory.
    pub(crate) fn reset(&mut self) {
        for (_, surface) in self.surfaces.drain() {
            unsafe {
                delete_surface(&self.gl, surface);
            }
        }
        self.surface_bytes = 0;
        for (_, entry) in self.textures.drain() {
            unsafe {
                self.gl.delete_texture(entry.texture);
            }
        }
        self.texture_bytes = 0;
        self.last_tev = None;
        self.last_lighting = None;
        self.last_fog = None;
    }

    /// a smaller cache for the tests, which cannot draw the thousands of
    /// distinct targets the real budget needs to turn the cache over
    #[cfg(test)]
    pub(crate) fn limit_surface_cache(&mut self, bytes: u64) {
        self.surface_budget = bytes;
    }

    /// a smaller texture cache for the tests, which cannot upload the thousands
    /// of distinct textures the real budget needs to turn it over
    #[cfg(test)]
    pub(crate) fn limit_texture_cache(&mut self, bytes: u64) {
        self.texture_budget = bytes;
    }

    /// how many uploaded textures the cache holds, for the eviction tests
    #[cfg(test)]
    pub(crate) fn texture_count(&self) -> usize {
        self.textures.len()
    }

    /// whether a target remains resident and dirty, for differential tests
    #[cfg(test)]
    pub(crate) fn surface_dirty(&self, addr: u32) -> Option<bool> {
        self.surfaces
            .iter()
            .find(|(key, _)| key.addr == addr)
            .map(|(_, surface)| surface.dirty)
    }

    fn evict_to_budget<M: GpuMemory>(&mut self, memory: &mut M, incoming_bytes: u64) {
        let limit = self.surface_budget.saturating_sub(incoming_bytes);
        while self.surface_bytes > limit {
            let Some(key) = self
                .surfaces
                .iter()
                .min_by_key(|(_, surface)| surface.last_used)
                .map(|(key, _)| *key)
            else {
                break;
            };
            let color_bytes = key
                .width
                .saturating_mul(key.height)
                .saturating_mul(color_format(key.format).bytes_per_pixel() as u32);
            self.sync(memory, key.addr, color_bytes, false);
            if key.depth_bytes != 0 {
                let depth_bytes = key
                    .width
                    .saturating_mul(key.height)
                    .saturating_mul(key.depth_bytes);
                self.sync(memory, key.depth_addr, depth_bytes, false);
            }
            if let Some(surface) = self.surfaces.remove(&key) {
                self.surface_bytes = self.surface_bytes.saturating_sub(surface.storage_bytes);
                unsafe {
                    delete_surface(&self.gl, surface);
                }
            }
        }
    }

    /// Resolve GPU-new pixels only when guest memory is about to be observed or overwritten.
    pub(crate) fn sync<M: GpuMemory>(
        &mut self,
        memory: &mut M,
        addr: u32,
        len: u32,
        invalidate: bool,
    ) {
        self.sync_except(memory, addr, len, invalidate, None);
    }

    fn sync_except<M: GpuMemory>(
        &mut self,
        memory: &mut M,
        addr: u32,
        len: u32,
        invalidate: bool,
        except: Option<SurfaceKey>,
    ) {
        let end = addr.saturating_add(len);
        let keys: Vec<_> = self
            .surfaces
            .iter()
            .filter_map(|(key, surface)| {
                if Some(*key) == except {
                    return None;
                }
                let bytes = key
                    .width
                    .saturating_mul(key.height)
                    .saturating_mul(color_format(key.format).bytes_per_pixel() as u32);
                let color_overlap = addr < key.addr.saturating_add(bytes) && key.addr < end;
                let color_hit = surface.dirty && color_overlap;
                let depth_size = key
                    .width
                    .saturating_mul(key.height)
                    .saturating_mul(key.depth_bytes);
                let depth_overlap = key.depth_bytes != 0
                    && addr < key.depth_addr.saturating_add(depth_size)
                    && key.depth_addr < end;
                let depth_hit = surface.depth_dirty && depth_overlap;
                (color_hit || depth_hit || (invalidate && (color_overlap || depth_overlap)))
                    .then_some(*key)
            })
            .collect();
        for key in keys {
            let surface = self.surfaces.get_mut(&key).unwrap();
            let (width, height, format) = (key.width, key.height, color_format(key.format));
            if surface.dirty {
                let mut linear = vec![0u8; (width * height * 4) as usize];
                let mut tiled =
                    vec![0u8; (width * height * format.bytes_per_pixel() as u32) as usize];
                unsafe {
                    self.gl
                        .bind_framebuffer(glow::FRAMEBUFFER, Some(surface.framebuffer));
                    self.gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
                    self.gl.read_pixels(
                        0,
                        0,
                        width as i32,
                        height as i32,
                        glow::RGBA,
                        glow::UNSIGNED_BYTE,
                        glow::PixelPackData::Slice(Some(&mut linear)),
                    );
                    self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                }
                if key.linear {
                    let mut rows = vec![0u8; tiled.len()];
                    linear_to_rows(&linear, &mut rows, width, height, format);
                    memory.write(key.addr, &rows);
                } else {
                    linear_to_tiled(&linear, &mut tiled, width, height, format);
                    memory.write(key.addr, &tiled);
                }
                surface.dirty = false;
            }
            if surface.depth_dirty && key.depth_bytes != 0 {
                let mut depth_linear = vec![0u8; (width * height * key.depth_bytes) as usize];
                unsafe {
                    self.gl
                        .bind_framebuffer(glow::FRAMEBUFFER, Some(surface.framebuffer));
                    if key.depth_bytes == 4 {
                        self.gl.read_pixels(
                            0,
                            0,
                            width as i32,
                            height as i32,
                            glow::DEPTH_STENCIL,
                            glow::UNSIGNED_INT_24_8,
                            glow::PixelPackData::Slice(Some(bytemuck::cast_slice_mut(
                                &mut depth_linear,
                            ))),
                        );
                    } else if key.depth_bytes == 2 {
                        let mut values = vec![0u16; (width * height) as usize];
                        self.gl.read_pixels(
                            0,
                            0,
                            width as i32,
                            height as i32,
                            glow::DEPTH_COMPONENT,
                            glow::UNSIGNED_SHORT,
                            glow::PixelPackData::Slice(Some(bytemuck::cast_slice_mut(&mut values))),
                        );
                        for (i, value) in values.into_iter().enumerate() {
                            depth_linear[i * 2..i * 2 + 2].copy_from_slice(&value.to_le_bytes());
                        }
                    } else {
                        let mut values = vec![0f32; (width * height) as usize];
                        self.gl.read_pixels(
                            0,
                            0,
                            width as i32,
                            height as i32,
                            glow::DEPTH_COMPONENT,
                            glow::FLOAT,
                            glow::PixelPackData::Slice(Some(bytemuck::cast_slice_mut(&mut values))),
                        );
                        for (i, value) in values.into_iter().enumerate() {
                            let raw = (value.clamp(0.0, 1.0) * 16_777_215.0 + 0.5) as u32;
                            depth_linear[i * 3..i * 3 + 3].copy_from_slice(&raw.to_le_bytes()[..3]);
                        }
                    }
                    self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                }
                let mut depth_tiled = vec![0; depth_linear.len()];
                linear_to_tiled_bytes(
                    &depth_linear,
                    &mut depth_tiled,
                    width,
                    height,
                    key.depth_bytes,
                );
                memory.write(key.depth_addr, &depth_tiled);
                surface.depth_dirty = false;
            }
            surface.dirty = false;
            if invalidate {
                surface.initialized = false;
            }
        }
    }

    /// a display transfer out of a buffer the renderer holds (one it drew or an
    /// earlier transfer wrote) done on the GPU, false when the input is anything
    /// else, for the CPU to do. The output stays in a surface the renderer
    /// keeps, guest memory untouched
    pub(crate) fn display_transfer<M: GpuMemory>(
        &mut self,
        memory: &mut M,
        transfer: &Transfer,
    ) -> Result<bool, String> {
        let t = transfer;
        let input_bpp = t.input_format.bytes_per_pixel() as u32;
        let output_bpp = t.output_format.bytes_per_pixel() as u32;
        // a tiled output of part of a tile goes to the CPU, which leaves out the
        // pixels the layout puts past its end, like draws do
        let whole = t.output_width.is_multiple_of(8) && t.output_height.is_multiple_of(8);
        // flipped, an input shorter than the output fills the output's last
        // rows, the CPU puts those where they go
        let short = t.flip && t.copy.1 < t.output_height;
        if t.copy.0 == 0 || t.copy.1 == 0 || (t.output_tiled && !whole) || short {
            return Ok(false);
        }
        let output_key = SurfaceKey {
            addr: t.output,
            format: t.output_format as u32,
            width: t.output_width,
            height: t.output_height,
            depth_addr: 0,
            depth_bytes: 0,
            linear: !t.output_tiled,
        };
        let output_bytes = t
            .output_width
            .saturating_mul(t.output_height)
            .saturating_mul(output_bpp);
        // room for the output before anything else looks the cache over, so
        // the input surface cannot be the victim the output evicts
        if !self.surfaces.contains_key(&output_key) {
            self.evict_to_budget(
                memory,
                surface_storage_bytes(output_key.width, output_key.height, 0),
            );
        }
        // the input rows read, from the first, which can start some rows into a
        // buffer the renderer holds (titles draw both screens into one), a
        // tiled buffer a row of tiles at a time
        let rows = t.copy.1 * t.scale.1;
        let step_rows = if t.input_linear { 1 } else { 8 };
        let step_bytes = step_rows * t.input_width * input_bpp;
        let row_in = |key: &SurfaceKey| -> Option<u32> {
            if !(key.linear == t.input_linear
                && key.format == t.input_format as u32
                && key.width == t.input_width)
            {
                return None;
            }
            let offset = t
                .input
                .checked_sub(key.addr)
                .filter(|offset| offset % step_bytes == 0)?;
            let row = offset / step_bytes * step_rows;
            (row + rows <= key.height).then_some(row)
        };
        let Some((input_key, row)) = self
            .surfaces
            .iter()
            .filter_map(|(key, surface)| row_in(key).map(|row| (row, surface.last_used, *key)))
            .max_by_key(|&(_, used, _)| used)
            .map(|(row, _, key)| (key, row))
        else {
            return Ok(false);
        };
        // rows read where the output goes, the CPU does in order; the rest of
        // the input's buffer can be under the output, and the output drawn
        // over those is the overdrawn below
        let row_bytes = u64::from(t.input_width * input_bpp);
        let read_start = u64::from(input_key.addr) + u64::from(row) * row_bytes;
        let read_end = read_start + u64::from(rows) * row_bytes;
        let output_end = u64::from(t.output) + u64::from(output_bytes);
        if read_start < output_end && u64::from(t.output) < read_end {
            return Ok(false);
        }
        // the input's rows not on the GPU yet upload from guest memory, which
        // has whatever overlapping surfaces hold resolved first
        let input_bytes = input_key
            .width
            .saturating_mul(input_key.height)
            .saturating_mul(input_bpp);
        self.surface_clock = self.surface_clock.wrapping_add(1);
        let input_uploaded = {
            let surface = self.surfaces.get_mut(&input_key).expect("the input was found");
            surface.last_used = self.surface_clock;
            let texture = surface.color;
            (surface.initialized, texture)
        };
        if !input_uploaded.0 {
            self.sync_except(memory, input_key.addr, input_bytes, true, Some(input_key));
            let mut guest = vec![0u8; input_bytes as usize];
            memory.read(input_key.addr, &mut guest);
            let linear = if input_key.linear {
                rows_to_linear(&guest, input_key.width, input_key.height, t.input_format)
            } else {
                tiled_to_linear(&guest, input_key.width, input_key.height, t.input_format)
            };
            unsafe {
                self.gl.bind_texture(glow::TEXTURE_2D, Some(input_uploaded.1));
                self.gl.tex_sub_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    0,
                    0,
                    input_key.width as i32,
                    input_key.height as i32,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(Some(&linear)),
                );
            }
            self.surfaces
                .get_mut(&input_key)
                .expect("the input was found")
                .initialized = true;
        }
        // the output surface, made fresh or kept, its rows not on the GPU
        // uploaded like the input's
        if !self.surfaces.contains_key(&output_key) {
            self.sync(memory, output_key.addr, output_bytes, true);
            let (framebuffer, color, depth) =
                unsafe { create_surface(&self.gl, output_key.width, output_key.height, 0)? };
            self.surface_clock = self.surface_clock.wrapping_add(1);
            let storage_bytes = surface_storage_bytes(output_key.width, output_key.height, 0);
            self.surfaces.insert(
                output_key,
                Surface {
                    framebuffer,
                    color,
                    depth,
                    initialized: false,
                    dirty: false,
                    depth_dirty: false,
                    storage_bytes,
                    last_used: self.surface_clock,
                    picture: 0,
                    upright: None,
                },
            );
            self.surface_bytes = self.surface_bytes.saturating_add(storage_bytes);
        }
        self.surface_clock = self.surface_clock.wrapping_add(1);
        let output_ready = {
            let surface = self.surfaces.get_mut(&output_key).expect("made or found above");
            surface.last_used = self.surface_clock;
            (surface.initialized, surface.framebuffer)
        };
        if !output_ready.0 {
            self.sync_except(memory, output_key.addr, output_bytes, true, Some(output_key));
            let mut guest = vec![0u8; output_bytes as usize];
            memory.read(output_key.addr, &mut guest);
            let linear = if output_key.linear {
                rows_to_linear(&guest, output_key.width, output_key.height, t.output_format)
            } else {
                tiled_to_linear(&guest, output_key.width, output_key.height, t.output_format)
            };
            unsafe {
                let color = self
                    .surfaces
                    .get(&output_key)
                    .expect("made or found above")
                    .color;
                self.gl.bind_texture(glow::TEXTURE_2D, Some(color));
                self.gl.tex_sub_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    0,
                    0,
                    output_key.width as i32,
                    output_key.height as i32,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(Some(&linear)),
                );
            }
            self.surfaces
                .get_mut(&output_key)
                .expect("made or found above")
                .initialized = true;
        }
        // the conversion drawn, every size and row the CPU's own engine writes
        unsafe {
            let gl = &self.gl;
            let uniform =
                |name: &str| gl.get_uniform_location(self.transfer_program, name);
            gl.use_program(Some(self.transfer_program));
            gl.bind_vertex_array(Some(self.utility_vertex_array));
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(output_ready.1));
            if gl.check_framebuffer_status(glow::FRAMEBUFFER) != glow::FRAMEBUFFER_COMPLETE {
                gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                gl.bind_vertex_array(None);
                return Err("OpenGL transfer framebuffer is incomplete".to_owned());
            }
            gl.viewport(0, 0, output_key.width as i32, output_key.height as i32);
            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::STENCIL_TEST);
            gl.disable(glow::BLEND);
            gl.disable(glow::CULL_FACE);
            gl.disable(glow::DITHER);
            gl.depth_mask(true);
            gl.stencil_mask(0xFF);
            gl.color_mask(true, true, true, true);
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(input_uploaded.1));
            if let Some(location) = uniform("copy_size") {
                gl.uniform_2_i32(Some(&location), t.copy.0 as i32, t.copy.1 as i32);
            }
            if let Some(location) = uniform("out_size") {
                gl.uniform_2_i32(
                    Some(&location),
                    output_key.width as i32,
                    output_key.height as i32,
                );
            }
            if let Some(location) = uniform("block") {
                gl.uniform_2_i32(Some(&location), t.scale.0 as i32, t.scale.1 as i32);
            }
            if let Some(location) = uniform("flip") {
                gl.uniform_1_i32(Some(&location), t.flip as i32);
            }
            if let Some(location) = uniform("out_tiled") {
                gl.uniform_1_i32(Some(&location), t.output_tiled as i32);
            }
            if let Some(location) = uniform("in_tiled") {
                gl.uniform_1_i32(Some(&location), (!t.input_linear) as i32);
            }
            if let Some(location) = uniform("input_row") {
                gl.uniform_1_i32(Some(&location), row as i32);
            }
            if let Some(location) = uniform("input_height") {
                gl.uniform_1_i32(Some(&location), input_key.height as i32);
            }
            if let Some(location) = uniform("alpha_forced") {
                let keeps_no_alpha = matches!(
                    t.output_format,
                    crate::format::ColorFormat::Rgb8 | crate::format::ColorFormat::Rgb565
                );
                gl.uniform_1_i32(Some(&location), keeps_no_alpha as i32);
            }
            gl.draw_arrays(glow::TRIANGLES, 0, 3);
            gl.bind_vertex_array(None);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
        let surface = self
            .surfaces
            .get_mut(&output_key)
            .expect("made or found above");
        surface.initialized = true;
        surface.dirty = true;
        surface.picture = surface.picture.wrapping_add(1);
        memory.guard(output_key.addr, output_bytes);
        memory.guard_writes(output_key.addr, output_bytes);
        Ok(true)
    }

    /// the newest picture a display transfer left in a buffer for a screen,
    /// while the renderer still holds it and guest memory has not been asked
    /// for it yet
    pub(crate) fn screen(
        &mut self,
        addr: u32,
        (row_pixels, rows): (u32, u32),
        stride: u32,
        format: crate::format::ColorFormat,
        guest: &[u8],
    ) -> Option<crate::ScreenRef> {
        // unlike the Vulkan renderer's, the GL one has no shadow of what it
        // wrote to compare the guest bytes against: the picture shows only
        // while the surface is dirty, the renderer still holding it
        let _ = guest;
        let row_bytes = u64::from(stride) * format.bytes_per_pixel() as u64;
        let (key, surface) = self.surfaces.iter().find(|(key, surface)| {
            key.linear
                && key.format == format as u32
                && key.width == stride
                && stride >= row_pixels
                && addr >= key.addr
                && u64::from(addr - key.addr) % row_bytes == 0
                && u64::from(addr - key.addr) / row_bytes + u64::from(rows) <= u64::from(key.height)
                && surface.dirty
        })?;
        let first = (u64::from(addr - key.addr) / row_bytes) as u32;
        Some(crate::ScreenRef {
            addr: key.addr,
            size: (key.width, key.height),
            format,
            batch: surface.picture,
            rows: (first, rows),
            columns: row_pixels,
        })
    }

    /// the image a screen's picture is upright in, for a presenter sharing the
    /// context, none once a later picture took its buffer
    pub(crate) fn screen_image(
        &mut self,
        screen: crate::ScreenRef,
    ) -> Result<Option<crate::GpuScreen>, String> {
        match self.upright(screen) {
            Some((texture, width, height)) => Ok(Some(crate::GpuScreen {
                #[cfg(feature = "vulkan")]
                view: ash::vk::ImageView::null(),
                gl_texture: Some(texture),
                area: [0.0, 0.0, 1.0, 1.0],
                bounds: [
                    0.5 / width as f32,
                    0.5 / height as f32,
                    1.0 - 0.5 / width as f32,
                    1.0 - 0.5 / height as f32,
                ],
            })),
            None => Ok(None),
        }
    }

    /// a screen's picture upright, RGBA like the screen shows it, read back
    /// from the GPU, none once a later picture took its buffer
    pub(crate) fn upright_picture(
        &mut self,
        screen: crate::ScreenRef,
        between: bool,
    ) -> Option<crate::Picture> {
        // no pictures in between draws here, the Vulkan path's interpolation
        if between {
            return None;
        }
        let (texture, width, height) = self.upright(screen)?;
        let mut pixels = vec![0u8; (width * height * 4) as usize];
        unsafe {
            self.gl.active_texture(glow::TEXTURE0);
            self.gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            self.gl.get_tex_image(
                glow::TEXTURE_2D,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(&mut pixels)),
            );
        }
        Some((std::sync::Arc::new(pixels), 1))
    }

    /// the upright picture of a screen's buffer, drawn into the surface's own
    /// texture when the picture is not the one already there
    fn upright(&mut self, screen: crate::ScreenRef) -> Option<(glow::Texture, u32, u32)> {
        let (row_pixels, rows_count) = screen.size;
        let (width, height) = (screen.rows.1, screen.columns);
        let (key, color, ready, kept) = {
            let key = *self.surfaces.keys().find(|key| {
                key.linear
                    && key.addr == screen.addr
                    && key.width == row_pixels
                    && key.height == rows_count
                    && key.format == screen.format as u32
            })?;
            let surface = self.surfaces.get_mut(&key)?;
            if surface.picture != screen.batch {
                return None;
            }
            let ready = surface.upright.as_ref().is_some_and(|upright| {
                upright.picture == screen.batch && upright.width == width && upright.height == height
            });
            let kept = surface
                .upright
                .as_ref()
                .map(|upright| (upright.texture, upright.width, upright.height));
            (key, surface.color, ready, kept)
        };
        if ready {
            let (texture, width, height) = kept.expect("ready means one is kept");
            return Some((texture, width, height));
        }
        // the texture the picture renders into, kept when its size fits
        let made = match kept {
            Some((texture, kept_width, kept_height)) if kept_width == width && kept_height == height => {
                texture
            }
            Some((texture, _, _)) => {
                unsafe {
                    self.gl.delete_texture(texture);
                    self.gl
                        .create_texture()
                        .expect("a texture for the upright picture")
                }
            }
            None => unsafe {
                self.gl
                    .create_texture()
                    .expect("a texture for the upright picture")
            },
        };
        unsafe {
            let gl = &self.gl;
            let previous_draw = gl.get_parameter_framebuffer(glow::DRAW_FRAMEBUFFER_BINDING);
            let previous_read = gl.get_parameter_framebuffer(glow::READ_FRAMEBUFFER_BINDING);
            gl.bind_texture(glow::TEXTURE_2D, Some(made));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA8 as i32,
                width as i32,
                height as i32,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(None),
            );
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.utility_framebuffer));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(made),
                0,
            );
            gl.viewport(0, 0, width as i32, height as i32);
            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::STENCIL_TEST);
            gl.disable(glow::BLEND);
            gl.disable(glow::CULL_FACE);
            gl.disable(glow::DITHER);
            gl.color_mask(true, true, true, true);
            gl.use_program(Some(self.upright_program));
            gl.bind_vertex_array(Some(self.utility_vertex_array));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(color));
            if let Some(location) = gl.get_uniform_location(self.upright_program, "column_pixels") {
                gl.uniform_1_i32(Some(&location), row_pixels as i32);
            }
            if let Some(location) = gl.get_uniform_location(self.upright_program, "first_column") {
                gl.uniform_1_i32(Some(&location), screen.rows.0 as i32);
            }
            gl.draw_arrays(glow::TRIANGLES, 0, 3);
            gl.bind_vertex_array(None);
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, previous_draw);
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, previous_read);
        }
        let surface = self.surfaces.get_mut(&key)?;
        let picture = surface.picture;
        surface.upright = Some(Upright { texture: made, picture, width, height });
        Some((made, width, height))
    }
}

fn surface_storage_bytes(width: u32, height: u32, depth_bytes: u32) -> u64 {
    let depth_storage = match depth_bytes {
        0 => 0,
        2 => 2,
        _ => 4,
    };
    u64::from(width)
        .saturating_mul(u64::from(height))
        .saturating_mul(4 + depth_storage)
}

unsafe fn create_surface(
    gl: &glow::Context,
    width: u32,
    height: u32,
    depth_bytes: u32,
) -> Result<(glow::Framebuffer, glow::Texture, Option<glow::Texture>), String> {
    unsafe {
        let previous_draw = gl.get_parameter_framebuffer(glow::DRAW_FRAMEBUFFER_BINDING);
        let previous_read = gl.get_parameter_framebuffer(glow::READ_FRAMEBUFFER_BINDING);
        let mut framebuffer = None;
        let mut color = None;
        let mut depth = None;
        let result = (|| {
            let made_framebuffer = gl.create_framebuffer().map_err(|e| e.to_string())?;
            framebuffer = Some(made_framebuffer);
            let made_color = gl.create_texture().map_err(|e| e.to_string())?;
            color = Some(made_color);
            gl.bind_texture(glow::TEXTURE_2D, Some(made_color));
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA8 as i32,
                width as i32,
                height as i32,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(None),
            );
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(made_framebuffer));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(made_color),
                0,
            );
            if depth_bytes != 0 {
                let made_depth = gl.create_texture().map_err(|e| e.to_string())?;
                depth = Some(made_depth);
                gl.bind_texture(glow::TEXTURE_2D, Some(made_depth));
                gl.tex_parameter_i32(
                    glow::TEXTURE_2D,
                    glow::TEXTURE_MIN_FILTER,
                    glow::NEAREST as i32,
                );
                gl.tex_parameter_i32(
                    glow::TEXTURE_2D,
                    glow::TEXTURE_MAG_FILTER,
                    glow::NEAREST as i32,
                );
                let internal = if depth_bytes == 2 {
                    glow::DEPTH_COMPONENT16
                } else if depth_bytes == 4 {
                    glow::DEPTH24_STENCIL8
                } else {
                    glow::DEPTH_COMPONENT24
                };
                gl.tex_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    internal as i32,
                    width as i32,
                    height as i32,
                    0,
                    if depth_bytes == 4 {
                        glow::DEPTH_STENCIL
                    } else {
                        glow::DEPTH_COMPONENT
                    },
                    if depth_bytes == 2 {
                        glow::UNSIGNED_SHORT
                    } else if depth_bytes == 4 {
                        glow::UNSIGNED_INT_24_8
                    } else {
                        glow::FLOAT
                    },
                    glow::PixelUnpackData::Slice(None),
                );
                gl.framebuffer_texture_2d(
                    glow::FRAMEBUFFER,
                    if depth_bytes == 4 {
                        glow::DEPTH_STENCIL_ATTACHMENT
                    } else {
                        glow::DEPTH_ATTACHMENT
                    },
                    glow::TEXTURE_2D,
                    Some(made_depth),
                    0,
                );
            }
            Ok(())
        })();
        gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, previous_draw);
        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, previous_read);
        if let Err(error) = result {
            if let Some(framebuffer) = framebuffer {
                gl.delete_framebuffer(framebuffer);
            }
            if let Some(color) = color {
                gl.delete_texture(color);
            }
            if let Some(depth) = depth {
                gl.delete_texture(depth);
            }
            return Err(error);
        }
        Ok((
            framebuffer.expect("created framebuffer"),
            color.expect("created color texture"),
            depth,
        ))
    }
}

unsafe fn delete_surface(gl: &glow::Context, surface: Surface) {
    unsafe {
        gl.delete_framebuffer(surface.framebuffer);
        gl.delete_texture(surface.color);
        if let Some(depth) = surface.depth {
            gl.delete_texture(depth);
        }
        if let Some(upright) = surface.upright {
            gl.delete_texture(upright.texture);
        }
    }
}

unsafe fn link_program(
    gl: &glow::Context,
    vertex_source: &str,
    fragment_source: &str,
) -> Result<glow::Program, String> {
    unsafe {
        let program = gl.create_program().map_err(|error| error.to_string())?;
        let mut shaders = Vec::with_capacity(2);
        let result = (|| {
            for (kind, source) in [
                (glow::VERTEX_SHADER, vertex_source),
                (glow::FRAGMENT_SHADER, fragment_source),
            ] {
                let shader = gl.create_shader(kind).map_err(|error| error.to_string())?;
                shaders.push(shader);
                gl.shader_source(shader, source);
                gl.compile_shader(shader);
                if !gl.get_shader_compile_status(shader) {
                    return Err(format!(
                        "OpenGL shader failed to compile: {}",
                        gl.get_shader_info_log(shader)
                    ));
                }
                gl.attach_shader(program, shader);
            }
            gl.link_program(program);
            if !gl.get_program_link_status(program) {
                return Err(format!(
                    "OpenGL program failed to link: {}",
                    gl.get_program_info_log(program)
                ));
            }
            Ok(())
        })();
        if let Err(error) = result {
            gl.delete_program(program);
            for shader in shaders {
                gl.delete_shader(shader);
            }
            return Err(error);
        }
        for shader in shaders {
            gl.detach_shader(program, shader);
            gl.delete_shader(shader);
        }
        Ok(program)
    }
}

fn compare(value: super::Compare) -> u32 {
    match value {
        super::Compare::Never => glow::NEVER,
        super::Compare::Always => glow::ALWAYS,
        super::Compare::Equal => glow::EQUAL,
        super::Compare::NotEqual => glow::NOTEQUAL,
        super::Compare::Less => glow::LESS,
        super::Compare::LessOrEqual => glow::LEQUAL,
        super::Compare::Greater => glow::GREATER,
        super::Compare::GreaterOrEqual => glow::GEQUAL,
    }
}

/// the shader's wrap numbers onto GL's, for the sampler state the shader's own
/// texel fetching rides above
fn wrap_sampler_mode(value: u32) -> u32 {
    match value {
        1 => glow::CLAMP_TO_BORDER,
        2 => glow::REPEAT,
        3 => glow::MIRRORED_REPEAT,
        _ => glow::CLAMP_TO_EDGE,
    }
}

fn stencil_op(value: super::StencilOp) -> u32 {
    match value {
        super::StencilOp::Keep => glow::KEEP,
        super::StencilOp::Zero => glow::ZERO,
        super::StencilOp::Replace => glow::REPLACE,
        super::StencilOp::IncrementSaturate => glow::INCR,
        super::StencilOp::DecrementSaturate => glow::DECR,
        super::StencilOp::Invert => glow::INVERT,
        super::StencilOp::IncrementWrap => glow::INCR_WRAP,
        super::StencilOp::DecrementWrap => glow::DECR_WRAP,
    }
}

/// Convert swizzled PICA depth pixels to linear GL upload bytes. D24S8 needs the
/// packed GL order (24 depth bits followed by 8 stencil bits).
fn depth_to_linear(tiled: &[u8], width: u32, height: u32, bytes: u32) -> Vec<u8> {
    let gl_bytes = if bytes == 2 { 2 } else { 4 };
    let mut linear = vec![0; (width * height * gl_bytes) as usize];
    for y in 0..height {
        for x in 0..width {
            let from = morton_offset(x, height - 1 - y, width, bytes) as usize;
            let to = ((y * width + x) * gl_bytes) as usize;
            match bytes {
                2 => {
                    let value = u16::from_le_bytes(tiled[from..from + 2].try_into().unwrap());
                    linear[to..to + 2].copy_from_slice(&value.to_ne_bytes());
                }
                3 => {
                    let value =
                        u32::from_le_bytes([tiled[from], tiled[from + 1], tiled[from + 2], 0]);
                    linear[to..to + 4]
                        .copy_from_slice(&((value as f32 / 16_777_215.0).to_ne_bytes()));
                }
                _ => {
                    let value = u32::from_le_bytes(tiled[from..from + 4].try_into().unwrap());
                    let packed = ((value & 0x00FF_FFFF) << 8) | (value >> 24);
                    linear[to..to + 4].copy_from_slice(&packed.to_ne_bytes());
                }
            }
        }
    }
    linear
}

fn linear_to_tiled_bytes(linear: &[u8], tiled: &mut [u8], width: u32, height: u32, bytes: u32) {
    for y in 0..height {
        for x in 0..width {
            let from = ((y * width + x) * bytes) as usize;
            let to = morton_offset(x, height - 1 - y, width, bytes) as usize;
            if bytes == 4 {
                let packed = u32::from_ne_bytes(linear[from..from + 4].try_into().unwrap());
                let guest = (packed >> 8) | ((packed & 0xFF) << 24);
                tiled[to..to + 4].copy_from_slice(&guest.to_le_bytes());
            } else {
                tiled[to..to + bytes as usize]
                    .copy_from_slice(&linear[from..from + bytes as usize]);
            }
        }
    }
}

/// the color format a surface key's raw number holds, the enum's own numbering
/// the key stores, not the registers'
fn color_format(raw: u32) -> crate::format::ColorFormat {
    match raw {
        1 => crate::format::ColorFormat::Rgb8,
        2 => crate::format::ColorFormat::Rgb565,
        3 => crate::format::ColorFormat::Rgb5A1,
        4 => crate::format::ColorFormat::Rgba4,
        _ => crate::format::ColorFormat::Rgba8,
    }
}

/// a surface's tiled guest pixels decode into linear RGBA rows for the upload
fn tiled_to_linear(
    tiled: &[u8],
    width: u32,
    height: u32,
    format: crate::format::ColorFormat,
) -> Vec<u8> {
    let bytes_per_pixel = format.bytes_per_pixel();
    let mut linear = vec![0; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let source = morton_offset(x, height - 1 - y, width, bytes_per_pixel as u32) as usize;
            let color = format.decode(&tiled[source..source + bytes_per_pixel]);
            let destination = ((y * width + x) * 4) as usize;
            linear[destination..destination + 4].copy_from_slice(&color);
        }
    }
    linear
}

/// a linear buffer's guest rows decode into linear RGBA rows for the upload,
/// the rows kept in memory order like the buffer lays them
fn rows_to_linear(
    guest: &[u8],
    width: u32,
    height: u32,
    format: crate::format::ColorFormat,
) -> Vec<u8> {
    let bytes_per_pixel = format.bytes_per_pixel() as usize;
    let mut linear = vec![0; (width * height * 4) as usize];
    for pixel in 0..(width * height) as usize {
        let source = pixel * bytes_per_pixel;
        let color = format.decode(&guest[source..source + bytes_per_pixel]);
        linear[pixel * 4..(pixel + 1) * 4].copy_from_slice(&color);
    }
    linear
}

/// linear RGBA rows pack back into a linear buffer's guest rows
fn linear_to_rows(
    linear: &[u8],
    guest: &mut [u8],
    width: u32,
    height: u32,
    format: crate::format::ColorFormat,
) {
    let bytes_per_pixel = format.bytes_per_pixel() as usize;
    for pixel in 0..(width * height) as usize {
        let source = pixel * 4;
        let mut encoded = [0; 4];
        format.encode(
            linear[source..source + 4].try_into().expect("RGBA pixel"),
            &mut encoded,
        );
        guest[pixel * bytes_per_pixel..(pixel + 1) * bytes_per_pixel]
            .copy_from_slice(&encoded[..bytes_per_pixel]);
    }
}

/// linear RGBA rows pack back into the surface's tiled guest bytes
fn linear_to_tiled(
    linear: &[u8],
    tiled: &mut [u8],
    width: u32,
    height: u32,
    format: crate::format::ColorFormat,
) {
    let bytes_per_pixel = format.bytes_per_pixel();
    for y in 0..height {
        for x in 0..width {
            let source = ((y * width + x) * 4) as usize;
            let mut encoded = [0; 4];
            format.encode(
                linear[source..source + 4].try_into().expect("RGBA pixel"),
                &mut encoded,
            );
            let destination =
                morton_offset(x, height - 1 - y, width, bytes_per_pixel as u32) as usize;
            tiled[destination..destination + bytes_per_pixel]
                .copy_from_slice(&encoded[..bytes_per_pixel]);
        }
    }
}

fn version_at_least(version: &str, major: u32, minor: u32) -> bool {
    let mut parts = version.split('.');
    let Some(found_major) = parts.next().and_then(|part| part.parse::<u32>().ok()) else {
        return false;
    };
    let Some(found_minor) = parts.next().and_then(|part| {
        part.split(|character: char| !character.is_ascii_digit())
            .next()
            .and_then(|part| part.parse::<u32>().ok())
    }) else {
        return false;
    };
    (found_major, found_minor) >= (major, minor)
}

#[cfg(test)]
mod tests {
    use super::{
        color_format, depth_to_linear, linear_to_tiled, linear_to_tiled_bytes, tiled_to_linear,
        version_at_least,
    };
    use crate::format::ColorFormat;

    #[test]
    fn recognizes_required_opengl_versions() {
        assert!(version_at_least("4.3.0 Intel", 4, 3));
        assert!(version_at_least("4.6 (Core Profile)", 4, 3));
        assert!(!version_at_least("4.2.0", 4, 3));
        assert!(!version_at_least("OpenGL ES 3.2", 4, 3));
    }

    /// the surface key holds the enum's own numbers, not the registers'
    #[test]
    fn surface_keys_number_formats_like_the_enum() {
        assert_eq!(color_format(ColorFormat::Rgba8 as u32), ColorFormat::Rgba8);
        assert_eq!(color_format(ColorFormat::Rgb8 as u32), ColorFormat::Rgb8);
        assert_eq!(
            color_format(ColorFormat::Rgb565 as u32),
            ColorFormat::Rgb565
        );
        assert_eq!(
            color_format(ColorFormat::Rgb5A1 as u32),
            ColorFormat::Rgb5A1
        );
        assert_eq!(color_format(ColorFormat::Rgba4 as u32), ColorFormat::Rgba4);
    }

    #[test]
    fn tiled_rgba8_round_trips_through_linear_gl_rows() {
        for (width, height) in [(8, 8), (16, 8), (16, 16)] {
            let length = (width * height * 4) as usize;
            let tiled: Vec<u8> = (0..length).map(|i| (i * 37) as u8).collect();
            let linear = tiled_to_linear(&tiled, width, height, ColorFormat::Rgba8);
            let mut restored = vec![0; length];
            linear_to_tiled(&linear, &mut restored, width, height, ColorFormat::Rgba8);
            assert_eq!(restored, tiled);
        }
    }

    /// an RGB8 target packs 3 bytes a texel into its tiles and unpacks the same
    /// texels back out, through the RGBA rows the GL side holds them in
    #[test]
    fn tiled_rgb8_round_trips_through_linear_gl_rows() {
        for (width, height) in [(8, 8), (16, 8), (24, 16)] {
            let length = (width * height * 3) as usize;
            let tiled: Vec<u8> = (0..length).map(|i| (i * 61 + 9) as u8).collect();
            let linear = tiled_to_linear(&tiled, width, height, ColorFormat::Rgb8);
            let mut restored = vec![0; length];
            linear_to_tiled(&linear, &mut restored, width, height, ColorFormat::Rgb8);
            assert_eq!(restored, tiled);
        }
        // and a decoded pixel is the format's own red, green and blue, opaque
        let tiled = [0x11, 0x22, 0x33];
        let linear = tiled_to_linear(&tiled, 1, 1, ColorFormat::Rgb8);
        assert_eq!(&linear[..4], &[0x33, 0x22, 0x11, 0xFF]);
    }

    /// the 16 bit targets pack 2 bytes a texel into their tiles and unpack the
    /// same texels back out, thru the RGBA rows the GL side holds them in, so
    /// what a draw leaves resolves into the guest's own bytes
    #[test]
    fn tiled_16bit_targets_round_trip_through_linear_gl_rows() {
        for format in [ColorFormat::Rgb565, ColorFormat::Rgb5A1, ColorFormat::Rgba4] {
            for (width, height) in [(8, 8), (16, 8), (24, 16)] {
                let length = (width * height * 2) as usize;
                let tiled: Vec<u8> = (0..length).map(|i| (i * 61 + 9) as u8).collect();
                let linear = tiled_to_linear(&tiled, width, height, format);
                let mut restored = vec![0; length];
                linear_to_tiled(&linear, &mut restored, width, height, format);
                assert_eq!(restored, tiled, "{format:?}");
            }
        }
    }

    /// what each 16 bit format's decode answers, the bytes the alpha channel of
    /// the GL copy holds for a target of it
    #[test]
    fn decoded_16bit_pixels_keep_their_own_precision() {
        // RGB565 keeps no alpha, its decode answers opaque
        let linear = tiled_to_linear(&[0xFF, 0xFF], 1, 1, ColorFormat::Rgb565);
        assert_eq!(&linear[..4], &[0xFF, 0xFF, 0xFF, 0xFF]);
        // RGB5A1 keeps a bit of it
        let linear = tiled_to_linear(&[0xFF, 0xFF], 1, 1, ColorFormat::Rgb5A1);
        assert_eq!(&linear[..4], &[0xFF, 0xFF, 0xFF, 0xFF]);
        let linear = tiled_to_linear(&[0xFE, 0xFF], 1, 1, ColorFormat::Rgb5A1);
        assert_eq!(&linear[3..4], &[0x00]);
        // RGBA4 keeps four bits, a stored 4 comes back as 4 times 17
        let linear = tiled_to_linear(&[0x44, 0x44], 1, 1, ColorFormat::Rgba4);
        assert_eq!(&linear[..4], &[0x44, 0x44, 0x44, 0x44]);
        // and a round trip thru the copy's precision lands on the stored value
        for format in [ColorFormat::Rgb565, ColorFormat::Rgb5A1, ColorFormat::Rgba4] {
            for value in [[0x00u8, 0x00], [0xFF, 0xFF], [0x21, 0x64], [0xBD, 0x10]] {
                let linear = tiled_to_linear(&value, 1, 1, format);
                let mut encoded = [0; 4];
                format.encode(linear[..4].try_into().expect("RGBA pixel"), &mut encoded);
                assert_eq!(&encoded[..2], &value, "{format:?}");
            }
        }
    }

    #[test]
    fn tiled_depth16_round_trips() {
        let (width, height) = (16, 8);
        let mut tiled = vec![0; (width * height * 2) as usize];
        for i in 0..width * height {
            tiled[(i * 2) as usize..(i * 2 + 2) as usize]
                .copy_from_slice(&((i * 997) as u16).to_le_bytes());
        }
        let linear = depth_to_linear(&tiled, width, height, 2);
        let mut restored = vec![0; tiled.len()];
        linear_to_tiled_bytes(&linear, &mut restored, width, height, 2);
        assert_eq!(restored, tiled);
    }

    #[test]
    fn tiled_depth24_stencil_round_trips() {
        let (width, height) = (16, 8);
        let mut tiled = vec![0; (width * height * 4) as usize];
        for i in 0..width * height {
            let value: u32 = ((i * 1237) & 0x00FF_FFFF) | ((i * 17 & 0xFF) << 24);
            tiled[(i * 4) as usize..(i * 4 + 4) as usize].copy_from_slice(&value.to_le_bytes());
        }
        let linear = depth_to_linear(&tiled, width, height, 4);
        let mut restored = vec![0; tiled.len()];
        linear_to_tiled_bytes(&linear, &mut restored, width, height, 4);
        assert_eq!(restored, tiled);
    }

    /// a D24S8 word reaches GL in its packed order, the 24 depth bits low and the
    /// stencil byte in the top, and the stencil comes back exactly as it went in
    #[test]
    fn depth24_stencil_reaches_gl_in_its_word_order() {
        use crate::format::morton_offset;
        let (width, height) = (8, 8);
        let mut tiled = vec![0; (width * height * 4) as usize];
        let mut sent = vec![0u32; (width * height) as usize];
        // the guest bytes in tile order, one depth and one stencil per pixel
        for y in 0..height {
            for x in 0..width {
                let depth = ((y * width + x) * 7137) & 0x00FF_FFFF;
                let stencil = ((y * width + x) * 31) & 0xFF;
                sent[(y * width + x) as usize] = (stencil << 24) | depth;
                let at = morton_offset(x, height - 1 - y, width, 4) as usize;
                tiled[at..at + 4].copy_from_slice(&((stencil << 24) | depth).to_le_bytes());
            }
        }
        let linear = depth_to_linear(&tiled, width, height, 4);
        for (i, word) in linear.as_chunks::<4>().0.iter().enumerate() {
            let packed = u32::from_ne_bytes(*word);
            let word = sent[i];
            assert_eq!(packed, (word << 8) | (word >> 24), "sample {i} is not in GL's packed order");
        }
    }

    /// D24 rides to GL as a float, which holds every 24 bit number but rounds the
    /// divisions; fed back thru the readback's conversion the samples land within
    /// one least significant bit, exactly for the ones a float splits evenly
    #[test]
    fn depth24_survives_the_float_round_trip() {
        use crate::format::morton_offset;
        let (width, height) = (16, 8);
        let mut tiled = vec![0; (width * height * 3) as usize];
        let mut sent = vec![0u32; (width * height) as usize];
        for y in 0..height {
            for x in 0..width {
                let value = (((y * width + x) * 4099) & 0x00FF_FFFF) as u32;
                sent[(y * width + x) as usize] = value;
                let at = morton_offset(x, height - 1 - y, width, 3) as usize;
                tiled[at..at + 3].copy_from_slice(&value.to_le_bytes()[..3]);
            }
        }
        // the upload, then the readback exactly as sync() does it
        let linear = depth_to_linear(&tiled, width, height, 3);
        for (i, word) in linear.as_chunks::<4>().0.iter().enumerate() {
            assert_eq!(
                f32::from_ne_bytes(*word).to_bits(),
                (sent[i] as f32 / 16_777_215.0).to_bits(),
                "sample {i} uploaded as a different float",
            );
            let back = (f32::from_ne_bytes(*word).clamp(0.0, 1.0) * 16_777_215.0 + 0.5) as u32;
            let off = back.abs_diff(sent[i]);
            assert!(
                off <= 1,
                "sample {} came back {off} steps off ({} against {})",
                i,
                sent[i],
                back,
            );
        }
        // the halfway-friendly samples, powers of two, come back exact
        for &value in &[0x800000u32, 0x400000, 0x200000, 1, 0x00FF_FFFF - 1] {
            let tiled = [(value & 0xFF) as u8, ((value >> 8) & 0xFF) as u8, ((value >> 16) & 0xFF) as u8];
            let linear = depth_to_linear(&tiled, 1, 1, 3);
            assert_eq!(linear.len(), 4, "one D24 sample uploads as a float word");
            let back =
                (f32::from_ne_bytes(*linear.as_chunks::<4>().0.first().unwrap()) * 16_777_215.0 + 0.5)
                    as u32;
            assert_eq!(back, value, "{value:#x} did not survive the float round trip exactly");
        }
    }
}
