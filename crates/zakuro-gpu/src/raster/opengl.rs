//! a small OpenGL rasterizer for vertex-colored, textured triangles running the
//! texture environment's combiners in one uber-shader

use glow::HasContext;
use std::collections::HashMap;

use super::{ColorTarget, DepthMap, DepthStencil, Screen};
use crate::{format::morton_offset, GpuMemory};

const VERTEX_SHADER: &str = r#"#version 330 core
layout(location = 0) in vec4 packed_a; // position over w, inverse w, z
layout(location = 1) in vec4 packed_b; // color
layout(location = 2) in vec4 packed_c; // texture coords of sets 0 and 1
layout(location = 3) in vec4 packed_d; // texture coord set 2, set 0's w
out vec4 vertex_color;
 noperspective out float pica_depth;
 noperspective out float pica_inverse_w;
out vec2 uv0;
out vec2 uv1;
out vec2 uv2;
out float q0;
void main() {
    gl_Position = vec4(packed_a.xy, 0.0, 1.0 / packed_a.z);
    vertex_color = packed_b;
    pica_depth = packed_a.w;
    pica_inverse_w = packed_a.z;
    uv0 = packed_c.xy;
    uv1 = packed_c.zw;
    uv2 = packed_d.xy;
    q0 = packed_d.z;
}
"#;

/// one fragment shader for every draw the OpenGL path takes: the texture
/// environment's six combiner stages and the three texture units run from
/// uniforms, so no draw ever waits on a shader compile. The combiner math and
/// the texel wrapping follow the software rasterizer's exactly — the wrap modes
/// wrap texel indices, not GL's normalized coordinates, and the level of
/// detail comes from the rate the coordinates change across the screen like
/// `BoundTexture::sample_at` works it
const FRAGMENT_SHADER: &str = r#"#version 330 core
in vec4 vertex_color;
 noperspective in float pica_depth;
 noperspective in float pica_inverse_w;
in vec2 uv0;
in vec2 uv1;
in vec2 uv2;
in float q0;
uniform float depth_scale;
uniform float depth_offset;
uniform bool w_buffer;
uniform bool alpha_test;
uniform int alpha_function;
uniform int alpha_reference;
// the texture units: samplers are named one by one (GLSL 330 cannot index an
// array of them dynamically), the rest of each unit's configuration is an array
uniform int tex_enabled[3];
uniform sampler2D tex_image0;
uniform sampler2D tex_image1;
uniform sampler2D tex_image2;
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
out vec4 color;
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
int wrap_coordinate(int mode, int coordinate, int size) {
    if (mode == 2) {
        int wrapped = coordinate % size;
        return wrapped < 0 ? wrapped + size : wrapped;
    }
    if (mode == 3) {
        int period = coordinate % (2 * size);
        if (period < 0) period += 2 * size;
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
    vec2 per_step_x = dFdx(coord) * tex_size[unit];
    vec2 per_step_y = dFdy(coord) * tex_size[unit];
    vec4 lod = tex_lod[unit];
    float level = 0.5 * log2(max(dot(per_step_x, per_step_x), dot(per_step_y, per_step_y)));
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
/// fragment lighting colors no draw that reaches this shader turns on
vec4 pick(int source, vec4 primary, vec4 t0, vec4 t1, vec4 t2, vec4 buffer, vec4 constant, vec4 previous) {
    if (source == 0) return primary;
    if (source == 1) return t0;
    if (source == 2) return t1;
    if (source == 3) return t2;
    if (source == 4) return vec4(0.0, 0.0, 0.0, 1.0);
    if (source == 5) return buffer;
    if (source == 6) return constant;
    if (source == 8) return primary;
    if (source == 9) return vec4(0.0, 0.0, 0.0, 1.0);
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
vec4 tev_combine(vec4 primary, vec4 t0, vec4 t1, vec4 t2) {
    vec4 previous = primary;
    vec4 buffer = vec4(0.0);
    vec4 next_buffer = tev_buffer_color;
    for (int i = 0; i < 6; i++) {
        if (tev_passthrough[i] == 0) {
            vec4 c0 = pick(tev_color_sources[i * 3 + 0], primary, t0, t1, t2, buffer, tev_constants[i], previous);
            vec4 c1 = pick(tev_color_sources[i * 3 + 1], primary, t0, t1, t2, buffer, tev_constants[i], previous);
            vec4 c2 = pick(tev_color_sources[i * 3 + 2], primary, t0, t1, t2, buffer, tev_constants[i], previous);
            vec4 a0 = pick(tev_alpha_sources[i * 3 + 0], primary, t0, t1, t2, buffer, tev_constants[i], previous);
            vec4 a1 = pick(tev_alpha_sources[i * 3 + 1], primary, t0, t1, t2, buffer, tev_constants[i], previous);
            vec4 a2 = pick(tev_alpha_sources[i * 3 + 2], primary, t0, t1, t2, buffer, tev_constants[i], previous);
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
        buffer = next_buffer;
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
    if (tev_active == 1) {
        vec4 t0 = vec4(0.0, 0.0, 0.0, 1.0);
        vec4 t1 = t0;
        vec4 t2 = t0;
        if (tex_enabled[0] == 1) {
            // a projected texture divides set 0's coords by its own w
            t0 = sample_unit(tex_image0, 0, tex0_projected == 1 ? uv0 / q0 : uv0);
        }
        if (tex_enabled[1] == 1) t1 = sample_unit(tex_image1, 1, uv1);
        if (tex_enabled[2] == 1) t2 = sample_unit(tex_image2, 2, uv2);
        result = tev_combine(primary, t0, t1, t2);
    }
    // the color leaves as whole bytes, like the software path truncates, so the
    // alpha test judges the same byte a fallback draw would
    vec4 quantized = floor(clamp(result, 0.0, 1.0) * 255.0);
    if (alpha_test && !alpha_passes(int(quantized.a), alpha_reference)) discard;
    color = quantized / 255.0;
}
"#;

const SURFACE_CACHE_BUDGET: u64 = 128 * 1024 * 1024;
/// decoded textures stay resident under this, about as much as the decoded
/// texel cache the CPU keeps draws from
const TEXTURE_CACHE_BUDGET: u64 = 64 * 1024 * 1024;

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

/// OpenGL resources are created and used on the thread where the context is current.
pub(crate) struct OpenGlRasterizer {
    gl: std::sync::Arc<glow::Context>,
    program: glow::Program,
    uniforms: Uniforms,
    vertex_array: glow::VertexArray,
    vertices: glow::Buffer,
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
            self.gl.delete_vertex_array(self.vertex_array);
            self.gl.delete_buffer(self.vertices);
            self.gl.delete_program(self.program);
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
            let program = link_program(&gl)?;
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

            // four whole vec4s a vertex: position over w, inverse w, z, the color,
            // then two texture coordinate sets and set 0's w
            gl.bind_vertex_array(Some(vertex_array));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vertices));
            let stride = (16 * size_of::<f32>()) as i32;
            for attribute in 0..4 {
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

            // the samplers ride units 0 to 2, set once after the link
            gl.use_program(Some(program));
            for unit in 0..3 {
                if let Some(location) =
                    gl.get_uniform_location(program, &format!("tex_image{unit}"))
                {
                    gl.uniform_1_i32(Some(&location), unit as i32);
                }
            }
            let uniforms = Uniforms {
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
            };

            Ok(Self {
                gl,
                program,
                uniforms,
                vertex_array,
                vertices,
                surfaces: HashMap::new(),
                surface_bytes: 0,
                surface_clock: 0,
                surface_budget: SURFACE_CACHE_BUDGET,
                textures: HashMap::new(),
                texture_bytes: 0,
                texture_budget: TEXTURE_CACHE_BUDGET,
                last_tev: None,
            })
        }
    }

    pub(super) fn draw<M: GpuMemory>(
        &mut self,
        memory: &mut M,
        target: &ColorTarget,
        depth: Option<&DepthStencil>,
        depth_map: DepthMap,
        output: &OutputState,
        textures: &TextureSetup,
        triangles: &[[Screen; 3]],
    ) -> Result<(), String> {
        if triangles.is_empty() || target.right <= target.left || target.top <= target.bottom {
            return Ok(());
        }
        let (width, height) = (target.buffer_width, target.buffer_height);
        let pixels = width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(4))
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or("OpenGL target dimensions overflow")?;
        let key = SurfaceKey {
            addr: target.addr,
            format: target.format as u32,
            width,
            height,
            depth_addr: depth.map_or(0, |d| d.addr),
            depth_bytes: depth.map_or(0, |d| d.bytes),
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
            tiled_to_linear(&tiled, width, height)
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

        let mut vertices = Vec::with_capacity(triangles.len() * 3 * 16);
        let set2 = if textures.texture2_uses_coord1 { 1 } else { 2 };
        for triangle in triangles {
            for vertex in triangle {
                let inverse_w = vertex.inv_w;
                vertices.extend([
                    (2.0 * vertex.x / width as f32 - 1.0) / inverse_w,
                    (2.0 * vertex.y / height as f32 - 1.0) / inverse_w,
                    inverse_w,
                    vertex.z,
                ]);
                vertices.extend(vertex.color_over_w.map(|component| component / inverse_w));
                let [u0, v0] = vertex.texcoords_over_w[0];
                let [u1, v1] = vertex.texcoords_over_w[1];
                let [u2, v2] = vertex.texcoords_over_w[set2];
                vertices.extend([
                    u0 / inverse_w,
                    v0 / inverse_w,
                    u1 / inverse_w,
                    v1 / inverse_w,
                    u2 / inverse_w,
                    v2 / inverse_w,
                    vertex.texcoord0_w_over_w / inverse_w,
                    0.0,
                ]);
            }
        }
        let vertex_bytes = bytemuck::cast_slice(&vertices);

        // the textures go into GL before the draw, each uploaded once for as
        // long as the CPU decoder hands out the same texels
        let bound = self.bind_textures(textures)?;
        let tev_changed = self.last_tev.as_ref() != Some(&textures.tev);

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
                gl.color_mask(
                    output.write[0],
                    output.write[1],
                    output.write[2],
                    output.write[3],
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
                gl.use_program(Some(self.program));
                gl.bind_vertex_array(Some(self.vertex_array));
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vertices));
                gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, vertex_bytes, glow::STREAM_DRAW);
                if let Some(location) = &self.uniforms.depth_scale {
                    gl.uniform_1_f32(Some(location), depth_map.scale);
                }
                if let Some(location) = &self.uniforms.depth_offset {
                    gl.uniform_1_f32(Some(location), depth_map.offset);
                }
                if let Some(location) = &self.uniforms.w_buffer {
                    gl.uniform_1_i32(Some(location), depth_map.w_buffer as i32);
                }
                if let Some(location) = &self.uniforms.alpha_test {
                    gl.uniform_1_i32(Some(location), output.alpha_test.is_some() as i32);
                }
                if let Some(location) = &self.uniforms.alpha_function {
                    gl.uniform_1_i32(
                        Some(location),
                        output.alpha_test.map_or(1, |(function, _)| function as i32),
                    );
                }
                if let Some(location) = &self.uniforms.alpha_reference {
                    gl.uniform_1_i32(
                        Some(location),
                        output
                            .alpha_test
                            .map_or(0, |(_, reference)| reference as i32),
                    );
                }
                self.set_texture_uniforms(textures, tev_changed);
                for (unit, texture) in bound.iter().enumerate() {
                    let Some(texture) = texture else { continue };
                    let Some(setup) = textures.units[unit].as_ref() else {
                        continue;
                    };
                    gl.active_texture(glow::TEXTURE0 + unit as u32);
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
                        if setup.min_linear {
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
                gl.draw_arrays(glow::TRIANGLES, 0, (vertices.len() / 16) as i32);
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
        memory.guard(target.addr, (width * height * 4) as u32);
        memory.guard_writes(target.addr, (width * height * 4) as u32);
        if let Some(depth) = depth {
            let bytes = width.saturating_mul(height).saturating_mul(depth.bytes);
            memory.guard(depth.addr, bytes);
            memory.guard_writes(depth.addr, bytes);
        }
        self.evict_to_budget(memory, 0);
        Ok(())
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
    unsafe fn set_texture_uniforms(&self, textures: &TextureSetup, tev_changed: bool) {
        unsafe {
            let gl = &self.gl;
            let uniforms = &self.uniforms;
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
            let color_bytes = key.width.saturating_mul(key.height).saturating_mul(4);
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
                let bytes = key.width.saturating_mul(key.height).saturating_mul(4);
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
            let (width, height) = (key.width, key.height);
            if surface.dirty {
                let mut linear = vec![0u8; (width * height * 4) as usize];
                let mut tiled = vec![0u8; linear.len()];
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
                linear_to_tiled(&linear, &mut tiled, width, height);
                memory.write(key.addr, &tiled);
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
    }
}

unsafe fn link_program(gl: &glow::Context) -> Result<glow::Program, String> {
    unsafe {
        let program = gl.create_program().map_err(|error| error.to_string())?;
        let mut shaders = Vec::with_capacity(2);
        let result = (|| {
            for (kind, source) in [
                (glow::VERTEX_SHADER, VERTEX_SHADER),
                (glow::FRAGMENT_SHADER, FRAGMENT_SHADER),
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

fn tiled_to_linear(tiled: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut linear = vec![0; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let source = morton_offset(x, height - 1 - y, width, 4) as usize;
            let color = crate::format::ColorFormat::Rgba8.decode(&tiled[source..source + 4]);
            let destination = ((y * width + x) * 4) as usize;
            linear[destination..destination + 4].copy_from_slice(&color);
        }
    }
    linear
}

fn linear_to_tiled(linear: &[u8], tiled: &mut [u8], width: u32, height: u32) {
    for y in 0..height {
        for x in 0..width {
            let source = ((y * width + x) * 4) as usize;
            let mut encoded = [0; 4];
            crate::format::ColorFormat::Rgba8.encode(
                linear[source..source + 4].try_into().expect("RGBA pixel"),
                &mut encoded,
            );
            let destination = morton_offset(x, height - 1 - y, width, 4) as usize;
            tiled[destination..destination + 4].copy_from_slice(&encoded);
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
        depth_to_linear, linear_to_tiled, linear_to_tiled_bytes, tiled_to_linear, version_at_least,
    };

    #[test]
    fn recognizes_required_opengl_versions() {
        assert!(version_at_least("4.3.0 Intel", 4, 3));
        assert!(version_at_least("4.6 (Core Profile)", 4, 3));
        assert!(!version_at_least("4.2.0", 4, 3));
        assert!(!version_at_least("OpenGL ES 3.2", 4, 3));
    }

    #[test]
    fn tiled_rgba8_round_trips_through_linear_gl_rows() {
        for (width, height) in [(8, 8), (16, 8), (16, 16)] {
            let length = (width * height * 4) as usize;
            let tiled: Vec<u8> = (0..length).map(|i| (i * 37) as u8).collect();
            let linear = tiled_to_linear(&tiled, width, height);
            let mut restored = vec![0; length];
            linear_to_tiled(&linear, &mut restored, width, height);
            assert_eq!(restored, tiled);
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
