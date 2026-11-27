# OpenGL rasterizer TODO

The current MVP only rasterizes untextured, vertex-colored triangles. Unsupported draws use Zakuro's complete CPU rasterizer.

Not implemented in the OpenGL path:

- PICA vertex shaders and geometry shaders on the GPU; vertex processing, clipping, and culling still run on the CPU.
- Texture sampling, mipmaps, texture replacement, and non-passthrough texture-environment (TEV) combiners.
- Fragment lighting, procedural textures, fog, and alpha testing.
- Blending, logic operations, and partial color masks; OpenGL depth/stencil currently supports D16, D24, and D24S8 only when the color path is otherwise supported.
- Framebuffer formats other than RGBA8 and dimensions that are not multiples of 8.
- Resolution scaling, direct GPU presentation, and frame interpolation.
- Full-range synchronization is used for an overlapping target; finer dirty tile/rectangle tracking could reduce writeback cost for sparse draws.
- OpenGL render-target cache eviction and GL object destruction are not yet bounded; long sessions with many distinct targets can retain resources.
- Direct GPU display transfer/presentation is not implemented, so transfers and scanout still synchronize to guest memory.
- Partial overlap and aliasing between differently sized color/depth views remain conservative: synchronization resolves full cached attachments, and unusual overlapping views should be covered by additional integration cases.
- OpenGL depth/stencil readback depends on driver support for the selected packed depth formats; unsupported/incomplete FBOs fall back to the CPU path.

# Differential test coverage

`crates/zakuro-gpu/src/raster/opengl_differential.rs` runs the CPU and OpenGL backends on identical guest memory and compares what each leaves behind. Covered:

- Rasterization: pixel coverage over random slanted triangles snapped to sixteenths, perspective-correct vertex-color interpolation, winding and cull-mode behavior, viewport/scissor rects including ones hanging off the buffer.
- Depth: all eight compare functions on D16, D24, and D24S8 against a stored sample, with and without depth writes; written depth agrees within a rounding step, and a D24S8 depth write preserves the stencil byte it shares the word with.
- Stencil: every stencil operation (keep, zero, replace, increment/decrement, invert, wrapping) in all three outcomes (stencil fail, depth fail, both passing), with partial compare and write masks, checked against the byte the state alone dictates.
- Depth mapping: z-buffer and w-buffer modes with scales and offsets that land in range and ones that clamp, over a slanted triangle with varying w.
- Lazy FBO caching: consecutive draws leave the target resident (one upload, no readbacks, CPU reads/writes guarded); explicit sync and the `sync_memory`/`sync_depth` hooks resolve and invalidate; memory fills and display transfers take the drawn pixels first; overlapping and aliased color/depth targets resolve each other; unsupported states (blending, logic ops, alpha test, a texture drawn into, a non-RGBA8 target, partial color masks) fall back after syncing, byte-identical to a CPU-only run.
- Full-path: a draw through a real command list (vertex shader, attribute arrays, `process_command_list`) with the OpenGL renderer attached matches the CPU run, and its display transfer converts the same pixels.
- Tiled conversions (pure unit tests in `raster/opengl.rs`, run everywhere): RGBA8, D16 and D24S8 round trips, the D24S8 packed GL word order with the stencil byte exact, and the D24 float round trip within one least significant bit.

Running them: the GL-dependent tests need an OpenGL 4.3 context. Headless Linux makes one through GLX on Mesa's software rasterizer — `xvfb-run -a cargo test -p zakuro-gpu --features opengl`. Without a display or driver they print the reason and pass; the conversion unit tests run everywhere. The CI's headless runners take the skip path.

Findings and deferred cases:

- Fragments testing exactly at a stored depth sample are deliberately left out: the CPU compares the fragment's float against the float of the stored word while the GL quantizes the fragment into the buffer first, so at the sample itself the two can answer interpolated pixels' float noise differently (observed up to half a D24 target). The tests keep an eight-LSB margin where the compare functions' answers are unambiguous. Games relying on exact-equal depth (shadow volumes) may see the GL path judge boundary fragments differently than the CPU did.
- The CPU rasterizer itself panics filling a target whose width is not a whole number of tiles: the spill tile of the last row of tiles is cut off the copy in `Surface::load` and `Band::at` indexes past it. Titles only draw into tile-aligned buffers so it never shows, and the GL path skips unaligned dimensions anyway; the ignored `cpu_draws_survive_an_unaligned_target_width` regression test is ready when it is fixed.
- A depth-only sync (`Gpu::sync_depth`) also writes back the color surface of any cached key it touches, because a selected key resolves color and depth together. Conservative rather than wrong; finer per-plane dirty tracking could avoid the extra writeback.
- Cache eviction under the 128 MiB surface budget is not covered by tests — exercising it needs thousands of distinct targets — so eviction under pressure stays a manual concern until a smaller budget hook exists for tests.
- Where a missing feature lands, the fallback tests extend to it; the ignored `opengl_partial_color_masks_accelerate_once_supported` marks the spot for partial color masks.

# Next work

See [PLAN.md](PLAN.md) for the current implementation plan.
