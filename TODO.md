# OpenGL rasterizer TODO

The current MVP only rasterizes untextured, vertex-colored triangles. Unsupported draws use Zakuro's complete CPU rasterizer.

Not implemented in the OpenGL path:

- PICA vertex shaders and geometry shaders on the GPU; vertex processing, clipping, and culling still run on the CPU.
- Texture sampling, mipmaps, texture replacement, and non-passthrough texture-environment (TEV) combiners.
- Fragment lighting, procedural textures, and fog.
- Logic operations, which stay on the CPU: the OpenGL binding does not wrap glLogicOp, so the GL path has no way to set one. OpenGL depth/stencil still supports D16, D24, and D24S8 only when the color path is otherwise supported.
- Framebuffer formats other than RGBA8 and dimensions that are not multiples of 8.
- Resolution scaling, direct GPU presentation, and frame interpolation.
- Full-range synchronization is used for an overlapping target; finer dirty tile/rectangle tracking could reduce writeback cost for sparse draws.
- The render-target cache is bounded by its 128 MiB budget with least-recently-used eviction, each victim resolved into guest memory before its GL objects are destroyed, so long sessions with many distinct targets stay bounded; only destruction itself is untested on real hardware.
- Direct GPU display transfer/presentation is not implemented, so transfers and scanout still synchronize to guest memory.
- Partial overlap and aliasing between differently sized color/depth views remain conservative: synchronization resolves full cached attachments, and unusual overlapping views should be covered by additional integration cases.
- OpenGL depth/stencil readback depends on driver support for the selected packed depth formats; unsupported/incomplete FBOs fall back to the CPU path.

# Differential test coverage

`crates/zakuro-gpu/src/raster/opengl_differential.rs` runs the CPU and OpenGL backends on identical guest memory and compares what each leaves behind. Covered:

- Rasterization: pixel coverage over random slanted triangles snapped to sixteenths, perspective-correct vertex-color interpolation, winding and cull-mode behavior, viewport/scissor rects including ones hanging off the buffer.
- Depth: all eight compare functions on D16, D24, and D24S8 against a stored sample, with and without depth writes; written depth agrees within a rounding step, and a D24S8 depth write preserves the stencil byte it shares the word with.
- Stencil: every stencil operation (keep, zero, replace, increment/decrement, invert, wrapping) in all three outcomes (stencil fail, depth fail, both passing), with partial compare and write masks, checked against the byte the state alone dictates.
- Depth mapping: z-buffer and w-buffer modes with scales and offsets that land in range and ones that clamp, over a slanted triangle with varying w.
- Output merger: the alpha test's eight compare functions over a flat alpha that passes and refuses as the reference dictates, plus one over a slanted triangle whose alpha slides across the reference, where the backends may split only on the reference's own byte; every blend equation and factor pairing GL takes (standard transparency, additive, subtract and reverse subtract, min and max, constant factors, the source alpha saturate, a separate alpha equation), the blended alpha channel included; partial color masks over a base quad, the masked channels holding what was under them, including a draw that may write nothing at all; and the merger whole — blending, an alpha test and a partial mask over a depth testing, stencil writing draw — once refused and once passed by the alpha.
- Cache eviction: the tests turn the surface budget down to a couple of targets (`limit_surface_cache`), the third distinct target resolves the first into guest memory with the bytes the CPU's run holds, and a later draw to it uploads what was resolved instead of pretending it stayed resident.
- Lazy FBO caching: consecutive draws leave the target resident (one upload, no readbacks, CPU reads/writes guarded); explicit sync and the `sync_memory`/`sync_depth` hooks resolve and invalidate; memory fills and display transfers take the drawn pixels first; overlapping and aliased color/depth targets resolve each other; unsupported states (logic ops, a texture drawn into, a non-RGBA8 target, a blend factor GL lacks) fall back after syncing, byte-identical to a CPU-only run.
- Full-path: a draw through a real command list (vertex shader, attribute arrays, `process_command_list`) with the OpenGL renderer attached matches the CPU run, and its display transfer converts the same pixels.
- Tiled conversions (pure unit tests in `raster/opengl.rs`, run everywhere): RGBA8, D16 and D24S8 round trips, the D24S8 packed GL word order with the stencil byte exact, and the D24 float round trip within one least significant bit.

Running them: the GL-dependent tests need an OpenGL 4.3 context. Headless Linux makes one through GLX on Mesa's software rasterizer — `xvfb-run -a cargo test -p zakuro-gpu --features opengl`. Without a display or driver they print the reason and pass; the conversion unit tests run everywhere. The CI's headless runners take the skip path.

Findings and deferred cases:

- Fragments testing exactly at a stored depth sample are deliberately left out: the CPU compares the fragment's float against the float of the stored word while the GL quantizes the fragment into the buffer first, so at the sample itself the two can answer interpolated pixels' float noise differently (observed up to half a D24 target). The tests keep an eight-LSB margin where the compare functions' answers are unambiguous. Games relying on exact-equal depth (shadow volumes) may see the GL path judge boundary fragments differently than the CPU did.
- The CPU rasterizer itself panics filling a target whose width is not a whole number of tiles: the spill tile of the last row of tiles is cut off the copy in `Surface::load` and `Band::at` indexes past it. Titles only draw into tile-aligned buffers so it never shows, and the GL path skips unaligned dimensions anyway; the ignored `cpu_draws_survive_an_unaligned_target_width` regression test is ready when it is fixed.
- A depth-only sync (`Gpu::sync_depth`) also writes back the color surface of any cached key it touches, because a selected key resolves color and depth together. Conservative rather than wrong; finer per-plane dirty tracking could avoid the extra writeback.
- Cache eviction is covered with a test-only budget (`limit_surface_cache` in the OpenGL rasterizer turns the 128 MiB down to a couple of targets); the real budget runs the same code at a different number, and exercising it for real would still need thousands of distinct targets.
- Where a missing feature lands, the fallback tests extend to it, as they did when the alpha test, blending and partial color masks accelerated: the draws they once used to pin the fallback became the acceleration differentials.
- Logic operations are the one output-merger piece still handed to the CPU: glow does not wrap glLogicOp, and OpenGL 4.3 offers no other way to read the destination in place, so a draw in logic-op mode falls back after syncing (pinned by the fallback test) until the binding grows the call.

# Next work

See [PLAN.md](PLAN.md) for the current implementation plan.
