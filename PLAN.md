# OpenGL rasterizer plan

## Current state

The MVP requires an OpenGL 4.3 context and accelerates draws that use RGBA8 targets with tile-aligned dimensions and no textures, lighting, fog, or logic operation. The fragment shader carries the alpha test, and the output merger's blending and partial color masks run in GL's fixed function, for every equation and factor pairing GL takes (a saturate factor on the destination side, which GL refuses, stays on the CPU). It supports depth/stencil tests and writes for D16, D24, and D24S8 attachments. The texture environment must be a passthrough. Zakuro still runs the vertex shader, clipping, and culling on the CPU; OpenGL rasterizes the resulting triangles and interpolates vertex color. Other draws use the existing CPU rasterizer.

The OpenGL path keeps keyed RGBA8/depth targets resident across draws. It synchronizes overlapping guest-memory ranges lazily through the existing GPU sync hooks. Depth/stencil formats D16, D24, and D24S8 are supported when the rest of the draw state is in the accelerated subset.

The differential test suite in `crates/zakuro-gpu/src/raster/opengl_differential.rs` runs the CPU and OpenGL backends on identical guest memory and compares the bytes each leaves behind, with explicit tolerances where float math and driver rounding legitimately differ (see TODO.md for coverage and findings). The suite covers the output merger — the alpha test, blending, partial color masks and their combination over depth and stencil writes, both refused and passed by the alpha — and cache eviction, which it exercises by turning the surface budget down. It needs an OpenGL 4.3 context: on headless Linux it builds one through GLX on the software rasterizer (`xvfb-run -a cargo test -p zakuro-gpu --features opengl`), and where no context can be made the GL-dependent tests report why and pass while the conversion unit tests run everywhere.

## Implementation sequence

1. **Done.** Focused comparison tests for OpenGL coverage, interpolation, culling, scissoring, tiled-buffer conversion, depth/stencil state, depth mapping, lazy FBO caching and synchronization, and CPU-fallback equivalence run against the CPU path. Coverage and deferred cases are listed in TODO.md.
2. Add common texture formats and sampling. Reuse the existing texture decoder, then add mip level selection and PICA wrap/filter behavior.
3. **Partly done.** The alpha test, blending, and color masks are accelerated and pinned by the differential suite; the fallback tests cover what GL cannot take (logic ops, a destination-side saturate factor). The TEV stages remain, with the CPU fallback kept for any unsupported state until each case is verified and extended per feature as it lands.
4. Move PICA vertex shader execution to OpenGL and handle geometry-shader draws through CPU preparation.
5. Add lighting, fog, procedural textures, and remaining framebuffer formats.
6. **Done as part of the MVP.** Keyed OpenGL color/depth targets already stay resident across draws, dirty GPU attachments are tracked, and synchronization resolves overlapping fallback draws, CPU reads/writes (the sync hooks), fills, and display transfers. The differential suite covers each of those paths; finer dirty-rectangle tracking to cut the writeback cost stays open in TODO.md.
7. Add resolution scaling, pipeline/resource caching, texture-pack support on the GPU path, and assess direct presentation and frame interpolation separately.
8. Validate the complete path on OpenGL 4.3 hardware across supported operating systems; retain software fallback for unsupported drivers and unimplemented states.
