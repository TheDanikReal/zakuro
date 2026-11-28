//! differential tests: the OpenGL 4.3 rasterizer against Zakuro's CPU rasterizer
//!
//! Both backends run thru the same `rasterize` entry (or the renderer's own `draw`
//! for the synchronization cases) on identical guest memory, and the tests compare
//! the bytes each leaves behind. Where the two may legitimately differ — slanted
//! edges, float interpolation, driver depth rounding — the test says so explicitly
//! and bounds the difference instead of demanding bit equality.
//!
//! The GL side needs an OpenGL 4.3 context. Where none can be made (no display, no
//! software GL), the GL-dependent tests report why and pass; the conversion unit
//! tests in `opengl.rs` keep running everywhere.
//!
//! On a headless machine the context comes from GLX on Mesa's llvmpipe:
//! `xvfb-run -a cargo test -p zakuro-gpu --features opengl`

use super::tests::{rasterize_shaded, rasterize_unshaded};
use crate::shader::INPUT_REGISTERS;
use glow::HasContext;
use crate::format::morton_offset;
use crate::raster::*;
use crate::{Gpu, GpuMemory, SharedOpenGlContext, SoftwareRenderer};

/// the inverse of `decode_float24`, safe for values below one, which the
/// shared helper's unsigned arithmetic cannot express
fn float24(value: f32) -> u32 {
    if value == 0.0 {
        return 0;
    }
    let bits = value.to_bits();
    let exponent = (((bits >> 23) & 0xFF) as i32 - 127 + 63) as u32;
    ((bits >> 31) << 23) | (exponent << 16) | ((bits >> 7) & 0xFFFF)
}

// ---------------------------------------------------------------------------
// the offscreen context
// ---------------------------------------------------------------------------

/// an offscreen OpenGL 4.3 core context on whatever X display `$DISPLAY` names,
/// GLX's pbuffer on Mesa's software rasterizer under `xvfb-run`
#[cfg(target_os = "linux")]
struct GlxSession {
    xlib: x11_dl::xlib::Xlib,
    glx: x11_dl::glx::Glx,
    display: *mut x11_dl::xlib::Display,
    context: x11_dl::glx::GLXContext,
    pbuffer: x11_dl::glx::GLXPbuffer,
}

// the session lives and dies wholly on the one test thread that made it
#[cfg(target_os = "linux")]
unsafe impl Send for GlxSession {}

#[cfg(target_os = "linux")]
impl Drop for GlxSession {
    fn drop(&mut self) {
        unsafe {
            (self.glx.glXDestroyPbuffer)(self.display, self.pbuffer);
            (self.glx.glXDestroyContext)(self.display, self.context);
            (self.xlib.XCloseDisplay)(self.display);
        }
    }
}

/// the context the GL-dependent tests share, none where the machine has no GL
fn software_opengl() -> Option<std::sync::Arc<glow::Context>> {
    #[cfg(target_os = "linux")]
    match try_software_opengl() {
        Ok(context) => Some(context),
        Err(reason) => {
            println!("skipping the OpenGL differential tests: {reason}");
            println!(
                "the CPU-only tests still ran; for these, install Mesa and run \
                 `xvfb-run -a cargo test -p zakuro-gpu --features opengl`"
            );
            None
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        println!(
            "skipping the OpenGL differential tests: no headless context here yet, \
             they run on Linux thru GLX (xvfb-run)"
        );
        None
    }
}

/// one GL context per test thread, the tests on it taking fresh rasterizers off it
fn shared_opengl() -> Option<std::sync::Arc<glow::Context>> {
    thread_local! {
        static CONTEXT: Option<std::sync::Arc<glow::Context>> = software_opengl();
    }
    CONTEXT.with(|context| context.clone())
}

/// the OpenGL resources for one test, none when there is no context to draw with
fn opengl_resources() -> Option<Resources> {
    let context = SharedOpenGlContext(shared_opengl()?);
    let renderer = unsafe { opengl::OpenGlRasterizer::new(context) };
    match renderer {
        Ok(renderer) => Some(Resources {
            opengl: Some(renderer),
            ..Default::default()
        }),
        // the context exists, so this is a real rasterizer failure, not a
        // machine that cannot run these tests: the tests fail on it instead of
        // passing untested
        Err(error) => panic!("the OpenGL rasterizer could not start: {error}"),
    }
}

#[cfg(target_os = "linux")]
fn try_software_opengl() -> Result<std::sync::Arc<glow::Context>, String> {
    use std::os::raw::c_int;
    use x11_dl::glx::{self, arb, Glx};
    use x11_dl::xlib::Xlib;

    let xlib = Xlib::open().map_err(|error| format!("libX11 could not be loaded ({error})"))?;
    let glx = Glx::open().map_err(|error| format!("libGL could not be loaded ({error})"))?;
    unsafe {
        let display = (xlib.XOpenDisplay)(std::ptr::null());
        if display.is_null() {
            return Err("no X display (headless runs want `xvfb-run -a`)".to_owned());
        }
        let mut session = GlxSession {
            xlib,
            glx,
            display,
            context: std::ptr::null_mut(),
            pbuffer: 0,
        };

        // a config that can back a pbuffer, RGBA8 w/ a 24/8 depth stencil
        let config_attribs: [c_int; 19] = [
            glx::GLX_DOUBLEBUFFER,
            0,
            glx::GLX_DRAWABLE_TYPE,
            glx::GLX_PBUFFER_BIT,
            glx::GLX_RENDER_TYPE,
            glx::GLX_RGBA_BIT,
            glx::GLX_RED_SIZE,
            8,
            glx::GLX_GREEN_SIZE,
            8,
            glx::GLX_BLUE_SIZE,
            8,
            glx::GLX_ALPHA_SIZE,
            8,
            glx::GLX_DEPTH_SIZE,
            24,
            glx::GLX_STENCIL_SIZE,
            8,
            0,
        ];
        let mut count: c_int = 0;
        let configs =
            (session.glx.glXChooseFBConfig)(display, 0, config_attribs.as_ptr(), &mut count);
        if configs.is_null() || count == 0 {
            return Err("GLX has no pbuffer-capable RGBA config".to_owned());
        }
        let config = *configs;

        // 4.3 core, what the rasterizer asks for
        type CreateContextAttribs = unsafe extern "C" fn(
            *mut x11_dl::xlib::Display,
            glx::GLXFBConfig,
            glx::GLXContext,
            c_int,
            *const c_int,
        ) -> glx::GLXContext;
        // the byte string reads clearer than a CString dance, clippy's lint notwithstanding
        #[allow(clippy::manual_c_str_literals)]
        let create: Option<CreateContextAttribs> = std::mem::transmute(
            (session.glx.glXGetProcAddress)(b"glXCreateContextAttribsARB\0".as_ptr())
                .map(|function| function as *const ())
                .unwrap_or(std::ptr::null()),
        );
        let context_attribs: [c_int; 7] = [
            arb::GLX_CONTEXT_MAJOR_VERSION_ARB,
            4,
            arb::GLX_CONTEXT_MINOR_VERSION_ARB,
            3,
            arb::GLX_CONTEXT_PROFILE_MASK_ARB,
            arb::GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
            0,
        ];
        if let Some(create) = create {
            session.context = create(
                display,
                config,
                std::ptr::null_mut(),
                1,
                context_attribs.as_ptr(),
            );
        }
        if session.context.is_null() {
            return Err("GLX would not give an OpenGL 4.3 core context".to_owned());
        }

        let pbuffer_attribs: [c_int; 5] =
            [glx::GLX_PBUFFER_WIDTH, 64, glx::GLX_PBUFFER_HEIGHT, 64, 0];
        session.pbuffer = (session.glx.glXCreatePbuffer)(display, config, pbuffer_attribs.as_ptr());
        if session.pbuffer == 0 {
            return Err("GLX would not make the offscreen pbuffer".to_owned());
        }
        if (session.glx.glXMakeContextCurrent)(
            display,
            session.pbuffer,
            session.pbuffer,
            session.context,
        ) == 0
        {
            return Err("GLX would not make the context current".to_owned());
        }

        let loader = |name: &str| -> *const std::os::raw::c_void {
            let name = std::ffi::CString::new(name).expect("no nul in a GL name");
            (session.glx.glXGetProcAddress)(name.as_ptr() as *const u8)
                .map(|function| function as *const std::os::raw::c_void)
                .unwrap_or(std::ptr::null())
        };
        let context = glow::Context::from_loader_function(loader);
        std::mem::forget(session); // the context lives for the test; the process exits after
        Ok(std::sync::Arc::new(context))
    }
}

// ---------------------------------------------------------------------------
// the memory the tests observe
// ---------------------------------------------------------------------------

/// guest memory that logs every read, write and guard the backends make,
/// so the tests can tell an upload from a readback from a fill
#[derive(Clone)]
struct CountingMemory {
    flat: Vec<u8>,
    reads: Vec<(u32, u32)>,
    writes: Vec<(u32, u32)>,
    guards: Vec<(u32, u32, bool)>,
}

impl CountingMemory {
    fn zeroed(bytes: usize) -> CountingMemory {
        CountingMemory {
            flat: vec![0; bytes],
            reads: Vec::new(),
            writes: Vec::new(),
            guards: Vec::new(),
        }
    }

    /// the same starting bytes for both backends: a color gradient over the target
    /// and a depth/stencil sample per pixel, the stencil in the top byte of a D24S8 word
    fn prefilled(depth_bytes: u32, depth_pattern: impl Fn(u32) -> u32) -> CountingMemory {
        let mut memory = CountingMemory::zeroed(0x1_0000);
        for y in 0..BUFFER {
            for x in 0..BUFFER {
                let pixel = [(x * 17 % 256) as u8, (y * 17 % 256) as u8, 0x30, 0xC0];
                let mut encoded = [0; 4];
                ColorFormat::Rgba8.encode(pixel, &mut encoded);
                let at = (COLOR + morton_offset(x, BUFFER - 1 - y, BUFFER, 4)) as usize;
                memory.flat[at..at + 4].copy_from_slice(&encoded);
            }
        }
        for sample in 0..BUFFER * BUFFER {
            let raw = depth_pattern(sample);
            let at = (DEPTH + sample * depth_bytes) as usize;
            match depth_bytes {
                2 => memory.flat[at..at + 2].copy_from_slice(&(raw as u16).to_le_bytes()),
                3 => memory.flat[at..at + 3].copy_from_slice(&raw.to_le_bytes()[..3]),
                _ => memory.flat[at..at + 4].copy_from_slice(&raw.to_le_bytes()),
            }
        }
        memory
    }

    fn reads_over(&self, addr: u32, len: u32) -> usize {
        self.reads
            .iter()
            .filter(|&&(at, bytes)| at < addr + len && addr < at + bytes)
            .count()
    }

    fn writes_over(&self, addr: u32, len: u32) -> usize {
        self.writes
            .iter()
            .filter(|&&(at, bytes)| at < addr + len && addr < at + bytes)
            .count()
    }

    fn guarded_reads(&self, addr: u32, len: u32) -> bool {
        self.guards
            .iter()
            .any(|&(at, bytes, writes)| !writes && at <= addr && addr + len <= at + bytes)
    }

    fn guarded_writes(&self, addr: u32, len: u32) -> bool {
        self.guards
            .iter()
            .any(|&(at, bytes, writes)| writes && at <= addr && addr + len <= at + bytes)
    }

    /// the same bytes, for the expected side of a comparison
    fn region(&self, addr: u32, len: u32) -> &[u8] {
        &self.flat[addr as usize..(addr + len) as usize]
    }

    /// overwrites a range, the way a fill lands after its sync ran
    fn fill(&mut self, addr: u32, len: u32, pattern: u8) {
        self.flat[addr as usize..(addr + len) as usize].fill(pattern);
    }
}

impl GpuMemory for CountingMemory {
    fn read(&mut self, addr: u32, out: &mut [u8]) {
        self.reads.push((addr, out.len() as u32));
        out.copy_from_slice(&self.flat[addr as usize..addr as usize + out.len()]);
    }

    fn write(&mut self, addr: u32, data: &[u8]) {
        self.writes.push((addr, data.len() as u32));
        self.flat[addr as usize..addr as usize + data.len()].copy_from_slice(data);
    }

    fn guard(&mut self, addr: u32, len: u32) {
        self.guards.push((addr, len, false));
    }

    fn guard_writes(&mut self, addr: u32, len: u32) {
        self.guards.push((addr, len, true));
    }
}

// ---------------------------------------------------------------------------
// the draws the tests describe
// ---------------------------------------------------------------------------

/// the size every differential draw uses, tiles whole like real targets are
const BUFFER: u32 = 16;
const COLOR: u32 = 0x1000;
const DEPTH: u32 = 0x3000;
const COLOR_BYTES: u32 = BUFFER * BUFFER * 4;

/// words a pixel of the depth buffer takes: 2 for D16, 3 for D24, 4 for D24S8
fn depth_bytes_of(depth_format: u32) -> u32 {
    match depth_format {
        0 => 2,
        2 => 3,
        _ => 4,
    }
}

/// registers for a BUFFER x BUFFER RGBA8 target and a depth buffer of `depth_format` beside it
fn registers(depth_format: u32) -> Vec<u32> {
    let mut registers = vec![0u32; 0x300];
    registers[REG_VIEWPORT_WIDTH] = float24(BUFFER as f32 / 2.0);
    registers[REG_VIEWPORT_HEIGHT] = float24(BUFFER as f32 / 2.0);
    registers[REG_VIEWPORT_DEPTH_RANGE] = float24(-1.0);
    registers[REG_VIEWPORT_DEPTH_NEAR] = float24(0.0);
    registers[REG_DEPTHMAP_ENABLE] = 1;
    registers[REG_COLOR_BUFFER_ADDRESS] = COLOR >> 3;
    registers[REG_DEPTH_BUFFER_ADDRESS] = DEPTH >> 3;
    registers[REG_DEPTH_BUFFER_FORMAT] = depth_format;
    registers[REG_FRAMEBUFFER_DIMENSIONS] = BUFFER | ((BUFFER - 1) << 12) | 1 << 24;
    registers[REG_COLOR_BUFFER_WRITE] = 0xF;
    registers[REG_DEPTH_STENCIL_WRITE] = 0x3;
    registers[REG_DEPTH_COLOR_MASK] = 0xF << 8;
    registers[REG_LOGIC_OP] = 3; // copy, which reads as no logic op at all
    registers
}

/// a vertex at an absolute window spot on the `viewport` (x, y, w, h) rect,
/// so edges land on pixel boundaries and ties are exact
fn corner_at(
    viewport: (f32, f32, f32, f32),
    x: f32,
    y: f32,
    z_over_w: f32,
    w: f32,
    color: Vec4,
) -> Vertex {
    let (vx, vy, vw, vh) = viewport;
    let ndc_x = (x - vx) / (vw / 2.0) - 1.0;
    let ndc_y = (y - vy) / (vh / 2.0) - 1.0;
    Vertex {
        clip: [ndc_x * w, ndc_y * w, z_over_w * w, w],
        color,
        texcoords: [[0.0; 2]; 3],
        quaternion: [0.0; 4],
        view: [0.0; 3],
        texcoord0_w: 1.0,
    }
}

/// a quad covering the whole target at clip depth `z_over_w`, drawn thru `rasterize`
fn cover(viewport: (f32, f32, f32, f32), z_over_w: f32, color: Vec4) -> Vec<Vertex> {
    let (vx, vy, vw, vh) = viewport;
    let corner = |x: f32, y: f32| corner_at(viewport, x, y, z_over_w, 1.0, color);
    let mut vertices = Vec::with_capacity(6);
    for triangle in [
        [
            (vx - 2.0, vy - 2.0),
            (vx + vw + 2.0, vy - 2.0),
            (vx + vw + 2.0, vy + vh + 2.0),
        ],
        [
            (vx - 2.0, vy - 2.0),
            (vx + vw + 2.0, vy + vh + 2.0),
            (vx - 2.0, vy + vh + 2.0),
        ],
    ] {
        vertices.extend(triangle.map(|(x, y)| corner(x, y)));
    }
    vertices
}

/// a slanted triangle with its own weight and color per corner, for interpolation
fn slanted(color: [Vec4; 3], ws: [f32; 3], z_over_w: f32) -> Vec<Vertex> {
    let spots = [(2.0, 2.0), (14.0, 4.0), (4.0, 14.0)];
    let viewport = (0.0, 0.0, BUFFER as f32, BUFFER as f32);
    spots
        .into_iter()
        .zip(color)
        .zip(ws)
        .map(|((spot, color), w)| corner_at(viewport, spot.0, spot.1, z_over_w, w, color))
        .collect()
}

/// the flat colors whose bytes both backends quantize to the same thing:
/// each sits far from an integer in `channel * 255`, so float noise never crosses a step
const FLAT: Vec4 = [0.5, 0.25, 0.75, 0.5];
const OTHER: Vec4 = [0.25, 0.75, 0.5, 0.75];

/// the triangles a draw of these vertices leaves for the backends, as the edge
/// distance check wants them
fn placed_triangles(registers: &[u32], shaded: &[Vertex]) -> Vec<[Screen; 3]> {
    let viewport = viewport(registers).expect("the tests always give a viewport");
    let placement = place(registers, shaded, None, viewport);
    placement
        .indices
        .as_chunks::<3>()
        .0
        .iter()
        .map(|&[a, b, c]| {
            [
                placement.placed[a as usize],
                placement.placed[b as usize],
                placement.placed[c as usize],
            ]
        })
        .collect()
}

/// runs the same draws twice — once on the CPU alone, once with the OpenGL
/// renderer attached — over identical starting memory, then resolves what the
/// GL still holds into guest memory, like its next reader would
fn run_both(
    depth_format: u32,
    depth_pattern: impl Fn(u32) -> u32,
    draws: impl Fn(&mut Vec<u32>, &mut CountingMemory, &mut CountingMemory, &mut Resources),
) -> Option<(CountingMemory, CountingMemory)> {
    let mut resources = opengl_resources()?;
    let mut registers = registers(depth_format);
    let mut cpu = CountingMemory::prefilled(depth_format, depth_pattern);
    let mut gl = cpu.clone();
    draws(&mut registers, &mut cpu, &mut gl, &mut resources);
    resolve(&mut resources, &mut gl, depth_bytes_of(depth_format));
    Some((cpu, gl))
}

/// pulls what the GL still holds into guest memory, color and depth
fn resolve(resources: &mut Resources, memory: &mut CountingMemory, depth_bytes: u32) {
    let renderer = resources
        .opengl
        .as_mut()
        .expect("the GL resources the test made");
    renderer.sync(memory, COLOR, COLOR_BYTES, true);
    renderer.sync(memory, DEPTH, depth_bytes * BUFFER * BUFFER, true);
}

// ---------------------------------------------------------------------------
// reading the results back
// ---------------------------------------------------------------------------

/// the target's pixels in window order, top row first, the raw guest bytes in
/// the buffer's own ABGR order, alpha first and red last
fn image(memory: &CountingMemory) -> Vec<[u8; 4]> {
    (0..BUFFER * BUFFER)
        .map(|sample| {
            let (x, y) = (sample % BUFFER, sample / BUFFER);
            let at = (COLOR + morton_offset(x, BUFFER - 1 - y, BUFFER, 4)) as usize;
            memory.flat[at..at + 4].try_into().expect("RGBA pixel")
        })
        .collect()
}

/// the RGB8 target's texels in window order, three raw guest bytes a pixel
fn rgb8_image(memory: &CountingMemory) -> Vec<[u8; 3]> {
    (0..BUFFER * BUFFER)
        .map(|sample| {
            let (x, y) = (sample % BUFFER, sample / BUFFER);
            let at = (COLOR + morton_offset(x, BUFFER - 1 - y, BUFFER, 3)) as usize;
            memory.flat[at..at + 3].try_into().expect("RGB pixel")
        })
        .collect()
}

/// a 16 bit target's texels in window order, the two raw guest bytes a pixel takes
fn packed_image(memory: &CountingMemory, format: ColorFormat) -> Vec<[u8; 2]> {
    let bytes_per_pixel = format.bytes_per_pixel() as u32;
    (0..BUFFER * BUFFER)
        .map(|sample| {
            let (x, y) = (sample % BUFFER, sample / BUFFER);
            let at = (COLOR + morton_offset(x, BUFFER - 1 - y, BUFFER, bytes_per_pixel)) as usize;
            memory.flat[at..at + 2].try_into().expect("packed pixel")
        })
        .collect()
}

/// the depth buffer's raw samples in window order, a u16/u24/u32 word per pixel
fn depth_image(memory: &CountingMemory, depth_bytes: u32) -> Vec<u32> {
    (0..BUFFER * BUFFER)
        .map(|sample| {
            let at = (DEPTH + sample * depth_bytes) as usize;
            match depth_bytes {
                2 => u16::from_le_bytes(memory.flat[at..at + 2].try_into().unwrap()) as u32,
                3 => u32::from_le_bytes([
                    memory.flat[at],
                    memory.flat[at + 1],
                    memory.flat[at + 2],
                    0,
                ]),
                _ => u32::from_le_bytes(memory.flat[at..at + 4].try_into().unwrap()),
            }
        })
        .collect()
}

/// which pixels the draw changed the color of
fn coverage(memory: &CountingMemory, start: &CountingMemory) -> Vec<bool> {
    image(memory)
        .iter()
        .zip(image(start).iter())
        .map(|(now, was)| now != was)
        .collect()
}

/// how far a pixel's center sits from the nearest edge of any triangle
fn edge_distance(x: u32, y: u32, triangles: &[[Screen; 3]]) -> f32 {
    let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
    let point_to_segment = |ax: f32, ay: f32, bx: f32, by: f32| -> f32 {
        let (dx, dy) = (bx - ax, by - ay);
        let length_squared = dx * dx + dy * dy;
        let t = if length_squared == 0.0 {
            0.0
        } else {
            (((px - ax) * dx + (py - ay) * dy) / length_squared).clamp(0.0, 1.0)
        };
        let (qx, qy) = (ax + t * dx, ay + t * dy);
        ((px - qx) * (px - qx) + (py - qy) * (py - qy)).sqrt()
    };
    triangles
        .iter()
        .flat_map(|triangle| (0..3).map(move |i| (i, triangle)))
        .map(|(i, triangle)| {
            let a = &triangle[i];
            let b = &triangle[(i + 1) % 3];
            point_to_segment(a.x, a.y, b.x, b.y)
        })
        .fold(f32::MAX, f32::min)
}

/// one pixel where the backends disagree past what this test allows
// the fields live for the Debug print in the assertions below
#[derive(Debug)]
#[allow(dead_code)]
struct Difference {
    x: u32,
    y: u32,
    cpu: [u8; 4],
    gl: [u8; 4],
    off: u8,
}

/// compares two results: interior pixels agree within `tolerance` a channel, and
/// whatever else differs must sit within a pixel of an edge (where coverage ties
/// and float interpolation live) in a count bounded by the edges' length
fn assert_images_close(
    cpu: &CountingMemory,
    gl: &CountingMemory,
    triangles: &[[Screen; 3]],
    tolerance: u8,
    what: &str,
) {
    let (cpu_pixels, gl_pixels) = (image(cpu), image(gl));
    let mut differences: Vec<Difference> = Vec::new();
    let mut edge_differences = 0usize;
    for y in 0..BUFFER {
        for x in 0..BUFFER {
            let index = (y * BUFFER + x) as usize;
            let off = cpu_pixels[index]
                .iter()
                .zip(gl_pixels[index].iter())
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap_or(0);
            if off <= tolerance {
                continue;
            }
            if edge_distance(x, y, triangles) <= 1.0 {
                edge_differences += 1;
            } else {
                differences.push(Difference {
                    x,
                    y,
                    cpu: cpu_pixels[index],
                    gl: gl_pixels[index],
                    off,
                });
            }
        }
    }
    let perimeter: f32 = triangles
        .iter()
        .map(|triangle| {
            (0..3)
                .map(|i| {
                    let a = &triangle[i];
                    let b = &triangle[(i + 1) % 3];
                    (b.x - a.x).hypot(b.y - a.y)
                })
                .sum::<f32>()
        })
        .sum();
    let bound = perimeter.ceil() as usize * 2 + 16;
    assert!(
        differences.is_empty() && edge_differences <= bound,
        "{what}: {edge_differences} edge pixels differ (bound {bound}) and {} pixels off the \
         edges do, worst first: {:?}",
        differences.len(),
        &differences[..differences.len().min(8)],
    );
}

/// the backends left the same color bytes, exactly
fn assert_images_identical(cpu: &CountingMemory, gl: &CountingMemory, what: &str) {
    let (cpu_pixels, gl_pixels) = (image(cpu), image(gl));
    let differing: Vec<Difference> = (0..(BUFFER * BUFFER) as usize)
        .filter(|&index| cpu_pixels[index] != gl_pixels[index])
        .map(|index| Difference {
            x: index as u32 % BUFFER,
            y: index as u32 / BUFFER,
            cpu: cpu_pixels[index],
            gl: gl_pixels[index],
            off: cpu_pixels[index]
                .iter()
                .zip(gl_pixels[index].iter())
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap_or(0),
        })
        .collect();
    assert!(
        differing.is_empty(),
        "{what}: {} pixels differ, worst first: {:?}",
        differing.len(),
        &differing[..differing.len().min(8)],
    );
}

/// the backends left the same RGB8 texels, exactly
fn assert_rgb8_identical(cpu: &CountingMemory, gl: &CountingMemory, what: &str) {
    let (cpu_pixels, gl_pixels) = (rgb8_image(cpu), rgb8_image(gl));
    let differing: Vec<(u32, u32, [u8; 3], [u8; 3])> = (0..(BUFFER * BUFFER) as usize)
        .filter(|&index| cpu_pixels[index] != gl_pixels[index])
        .map(|index| {
            (
                index as u32 % BUFFER,
                index as u32 / BUFFER,
                cpu_pixels[index],
                gl_pixels[index],
            )
        })
        .collect();
    assert!(
        differing.is_empty(),
        "{what}: {} pixels differ, worst first: {:?}",
        differing.len(),
        &differing[..differing.len().min(8)],
    );
}

/// interior RGB8 texels agree within `tolerance` a channel, and whatever else
/// differs sits within a pixel of an edge, in the usual bounded count
fn assert_rgb8_close(
    cpu: &CountingMemory,
    gl: &CountingMemory,
    triangles: &[[Screen; 3]],
    tolerance: u8,
    what: &str,
) {
    let (cpu_pixels, gl_pixels) = (rgb8_image(cpu), rgb8_image(gl));
    let mut differences: Vec<(u32, u32, [u8; 3], [u8; 3])> = Vec::new();
    let mut edge_differences = 0usize;
    for y in 0..BUFFER {
        for x in 0..BUFFER {
            let index = (y * BUFFER + x) as usize;
            let off = cpu_pixels[index]
                .iter()
                .zip(gl_pixels[index].iter())
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap_or(0);
            if off <= tolerance {
                continue;
            }
            if edge_distance(x, y, triangles) <= 1.0 {
                edge_differences += 1;
            } else {
                differences.push((x, y, cpu_pixels[index], gl_pixels[index]));
            }
        }
    }
    let perimeter: f32 = triangles
        .iter()
        .map(|triangle| {
            (0..3)
                .map(|i| {
                    let a = &triangle[i];
                    let b = &triangle[(i + 1) % 3];
                    (b.x - a.x).hypot(b.y - a.y)
                })
                .sum::<f32>()
        })
        .sum();
    let bound = perimeter.ceil() as usize * 2 + 16;
    assert!(
        differences.is_empty() && edge_differences <= bound,
        "{what}: {edge_differences} edge pixels differ (bound {bound}) and {} pixels off the \
         edges do, worst first: {:?}",
        differences.len(),
        &differences[..differences.len().min(8)],
    );
}

/// the backends left the same 16 bit texels, exactly
fn assert_packed_identical(cpu: &CountingMemory, gl: &CountingMemory, format: ColorFormat, what: &str) {
    let (cpu_pixels, gl_pixels) = (packed_image(cpu, format), packed_image(gl, format));
    let differing: Vec<(u32, u32, [u8; 2], [u8; 2])> = (0..(BUFFER * BUFFER) as usize)
        .filter(|&index| cpu_pixels[index] != gl_pixels[index])
        .map(|index| {
            (
                index as u32 % BUFFER,
                index as u32 / BUFFER,
                cpu_pixels[index],
                gl_pixels[index],
            )
        })
        .collect();
    assert!(
        differing.is_empty(),
        "{what:?}: {} pixels differ, worst first: {:?}",
        differing.len(),
        &differing[..differing.len().min(8)],
    );
}

/// interior 16 bit texels agree within `tolerance` a decoded channel, and whatever
/// else differs sits within a pixel of an edge, in the usual bounded count
fn assert_packed_close(
    cpu: &CountingMemory,
    gl: &CountingMemory,
    format: ColorFormat,
    triangles: &[[Screen; 3]],
    tolerance: u8,
    what: &str,
) {
    let expand = |texel: [u8; 2]| format.decode(&texel);
    let (cpu_pixels, gl_pixels) = (packed_image(cpu, format), packed_image(gl, format));
    let mut differences: Vec<(u32, u32, [u8; 4], [u8; 4])> = Vec::new();
    let mut edge_differences = 0usize;
    for y in 0..BUFFER {
        for x in 0..BUFFER {
            let index = (y * BUFFER + x) as usize;
            let (cpu_pixel, gl_pixel) = (expand(cpu_pixels[index]), expand(gl_pixels[index]));
            let off = cpu_pixel
                .iter()
                .zip(gl_pixel.iter())
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap_or(0);
            if off <= tolerance {
                continue;
            }
            if edge_distance(x, y, triangles) <= 1.0 {
                edge_differences += 1;
            } else {
                differences.push((x, y, cpu_pixel, gl_pixel));
            }
        }
    }
    let perimeter: f32 = triangles
        .iter()
        .map(|triangle| {
            (0..3)
                .map(|i| {
                    let a = &triangle[i];
                    let b = &triangle[(i + 1) % 3];
                    (b.x - a.x).hypot(b.y - a.y)
                })
                .sum::<f32>()
        })
        .sum();
    let bound = perimeter.ceil() as usize * 2 + 16;
    assert!(
        differences.is_empty() && edge_differences <= bound,
        "{what:?}: {edge_differences} edge pixels differ (bound {bound}) and {} pixels off the \
         edges do, worst first: {:?}",
        differences.len(),
        &differences[..differences.len().min(8)],
    );
}

/// the coverage masks agree, so depth comparisons stand on solid ground
fn assert_coverage_agrees(
    cpu: &CountingMemory,
    gl: &CountingMemory,
    start: &CountingMemory,
    what: &str,
) {
    let disagreeing: Vec<(u32, u32)> = coverage(cpu, start)
        .iter()
        .zip(coverage(gl, start).iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(index, _)| (index as u32 % BUFFER, index as u32 / BUFFER))
        .collect();
    assert!(
        disagreeing.is_empty(),
        "{what}: the backends disagree on {} pixels' coverage, at {disagreeing:?}",
        disagreeing.len(),
    );
}

/// the depth samples of drawn pixels sit within `tolerance` raw units a side,
/// and where nothing passed both left the same word
fn assert_depth_close(
    cpu: &CountingMemory,
    gl: &CountingMemory,
    depth_bytes: u32,
    covered: &[bool],
    tolerance: u32,
    what: &str,
) {
    let (cpu_depth, gl_depth) = (depth_image(cpu, depth_bytes), depth_image(gl, depth_bytes));
    let mut worst = 0u32;
    let mut worst_at = 0usize;
    for (index, &covered) in covered.iter().enumerate() {
        // an undrawn sample holds what it started with, back thru the GL's float
        // readback where the depth rides as a float, so a few steps of noise can show
        let off = cpu_depth[index].abs_diff(gl_depth[index]);
        if off > tolerance {
            assert!(
                covered,
                "{what}: sample ({}, {}) was not drawn but the backends differ by {off}",
                index as u32 % BUFFER,
                index as u32 / BUFFER,
            );
        }
        if off > worst {
            worst = off;
            worst_at = index;
        }
    }
    assert!(
        worst <= tolerance,
        "{what}: depth differs by {worst} raw units at ({}, {}) ({} against {}), bound {tolerance}",
        worst_at as u32 % BUFFER,
        worst_at as u32 / BUFFER,
        cpu_depth[worst_at],
        gl_depth[worst_at],
    );
}

// ---------------------------------------------------------------------------
// rasterization, interpolation, culling and scissoring
// ---------------------------------------------------------------------------

/// a slanted triangle with a color and a weight per corner interpolates the same
/// on both sides: interior pixels agree within float noise, and whatever else
/// differs stays on the edge, in a count the edge's length bounds
#[test]
fn opengl_interpolates_vertex_colors_like_the_cpu() {
    let Some(resources) = opengl_resources() else {
        return;
    };
    let registers = registers(0);
    let triangle = slanted(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        [1.0, 0.7, 1.4],
        -0.5,
    );
    let placed = placed_triangles(&registers, &triangle);
    let mut cpu = CountingMemory::zeroed(0x1_0000);
    let mut gl = CountingMemory::zeroed(0x1_0000);
    rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &triangle);
    let mut resources = resources;
    rasterize_shaded(&registers, &mut gl, &mut resources, &triangle);
    resolve(&mut resources, &mut gl, 0);
    assert_images_close(&cpu, &gl, &placed, 2, "the interpolated triangle");
}

/// corners the PICA snaps onto sixteenths cover the same pixels both ways,
/// over a spread of slants, one triangle at a time over a cleared target
#[test]
fn opengl_covers_the_same_pixels() {
    let Some(mut resources) = opengl_resources() else {
        return;
    };
    let mut seed = 1u32;
    let mut random = move |range: u32| {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 8) % range
    };
    // corners on a sixteenth of a pixel, half of them on pixel centers
    let mut coordinate = || {
        let sixteenths = random(BUFFER * 16 + 1);
        let sixteenths = if random(2) == 0 {
            sixteenths / 8 * 8 + 8
        } else {
            sixteenths
        };
        sixteenths as f32 / 16.0
    };
    let viewport = (0.0, 0.0, BUFFER as f32, BUFFER as f32);
    for case in 0..120 {
        let registers = registers(0);
        let mut cpu = CountingMemory::zeroed(0x1_0000);
        let mut gl = CountingMemory::zeroed(0x1_0000);
        let triangle: Vec<Vertex> = (0..3)
            .map(|_| corner_at(viewport, coordinate(), coordinate(), -0.5, 1.0, FLAT))
            .collect();
        let placed = placed_triangles(&registers, &triangle);
        rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &triangle);
        rasterize_shaded(&registers, &mut gl, &mut resources, &triangle);
        resolve(&mut resources, &mut gl, 0);
        assert_images_close(&cpu, &gl, &placed, 1, &format!("coverage case {case}"));
    }
}

/// winding the PICA culls stays culled and winding it keeps gets drawn,
/// the same on both sides, byte for byte
#[test]
fn opengl_culls_the_same_triangles() {
    for cull_mode in 0..3u32 {
        for (winding, spots) in [
            ("counter-clockwise", [(2.0, 2.0), (2.0, 13.0), (13.0, 13.0)]),
            ("clockwise", [(2.0, 2.0), (13.0, 13.0), (2.0, 13.0)]),
        ] {
            let viewport = (0.0, 0.0, BUFFER as f32, BUFFER as f32);
            let Some((cpu, gl)) = run_both(
                0,
                |_| 0,
                |registers, cpu_memory, gl_memory, resources| {
                    registers[REG_FACE_CULLING] = cull_mode;
                    let triangle: Vec<Vertex> = spots
                        .map(|(x, y)| corner_at(viewport, x, y, -0.5, 1.0, FLAT))
                        .into();
                    rasterize_shaded(registers, cpu_memory, &mut Resources::default(), &triangle);
                    rasterize_shaded(registers, gl_memory, resources, &triangle);
                },
            ) else {
                return;
            };
            assert_images_identical(&cpu, &gl, &format!("cull mode {cull_mode}, {winding}"));
        }
    }
}

/// the viewport cuts the draw to its rect on the buffer, the same rect both
/// backends scissor to, at every corner of the buffer and past its edges
#[test]
fn opengl_scissored_viewports_agree() {
    let bits = |value: f32| ((value as i32) & 0x3FF) as u32;
    for (name, viewport) in [
        ("the whole buffer", (0.0, 0.0, 16.0, 16.0)),
        ("an inset rect", (4.0, 4.0, 8.0, 8.0)),
        ("one hanging off the bottom left", (-4.0, -4.0, 16.0, 16.0)),
        ("one hanging off the top right", (8.0, 8.0, 16.0, 16.0)),
    ] {
        let Some((cpu, gl)) = run_both(
            0,
            |_| 0,
            |registers, cpu_memory, gl_memory, resources| {
                registers[REG_VIEWPORT_XY] = bits(viewport.0) | (bits(viewport.1) << 16);
                registers[REG_VIEWPORT_WIDTH] = float24(viewport.2 / 2.0);
                registers[REG_VIEWPORT_HEIGHT] = float24(viewport.3 / 2.0);
                let triangle = cover(viewport, -0.5, FLAT);
                rasterize_shaded(registers, cpu_memory, &mut Resources::default(), &triangle);
                rasterize_shaded(registers, gl_memory, resources, &triangle);
            },
        ) else {
            return;
        };
        assert_images_identical(&cpu, &gl, &format!("scissor for {name}"));
    }
}

// ---------------------------------------------------------------------------
// depth tests and depth writes
// ---------------------------------------------------------------------------

/// the stored sample the depth tests test against, a raw word both formats
/// quantize to the same float the CPU reads back, so `Equal` means the same
/// thing on both sides. D24's word avoids half scale, where the quantized
/// fragment sits exactly on a rounding tie and a float step of interpolation
/// noise flips whole rows of pixels
const STORED_D16: u32 = 0x8000;
const STORED_D24: u32 = 0x600000;

fn stored_raw(depth_format: u32) -> u32 {
    match depth_format {
        0 => STORED_D16,
        _ => STORED_D24,
    }
}

/// the depth as the CPU's `read_sample` hands the stored word back
fn stored_depth(depth_format: u32) -> f32 {
    match depth_format {
        0 => STORED_D16 as f32 / 65535.0,
        _ => STORED_D24 as f32 / 16_777_215.0,
    }
}

/// what `Compare::passes` says of a fragment's depth against the stored sample,
/// the analytic answer the coverage is checked against
fn analytically(test: Compare, depth_format: u32, fragment: f32) -> bool {
    test.passes(fragment, stored_depth(depth_format))
}

/// a sample's worth of the depth format, the LSB the comparisons are counted in
fn depth_lsb(depth_format: u32) -> f32 {
    match depth_format {
        0 => 1.0 / 65535.0,
        _ => 1.0 / 16_777_215.0,
    }
}

/// every compare function takes the same fragments on both sides, on all three
/// depth formats, at fragment depths a clear margin below and above the stored
/// sample, plus the far side of the range
///
/// there is deliberately no fragment exactly at the stored sample: the CPU
/// compares the fragment's float against the float of the stored word, while the
/// GL quantizes the fragment into the buffer first, so at the sample itself the
/// two answer interpolated pixels' float noise differently — on D24 half the
/// target can land each way. That boundary is implementation defined, not a bug
/// either side can fix, and TODO.md records it
#[test]
fn opengl_depth_tests_agree() {
    let tests = [
        (0u32, Compare::Never),
        (1, Compare::Always),
        (2, Compare::Equal),
        (3, Compare::NotEqual),
        (4, Compare::Less),
        (5, Compare::LessOrEqual),
        (6, Compare::Greater),
        (7, Compare::GreaterOrEqual),
    ];
    for depth_format in [0u32, 2, 3] {
        let depth_bytes = depth_bytes_of(depth_format);
        let stored = stored_depth(depth_format);
        let lsb = depth_lsb(depth_format);
        // eight least significant bits of margin, far past the readback's float noise
        let fragments = [
            ("far below", 0.25),
            ("just below", stored - 8.0 * lsb),
            ("just above", stored + 8.0 * lsb),
            ("far above", 0.75),
        ];
        for (test_raw, test) in tests {
            for &(name, fragment) in fragments.iter() {
                let Some((cpu, gl, start)) = (|| {
                    let mut resources = opengl_resources()?;
                    let mut registers = registers(depth_format);
                    registers[REG_DEPTH_COLOR_MASK] = 0xF << 8 | 1 << 12 | 1 | (test_raw << 4);
                    let triangle = cover((0.0, 0.0, 16.0, 16.0), -fragment, FLAT);
                    let mut cpu =
                        CountingMemory::prefilled(depth_bytes, |_| stored_raw(depth_format));
                    let mut gl = cpu.clone();
                    let start = cpu.clone();
                    rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &triangle);
                    rasterize_shaded(&registers, &mut gl, &mut resources, &triangle);
                    resolve(&mut resources, &mut gl, depth_bytes);
                    Some((cpu, gl, start))
                })() else {
                    return;
                };
                let what = format!("depth format {depth_format}, test {test:?}, fragment {name}");
                let cpu_coverage = coverage(&cpu, &start);
                // the coverage the state alone dictates of any correct rasterizer
                let expected = analytically(test, depth_format, fragment);
                let off: Vec<(u32, u32)> = cpu_coverage
                    .iter()
                    .enumerate()
                    .filter(|(_, &covered)| covered != expected)
                    .map(|(index, _)| (index as u32 % BUFFER, index as u32 / BUFFER))
                    .collect();
                assert!(
                    off.is_empty(),
                    "{what}: the CPU's coverage disagrees with the state's answer ({expected}) at {off:?}",
                );
                assert_coverage_agrees(&cpu, &gl, &start, &what);
                assert_images_identical(&cpu, &gl, &what);
                if expected {
                    // written depth rounds on the GL and truncates on the CPU, a step apart
                    assert_depth_close(&cpu, &gl, depth_bytes, &cpu_coverage, 2, &what);
                }
            }
        }
    }
}

/// depth writes round on the GL and truncate on the CPU, every format a step
/// apart at most, and a D24S8 write keeps the stencil byte it shares the word with
#[test]
fn opengl_depth_writes_keep_the_stencil_byte() {
    for depth_format in [0u32, 2, 3] {
        let depth_bytes = depth_bytes_of(depth_format);
        let pattern = |sample: u32| match depth_format {
            3 => (0x5A << 24) | STORED_D24,
            _ => stored_raw(depth_format) + sample,
        };
        let Some((cpu, gl, start)) = (|| {
            let mut resources = opengl_resources()?;
            let mut registers = registers(depth_format);
            // the depth test on and writing, at a depth clear of the stored one
            registers[REG_DEPTH_COLOR_MASK] = 0xF << 8 | 1 << 12 | 1 | (5 << 4);
            let triangle = cover((0.0, 0.0, 16.0, 16.0), -0.25, FLAT);
            let mut cpu = CountingMemory::prefilled(depth_bytes, pattern);
            let mut gl = cpu.clone();
            let start = cpu.clone();
            rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &triangle);
            rasterize_shaded(&registers, &mut gl, &mut resources, &triangle);
            resolve(&mut resources, &mut gl, depth_bytes);
            Some((cpu, gl, start))
        })() else {
            return;
        };
        let what = format!("depth write in format {depth_format}");
        let covered = coverage(&cpu, &start);
        assert_coverage_agrees(&cpu, &gl, &start, &what);
        assert_images_identical(&cpu, &gl, &what);
        assert_depth_close(&cpu, &gl, depth_bytes, &covered, 2, &what);
        // the stencil byte survives a depth-only draw, on both sides
        if depth_format == 3 {
            for (name, memory) in [("the CPU", &cpu), ("the GL", &gl)] {
                for (sample, &word) in depth_image(memory, 4).iter().enumerate() {
                    assert_eq!(
                        (word >> 24) as u8,
                        0x5A,
                        "{what}: {name} disturbed the stencil byte at ({}, {})",
                        sample as u32 % BUFFER,
                        sample as u32 / BUFFER,
                    );
                }
            }
        }
    }

    // and w/ the write bit off, neither side touches the depth buffer
    let Some((cpu, gl, start)) = (|| {
        let mut resources = opengl_resources()?;
        let mut registers = registers(3);
        registers[REG_DEPTH_COLOR_MASK] = 0xF << 8 | 1 | (5 << 4);
        let triangle = cover((0.0, 0.0, 16.0, 16.0), -0.25, FLAT);
        let mut cpu = CountingMemory::prefilled(4, |_| (0x5A << 24) | STORED_D24);
        let mut gl = cpu.clone();
        let start = cpu.clone();
        rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &triangle);
        rasterize_shaded(&registers, &mut gl, &mut resources, &triangle);
        resolve(&mut resources, &mut gl, 4);
        Some((cpu, gl, start))
    })() else {
        return;
    };
    assert_images_identical(&cpu, &gl, "a depth read-only draw");
    assert_eq!(
        depth_image(&cpu, 4),
        depth_image(&gl, 4),
        "a depth read-only draw moved the depth"
    );
    assert_eq!(
        depth_image(&cpu, 4),
        depth_image(&start, 4),
        "the CPU wrote depth it may not"
    );
}

// ---------------------------------------------------------------------------
// stencil tests and stencil operations
// ---------------------------------------------------------------------------

/// what one stencil operation does to the bits its write mask uncovers
fn apply_stencil_op(stored: u8, reference: u8, raw: u32, write_mask: u8) -> u8 {
    let updated = match raw & 0x7 {
        0 => stored,
        1 => 0,
        2 => reference,
        3 => stored.saturating_add(1),
        4 => stored.saturating_sub(1),
        5 => !stored,
        6 => stored.wrapping_add(1),
        _ => stored.wrapping_sub(1),
    };
    (stored & !write_mask) | (updated & write_mask)
}

/// the final stencil byte the state dictates here: the setup gives the fail,
/// depth fail and pass ops all the same raw value, so whichever outcome a pixel
/// meets, the same masked op lands, on both sides
fn expected_stencil(stored: u8, reference: u8, write_mask: u8, op: u32) -> u8 {
    apply_stencil_op(stored, reference, op, write_mask)
}

/// every stencil operation, in every outcome it can meet — the stencil test failing,
/// the depth test failing, both passing — with compare and write masks partial and
/// every starting stencil byte; the stencil byte comes out the same on both sides
#[test]
fn opengl_stencil_operations_agree() {
    let ops: [(&str, u32); 8] = [
        ("keep", 0),
        ("zero", 1),
        ("replace", 2),
        ("increment saturate", 3),
        ("decrement saturate", 4),
        ("invert", 5),
        ("increment wrap", 6),
        ("decrement wrap", 7),
    ];
    // one scenario per outcome: which of the three ops applies
    let scenarios: [(&str, Compare, Compare, bool); 3] = [
        // the stencil test fails first: Equal against a buffer holding something else
        (
            "the stencil test fails",
            Compare::Equal,
            Compare::Always,
            false,
        ),
        // the stencil passes, the depth test is Never
        (
            "the depth test fails",
            Compare::Always,
            Compare::Never,
            false,
        ),
        // both tests pass and the depth writes
        (
            "both tests pass",
            Compare::Always,
            Compare::LessOrEqual,
            true,
        ),
    ];
    let depth_of = |test: Compare, fragment: f32| analytically(test, 2, fragment);

    for (scenario_name, function, depth_test, depth_writes) in scenarios {
        for (op_name, op_raw) in ops {
            for (compare_mask, write_mask) in [(0xFFu8, 0xFFu8), (0x0F, 0xF0)] {
                for stored in [0x00u8, 0xFF, 0x0F, 0xF0] {
                    for reference in [0x3Cu8, 0x0A] {
                        let Some((cpu, gl, start)) = (|| {
                            let mut resources = opengl_resources()?;
                            let mut registers = registers(3);
                            registers[REG_STENCIL_TEST] = 1
                                | ((function as u32 & 0x7) << 4)
                                | ((write_mask as u32) << 8)
                                | ((reference as u32) << 16)
                                | ((compare_mask as u32) << 24);
                            registers[REG_STENCIL_OP] = op_raw | (op_raw << 4) | (op_raw << 8);
                            registers[REG_DEPTH_COLOR_MASK] = 0xF << 8
                                | 1
                                | ((depth_test as u32 & 0x7) << 4)
                                | if depth_writes { 1 << 12 } else { 0 };
                            let triangle = cover((0.0, 0.0, 16.0, 16.0), -0.25, FLAT);
                            let mut cpu = CountingMemory::prefilled(4, |_| {
                                (stored as u32) << 24 | STORED_D24
                            });
                            let mut gl = cpu.clone();
                            let start = cpu.clone();
                            rasterize_shaded(
                                &registers,
                                &mut cpu,
                                &mut Resources::default(),
                                &triangle,
                            );
                            rasterize_shaded(&registers, &mut gl, &mut resources, &triangle);
                            resolve(&mut resources, &mut gl, 4);
                            Some((cpu, gl, start))
                        })() else {
                            return;
                        };
                        let what = format!(
                            "stencil {op_name} where {scenario_name} (stored {stored:#04x}, ref \
                             {reference:#04x}, compare {compare_mask:#04x}, write {write_mask:#04x})",
                        );
                        // which pixels write color: only where both tests passed.
                        // the Equal scenario's ref never matches any stored byte masked or not,
                        // so its coverage is empty, and Always's is the depth test's answer
                        let stencil_passes = analytically(function, 2, 0.0) || {
                            let (left, right) = (reference & compare_mask, stored & compare_mask);
                            match function {
                                Compare::Equal => left == right,
                                _ => true,
                            }
                        };
                        let depth_passes = depth_of(depth_test, 0.25);
                        let expected_written = stencil_passes && depth_passes;
                        let cpu_coverage = coverage(&cpu, &start);
                        assert!(
                            cpu_coverage.iter().all(|&covered| covered == expected_written),
                            "{what}: the CPU's coverage disagrees with the state's ({expected_written})",
                        );
                        assert_coverage_agrees(&cpu, &gl, &start, &what);
                        assert_images_identical(&cpu, &gl, &what);
                        // the depth buffer: written only where both tests passed.
                        // the stencil byte shares the word, so compare depth bits alone
                        if expected_written {
                            assert_depth_close(&cpu, &gl, 4, &cpu_coverage, 2, &what);
                        } else {
                            for (sample, (word, was)) in depth_image(&cpu, 4)
                                .iter()
                                .zip(depth_image(&start, 4))
                                .enumerate()
                            {
                                assert_eq!(
                                    word & 0xFF_FFFF,
                                    was & 0xFF_FFFF,
                                    "{what}: the CPU wrote depth it may not at ({}, {})",
                                    sample as u32 % BUFFER,
                                    sample as u32 / BUFFER,
                                );
                            }
                            for (sample, (word, was)) in depth_image(&gl, 4)
                                .iter()
                                .zip(depth_image(&start, 4))
                                .enumerate()
                            {
                                assert_eq!(
                                    word & 0xFF_FFFF,
                                    was & 0xFF_FFFF,
                                    "{what}: the GL wrote depth it may not at ({}, {})",
                                    sample as u32 % BUFFER,
                                    sample as u32 / BUFFER,
                                );
                            }
                        }
                        // the draw covers the whole target, so every pixel's stencil
                        // takes the scenario's op, masked, on both sides
                        let expected = expected_stencil(stored, reference, write_mask, op_raw);
                        for (name, memory) in [("the CPU", &cpu), ("the GL", &gl)] {
                            for (sample, &word) in depth_image(memory, 4).iter().enumerate() {
                                assert_eq!(
                                    (word >> 24) as u8,
                                    expected,
                                    "{what}: {name} left the stencil at ({}, {}) wrong",
                                    sample as u32 % BUFFER,
                                    sample as u32 / BUFFER,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// the depth map: scale, offset and the w buffer
// ---------------------------------------------------------------------------

/// z and w buffers, scales and offsets that land in range and ones that clamp,
/// over a slanted triangle whose w changes across it: the mapped depths agree
/// within the interpolation's float noise, a few raw units of D24
#[test]
fn opengl_depth_maps_match_the_cpu() {
    let maps: [(&str, f32, f32, bool, u32); 5] = [
        ("a z buffer, scale -1", -1.0, 0.0, false, 2),
        ("a w buffer, scale -1", -1.0, 0.0, true, 5),
        ("a z buffer pulled to the top half", -0.5, 0.5, false, 2),
        ("a z buffer that clamps low", 1.0, 0.0, false, 2),
        ("a w buffer that clamps", -2.0, -1.0, true, 5),
    ];
    for (name, scale, offset, w_buffer, tolerance) in maps {
        for depth_format in [2u32, 0] {
            let depth_bytes = depth_bytes_of(depth_format);
            let Some((cpu, gl, start)) = (|| {
                let mut resources = opengl_resources()?;
                let mut registers = registers(depth_format);
                registers[REG_VIEWPORT_DEPTH_RANGE] = float24(scale);
                registers[REG_VIEWPORT_DEPTH_NEAR] = float24(offset);
                registers[REG_DEPTHMAP_ENABLE] = !w_buffer as u32;
                registers[REG_DEPTH_COLOR_MASK] = 0xF << 8 | 1 << 12;
                // a slanted triangle whose w changes across it, mid-range depth
                let triangle = slanted([FLAT; 3], [1.0, 0.6, 1.4], -0.5);
                let mut cpu = CountingMemory::prefilled(depth_bytes, |sample| {
                    stored_raw(depth_format) + sample
                });
                let mut gl = cpu.clone();
                let start = cpu.clone();
                rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &triangle);
                rasterize_shaded(&registers, &mut gl, &mut resources, &triangle);
                resolve(&mut resources, &mut gl, depth_bytes);
                Some((cpu, gl, start))
            })() else {
                return;
            };
            let what = format!("{name}, format {depth_format}");
            let covered = coverage(&cpu, &start);
            assert_coverage_agrees(&cpu, &gl, &start, &what);
            assert_images_identical(&cpu, &gl, &what);
            assert_depth_close(&cpu, &gl, depth_bytes, &covered, tolerance, &what);
        }
    }
}

// ---------------------------------------------------------------------------
// the output merger: the alpha test, blending and partial color masks
// ---------------------------------------------------------------------------

/// the GPUREG_BLEND_FUNC halves packed: each side's equation, then its source and
/// destination factors, the encoding `Blend::read` takes apart
fn blend_func(color: (u32, u32, u32), alpha: (u32, u32, u32)) -> u32 {
    color.0 | (alpha.0 << 8) | (color.1 << 16) | (color.2 << 20) | (alpha.1 << 24)
        | (alpha.2 << 28)
}

/// the register the blend constant sits in, and a constant unlike the quads'
/// colors, so the Constant* factors show their own bytes, packed red low
const REG_BLEND_COLOR: usize = 0x0103;
const BLEND_CONSTANT: u32 = 0xFF80_40C0;

/// every compare function takes and leaves the same fragments when the alpha
/// under test is flat and sits clear of the reference: FLAT's alpha quantizes to
/// 127 on both sides, so each function gets one reference it must pass and one
/// it must refuse, byte for byte
#[test]
fn opengl_alpha_test_agrees_on_flat_alphas() {
    let cases: [(&str, u32, u32, bool); 14] = [
        ("never", 0, 0, false),
        ("always", 1, 0, true),
        ("equal passing", 2, 127, true),
        ("equal refusing", 2, 126, false),
        ("not equal passing", 3, 126, true),
        ("not equal refusing", 3, 127, false),
        ("less passing", 4, 200, true),
        ("less refusing", 4, 127, false),
        ("less or equal passing", 5, 127, true),
        ("less or equal refusing", 5, 126, false),
        ("greater passing", 6, 126, true),
        ("greater refusing", 6, 127, false),
        ("greater or equal passing", 7, 127, true),
        ("greater or equal refusing", 7, 200, false),
    ];
    for (name, function, reference, passes) in cases {
        let Some((cpu, gl, start)) = (|| {
            let mut resources = opengl_resources()?;
            let mut registers = registers(0);
            registers[REG_ALPHA_TEST] = 1 | (function << 4) | (reference << 8);
            let triangle = cover((0.0, 0.0, 16.0, 16.0), -0.5, FLAT);
            let mut cpu = CountingMemory::zeroed(0x1_0000);
            let mut gl = CountingMemory::zeroed(0x1_0000);
            let start = cpu.clone();
            rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &triangle);
            rasterize_shaded(&registers, &mut gl, &mut resources, &triangle);
            resolve(&mut resources, &mut gl, 0);
            Some((cpu, gl, start))
        })() else {
            return;
        };
        let what = format!("the alpha test {name} (function {function}, reference {reference})");
        assert_coverage_agrees(&cpu, &gl, &start, &what);
        assert_images_identical(&cpu, &gl, &what);
        // the state alone dictates whether anything survives, whichever backend drew it
        let drew_any = coverage(&cpu, &start).iter().any(|&covered| covered);
        assert_eq!(
            drew_any, passes,
            "{what}: the backends drew {drew_any} where the reference says {passes}",
        );
    }
}

/// the alpha test over a triangle whose alpha slides across the reference: the
/// backends may only disagree where the fragment's alpha sits on the reference
/// itself, the one byte a step of float noise flips, and pixels both drew agree
/// within that noise
#[test]
fn opengl_alpha_test_takes_the_same_fragments_of_a_varying_alpha() {
    let Some(mut resources) = opengl_resources() else {
        return;
    };
    let mut registers = registers(0);
    // on, Less, against 128: fragments below 128 continue, the rest go
    registers[REG_ALPHA_TEST] = 1 | (4 << 4) | (128 << 8);
    let triangle = slanted(
        [
            [0.5, 0.5, 0.5, 0.05],
            [0.5, 0.5, 0.5, 0.55],
            [0.5, 0.5, 0.5, 0.95],
        ],
        [1.0, 1.0, 1.0],
        -0.5,
    );
    let mut cpu = CountingMemory::zeroed(0x1_0000);
    let mut gl = CountingMemory::zeroed(0x1_0000);
    rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &triangle);
    rasterize_shaded(&registers, &mut gl, &mut resources, &triangle);
    resolve(&mut resources, &mut gl, 0);
    let (cpu_pixels, gl_pixels) = (image(&cpu), image(&gl));
    // the memories keep the bytes in the buffer's ABGR order, so the alpha is
    // byte zero, and zero bytes mean the pixel drew nothing
    let alpha_of = |pixel: [u8; 4]| pixel[0];
    let mut disagreements: Vec<(u32, u32, [u8; 4], [u8; 4])> = Vec::new();
    for index in 0..(BUFFER * BUFFER) as usize {
        let (drawn_cpu, drawn_gl) = (cpu_pixels[index], gl_pixels[index]);
        let (cpu_drew, gl_drew) = (drawn_cpu != [0; 4], drawn_gl != [0; 4]);
        if drawn_cpu == drawn_gl || (cpu_drew && gl_drew) {
            // both drew or neither did; the both-drew noise is checked below
            continue;
        }
        // a pixel only one backend drew has to hold the reference in its alpha,
        // the rounding boundary where the two interpolations may land apart.
        // the quad's other bytes sit far from zero, so zero means not drawn
        let drawn = if cpu_drew { drawn_cpu } else { drawn_gl };
        if alpha_of(drawn).abs_diff(128) > 1 {
            disagreements.push((
                index as u32 % BUFFER,
                index as u32 / BUFFER,
                drawn_cpu,
                drawn_gl,
            ));
        }
    }
    assert!(
        disagreements.is_empty(),
        "the alpha test split the triangle where the state does not: {disagreements:?}",
    );
    // pixels both drew agree on the alpha the test judged within a noise step,
    // the RGB within two, the interpolation's usual slack
    for index in 0..(BUFFER * BUFFER) as usize {
        if cpu_pixels[index] == [0; 4] || gl_pixels[index] == [0; 4] {
            continue;
        }
        let (cpu_pixel, gl_pixel) = (cpu_pixels[index], gl_pixels[index]);
        let off = cpu_pixel
            .iter()
            .zip(gl_pixel.iter())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0);
        assert!(
            off <= 2 && alpha_of(cpu_pixel).abs_diff(alpha_of(gl_pixel)) <= 1,
            "both backends drew ({}, {}) but left {:?} against {:?}",
            index as u32 % BUFFER,
            index as u32 / BUFFER,
            cpu_pixel,
            gl_pixel,
        );
    }
}

/// the blend equations and factors, over a base quad each blends into: the GL
/// blends from the same bytes thru its fixed function, and the two answers sit
/// a rounding step apart at most, plus the edge pixels' usual slack
#[test]
fn opengl_blends_like_the_cpu() {
    let viewport = (0.0, 0.0, 16.0, 16.0);
    let cases: [(&str, u32); 9] = [
        // the transparency almost every UI draw sets up
        ("standard transparency", blend_func((0, 6, 7), (0, 1, 0))),
        ("additive", blend_func((0, 1, 1), (0, 1, 1))),
        ("subtractive", blend_func((1, 6, 1), (0, 1, 0))),
        ("reverse subtract", blend_func((2, 1, 1), (0, 1, 1))),
        // min and max read the factors, which sit at zero to say so
        ("minimum", blend_func((3, 0, 0), (3, 0, 0))),
        ("maximum", blend_func((4, 0, 0), (4, 0, 0))),
        ("constant factors", blend_func((0, 10, 11), (0, 12, 13))),
        // GL takes the saturate factor as a color side's source
        ("source alpha saturate", blend_func((0, 14, 1), (0, 1, 0))),
        // a different equation on the alpha, subtracting it while the color adds
        ("a separate alpha equation", blend_func((0, 6, 7), (1, 1, 6))),
    ];
    for (name, function) in cases {
        let Some(mut resources) = opengl_resources() else {
            return;
        };
        let mut plain = registers(0);
        plain[REG_BLEND_COLOR] = BLEND_CONSTANT;
        let mut blended = plain.clone();
        blended[REG_COLOR_OPERATION] = 0x100;
        blended[REG_BLEND_FUNC] = function;
        let base = cover(viewport, -0.5, FLAT);
        let quad = slanted([OTHER, OTHER, OTHER], [1.2, 0.8, 1.0], -0.4);
        let placed = placed_triangles(&blended, &quad);
        let mut cpu = CountingMemory::zeroed(0x1_0000);
        let mut gl = CountingMemory::zeroed(0x1_0000);
        // the base quad goes in plain, so the blend works over colors of its own choosing
        rasterize_shaded(&plain, &mut cpu, &mut Resources::default(), &base);
        rasterize_shaded(&plain, &mut gl, &mut Resources::default(), &base);
        rasterize_shaded(&blended, &mut cpu, &mut Resources::default(), &quad);
        rasterize_shaded(&blended, &mut gl, &mut resources, &quad);
        resolve(&mut resources, &mut gl, 0);
        assert_images_close(&cpu, &gl, &placed, 1, &format!("the {name} blend"));
    }
}

/// the blended alpha channel too: a base quad's alpha and the blend's answer are
/// the same bytes on both sides, including where the alpha equation is its own
#[test]
fn opengl_blended_alphas_come_out_the_same() {
    for (name, function) in [
        ("standard transparency", blend_func((0, 6, 7), (0, 1, 0))),
        ("an alpha of its own", blend_func((0, 6, 7), (0, 1, 6))),
        ("an additive alpha", blend_func((0, 6, 7), (0, 1, 1))),
    ] {
        let Some(mut resources) = opengl_resources() else {
            return;
        };
        let mut plain = registers(0);
        plain[REG_BLEND_COLOR] = BLEND_CONSTANT;
        let mut blended = plain.clone();
        blended[REG_COLOR_OPERATION] = 0x100;
        blended[REG_BLEND_FUNC] = function;
        let base = cover((0.0, 0.0, 16.0, 16.0), -0.5, FLAT);
        let quad = slanted([OTHER, OTHER, OTHER], [1.0, 1.0, 1.0], -0.4);
        let mut cpu = CountingMemory::zeroed(0x1_0000);
        let mut gl = CountingMemory::zeroed(0x1_0000);
        rasterize_shaded(&plain, &mut cpu, &mut Resources::default(), &base);
        rasterize_shaded(&plain, &mut gl, &mut Resources::default(), &base);
        rasterize_shaded(&blended, &mut cpu, &mut Resources::default(), &quad);
        rasterize_shaded(&blended, &mut gl, &mut resources, &quad);
        resolve(&mut resources, &mut gl, 0);
        let (cpu_pixels, gl_pixels) = (image(&cpu), image(&gl));
        let mut worst = 0u8;
        let mut worst_at = 0usize;
        for index in 0..(BUFFER * BUFFER) as usize {
            // image() keeps the guest's ABGR order, so the alpha is byte 0
            let off = cpu_pixels[index][0].abs_diff(gl_pixels[index][0]);
            if off > worst {
                worst = off;
                worst_at = index;
            }
        }
        assert!(
            worst <= 1,
            "the {name} blend left alphas {} apart at ({}, {}) ({} against {})",
            worst,
            worst_at as u32 % BUFFER,
            worst_at as u32 / BUFFER,
            cpu_pixels[worst_at][0],
            gl_pixels[worst_at][0],
        );
    }
}

/// partial color masks change only the channels they leave on, byte for byte,
/// the masked ones holding what the base quad put there
#[test]
fn opengl_partial_color_masks_keep_the_masked_channels() {
    let masks: [(&str, u32, [bool; 4]); 6] = [
        ("alpha off", 0xF << 8 & !(1 << 11), [true, true, true, false]),
        ("green off", 0xF << 8 & !(1 << 9), [true, false, true, true]),
        (
            "red and blue off",
            0xF << 8 & !(1 << 8 | 1 << 10),
            [false, true, false, true],
        ),
        ("blue only", 0xF << 8 & (1 << 10), [false, false, true, false]),
        (
            "green and alpha only",
            0xF << 8 & (1 << 9 | 1 << 11),
            [false, true, false, true],
        ),
        ("nothing writable", 0, [false, false, false, false]),
    ];
    for (name, mask, writable) in masks {
        let Some(mut resources) = opengl_resources() else {
            return;
        };
        let plain = registers(0);
        let mut masked = plain.clone();
        masked[REG_DEPTH_COLOR_MASK] = mask;
        let base = cover((0.0, 0.0, 16.0, 16.0), -0.5, FLAT);
        let quad = slanted([OTHER, OTHER, OTHER], [1.0, 1.1, 0.9], -0.4);
        let mut cpu = CountingMemory::zeroed(0x1_0000);
        let mut gl = CountingMemory::zeroed(0x1_0000);
        rasterize_shaded(&plain, &mut cpu, &mut Resources::default(), &base);
        rasterize_shaded(&plain, &mut gl, &mut Resources::default(), &base);
        let underneath = gl.clone();
        rasterize_shaded(&masked, &mut cpu, &mut Resources::default(), &quad);
        rasterize_shaded(&masked, &mut gl, &mut resources, &quad);
        resolve(&mut resources, &mut gl, 0);
        let what = format!("the {name} mask");
        assert_images_identical(&cpu, &gl, &what);
        // and the masked channels hold what was under the draw, on both sides.
        // the images keep the bytes in the buffer's ABGR order, red last, so the
        // writable mask flips over before it can pick bytes
        let writable_bytes = [writable[3], writable[2], writable[1], writable[0]];
        let under_pixels = image(&underneath);
        for (label, memory) in [("the CPU", &cpu), ("the GL", &gl)] {
            for (index, pixel) in image(memory).iter().enumerate() {
                for (channel, &writable) in writable_bytes.iter().enumerate() {
                    assert!(
                        writable || pixel[channel] == under_pixels[index][channel],
                        "{what}: {label} changed channel {channel} at ({}, {})",
                        index as u32 % BUFFER,
                        index as u32 / BUFFER,
                    );
                }
            }
        }
    }
}

/// a draw with the whole output merger on — blending, an alpha test, a partial
/// color mask, over a depth test and a stencil that writes — rides in behind a
/// plain base quad, and the quad's alpha fails the test: nothing it does may
/// touch either buffer, on either backend
#[test]
fn opengl_output_merger_states_work_together() {
    let Some(mut resources) = opengl_resources() else {
        return;
    };
    let mut plain = registers(3);
    // the base quad's depth clears the Greater test and writes, and no stencil is
    // on yet, so the byte it leaves alone is the one the prefill put there
    plain[REG_DEPTH_COLOR_MASK] = 0xF << 8 | 1 | (6 << 4) | 1 << 12;
    let mut full = plain.clone();
    // the quad blends w/ standard transparency over the constant
    full[REG_COLOR_OPERATION] = 0x100;
    full[REG_BLEND_FUNC] = blend_func((0, 6, 7), (0, 1, 0));
    full[REG_BLEND_COLOR] = BLEND_CONSTANT;
    // the alpha test holds FLAT's alpha (127) back: every fragment of the quad goes
    full[REG_ALPHA_TEST] = 1 | (4 << 4) | (100 << 8);
    // the stencil would take the replace where fragments survive, and the color
    // mask sits over blue, though nothing should reach either
    full[REG_DEPTH_COLOR_MASK] = 0xF << 8 & !(1 << 10) | 1 | (6 << 4) | 1 << 12;
    full[REG_STENCIL_TEST] = 1
        | ((Compare::Always as u32 & 0x7) << 4)
        | (0xFF << 8)
        | (0x5A << 16)
        | (0xFF << 24);
    full[REG_STENCIL_OP] = 2 | (2 << 4) | (2 << 8);
    let base = cover((0.0, 0.0, 16.0, 16.0), -0.75, FLAT);
    let quad = slanted([FLAT, FLAT, FLAT], [1.0, 0.9, 1.1], -0.85);
    let depth_bytes = 4u32;
    let mut cpu = CountingMemory::prefilled(depth_bytes, |_| (0x33 << 24) | STORED_D24);
    let mut gl = cpu.clone();
    rasterize_shaded(&plain, &mut cpu, &mut Resources::default(), &base);
    rasterize_shaded(&plain, &mut gl, &mut resources, &base);
    // a reader pulls the base's frame out of the GL before the next draw goes in
    resolve(&mut resources, &mut gl, depth_bytes);
    let after_base = (cpu.clone(), gl.clone());
    rasterize_shaded(&full, &mut cpu, &mut Resources::default(), &quad);
    rasterize_shaded(&full, &mut gl, &mut resources, &quad);
    resolve(&mut resources, &mut gl, depth_bytes);
    let what = "the refused output merger draw";
    // neither backend moved a byte: the alpha test came before everything else
    assert_images_identical(&cpu, &after_base.0, &format!("{what}, the CPU stood still"));
    assert_images_identical(&gl, &after_base.1, &format!("{what}, the GL stood still"));
    assert_images_identical(&cpu, &gl, what);
    for (label, memory) in [("the CPU", &cpu), ("the GL", &gl)] {
        for (sample, &word) in depth_image(memory, 4).iter().enumerate() {
            assert_eq!(
                (word >> 24) as u8,
                0x33,
                "{what}: {} left the stencil at ({}, {}) off the prefill, where only the alpha test was asked",
                label,
                sample as u32 % BUFFER,
                sample as u32 / BUFFER,
            );
        }
    }
}

/// the same output merger draw, but the quad's alpha passes, so the blend, the
/// mask, the stencil and the depth write all land where the quad covers: the
/// backends agree pixel for pixel and on the stencil byte's spread
#[test]
fn opengl_output_merger_states_work_together_when_the_alpha_passes() {
    let Some(mut resources) = opengl_resources() else {
        return;
    };
    let mut plain = registers(3);
    // the base quad's depth clears the Greater test and writes, no stencil yet
    plain[REG_DEPTH_COLOR_MASK] = 0xF << 8 | 1 | (6 << 4) | 1 << 12;
    let mut full = plain.clone();
    // source alpha over one minus source alpha
    full[REG_COLOR_OPERATION] = 0x100;
    full[REG_BLEND_FUNC] = blend_func((0, 6, 7), (0, 1, 0));
    full[REG_BLEND_COLOR] = BLEND_CONSTANT;
    // the alpha test lets an alpha of 229 past the Greater against 150
    full[REG_ALPHA_TEST] = 1 | (6 << 4) | (150 << 8);
    // green stays as the blend leaves the rest of it, and the stencil takes the
    // replace where the quad's fragments survive
    full[REG_DEPTH_COLOR_MASK] = 0xF << 8 & !(1 << 9) | 1 | (6 << 4) | 1 << 12;
    full[REG_STENCIL_TEST] = 1
        | ((Compare::Always as u32 & 0x7) << 4)
        | (0xFF << 8)
        | (0x5A << 16)
        | (0xFF << 24);
    full[REG_STENCIL_OP] = 2 | (2 << 4) | (2 << 8);
    let base = cover((0.0, 0.0, 16.0, 16.0), -0.75, FLAT);
    let quad = slanted(
        [
            [0.7, 0.3, 0.9, 0.9],
            [0.4, 0.8, 0.2, 0.9],
            [0.9, 0.5, 0.6, 0.9],
        ],
        [1.0, 1.0, 1.0],
        -0.85,
    );
    let placed = placed_triangles(&full, &quad);
    let depth_bytes = 4u32;
    let mut cpu = CountingMemory::prefilled(depth_bytes, |_| (0x33 << 24) | STORED_D24);
    let mut gl = cpu.clone();
    rasterize_shaded(&plain, &mut cpu, &mut Resources::default(), &base);
    rasterize_shaded(&plain, &mut gl, &mut resources, &base);
    // the base's frame is read back before the quad's draw goes in
    resolve(&mut resources, &mut gl, depth_bytes);
    let after_base = gl.clone();
    rasterize_shaded(&full, &mut cpu, &mut Resources::default(), &quad);
    rasterize_shaded(&full, &mut gl, &mut resources, &quad);
    resolve(&mut resources, &mut gl, depth_bytes);
    let what = "the passing output merger draw";
    let covered = coverage(&cpu, &after_base);
    assert!(covered.iter().any(|&covered| covered), "{what}: nothing drew");
    assert_coverage_agrees(&cpu, &gl, &after_base, what);
    assert_images_close(&cpu, &gl, &placed, 1, what);
    // where the quad drew, the depth moved a rounding step apart at most, and the
    // stencil took the replace there while keeping the base's byte elsewhere
    assert_depth_close(&cpu, &gl, depth_bytes, &covered, 2, what);
    // the depth buffer sits tiled in memory, so the stencil byte of the pixel at
    // (x, y) lives at its morton spot, not at the sample's linear offset
    let stencil_at = |memory: &CountingMemory, x: u32, y: u32| -> u8 {
        let at = (DEPTH + morton_offset(x, BUFFER - 1 - y, BUFFER, 4)) as usize;
        (u32::from_le_bytes(memory.flat[at..at + 4].try_into().expect("D24S8 word")) >> 24) as u8
    };
    for (label, memory) in [("the CPU", &cpu), ("the GL", &gl)] {
        for y in 0..BUFFER {
            for x in 0..BUFFER {
                let expected = if covered[(y * BUFFER + x) as usize] {
                    0x5A
                } else {
                    0x33
                };
                assert_eq!(
                    stencil_at(memory, x, y),
                    expected,
                    "{what}: {label} left the stencil at ({x}, {y}) off the state's answer",
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// the lazy FBO cache: uploads, readbacks and guards
// ---------------------------------------------------------------------------

/// consecutive draws to the same target keep it resident: one upload of the color
/// and one of the depth at the start, no uploads or readbacks between draws, and
/// the CPU's later reads and writes over the ranges guarded each draw
#[test]
fn opengl_consecutive_draws_stay_resident() {
    let Some(mut resources) = opengl_resources() else {
        return;
    };
    let mut registers = registers(0);
    // the depth buffer takes part: test on, writing
    registers[REG_DEPTH_COLOR_MASK] = 0xF << 8 | 1 << 12 | 1;
    let depth_bytes = depth_bytes_of(0);
    let mut gl = CountingMemory::prefilled(2, |sample| STORED_D16 + sample);
    let viewport = (0.0, 0.0, 16.0, 16.0);
    let full = cover(viewport, -0.5, FLAT);
    let half = vec![
        corner_at(viewport, 0.0, 0.0, -0.5, 1.0, OTHER),
        corner_at(viewport, 32.0, 0.0, -0.5, 1.0, OTHER),
        corner_at(viewport, 0.0, 8.0, -0.5, 1.0, OTHER),
    ];
    rasterize_shaded(&registers, &mut gl, &mut resources, &full);
    assert_eq!(
        gl.reads_over(COLOR, COLOR_BYTES),
        1,
        "the first draw uploads the color once"
    );
    assert_eq!(
        gl.reads_over(DEPTH, depth_bytes * 256),
        1,
        "the first draw uploads the depth once"
    );
    assert_eq!(
        gl.writes_over(COLOR, COLOR_BYTES),
        0,
        "nothing was read back yet"
    );
    assert_eq!(
        gl.writes_over(DEPTH, depth_bytes * 256),
        0,
        "nothing was read back yet"
    );
    assert!(
        gl.guarded_reads(COLOR, COLOR_BYTES),
        "the draw guards the CPU's reads"
    );
    assert!(
        gl.guarded_writes(COLOR, COLOR_BYTES),
        "the draw guards the CPU's writes"
    );
    assert!(
        gl.guarded_reads(DEPTH, depth_bytes * 256),
        "the draw guards the depth's reads"
    );
    assert!(
        gl.guarded_writes(DEPTH, depth_bytes * 256),
        "the draw guards the depth's writes"
    );

    rasterize_shaded(&registers, &mut gl, &mut resources, &half);
    assert_eq!(
        gl.reads_over(COLOR, COLOR_BYTES),
        1,
        "the second draw kept the color resident, no upload again",
    );
    assert_eq!(
        gl.reads_over(DEPTH, depth_bytes * 256),
        1,
        "the second draw kept the depth resident, no upload again",
    );
    assert_eq!(
        gl.writes_over(COLOR, COLOR_BYTES),
        0,
        "no readback between draws"
    );
    assert_eq!(
        gl.writes_over(DEPTH, depth_bytes * 256),
        0,
        "no readback between draws"
    );
}

/// an explicit sync writes the drawn pixels into guest memory and invalidates the
/// cache, so the next draw uploads what the sync left; a depth-only sync leaves
/// the color resident and the next draw skips its upload
#[test]
fn opengl_sync_resolves_and_invalidates() {
    let Some(mut resources) = opengl_resources() else {
        return;
    };
    let mut registers = registers(0);
    registers[REG_DEPTH_COLOR_MASK] = 0xF << 8 | 1 << 12 | 1;
    let depth_bytes = depth_bytes_of(0);
    let mut gl = CountingMemory::prefilled(2, |sample| STORED_D16 + sample);
    let full = cover((0.0, 0.0, 16.0, 16.0), -0.5, FLAT);
    rasterize_shaded(&registers, &mut gl, &mut resources, &full);

    resources
        .opengl
        .as_mut()
        .unwrap()
        .sync(&mut gl, COLOR, COLOR_BYTES, true);
    assert_eq!(
        gl.writes_over(COLOR, COLOR_BYTES),
        1,
        "the sync wrote the drawn pixels back"
    );
    rasterize_shaded(
        &registers,
        &mut gl,
        &mut resources,
        &cover((0.0, 0.0, 16.0, 16.0), -0.5, OTHER),
    );
    assert_eq!(
        gl.reads_over(COLOR, COLOR_BYTES),
        2,
        "the invalidated surface uploads again on the next draw",
    );

    // a depth-only sync resolves the depth; the same key's color may ride along
    // (a dirty key resolves whole), but nothing uploads and the depth lands
    let before_reads = gl.reads_over(COLOR, COLOR_BYTES);
    resources
        .opengl
        .as_mut()
        .unwrap()
        .sync(&mut gl, DEPTH, depth_bytes * 256, false);
    assert!(
        gl.writes_over(DEPTH, depth_bytes * 256) >= 1,
        "the depth sync wrote the drawn depth back",
    );
    assert_eq!(
        gl.reads_over(COLOR, COLOR_BYTES),
        before_reads,
        "the depth sync uploaded something it should not have",
    );
    rasterize_shaded(
        &registers,
        &mut gl,
        &mut resources,
        &cover((0.0, 0.0, 16.0, 16.0), -0.75, FLAT),
    );
    assert_eq!(
        gl.reads_over(COLOR, COLOR_BYTES),
        before_reads,
        "a depth-only sync keeps the color resident",
    );
}

/// the textured quad the render-to-texture test draws, cover() w/ the texture
/// environment passing the vertex color thru
fn textured_draw(registers: &mut Vec<u32>, memory: &mut CountingMemory, resources: &mut Resources) {
    let viewport = (0.0, 0.0, 16.0, 16.0);
    let quad: Vec<Vertex> = cover(viewport, -0.75, OTHER);
    rasterize_shaded(registers, memory, resources, &quad);
}

/// a textured draw matches the CPU, including the render-to-texture order: a
/// texture that overlaps the target the last GL draw still holds samples what
/// that draw left, resolved into guest memory before the decode
#[test]
fn opengl_textured_draws_match_the_cpu() {
    let Some(mut resources) = opengl_resources() else {
        return;
    };
    let mut registers = registers(0);
    let viewport = (0.0, 0.0, 16.0, 16.0);
    let mut cpu = CountingMemory::zeroed(0x1_0000);
    let mut gl = CountingMemory::zeroed(0x1_0000);
    // first a plain draw the GL keeps resident
    let first = cover(viewport, -0.5, FLAT);
    rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &first);
    rasterize_shaded(&registers, &mut gl, &mut resources, &first);
    // then the texture registers arrive and the draw samples what it sits on:
    // stage 0 hands texture 0 straight thru, so the second draw's color is the
    // first draw's, not the second draw's own OTHER vertex color, and the
    // test fails when the resident target's bytes never reach the decode
    registers[REG_TEXTURE_CONFIG] = 1;
    registers[TEXTURE_UNIT_BASES[0] + 1] = 8 | (8 << 16);
    registers[TEXTURE_UNIT_BASES[0] + 4] = COLOR >> 3;
    registers[TEXTURE_UNIT_BASES[0] + 13] = 0; // RGBA8
    tev_stage(
        &mut registers,
        0,
        [3, 0, 0],
        [3, 0, 0],
        [0; 3],
        [0; 3],
        0,
        0,
        0,
    );
    tev_passthrough(&mut registers, &[0]);
    textured_draw(&mut registers, &mut cpu, &mut Resources::default());
    textured_draw(&mut registers, &mut gl, &mut resources);
    resolve(&mut resources, &mut gl, 0);
    assert_images_identical(&cpu, &gl, "a draw sampling its own target");
    // FLAT quantizes to [red, green, blue, alpha] 127, 63, 191, 127; image()
    // returns the guest's ABGR order, so the known covered pixel reads
    // [alpha, blue, green, red]
    assert_eq!(
        image(&gl)[8 * BUFFER as usize + 8],
        [127, 191, 63, 127],
        "the textured draw did not sample the target's FLAT color",
    );
}

// ---------------------------------------------------------------------------
// texture formats, wrap modes, filtering, mipmaps and the combiners
// ---------------------------------------------------------------------------

/// the address the texture tests lay their texture at, past the target and its
/// depth buffer
const TEXTURE: u32 = 0x5000;
/// the second texture's address
const OTHER_TEXTURE: u32 = 0x6000;

/// where each combiner stage's five registers start, the same list `tev.rs` runs on
const TEV_STAGES: [usize; 6] = [0x0C0, 0x0C8, 0x0D0, 0x0D8, 0x0F0, 0x0F8];
/// the register holding which stages write the combiner buffer
const TEV_UPDATE_BUFFER: usize = 0x0E0;
/// the register holding the combiner buffer's initial color
const TEV_BUFFER_COLOR: usize = 0x0FD;

/// a deterministic byte per index, so every test lays the same texture down
fn pattern_byte(index: usize) -> u8 {
    let mut word = (index as u32).wrapping_mul(0x9E37_79B9) ^ 0x1234_5678;
    word ^= word << 13;
    word ^= word >> 17;
    word ^= word << 5;
    word as u8
}

/// scatters a texture's linear texel bytes over its guest memory, Morton tiled
/// the way an uncompressed texture sits in it. The texels run one row after
/// another from the top, each in the format's own bytes, the byte order
/// `decode_bytes` reads back
fn tiled_texture(texels: &[u8], raw_format: u32, width: u32, height: u32) -> Vec<u8> {
    let format = crate::texture::TextureFormat::from_raw(raw_format);
    let bits = format.bits_per_pixel();
    let bytes_per_texel = (bits / 8).max(1) as usize;
    let mut tiled = vec![0u8; texture_bytes(format, width, height, 1)];
    for y in 0..height {
        for x in 0..width {
            let source = ((y * width + x) as usize) * bytes_per_texel;
            if bits >= 8 {
                let at = morton_offset(x, y, width, bits / 8) as usize;
                tiled[at..at + bytes_per_texel]
                    .copy_from_slice(&texels[source..source + bytes_per_texel]);
            } else {
                // a nibble a texel, the low nibble first
                let at = morton_offset(x, y, width, bits) as usize;
                let value = texels[source];
                if at.is_multiple_of(8) {
                    tiled[at / 8] = (tiled[at / 8] & 0xF0) | (value & 0x0F);
                } else {
                    tiled[at / 8] = (tiled[at / 8] & 0x0F) | (value << 4);
                }
            }
        }
    }
    tiled
}

/// lays a texture of `texels` down in a test's memory, every level of it
fn lay_texture(
    memory: &mut CountingMemory,
    addr: u32,
    texels: &[u8],
    raw_format: u32,
    width: u32,
    height: u32,
    levels: u32,
) {
    let format = crate::texture::TextureFormat::from_raw(raw_format);
    let bytes_per_texel = (format.bits_per_pixel() / 8).max(1) as usize;
    let mut offset = 0usize;
    for level in 0..levels {
        let (w, h) = (width >> level, height >> level);
        let count = (w * h) as usize * bytes_per_texel;
        let tiled = tiled_texture(&texels[offset..offset + count], raw_format, w, h);
        let at = (addr + texture_bytes(format, width, height, level) as u32) as usize;
        memory.flat[at..at + tiled.len()].copy_from_slice(&tiled);
        offset += count;
    }
}

/// points texture unit `unit` at a texture at `addr`, w/ the unit config and
/// level of detail words the registers hold
fn bind_texture_unit(
    registers: &mut [u32],
    unit: usize,
    addr: u32,
    width: u32,
    height: u32,
    raw_format: u32,
    config: u32,
    lod: u32,
) {
    registers[REG_TEXTURE_CONFIG] |= 1 << unit;
    let base = TEXTURE_UNIT_BASES[unit];
    registers[base + 1] = height | (width << 16);
    registers[base + 2] = config;
    registers[base + 3] = lod;
    registers[base + 4] = addr >> 3;
    if unit == 0 {
        registers[base + 13] = raw_format;
    } else {
        registers[base + 5] = raw_format;
    }
}

/// sets one combiner stage from the fields, by the register numbers the PICA
/// packs them by
#[allow(clippy::too_many_arguments)]
fn tev_stage(
    registers: &mut [u32],
    stage: usize,
    color_sources: [u32; 3],
    alpha_sources: [u32; 3],
    color_operands: [u32; 3],
    alpha_operands: [u32; 3],
    ops: u32,
    constant: u32,
    scale: u32,
) {
    let base = TEV_STAGES[stage];
    registers[base] = color_sources[0]
        | (color_sources[1] << 4)
        | (color_sources[2] << 8)
        | (alpha_sources[0] << 16)
        | (alpha_sources[1] << 20)
        | (alpha_sources[2] << 24);
    registers[base + 1] = color_operands[0]
        | (color_operands[1] << 4)
        | (color_operands[2] << 8)
        | (alpha_operands[0] << 12)
        | (alpha_operands[1] << 16)
        | (alpha_operands[2] << 20);
    registers[base + 2] = ops;
    registers[base + 3] = constant;
    registers[base + 4] = scale;
}

/// every stage but the given one hands the previous result on untouched
fn tev_passthrough(registers: &mut [u32], except: &[usize]) {
    for (stage, base) in TEV_STAGES.iter().enumerate() {
        if !except.contains(&stage) {
            registers[*base] = 0x000F_000F;
        }
    }
}

/// a quad exactly over the target whose texture coordinates on the named sets
/// run from (u0, v0) at the bottom left to (u1, v1) at the top right. The
/// corners sit on the viewport's own edges, so every pixel center's coordinates
/// land inside a texel, never on the boundary a rounding step would cross.
/// Each set interpolates between its own span's ends, so one asking for
/// coordinates out of range keeps them there
fn uv_quad(viewport: (f32, f32, f32, f32), spans: &[(&usize, f32, f32, f32, f32)]) -> Vec<Vertex> {
    let (vx, vy, vw, vh) = viewport;
    let mut vertices = Vec::with_capacity(6);
    for triangle in [
        [(vx, vy), (vx + vw, vy), (vx + vw, vy + vh)],
        [(vx, vy), (vx + vw, vy + vh), (vx, vy + vh)],
    ] {
        vertices.extend(triangle.map(|(x, y)| {
            let mut vertex = corner_at(viewport, x, y, -0.5, 1.0, FLAT);
            // where the corner sits across the quad, zero at the first corner
            let (u, v) = ((x - vx) / vw, (y - vy) / vh);
            for (set, span_u0, span_v0, span_u1, span_v1) in spans.iter() {
                vertex.texcoords[**set] = [
                    span_u0 + (span_u1 - span_u0) * u,
                    span_v0 + (span_v1 - span_v0) * v,
                ];
            }
            vertex
        }));
    }
    vertices
}

/// every texture format decodes the same guest bytes into both backends'
/// samples, Zakuro's own decoder feeding each its texels
#[test]
fn opengl_texture_formats_match_the_cpu() {
    // (the format's raw number, bytes a texel takes in memory, a name)
    let formats = [
        (0u32, 4usize, "RGBA8"),
        (1, 3, "RGB8"),
        (2, 2, "RGBA5551"),
        (3, 2, "RGB565"),
        (4, 2, "RGBA4"),
        (5, 2, "LA8"),
        (6, 2, "HILO8"),
        (7, 1, "L8"),
        (8, 1, "A8"),
        (9, 1, "LA4"),
        (10, 1, "L4"),
        (11, 1, "A4"),
    ];
    let viewport = (0.0, 0.0, 16.0, 16.0);
    for (raw, bytes_per_texel, name) in formats {
        let Some(mut resources) = opengl_resources() else {
            return;
        };
        let mut registers = registers(0);
        // an 8x8 texture of every byte value the format's texels hold
        let texels: Vec<u8> = (0..64 * bytes_per_texel).map(pattern_byte).collect();
        let mut cpu = CountingMemory::zeroed(0x1_0000);
        let mut gl = CountingMemory::zeroed(0x1_0000);
        lay_texture(&mut cpu, TEXTURE, &texels, raw, 8, 8, 1);
        lay_texture(&mut gl, TEXTURE, &texels, raw, 8, 8, 1);
        bind_texture_unit(&mut registers, 0, TEXTURE, 8, 8, raw, 0, 0);
        // one combiner stage taking the sample straight thru
        tev_stage(
            &mut registers,
            0,
            [3, 3, 3],
            [3, 3, 3],
            [0; 3],
            [0; 3],
            0,
            0,
            0,
        );
        tev_passthrough(&mut registers, &[0]);
        let quad = uv_quad(viewport, &[(&0, 0.0, 0.0, 1.0, 1.0)]);
        rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &quad);
        rasterize_shaded(&registers, &mut gl, &mut resources, &quad);
        resolve(&mut resources, &mut gl, 0);
        assert_images_identical(&cpu, &gl, &format!("{name} samples the same"));
    }
    // the compressed formats, raw blocks of every bit pattern the decoder reads
    for (raw, block_bytes, name) in [(12u32, 8usize, "ETC1"), (13, 16, "ETC1A4")] {
        let Some(mut resources) = opengl_resources() else {
            return;
        };
        let mut registers = registers(0);
        // an 8x8 texture is one 8x8 tile of four 4x4 blocks, laid out in order
        let raw_blocks: Vec<u8> = (0..4 * block_bytes).map(pattern_byte).collect();
        let mut cpu = CountingMemory::zeroed(0x1_0000);
        let mut gl = CountingMemory::zeroed(0x1_0000);
        let at = TEXTURE as usize;
        cpu.flat[at..at + raw_blocks.len()].copy_from_slice(&raw_blocks);
        gl.flat[at..at + raw_blocks.len()].copy_from_slice(&raw_blocks);
        bind_texture_unit(&mut registers, 0, TEXTURE, 8, 8, raw, 0, 0);
        tev_stage(
            &mut registers,
            0,
            [3, 3, 3],
            [3, 3, 3],
            [0; 3],
            [0; 3],
            0,
            0,
            0,
        );
        tev_passthrough(&mut registers, &[0]);
        let quad = uv_quad(viewport, &[(&0, 0.0, 0.0, 1.0, 1.0)]);
        rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &quad);
        rasterize_shaded(&registers, &mut gl, &mut resources, &quad);
        resolve(&mut resources, &mut gl, 0);
        assert_images_identical(&cpu, &gl, &format!("{name} samples the same"));
    }
}

/// wrap modes wrap the same texels on both sides, over coordinates hanging off
/// every side of the texture, and a border clamp answers the unit's border color
#[test]
fn opengl_wrap_modes_and_border_colors_match_the_cpu() {
    let viewport = (0.0, 0.0, 16.0, 16.0);
    let texels: Vec<u8> = (0..64 * 4).map(pattern_byte).collect();
    for wrap_s in 0..4u32 {
        for wrap_t in 0..4u32 {
            let Some(mut resources) = opengl_resources() else {
                return;
            };
            let mut registers = registers(0);
            let mut cpu = CountingMemory::zeroed(0x1_0000);
            let mut gl = CountingMemory::zeroed(0x1_0000);
            lay_texture(&mut cpu, TEXTURE, &texels, 0, 8, 8, 1);
            lay_texture(&mut gl, TEXTURE, &texels, 0, 8, 8, 1);
            bind_texture_unit(
                &mut registers,
                0,
                TEXTURE,
                8,
                8,
                0,
                (wrap_t << 8) | (wrap_s << 12),
                0,
            );
            // a border color nothing else in the image answers
            registers[TEXTURE_UNIT_BASES[0]] = 0x4080_C020; // red 0x20, green 0xC0, blue 0x80, alpha 0x40
            tev_stage(
                &mut registers,
                0,
                [3, 3, 3],
                [3, 3, 3],
                [0; 3],
                [0; 3],
                0,
                0,
                0,
            );
            tev_passthrough(&mut registers, &[0]);
            let quad = uv_quad(viewport, &[(&0, -0.25, -0.25, 1.25, 1.25)]);
            rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &quad);
            rasterize_shaded(&registers, &mut gl, &mut resources, &quad);
            resolve(&mut resources, &mut gl, 0);
            assert_images_identical(&cpu, &gl, &format!("wrap s{wrap_s} t{wrap_t}"));
        }
    }
}

/// bilinear filtering lands between texels the same way on both sides,
/// magnifying and minifying
#[test]
fn opengl_linear_filtering_matches_the_cpu() {
    let viewport = (0.0, 0.0, 16.0, 16.0);
    // (config, span, levels, size, what): the minifying case needs a mip chain
    // and a coordinate rate above one texel a pixel to select minification; the
    // span is then no whole texel a pixel at the level it picks, so bilinear and
    // nearest sampling cannot land on the same texel centers
    for (config, span, levels, size, what) in [
        (2u32, (0.0f32, 0.0, 1.0, 1.0), 1u32, 8u32, "magnifying"), // bit 1, linear when magnifying
        (4, (0.0, 0.0, 2.2, 2.2), 2, 16, "minifying"),             // bit 2, linear when minifying
    ] {
        let Some(mut resources) = opengl_resources() else {
            return;
        };
        let texel_count: usize = (0..levels)
            .map(|level| ((size >> level) * (size >> level)) as usize)
            .sum();
        let texels: Vec<u8> = (0..texel_count * 4).map(pattern_byte).collect();
        let lod = (levels - 1) << 16;
        let mut registers = registers(0);
        let mut cpu = CountingMemory::zeroed(0x1_0000);
        let mut gl = CountingMemory::zeroed(0x1_0000);
        lay_texture(&mut cpu, TEXTURE, &texels, 0, size, size, levels);
        lay_texture(&mut gl, TEXTURE, &texels, 0, size, size, levels);
        bind_texture_unit(&mut registers, 0, TEXTURE, size, size, 0, config, lod);
        tev_stage(
            &mut registers,
            0,
            [3, 3, 3],
            [3, 3, 3],
            [0; 3],
            [0; 3],
            0,
            0,
            0,
        );
        tev_passthrough(&mut registers, &[0]);
        let quad = uv_quad(viewport, &[(&0, span.0, span.1, span.2, span.3)]);
        let placed = placed_triangles(&registers, &quad);
        rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &quad);
        rasterize_shaded(&registers, &mut gl, &mut resources, &quad);
        resolve(&mut resources, &mut gl, 0);
        assert_images_close(&cpu, &gl, &placed, 1, &format!("bilinear {what}"));
        if what == "minifying" {
            // the same draw with nearest minification, so the filter's work is
            // visible and not just two backends agreeing on the wrong one
            let mut nearest_registers = registers.clone();
            nearest_registers[TEXTURE_UNIT_BASES[0] + 2] = 0;
            let mut nearest = CountingMemory::zeroed(0x1_0000);
            lay_texture(&mut nearest, TEXTURE, &texels, 0, size, size, levels);
            rasterize_shaded(
                &nearest_registers,
                &mut nearest,
                &mut Resources::default(),
                &quad,
            );
            assert!(
                image(&cpu)
                    .iter()
                    .zip(image(&nearest).iter())
                    .any(|(linear, nearest)| linear != nearest),
                "minifying with linear filtering matched nearest sampling",
            );
        }
    }
}

/// the level of detail the coordinate rate picks is the level both sides
/// sample: a 32x32 texture of three flat sizes answers its own color at each
/// span, a bias moves the level, and blending the two levels about the level
/// blends the same on both sides
#[test]
fn opengl_mipmap_levels_match_the_cpu() {
    let viewport = (0.0, 0.0, 16.0, 16.0);
    // three sizes, each flat: red, green and blue as the decoder reads the bytes
    let mut texels = vec![0u8; (32 * 32 + 16 * 16 + 8 * 8) * 4];
    for texel in texels[..32 * 32 * 4].chunks_mut(4) {
        texel.copy_from_slice(&[255, 0, 0, 255]);
    }
    for texel in texels[32 * 32 * 4..32 * 32 * 4 + 16 * 16 * 4].chunks_mut(4) {
        texel.copy_from_slice(&[255, 0, 255, 0]); // decodes to green
    }
    for texel in texels[32 * 32 * 4 + 16 * 16 * 4..].chunks_mut(4) {
        texel.copy_from_slice(&[255, 255, 0, 0]); // decodes to blue
    }
    let red = [255u8, 0, 0, 255];
    let green = [255u8, 0, 255, 0];
    let blue = [255u8, 255, 0, 0];
    let mixed = [255u8, 127, 127, 0];
    // (uv span, config, lod word, the color the picked level answers in the
    // decoder's own RGBA bytes, how far a channel may sit from it: flat levels
    // answer exactly, the blended one a rounding step)
    let cases: [((f32, f32, f32, f32), u32, u32, [u8; 4], u8, &str); 6] = [
        ((0.0, 0.0, 0.5, 0.5), 0, 2 << 16, red, 0, "level 0, magnified"),
        (
            (0.0, 0.0, 1.0, 1.0),
            0,
            2 << 16,
            green,
            0,
            "level 1, one texel a pixel",
        ),
        (
            (0.0, 0.0, 2.0, 2.0),
            0,
            2 << 16,
            blue,
            0,
            "level 2, four texels a pixel",
        ),
        (
            (0.0, 0.0, 1.41421354, 1.41421354),
            0,
            // just under the halfway point, so float noise in the derivatives
            // cannot tip the rounding over the tie
            (2 << 16) | 0x1FFF,
            green,
            0,
            "near halfway takes the bigger level",
        ),
        (
            (0.0, 0.0, 1.0, 1.0),
            0,
            (2 << 16) | 0xFF00,
            red,
            0,
            "a bias of minus one",
        ),
        (
            (0.0, 0.0, 1.41421354, 1.41421354),
            1 << 24,
            2 << 16,
            mixed,
            1,
            "the two levels about the level blend",
        ),
    ];
    for (span, config, lod, expected, tolerance, name) in cases {
        let Some(mut resources) = opengl_resources() else {
            return;
        };
        let mut registers = registers(0);
        let mut cpu = CountingMemory::zeroed(0x1_0000);
        let mut gl = CountingMemory::zeroed(0x1_0000);
        lay_texture(&mut cpu, TEXTURE, &texels, 0, 32, 32, 3);
        lay_texture(&mut gl, TEXTURE, &texels, 0, 32, 32, 3);
        bind_texture_unit(&mut registers, 0, TEXTURE, 32, 32, 0, config, lod);
        tev_stage(
            &mut registers,
            0,
            [3, 3, 3],
            [3, 3, 3],
            [0; 3],
            [0; 3],
            0,
            0,
            0,
        );
        tev_passthrough(&mut registers, &[0]);
        let quad = uv_quad(viewport, &[(&0, span.0, span.1, span.2, span.3)]);
        rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &quad);
        rasterize_shaded(&registers, &mut gl, &mut resources, &quad);
        resolve(&mut resources, &mut gl, 0);
        assert_images_identical(&cpu, &gl, name);
        // the raw guest bytes run ABGR, the colors the fixtures lay down with
        // them, so every channel of every pixel answers the level's own color
        for (index, pixel) in image(&cpu).iter().enumerate() {
            for (channel, (&got, &want)) in pixel.iter().zip(&expected).enumerate() {
                assert!(
                    got.abs_diff(want) <= tolerance,
                    "{name}: pixel {index} channel {channel} is {got}, not the level the rate picks ({want})",
                );
            }
        }
    }
}

/// every combiner operation folds its three sources the same on both sides, the
/// operands picking which channels feed it
#[test]
fn opengl_tev_combiner_operations_match_the_cpu() {
    let viewport = (0.0, 0.0, 16.0, 16.0);
    let texels: Vec<u8> = (0..64 * 4).map(pattern_byte).collect();
    for (op, name) in [
        (0u32, "replace"),
        (1, "modulate"),
        (2, "add"),
        (3, "add signed"),
        (4, "lerp"),
        (5, "subtract"),
        (6, "dot3 rgb"),
        (7, "dot3 rgba"),
        (8, "multiply add"),
        (9, "add multiply"),
    ] {
        let Some(mut resources) = opengl_resources() else {
            return;
        };
        let mut registers = registers(0);
        let mut cpu = CountingMemory::zeroed(0x1_0000);
        let mut gl = CountingMemory::zeroed(0x1_0000);
        lay_texture(&mut cpu, TEXTURE, &texels, 0, 8, 8, 1);
        lay_texture(&mut gl, TEXTURE, &texels, 0, 8, 8, 1);
        bind_texture_unit(&mut registers, 0, TEXTURE, 8, 8, 0, 0, 0);
        // texture 0, the stage constant and the vertex color in the three inputs
        tev_stage(
            &mut registers,
            0,
            [3, 0xE, 0x0],
            [3, 0xE, 0x0],
            [0; 3],
            [0; 3],
            op | (op << 16),
            0x4080_C020,
            0,
        );
        tev_passthrough(&mut registers, &[0]);
        let quad = uv_quad(viewport, &[(&0, 0.0, 0.0, 1.0, 1.0)]);
        let placed = placed_triangles(&registers, &quad);
        rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &quad);
        rasterize_shaded(&registers, &mut gl, &mut resources, &quad);
        resolve(&mut resources, &mut gl, 0);
        assert_images_close(&cpu, &gl, &placed, 1, &format!("combiner {name}"));
    }
}

/// the combiner buffer hands a stage's result to the stage after next, the
/// buffer's own color in between, color and alpha updating separately, and the
/// output scales multiplying past one
#[test]
fn opengl_tev_buffer_and_scales_match_the_cpu() {
    let viewport = (0.0, 0.0, 16.0, 16.0);
    let texels: Vec<u8> = (0..64 * 4).map(pattern_byte).collect();
    let cases: [(u32, u32, &str); 3] = [
        (0x1100, 0, "color and alpha update the buffer"),
        (0x0100, 0, "color alone updates the buffer"),
        (0x1100, 0x0001_0001, "the output scale doubles"),
    ];
    for (update, scale, name) in cases {
        let Some(mut resources) = opengl_resources() else {
            return;
        };
        let mut registers = registers(0);
        let mut cpu = CountingMemory::zeroed(0x1_0000);
        let mut gl = CountingMemory::zeroed(0x1_0000);
        lay_texture(&mut cpu, TEXTURE, &texels, 0, 8, 8, 1);
        lay_texture(&mut gl, TEXTURE, &texels, 0, 8, 8, 1);
        bind_texture_unit(&mut registers, 0, TEXTURE, 8, 8, 0, 0, 0);
        // stage 0 takes the texture thru, writing the buffer
        tev_stage(
            &mut registers,
            0,
            [3, 3, 3],
            [3, 3, 3],
            [0; 3],
            [0; 3],
            0,
            0,
            0,
        );
        registers[TEV_UPDATE_BUFFER] = update;
        registers[TEV_BUFFER_COLOR] = 0x4080_C020;
        // stage 2 folds the buffer stage 0 wrote over the constant, signed, at a
        // scale: reading it on both the color and the alpha sides, so the test
        // fails when a backend ignores the buffer's updates, and the color-only
        // case leaves the alpha at the configured buffer color's own alpha
        tev_stage(
            &mut registers,
            2,
            [0xD, 0xE, 0xF],
            [0xD, 0xE, 0xF],
            [0; 3],
            [0; 3],
            3 | (3 << 16),
            0x4080_C020,
            scale,
        );
        tev_passthrough(&mut registers, &[0, 2]);
        let quad = uv_quad(viewport, &[(&0, 0.0, 0.0, 1.0, 1.0)]);
        let placed = placed_triangles(&registers, &quad);
        rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &quad);
        rasterize_shaded(&registers, &mut gl, &mut resources, &quad);
        resolve(&mut resources, &mut gl, 0);
        assert_images_close(&cpu, &gl, &placed, 1, name);
    }
}

/// two units sample in one draw, unit 2 reads coordinate set 1 instead of its
/// own, and a projected unit 0 divides its coordinates by set 0's w
#[test]
fn opengl_texture_units_and_coordinates_match_the_cpu() {
    let viewport = (0.0, 0.0, 16.0, 16.0);
    let first: Vec<u8> = (0..64 * 4).map(pattern_byte).collect();
    let second: Vec<u8> = (0..64 * 4)
        .map(|index| pattern_byte(index ^ 0x55))
        .collect();
    // (a) units 0 and 1 modulate each other, each on its own coordinate set
    {
        let Some(mut resources) = opengl_resources() else {
            return;
        };
        let mut registers = registers(0);
        let mut cpu = CountingMemory::zeroed(0x1_0000);
        let mut gl = CountingMemory::zeroed(0x1_0000);
        lay_texture(&mut cpu, TEXTURE, &first, 0, 8, 8, 1);
        lay_texture(&mut gl, TEXTURE, &first, 0, 8, 8, 1);
        lay_texture(&mut cpu, OTHER_TEXTURE, &second, 0, 8, 8, 1);
        lay_texture(&mut gl, OTHER_TEXTURE, &second, 0, 8, 8, 1);
        bind_texture_unit(&mut registers, 0, TEXTURE, 8, 8, 0, 0, 0);
        bind_texture_unit(&mut registers, 1, OTHER_TEXTURE, 8, 8, 0, 0, 0);
        tev_stage(
            &mut registers,
            0,
            [3, 4, 3],
            [3, 4, 3],
            [0; 3],
            [0; 3],
            1 | (1 << 16),
            0,
            0,
        );
        tev_passthrough(&mut registers, &[0]);
        let quad = uv_quad(
            viewport,
            &[(&0, 0.0, 0.0, 1.0, 1.0), (&1, 0.25, 0.125, 0.75, 0.875)],
        );
        let placed = placed_triangles(&registers, &quad);
        rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &quad);
        rasterize_shaded(&registers, &mut gl, &mut resources, &quad);
        resolve(&mut resources, &mut gl, 0);
        assert_images_close(&cpu, &gl, &placed, 1, "units 0 and 1 modulate");
    }
    // (b) unit 2 reads coordinate set 1
    {
        let Some(mut resources) = opengl_resources() else {
            return;
        };
        let mut registers = registers(0);
        registers[REG_TEXTURE_CONFIG] |= 1 << 13;
        let mut cpu = CountingMemory::zeroed(0x1_0000);
        let mut gl = CountingMemory::zeroed(0x1_0000);
        lay_texture(&mut cpu, TEXTURE, &first, 0, 8, 8, 1);
        lay_texture(&mut gl, TEXTURE, &first, 0, 8, 8, 1);
        bind_texture_unit(&mut registers, 2, TEXTURE, 8, 8, 0, 0, 0);
        tev_stage(
            &mut registers,
            0,
            [5, 5, 5],
            [5, 5, 5],
            [0; 3],
            [0; 3],
            0,
            0,
            0,
        );
        tev_passthrough(&mut registers, &[0]);
        let quad = uv_quad(viewport, &[(&1, 0.0, 0.0, 1.0, 1.0)]);
        rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &quad);
        rasterize_shaded(&registers, &mut gl, &mut resources, &quad);
        resolve(&mut resources, &mut gl, 0);
        assert_images_identical(&cpu, &gl, "unit 2 on set 1");
    }
    // (c) a projected unit 0 divides its coordinates by set 0's w
    {
        let Some(mut resources) = opengl_resources() else {
            return;
        };
        let mut registers = registers(0);
        let mut cpu = CountingMemory::zeroed(0x1_0000);
        let mut gl = CountingMemory::zeroed(0x1_0000);
        lay_texture(&mut cpu, TEXTURE, &first, 0, 8, 8, 1);
        lay_texture(&mut gl, TEXTURE, &first, 0, 8, 8, 1);
        bind_texture_unit(&mut registers, 0, TEXTURE, 8, 8, 0, 3 << 28, 0);
        tev_stage(
            &mut registers,
            0,
            [3, 3, 3],
            [3, 3, 3],
            [0; 3],
            [0; 3],
            0,
            0,
            0,
        );
        tev_passthrough(&mut registers, &[0]);
        let mut quad = uv_quad(viewport, &[(&0, 0.0, 0.0, 2.0, 2.0)]);
        for vertex in &mut quad {
            vertex.texcoord0_w = 2.0;
        }
        let placed = placed_triangles(&registers, &quad);
        rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &quad);
        rasterize_shaded(&registers, &mut gl, &mut resources, &quad);
        resolve(&mut resources, &mut gl, 0);
        assert_images_close(&cpu, &gl, &placed, 1, "projected coordinates");
    }
}

/// a texture decoded once rides its GL upload across draws without a second
/// read of guest memory; bytes written over it between command lists decode
/// again; and a small budget turns textures over with the pixels staying right
#[test]
fn opengl_texture_cache_reuse_invalidation_and_eviction() {
    let viewport = (0.0, 0.0, 16.0, 16.0);
    let first: Vec<u8> = (0..64 * 4).map(pattern_byte).collect();
    let second: Vec<u8> = (0..64 * 4)
        .map(|index| pattern_byte(index ^ 0xAA))
        .collect();
    let draw = |registers: &[u32], memory: &mut CountingMemory, resources: &mut Resources| {
        let quad = uv_quad(viewport, &[(&0, 0.0, 0.0, 1.0, 1.0)]);
        rasterize_shaded(registers, memory, resources, &quad);
    };
    // (a) the second draw over the same texture decodes nothing
    {
        let Some(mut resources) = opengl_resources() else {
            return;
        };
        let mut registers = registers(0);
        let mut gl = CountingMemory::zeroed(0x1_0000);
        lay_texture(&mut gl, TEXTURE, &first, 0, 8, 8, 1);
        bind_texture_unit(&mut registers, 0, TEXTURE, 8, 8, 0, 0, 0);
        tev_stage(
            &mut registers,
            0,
            [3, 3, 3],
            [3, 3, 3],
            [0; 3],
            [0; 3],
            0,
            0,
            0,
        );
        tev_passthrough(&mut registers, &[0]);
        draw(&registers, &mut gl, &mut resources);
        let reads_after_first = gl.reads_over(TEXTURE, 256);
        assert!(reads_after_first >= 1, "the first draw decoded the texture");
        draw(&registers, &mut gl, &mut resources);
        assert_eq!(
            gl.reads_over(TEXTURE, 256),
            reads_after_first,
            "the second draw re-decoded a texture nothing touched"
        );
        resolve(&mut resources, &mut gl, 0);
    }
    // (b) bytes written over the texture between command lists decode again
    {
        let Some(mut resources) = opengl_resources() else {
            return;
        };
        let mut registers = registers(0);
        let mut cpu = CountingMemory::zeroed(0x1_0000);
        let mut gl = CountingMemory::zeroed(0x1_0000);
        lay_texture(&mut cpu, TEXTURE, &first, 0, 8, 8, 1);
        lay_texture(&mut gl, TEXTURE, &first, 0, 8, 8, 1);
        bind_texture_unit(&mut registers, 0, TEXTURE, 8, 8, 0, 0, 0);
        tev_stage(
            &mut registers,
            0,
            [3, 3, 3],
            [3, 3, 3],
            [0; 3],
            [0; 3],
            0,
            0,
            0,
        );
        tev_passthrough(&mut registers, &[0]);
        draw(&registers, &mut cpu, &mut Resources::default());
        draw(&registers, &mut gl, &mut resources);
        // the CPU writes new bytes over the texture, then a new command list starts
        let replacement = tiled_texture(&second, 0, 8, 8);
        cpu.flat[TEXTURE as usize..TEXTURE as usize + replacement.len()]
            .copy_from_slice(&replacement);
        gl.flat[TEXTURE as usize..TEXTURE as usize + replacement.len()]
            .copy_from_slice(&replacement);
        let before = image(&cpu);
        resources.textures.begin_list();
        draw(&registers, &mut cpu, &mut Resources::default());
        draw(&registers, &mut gl, &mut resources);
        resolve(&mut resources, &mut gl, 0);
        assert_images_identical(&cpu, &gl, "rewritten texture bytes");
        assert_ne!(
            image(&cpu),
            before,
            "the rewritten texture answered with different bytes"
        );
    }
    // (c) a budget of one texture turns them over, the pixels stay right: the
    // second texture evicts the first, and revisiting the first uploads it
    // again instead of riding a stale copy
    {
        let Some(mut resources) = opengl_resources() else {
            return;
        };
        resources
            .opengl
            .as_mut()
            .unwrap()
            .limit_texture_cache(8 * 8 * 4);
        let mut registers = registers(0);
        let mut gl = CountingMemory::zeroed(0x1_0000);
        lay_texture(&mut gl, TEXTURE, &first, 0, 8, 8, 1);
        lay_texture(&mut gl, OTHER_TEXTURE, &second, 0, 8, 8, 1);
        tev_stage(
            &mut registers,
            0,
            [3, 3, 3],
            [3, 3, 3],
            [0; 3],
            [0; 3],
            0,
            0,
            0,
        );
        tev_passthrough(&mut registers, &[0]);
        let mut cpu = CountingMemory::zeroed(0x1_0000);
        for (visit, (addr, texels)) in [
            (TEXTURE, &first),
            (OTHER_TEXTURE, &second),
            (TEXTURE, &first),
        ]
        .into_iter()
        .enumerate()
        {
            let mut registers = registers.clone();
            lay_texture(&mut gl, addr, texels, 0, 8, 8, 1);
            lay_texture(&mut cpu, addr, texels, 0, 8, 8, 1);
            registers[REG_TEXTURE_CONFIG] = 1;
            registers[TEXTURE_UNIT_BASES[0] + 1] = 8 | (8 << 16);
            registers[TEXTURE_UNIT_BASES[0] + 2] = 0;
            registers[TEXTURE_UNIT_BASES[0] + 3] = 0;
            registers[TEXTURE_UNIT_BASES[0] + 4] = addr >> 3;
            registers[TEXTURE_UNIT_BASES[0] + 13] = 0;
            draw(&registers, &mut cpu, &mut Resources::default());
            draw(&registers, &mut gl, &mut resources);
            assert_eq!(
                resources.opengl.as_ref().unwrap().texture_count(),
                1,
                "visit {visit} did not turn the one-texture cache over",
            );
            resolve(&mut resources, &mut gl, 0);
            assert_images_identical(
                &cpu,
                &gl,
                &format!("after turning the cache over to {addr:#x}"),
            );
        }
    }
}
/// a draw the OpenGL path cannot take — a logic op, a depth buffer over its own
/// color target, a blend factor GL lacks or procedural textures — falls back to
/// the CPU rasterizer
/// after syncing what the GL drew, and the bytes match a run that never saw the
/// GL at all. Fragment lighting, the fog and RGB8 targets used to be on this
/// list; they draw in the GL now, and their tests are differentials of their own.
#[test]
fn opengl_unsupported_draws_fall_back_after_syncing_and_match_the_cpu() {
    // (an unaligned target would join these, but the CPU rasterizer itself panics
    // on tile-unaligned widths today, see the ignored regression test and TODO.md)
    type StateChange = fn(&mut Vec<u32>);
    let cases: [(&str, StateChange); 4] = [
        ("a logic op", |registers| {
            registers[REG_COLOR_OPERATION] = 0;
            registers[REG_LOGIC_OP] = 6; // XOR, not the copy that reads as none
        }),
        ("a depth buffer over its own color target", |registers| {
            // the GL path refuses a target whose depth shares its color bytes,
            // the software fills both from the same memory in order
            registers[REG_DEPTH_COLOR_MASK] = 0xF << 8 | 1 << 12; // the depth test on and writing
            registers[REG_DEPTH_BUFFER_ADDRESS] = (COLOR + 256) >> 3;
        }),
        ("a blend factor GL lacks", |registers| {
            // GL takes the source alpha saturate factor as the color side's
            // source alone, a draw using it as a destination stays on the CPU
            registers[REG_COLOR_OPERATION] = 0x100;
            registers[REG_BLEND_FUNC] = 0x11E1_0000; // color side: add, one, the saturate factor as the destination
        }),
        ("procedural textures", |registers| {
            registers[REG_TEXTURE_CONFIG] |= 1 << 10;
        }),
    ];
    for (name, apply) in cases {
        let Some((cpu, gl, accelerated_before)) = (|| {
            let mut resources = opengl_resources()?;
            let mut registers = registers(0);
            let viewport = (0.0, 0.0, 16.0, 16.0);
            let mut cpu = CountingMemory::zeroed(0x1_0000);
            let mut gl = CountingMemory::zeroed(0x1_0000);
            // first a draw the GL takes, so the fallback has something to sync over
            let first = cover(viewport, -0.5, FLAT);
            rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &first);
            rasterize_shaded(&registers, &mut gl, &mut resources, &first);
            assert_eq!(
                gl.writes_over(COLOR, COLOR_BYTES),
                0,
                "{name}: the accelerated draw read back when nothing asked",
            );
            let accelerated_before = gl.writes_over(COLOR, COLOR_BYTES);
            // then the draw whose state the GL path does not take
            apply(&mut registers);
            let second = cover(viewport, -0.75, OTHER);
            rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &second);
            rasterize_shaded(&registers, &mut gl, &mut resources, &second);
            Some((cpu, gl, accelerated_before))
        })() else {
            return;
        };
        assert!(
            gl.writes_over(COLOR, COLOR_BYTES) > accelerated_before,
            "{name}: the fallback ran without syncing what the GL drew first",
        );
        assert_images_identical(&cpu, &gl, &format!("fallback on {name}"));
    }
}

/// a second target overlapping the first's color and depth ranges resolves it:
/// after the second draw, guest memory holds the first's pixels where the second
/// never reached, and the second drew over the first's own bytes on both sides
#[test]
fn opengl_overlapping_surfaces_resolve_each_other() {
    let Some(mut resources) = opengl_resources() else {
        return;
    };
    let map = DepthMap {
        scale: -1.0,
        offset: 0.0,
        w_buffer: false,
    };
    let screen_triangle = [
        Screen {
            x: 0.0,
            y: 0.0,
            z: -0.5,
            inv_w: 1.0,
            color_over_w: FLAT,
            texcoords_over_w: [[0.0; 2]; 3],
            quaternion_over_w: [0.0; 4],
            view_over_w: [0.0; 3],
            texcoord0_w_over_w: 1.0,
        },
        Screen {
            x: 32.0,
            y: 0.0,
            z: -0.5,
            inv_w: 1.0,
            color_over_w: FLAT,
            texcoords_over_w: [[0.0; 2]; 3],
            quaternion_over_w: [0.0; 4],
            view_over_w: [0.0; 3],
            texcoord0_w_over_w: 1.0,
        },
        Screen {
            x: 0.0,
            y: 32.0,
            z: -0.5,
            inv_w: 1.0,
            color_over_w: FLAT,
            texcoords_over_w: [[0.0; 2]; 3],
            quaternion_over_w: [0.0; 4],
            view_over_w: [0.0; 3],
            texcoord0_w_over_w: 1.0,
        },
    ];
    let target = |addr: u32| ColorTarget {
        addr,
        left: 0,
        bottom: 0,
        right: 16,
        top: 16,
        buffer_width: 16,
        buffer_height: 16,
        format: ColorFormat::Rgba8,
        write: [true; 4],
    };
    let depth = |addr: u32| DepthStencil {
        addr,
        bytes: 2,
        test: None,
        write_depth: true,
        stencil: None,
        write_stencil: false,
    };
    let mut gl = CountingMemory::zeroed(0x1_0000);
    let mut cpu = CountingMemory::zeroed(0x1_0000);
    let output = opengl::OutputState::default();
    let textures = opengl::TextureSetup {
        units: [None, None, None],
        texture0_projected: false,
        texture2_uses_coord1: false,
        tev: crate::tev::TevUniforms {
            all_passthrough: true,
            ..Default::default()
        },
    };
    let fragment = opengl::FragmentSetup {
        lighting: None,
        tables: &resources.light_tables,
        fog: None,
    };
    {
        let renderer = resources.opengl.as_mut().unwrap();
        // A: color at 0x1000 and D16 depth at 0x3000
        renderer
            .draw(
                &mut gl,
                &target(0x1000),
                Some(&depth(0x3000)),
                map,
                &output,
                &textures,
                &fragment,
                &[screen_triangle],
            )
            .expect("the first draw");
        // B's color range overlaps A's tail and its depth overlaps A's depth
        renderer
            .draw(
                &mut gl,
                &target(0x1100),
                Some(&depth(0x3100)),
                map,
                &output,
                &textures,
                &fragment,
                &[screen_triangle],
            )
            .expect("the second draw");
        // what a later reader gets: B, and with it the tail of A it resolves over
        renderer.sync(&mut gl, 0x1100, 0x400, true);
        renderer.sync(&mut gl, 0x3100, 0x200, true);
    }
    // the CPU's own sequential run of the same two draws, the same single
    // triangle the renderer was handed
    {
        let mut registers = registers(0);
        registers[REG_DEPTH_COLOR_MASK] = 0xF << 8 | 1 << 12;
        let vertices: Vec<Vertex> = [[0.0f32, 0.0], [32.0, 0.0], [0.0, 32.0]]
            .map(|[x, y]| corner_at((0.0, 0.0, 16.0, 16.0), x, y, -0.5, 1.0, FLAT))
            .into();
        rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &vertices);
        registers[REG_COLOR_BUFFER_ADDRESS] = 0x1100 >> 3;
        registers[REG_DEPTH_BUFFER_ADDRESS] = 0x3100 >> 3;
        rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &vertices);
    }
    assert!(
        gl.writes_over(0x1000, 0x400) >= 1,
        "A's color was resolved into memory when B overlapped it",
    );
    assert!(
        gl.writes_over(0x3000, 0x200) >= 1,
        "A's depth was resolved into memory when B overlapped it",
    );
    // color bytes equal, the whole span A and B cover; the depths they wrote
    // round one step apart, like every depth write
    assert_eq!(
        gl.region(0x1000, 0x500),
        cpu.region(0x1000, 0x500),
        "overlapping targets left different color bytes",
    );
    for at in (0x3000..0x3300u32).step_by(2) {
        let cpu_word = u16::from_le_bytes(cpu.region(at, 2).try_into().unwrap()) as i32;
        let gl_word = u16::from_le_bytes(gl.region(at, 2).try_into().unwrap()) as i32;
        assert!(
            (cpu_word - gl_word).abs() <= 2,
            "overlapping targets left depth {cpu_word} against {gl_word} at 0x{at:04X}",
        );
    }
}

/// the CPU rasterizer itself panics filling a target whose width is not a whole
/// number of tiles: the spill tile of the last row of tiles is cut off the copy
/// (Surface::load clamps it to `total`, then Band::at indexes past it). Titles
/// only ever draw into tile-aligned buffers, so it never shows, but the GL path
/// skips unaligned dimensions for the same reason. When this is fixed on the CPU
/// side, un-ignore and extend it into a full GL fallback differential like the
/// rest of the fallback cases.
#[test]
#[ignore = "the CPU rasterizer panics on widths that are not multiples of 8 (see TODO.md)"]
fn cpu_draws_survive_an_unaligned_target_width() {
    let mut registers = registers(0);
    registers[REG_FRAMEBUFFER_DIMENSIONS] = 12 | (11 << 12) | 1 << 24;
    registers[REG_VIEWPORT_WIDTH] = float24(6.0);
    registers[REG_VIEWPORT_HEIGHT] = float24(6.0);
    let mut memory = CountingMemory::zeroed(0x1_0000);
    let triangle = cover((0.0, 0.0, 12.0, 12.0), -0.5, FLAT);
    rasterize_shaded(
        &registers,
        &mut memory,
        &mut Resources::default(),
        &triangle,
    );
}

/// a memory fill over a buffer the GL still holds resolves it first (what
/// Gpu::memory_fill's hook does), the fill lands on the drawn pixels, and the
/// next draw starts from the filled bytes on both sides
#[test]
fn opengl_fills_take_the_drawn_pixels_first() {
    let Some(mut resources) = opengl_resources() else {
        return;
    };
    let registers = registers(0);
    let viewport = (0.0, 0.0, 16.0, 16.0);
    let mut cpu = CountingMemory::zeroed(0x1_0000);
    let mut gl = CountingMemory::zeroed(0x1_0000);
    // the top half, drawn and left resident
    let top = vec![
        corner_at(viewport, 0.0, 8.0, -0.5, 1.0, FLAT),
        corner_at(viewport, 32.0, 8.0, -0.5, 1.0, FLAT),
        corner_at(viewport, 0.0, 32.0, -0.5, 1.0, FLAT),
    ];
    rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &top);
    rasterize_shaded(&registers, &mut gl, &mut resources, &top);
    assert_eq!(
        gl.writes_over(COLOR, COLOR_BYTES),
        0,
        "the draw kept its pixels"
    );

    // the fill: resolve, then the pattern lands over everything
    resources
        .opengl
        .as_mut()
        .unwrap()
        .sync(&mut gl, COLOR, COLOR_BYTES, true);
    cpu.fill(COLOR, COLOR_BYTES, 0x11);
    gl.fill(COLOR, COLOR_BYTES, 0x11);
    assert_eq!(
        gl.writes_over(COLOR, COLOR_BYTES),
        1,
        "the fill's sync wrote the drawn half back first",
    );
    assert_eq!(
        gl.region(COLOR, COLOR_BYTES),
        cpu.region(COLOR, COLOR_BYTES),
        "the fill"
    );

    // the next draw uploads the filled memory and draws over its bottom half
    let bottom = vec![
        corner_at(viewport, 0.0, 0.0, -0.5, 1.0, OTHER),
        corner_at(viewport, 32.0, 0.0, -0.5, 1.0, OTHER),
        corner_at(viewport, 0.0, 8.0, -0.5, 1.0, OTHER),
    ];
    rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &bottom);
    rasterize_shaded(&registers, &mut gl, &mut resources, &bottom);
    assert_eq!(
        gl.reads_over(COLOR, COLOR_BYTES),
        2,
        "the invalidated surface uploaded the filled bytes anew",
    );
    resolve(&mut resources, &mut gl, 0);
    assert_eq!(
        gl.reads_over(COLOR, COLOR_BYTES),
        2,
        "resolving never uploads",
    );
    assert_images_identical(&cpu, &gl, "a draw after a fill");
}

// ---------------------------------------------------------------------------
// the whole GPU path: a command list thru Gpu, then the display transfer and
// the sync hooks the other engine blocks call
// ---------------------------------------------------------------------------

/// one register write, as a command list encodes it
fn command(register: usize, value: u32) -> [u32; 2] {
    [value, register as u32 | (0xF << 16)]
}

const VERTS: u32 = 0x9000;
const LIST: u32 = 0x8000;
const OUT: u32 = 0x5000;

/// a Gpu whose registers and vertex shader draw two flat triangles over the
/// target thru a command list, with the OpenGL renderer attached or not
fn gpu_with_draw(
    depth_format: u32,
    depth_writes: bool,
    attach_opengl: bool,
) -> Option<(Gpu, CountingMemory)> {
    let mut gpu = Gpu::new();
    if attach_opengl {
        let context = SharedOpenGlContext(shared_opengl()?);
        gpu.enable_opengl_renderer(context)
            .expect("the context works");
    }
    // the vertex shader: position from v0, color from v1
    const IDENTITY: u32 = 0x1B << 5 | 0x1B << 14 | 0x1B << 23;
    gpu.vertex_shader.descriptors[0] = 0xF | IDENTITY;
    let mov = |destination: u32, source: u32| 0x13 << 26 | destination << 21 | source << 12;
    // 0x01 is input v1, while 0x20 is float constant c0
    gpu.vertex_shader.program[..3].copy_from_slice(&[mov(0, 0x00), mov(1, 0x01), 0x22 << 26]);

    let mut registers = registers(depth_format);
    if depth_writes {
        // the depth test off, the writes on: every fragment passes and writes
        registers[REG_DEPTH_COLOR_MASK] = 0xF << 8 | 1 << 12;
    }
    // shader wiring: two inputs into v0 and v1, position out of register 0 and color out of 2
    registers[REG_SHADER_OUTPUT_TOTAL] = 2;
    registers[REG_SHADER_OUTPUT_MAP] = 0x0302_0100;
    registers[REG_SHADER_OUTPUT_MAP + 1] = 0x0B0A_0908;
    registers[REG_VS_OUTPUT_MASK] = 0b11;
    registers[REG_VS_NUM_INPUT_ATTRIBUTES] = 1; // two inputs
    registers[REG_VS_BLOCK + SHADER_INPUT_MAP_LOW] = 0x10;
    // one array feeding both attributes: position then color, 32 bytes a vertex
    registers[REG_ATTRIBUTE_BASE] = VERTS >> 3;
    registers[REG_ATTRIBUTE_FORMAT_LOW] = 0xFF;
    registers[REG_ATTRIBUTE_LOADER] = 0;
    registers[REG_ATTRIBUTE_LOADER + 1] = 0x10; // the array's slots: attribute 0, then 1
    registers[REG_ATTRIBUTE_LOADER + 2] = (2 << 28) | (32 << 16);
    gpu.internal[..registers.len()].copy_from_slice(&registers);

    // six vertices: two triangles covering the target, flat color
    let viewport = (0.0, 0.0, 16.0, 16.0);
    let spots = [
        [(0.0f32, 0.0), (32.0, 0.0), (0.0, 32.0)],
        [(32.0, 0.0), (32.0, 32.0), (0.0, 32.0)],
    ];
    let mut memory = CountingMemory::zeroed(0x1_0000);
    let mut vertex_bytes = Vec::with_capacity(6 * 32);
    for triangle in spots.iter() {
        for &(x, y) in triangle.iter() {
            let vertex = corner_at(viewport, x, y, -0.5, 1.0, FLAT);
            vertex_bytes.extend_from_slice(bytemuck::cast_slice::<f32, u8>(&vertex.clip));
            vertex_bytes.extend_from_slice(bytemuck::cast_slice::<f32, u8>(&vertex.color));
        }
    }
    memory.flat[VERTS as usize..VERTS as usize + vertex_bytes.len()].copy_from_slice(&vertex_bytes);

    let list: Vec<u32> = [command(REG_VERTEX_COUNT, 6), command(REG_DRAW_ARRAYS, 1)].concat();
    for (i, word) in list.iter().enumerate() {
        let at = LIST as usize + i * 4;
        memory.flat[at..at + 4].copy_from_slice(&word.to_le_bytes());
    }
    gpu.process_command_list(
        &mut memory,
        &mut SoftwareRenderer::default(),
        LIST,
        list.len() as u32 * 4,
    );
    Some((gpu, memory))
}

/// a draw that went thru the whole Gpu — command list, vertex shader, placement,
/// the OpenGL rasterizer — leaves the same bytes the CPU run does, and the
/// display transfer that reads the drawn buffer converts the same pixels
#[test]
fn opengl_command_list_draws_and_transfers_like_the_cpu() {
    let Some((mut gl_gpu, mut gl_memory)) = gpu_with_draw(0, true, true) else {
        return;
    };
    let Some((mut cpu_gpu, mut cpu_memory)) = gpu_with_draw(0, true, false) else {
        return;
    };
    assert_eq!(gl_gpu.draw_calls, 1, "the draw ran");
    assert_eq!(gl_gpu.vertices_drawn, 6, "the six vertices drew");
    assert_eq!(
        gl_memory.writes_over(COLOR, COLOR_BYTES),
        0,
        "the drawn target stayed resident",
    );

    // the display transfer reads the drawn buffer on the GPU, the drawn target
    // staying resident and the output held in a surface of its own, and the
    // converted rows match the CPU's byte for byte once they reach memory
    let dimensions = 16 | (16 << 16);
    gl_gpu.display_transfer(&mut gl_memory, COLOR, OUT, dimensions, dimensions, 0);
    cpu_gpu.display_transfer(&mut cpu_memory, COLOR, OUT, dimensions, dimensions, 0);
    assert_eq!(
        gl_memory.writes_over(COLOR, COLOR_BYTES),
        0,
        "the transfer read the drawn buffer on the GPU but resolved it",
    );
    assert!(
        gl_memory.guarded_writes(OUT, 16 * 16 * 4),
        "the transferred buffer is not guarded against CPU writes",
    );
    gl_gpu.sync_memory(&mut gl_memory, OUT, 16 * 16 * 4);
    assert_eq!(
        gl_memory.region(OUT, 16 * 16 * 4),
        cpu_memory.region(OUT, 16 * 16 * 4),
        "the transferred picture differs",
    );
    // and a sync of the drawn target resolves it like the CPU's run holds it
    gl_gpu.sync_memory(&mut gl_memory, COLOR, COLOR_BYTES);
    assert_eq!(
        gl_memory.region(COLOR, COLOR_BYTES),
        cpu_memory.region(COLOR, COLOR_BYTES),
        "the resolved target differs",
    );
    // FLAT is [0.5, 0.25, 0.75, 0.5], which both backends floor to
    // [127, 63, 191, 127] in RGBA; image() gives the guest's ABGR bytes, so the
    // known covered pixel reads [alpha, blue, green, red], the vertex shader
    // having carried the color from the array to the pixel
    assert_eq!(
        image(&gl_memory)[8 * BUFFER as usize + 8],
        [127, 191, 63, 127],
        "the command-list draw did not use the per-vertex FLAT color",
    );
}

/// draws the GPU shades itself: the vertex shader runs translated on the
/// OpenGL side and interpreted on the CPU, its math moving the positions and
/// mixing the colors, and the pixels come out the same
#[test]
fn opengl_vertex_shaders_match_the_cpu() {
    let Some(mut resources) = opengl_resources() else {
        return;
    };
    let mut registers = registers(0);
    const IDENTITY: u32 = 0x1B << 5 | 0x1B << 14 | 0x1B << 23;
    let mut unit = ShaderUnit::new();
    unit.descriptors[0] = 0xF | IDENTITY;
    // a temp = v1 * c1, o1 = temp + c2, o0 = v0 * c0, end: the color and the
    // position both come out of the program's math, not of the inputs as is
    let mul = |destination: u32, wide: u32, narrow: u32| (0x08 << 26) | (destination << 21) | (wide << 12) | (narrow << 7);
    let add = |destination: u32, wide: u32, narrow: u32| (0x00 << 26) | (destination << 21) | (wide << 12) | (narrow << 7);
    unit.program[..4].copy_from_slice(&[
        mul(0x10, 0x21, 0x01),
        add(0x01, 0x22, 0x10),
        mul(0x00, 0x20, 0x00),
        0x22 << 26,
    ]);
    unit.float_uniforms[0] = [0.75, 0.75, 0.75, 1.0];
    unit.float_uniforms[1] = [1.0, 0.5, 0.25, 0.125];
    unit.float_uniforms[2] = [0.25, 0.25, 0.25, 0.0];
    unit.prepare();

    // position out of register 0 and color out of 1, both enabled
    registers[REG_SHADER_OUTPUT_TOTAL] = 2;
    registers[REG_SHADER_OUTPUT_MAP] = 0x0302_0100;
    registers[REG_SHADER_OUTPUT_MAP + 1] = 0x0B0A_0908;
    registers[REG_VS_OUTPUT_MASK] = 0b11;

    // one triangle over the target, uneven colors and w of one
    let inputs: Vec<[Vec4; INPUT_REGISTERS]> = [
        ([-1.0, -1.0], [1.0, 0.0, 0.0, 1.0]),
        ([3.0, -1.0], [0.0, 1.0, 0.0, 1.0]),
        ([-1.0, 3.0], [0.0, 0.0, 1.0, 1.0]),
    ]
    .into_iter()
    .map(|([x, y], color)| {
        let mut input = [shader::ZERO; INPUT_REGISTERS];
        input[0] = [x, y, -0.5, 1.0];
        input[1] = color;
        input
    })
    .collect();

    // the shaded vertices, whose placed triangles bound the comparison's edges:
    // the program scales the position by c0 and mixes the color by c1 and c2
    let placed = placed_triangles(
        &registers,
        &inputs
            .iter()
            .map(|input| Vertex {
                clip: [
                    input[0][0] * 0.75,
                    input[0][1] * 0.75,
                    input[0][2] * 0.75,
                    input[0][3],
                ],
                color: std::array::from_fn(|i| input[1][i] * unit.float_uniforms[1][i] + unit.float_uniforms[2][i]),
                texcoords: [[0.0; 2]; 3],
                quaternion: [0.0; 4],
                view: [0.0; 3],
                texcoord0_w: 1.0,
            })
            .collect::<Vec<_>>(),
    );
    let mut cpu = CountingMemory::zeroed(0x1_0000);
    let mut gl = CountingMemory::zeroed(0x1_0000);
    rasterize_unshaded(&registers, &mut cpu, &mut Resources::default(), &unit, &unit, &inputs);
    rasterize_unshaded(&registers, &mut gl, &mut resources, &unit, &unit, &inputs);
    // the draw reached the GL's vertex stage at all: a silent fallback to the
    // CPU would make the comparison prove nothing
    assert_eq!(
        resources
            .opengl
            .as_ref()
            .and_then(|renderer| renderer.surface_dirty(COLOR))
            .map(|dirty| dirty as u32),
        Some(1),
        "the shaded draw did not submit to OpenGL",
    );
    resolve(&mut resources, &mut gl, 0);
    // the interpolated colors round a step apart between the sides, like the
    // placed tests' vertex colors do
    assert_images_close(&cpu, &gl, &placed, 1, "vertex shaders shaded on the GPU");
    // the combiner's alpha is the program's own 0.125, floored to 31 on both
    // sides, and the scaled-away corner stays unwritten
    assert_eq!(image(&gl)[4 * BUFFER as usize + 4][0], 31, "the covered pixel is not lit");
    assert_eq!(image(&gl)[15 * BUFFER as usize + 15][0], 0, "the scaled-away corner drew");
}

/// a draw with a geometry shader after its vertex shader prepares on the CPU
/// and draws thru the placed OpenGL path, matching the CPU's own run
#[test]
fn opengl_geometry_shader_draws_match_the_cpu() {
    let Some(mut resources) = opengl_resources() else {
        return;
    };
    let mut registers = registers(0);
    const IDENTITY: u32 = 0x1B << 5 | 0x1B << 14 | 0x1B << 23;

    // the vertex shader hands the position and the two offsets the geometry
    // shader spreads the triangle's corners apart by
    let mut vertex_shader = ShaderUnit::new();
    vertex_shader.descriptors[0] = 0xF | IDENTITY;
    let mov = |destination: u32, source: u32| 0x13 << 26 | destination << 21 | source << 12;
    vertex_shader.program[..4].copy_from_slice(&[
        mov(0, 0x00),
        mov(1, 0x01),
        mov(2, 0x02),
        0x22 << 26,
    ]);
    vertex_shader.prepare();

    // the geometry shader turns the one invocation into a triangle: the
    // position as is, then over by the first offset, then up by the second
    let mut geometry_shader = ShaderUnit::new();
    geometry_shader.descriptors[0] = 0xF | IDENTITY;
    let setemit = |slot: u32, primitive: bool| (0x2B << 26) | (slot << 24) | ((primitive as u32) << 23);
    let emit = 0x2A << 26;
    let add = |destination: u32, wide: u32, narrow: u32| (0x00 << 26) | (destination << 21) | (wide << 12) | (narrow << 7);
    geometry_shader.program[..10].copy_from_slice(&[
        setemit(0, false),
        mov(0, 0x00),
        emit,
        setemit(1, false),
        add(0, 0x00, 0x01),
        emit,
        setemit(2, true),
        add(0, 0x00, 0x02),
        emit,
        0x22 << 26,
    ]);
    geometry_shader.prepare();

    // the geometry stage on, three attribute vec4s an invocation, its one
    // output register carrying the position
    registers[REG_GEOSTAGE_CONFIG] = 2;
    registers[REG_GS_CONFIG] = 0;
    registers[REG_VS_OUTPUT_TOTAL] = 2; // three packed outputs a vertex
    registers[REG_VS_OUTPUT_MASK] = 0b111;
    registers[REG_GS_BLOCK + SHADER_INPUT_CONFIG] = 2; // three attribute vec4s an invocation
    registers[REG_GS_BLOCK + SHADER_INPUT_MAP_LOW] = 0x210; // v0, v1 and v2
    registers[REG_GS_BLOCK + SHADER_OUTPUT_MASK] = 0b1;
    // position out of register 0, for the geometry stage's own outputs
    registers[REG_SHADER_OUTPUT_TOTAL] = 1;
    registers[REG_SHADER_OUTPUT_MAP] = 0x0302_0100;
    registers[REG_SHADER_OUTPUT_MAP + 1] = 0x1F1F_1F1F;

    let mut input = [shader::ZERO; INPUT_REGISTERS];
    input[0] = [-0.5, -0.5, -0.5, 1.0];
    input[1] = [1.0, 0.25, 0.0, 0.0]; // the first corner's offset
    input[2] = [0.25, 1.0, 0.0, 0.0]; // the second corner's offset
    let inputs = vec![input];

    let mut cpu = CountingMemory::zeroed(0x1_0000);
    let mut gl = CountingMemory::zeroed(0x1_0000);
    rasterize_unshaded(&registers, &mut cpu, &mut Resources::default(), &vertex_shader, &geometry_shader, &inputs);
    rasterize_unshaded(&registers, &mut gl, &mut resources, &vertex_shader, &geometry_shader, &inputs);
    assert!(
        resources
            .opengl
            .as_ref()
            .and_then(|renderer| renderer.surface_dirty(COLOR))
            .map(|dirty| dirty as u32)
            == Some(1),
        "the geometry shader's draw did not submit to OpenGL",
    );
    resolve(&mut resources, &mut gl, 0);
    assert_images_identical(&cpu, &gl, "a geometry shader's draw");
    // one triangle out of the one invocation: something drew, and it drew
    // where the shader's offsets put it, not where the input vertex sat
    let drawn = image(&gl).iter().filter(|pixel| pixel[0] == 255).count();
    assert!(drawn > 0, "the geometry shader emitted nothing");
}

/// the display transfers the OpenGL renderer takes run on the GPU — the input
/// staying resident, the output held or written byte for byte — and the ones it
/// refuses still reach the CPU engine, which the run matches
#[test]
fn opengl_display_transfers_run_on_the_gpu_and_match_the_cpu() {
    // every case draws the target thru a command list first, like the
    // full-path test does, then transfers it with its flags
    let case = |flags: u32, input_dimensions: u32, output_dimensions: u32| -> Option<(CountingMemory, CountingMemory)> {
        let (mut gl_gpu, mut gl_memory) = gpu_with_draw(0, true, true)?;
        let (mut cpu_gpu, mut cpu_memory) = gpu_with_draw(0, true, false)?;
        gl_gpu.display_transfer(&mut gl_memory, COLOR, OUT, input_dimensions, output_dimensions, flags);
        cpu_gpu.display_transfer(&mut cpu_memory, COLOR, OUT, input_dimensions, output_dimensions, flags);
        // what the GPU kept only reaches memory when something asks for it
        gl_gpu.sync_memory(&mut gl_memory, OUT, 16 * 16 * 4);
        Some((cpu_memory, gl_memory))
    };
    let dimensions = 16 | (16 << 16);
    // a straight transfer out of the drawn buffer, tiled into the screen's
    // rows, the GPU reading its own surface
    {
        let Some((cpu, gl)) = case(0, dimensions, dimensions) else {
            return;
        };
        assert_eq!(
            gl.writes_over(COLOR, COLOR_BYTES),
            0,
            "the transfer read the drawn buffer on the GPU but resolved it",
        );
        assert_eq!(
            gl.region(OUT, 16 * 16 * 4),
            cpu.region(OUT, 16 * 16 * 4),
            "the straight transfer differs",
        );
    }
    // a tiled output, the effects-buffer kind the next draw renders into
    {
        let Some((cpu, gl)) = case(1 << 5, dimensions, dimensions) else {
            return;
        };
        assert_eq!(
            gl.writes_over(COLOR, COLOR_BYTES),
            0,
            "the tiled transfer resolved the drawn buffer",
        );
        assert_eq!(
            gl.region(OUT, 16 * 16 * 4),
            cpu.region(OUT, 16 * 16 * 4),
            "the tiled transfer differs",
        );
    }
    // flipped, the rows read from the other end
    {
        let Some((cpu, gl)) = case(1, dimensions, dimensions) else {
            return;
        };
        assert_eq!(
            gl.writes_over(COLOR, COLOR_BYTES),
            0,
            "the flipped transfer resolved the drawn buffer",
        );
        assert_eq!(
            gl.region(OUT, 16 * 16 * 4),
            cpu.region(OUT, 16 * 16 * 4),
            "the flipped transfer differs",
        );
    }
    // halved, each output pixel the average of its 2x2 block the way the CPU
    // rounds it, the output's register counting input pixels
    {
        let Some((cpu, gl)) = case(2 << 24, dimensions, dimensions) else {
            return;
        };
        assert_eq!(
            gl.writes_over(COLOR, COLOR_BYTES),
            0,
            "the halved transfer resolved the drawn buffer",
        );
        assert_eq!(
            gl.region(OUT, 8 * 8 * 4),
            cpu.region(OUT, 8 * 8 * 4),
            "the halved transfer differs",
        );
    }
    // converted into the screen's own format, RGB565, the guest bytes packing
    // the same pixels
    {
        let Some((cpu, gl)) = case(2 << 12, dimensions, dimensions) else {
            return;
        };
        assert_eq!(
            gl.writes_over(COLOR, COLOR_BYTES),
            0,
            "the RGB565 transfer resolved the drawn buffer",
        );
        assert_eq!(
            gl.region(OUT, 16 * 16 * 2),
            cpu.region(OUT, 16 * 16 * 2),
            "the RGB565 transfer differs",
        );
    }
    // chained: the linear buffer an earlier transfer wrote is the next one's
    // input, read from the GPU it stayed in
    {
        let Some((mut gl_gpu, mut gl_memory)) = gpu_with_draw(0, true, true) else {
            return;
        };
        let Some((mut cpu_gpu, mut cpu_memory)) = gpu_with_draw(0, true, false) else {
            return;
        };
        gl_gpu.display_transfer(&mut gl_memory, COLOR, OUT, dimensions, dimensions, 0);
        cpu_gpu.display_transfer(&mut cpu_memory, COLOR, OUT, dimensions, dimensions, 0);
        // the second transfer reads the first's linear output back into tiles
        gl_gpu.display_transfer(&mut gl_memory, OUT, OUT + 16 * 16 * 4, dimensions, dimensions, 1 << 1);
        cpu_gpu.display_transfer(&mut cpu_memory, OUT, OUT + 16 * 16 * 4, dimensions, dimensions, 1 << 1);
        assert_eq!(
            gl_memory.writes_over(OUT, 16 * 16 * 4),
            0,
            "the chained transfer resolved the first picture",
        );
        gl_gpu.sync_memory(&mut gl_memory, OUT + 16 * 16 * 4, 16 * 16 * 4);
        assert_eq!(
            gl_memory.region(OUT + 16 * 16 * 4, 16 * 16 * 4),
            cpu_memory.region(OUT + 16 * 16 * 4, 16 * 16 * 4),
            "the chained transfer differs",
        );
    }
    // flipped and cropped, the output's rows starting the skew the CPU applies:
    // the input is shorter than the output is tall, which the GPU refuses
    {
        let Some((cpu, gl)) = case(1 | (1 << 2), 16 | (8 << 16), 8 | (16 << 16)) else {
            return;
        };
        assert!(
            gl.writes_over(COLOR, COLOR_BYTES) >= 1,
            "the skewed transfer ran on the GPU",
        );
        assert_eq!(
            gl.region(OUT, 8 * 16 * 4),
            cpu.region(OUT, 8 * 16 * 4),
            "the skewed transfer differs",
        );
    }
    // a tiled output of part of a tile, which the GPU leaves to the CPU's
    // layout, quirks and all
    {
        let Some((cpu, gl)) = case(1 << 5, dimensions, 12 | (16 << 16)) else {
            return;
        };
        assert!(
            gl.writes_over(COLOR, COLOR_BYTES) >= 1,
            "the unaligned transfer ran on the GPU",
        );
        assert_eq!(
            gl.region(OUT, 12 * 16 * 4),
            cpu.region(OUT, 12 * 16 * 4),
            "the unaligned transfer differs",
        );
    }
}

/// the picture a transfer left for a screen shows straight from the GPU: an
/// upright texture like the presenter takes, equal to the CPU's own column
/// reading, and gone once guest memory took the buffer back
#[test]
fn opengl_shows_transferred_screens_directly() {
    let Some((mut gl_gpu, mut gl_memory)) = gpu_with_draw(0, true, true) else {
        return;
    };
    let Some((mut cpu_gpu, mut cpu_memory)) = gpu_with_draw(0, true, false) else {
        return;
    };
    let dimensions = 16 | (16 << 16);
    gl_gpu.display_transfer(&mut gl_memory, COLOR, OUT, dimensions, dimensions, 0);
    cpu_gpu.display_transfer(&mut cpu_memory, COLOR, OUT, dimensions, dimensions, 0);
    // the screen shows the whole buffer, its rows the screen's columns
    let Some(screen) = gl_gpu.scaled_screen(OUT, (16, 16), 16, crate::format::ColorFormat::Rgba8, gl_memory.region(OUT, 16 * 16 * 4)) else {
        panic!("the transfer's picture was not shown straight from the GPU");
    };
    assert_eq!(screen.size, (16, 16), "the screen ref names the whole buffer");
    assert_eq!(screen.rows, (0, 16), "the buffer's first row is the screen's first column");
    let Some(image) = gl_gpu.screen_image(screen) else {
        panic!("no picture came back for the screen");
    };
    let Some(texture) = image.gl_texture else {
        panic!("the picture is not an OpenGL texture");
    };
    // read the upright texture back, the screen's rows top first
    let mut upright = vec![0u8; 16 * 16 * 4];
    let gl = shared_opengl().expect("the context the rasterizer drew on");
    unsafe {
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.get_tex_image(
            glow::TEXTURE_2D,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            glow::PixelPackData::Slice(Some(&mut upright)),
        );
    }
    // the CPU's own reading of the transferred buffer: column major in memory
    // and the panel scans bottom to top, like read_screen_at works it
    let source = cpu_memory.region(OUT, 16 * 16 * 4);
    for x in 0..16u32 {
        for y in 0..16u32 {
            // guest memory holds the format's ABGR bytes; the texture is RGBA
            let at = ((x * 16 + (15 - y)) * 4) as usize;
            let pixel = crate::format::ColorFormat::Rgba8.decode(&source[at..at + 4]);
            let shown = &upright[((y * 16 + x) * 4) as usize..((y * 16 + x) * 4 + 4) as usize];
            assert_eq!(shown, pixel.as_slice(), "the upright picture differs at ({x},{y})");
        }
    }
    // and once guest memory takes the buffer back, the GPU shows nothing
    gl_gpu.sync_memory(&mut gl_memory, OUT, 16 * 16 * 4);
    assert!(
        gl_gpu
            .scaled_screen(OUT, (16, 16), 16, crate::format::ColorFormat::Rgba8, gl_memory.region(OUT, 16 * 16 * 4))
            .is_none(),
        "a resolved buffer still showed straight from the GPU",
    );
    assert_eq!(
        gl_memory.region(OUT, 16 * 16 * 4),
        cpu_memory.region(OUT, 16 * 16 * 4),
        "the resolved screen buffer differs",
    );
}

/// Gpu::sync_memory writes the drawn color back into guest memory, and
/// Gpu::sync_depth the drawn depth, what the services reading the buffers call
#[test]
fn opengl_sync_memory_and_sync_depth_hooks_write_back() {
    // color: after the draw, guest memory only holds what sync_memory brought
    let Some((mut gl_gpu, mut gl_memory)) = gpu_with_draw(0, false, true) else {
        return;
    };
    let Some((_, cpu_memory)) = gpu_with_draw(0, false, false) else {
        return;
    };
    assert_eq!(
        gl_memory.region(COLOR, COLOR_BYTES),
        &[0u8; 1024][..],
        "nothing written back yet"
    );
    gl_gpu.sync_memory(&mut gl_memory, COLOR, COLOR_BYTES);
    assert_eq!(
        gl_memory.writes_over(COLOR, COLOR_BYTES),
        1,
        "the sync wrote the drawn target back once",
    );
    assert_eq!(
        gl_memory.region(COLOR, COLOR_BYTES),
        cpu_memory.region(COLOR, COLOR_BYTES),
        "the synced target differs",
    );

    // depth: sync_depth brings the drawn depth back, the D24S8 stencil byte intact
    let Some((mut gl_gpu, mut gl_memory)) = gpu_with_draw(3, true, true) else {
        return;
    };
    let Some((_, cpu_memory)) = gpu_with_draw(3, true, false) else {
        return;
    };
    let depth_bytes = 16 * 16 * 4;
    gl_gpu.sync_depth(&mut gl_memory, DEPTH, depth_bytes);
    assert!(
        gl_memory.writes_over(DEPTH, depth_bytes) >= 1,
        "the depth sync wrote nothing back",
    );
    for (sample, (gl_word, cpu_word)) in depth_image(&gl_memory, 4)
        .iter()
        .zip(depth_image(&cpu_memory, 4))
        .enumerate()
    {
        assert_eq!(
            (gl_word >> 24) as u8,
            (cpu_word >> 24) as u8,
            "the stencil byte at ({}, {}) differs after the depth sync",
            sample as u32 % BUFFER,
            sample as u32 / BUFFER,
        );
        assert!(
            (gl_word & 0xFF_FFFF).abs_diff(cpu_word & 0xFF_FFFF) <= 2,
            "the depth at ({}, {}) differs by more than a rounding step",
            sample as u32 % BUFFER,
            sample as u32 / BUFFER,
        );
    }
}

/// a surface the cache gives up under its budget is resolved into guest memory at
/// the eviction, and a later draw to it uploads that resolved state again rather
/// than pretending it was resident the whole time. The real budget needs
/// thousands of targets to turn over, so the test turns it down to a few.
#[test]
fn opengl_evicted_surfaces_resolve_into_guest_memory() {
    let Some(mut resources) = opengl_resources() else {
        return;
    };
    resources
        .opengl
        .as_mut()
        .unwrap()
        .limit_surface_cache(2_500); // two 16x16 targets w/ no depth fit, not three
    let viewport = (0.0, 0.0, 16.0, 16.0);
    let targets = [(0x1000u32, FLAT), (0x4000, OTHER), (0x8000, FLAT)];
    let quads: Vec<Vec<Vertex>> = targets
        .iter()
        .map(|(_, color)| cover(viewport, -0.5, *color))
        .collect();
    let mut cpu = CountingMemory::zeroed(0x1_0000);
    let mut gl = CountingMemory::zeroed(0x1_0000);
    // three distinct targets: the third evicts the first, writing its pixels back
    for ((addr, _), quad) in targets.iter().zip(quads.iter()) {
        let mut registers = registers(0);
        registers[REG_COLOR_BUFFER_ADDRESS] = addr >> 3;
        rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), quad);
        rasterize_shaded(&registers, &mut gl, &mut resources, quad);
    }
    assert_eq!(
        gl.writes_over(0x1000, COLOR_BYTES),
        1,
        "the first target should have been resolved into memory once, at the eviction",
    );
    assert_eq!(
        gl.region(0x1000, COLOR_BYTES),
        cpu.region(0x1000, COLOR_BYTES),
        "the eviction left the first target's guest bytes different from the CPU's",
    );
    // and the second still holds what it drew, with no readback asked of it yet
    assert_eq!(
        gl.writes_over(0x4000, COLOR_BYTES),
        0,
        "the second target should still be resident",
    );
    // drawing to the first again uploads what the eviction wrote back
    let before = gl.reads_over(0x1000, COLOR_BYTES);
    let second_visit = slanted([FLAT, FLAT, FLAT], [1.0, 1.2, 0.8], -0.4);
    let mut registers = registers(0);
    registers[REG_COLOR_BUFFER_ADDRESS] = 0x1000 >> 3;
    let placed = placed_triangles(&registers, &second_visit);
    rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &second_visit);
    rasterize_shaded(&registers, &mut gl, &mut resources, &second_visit);
    assert!(
        gl.reads_over(0x1000, COLOR_BYTES) > before,
        "the second visit should have uploaded the target it was evicted from",
    );
    resolve(&mut resources, &mut gl, 0);
    assert_images_close(&cpu, &gl, &placed, 1, "the target after its eviction and return");
}

// ---------------------------------------------------------------------------
// RGB8 targets
// ---------------------------------------------------------------------------

/// an RGB8 target draws like an RGBA8 one: the tiles pack 3 bytes a texel, the
/// GL keeps them in an RGBA8 copy whose alpha stays at the 255 the format
/// decodes, and what the GL holds resolves back into the guest's own bytes
#[test]
fn opengl_rgb8_targets_match_the_cpu() {
    let viewport = (0.0, 0.0, 16.0, 16.0);
    // a plain cover, then a slanted triangle over it
    {
        let Some(mut resources) = opengl_resources() else {
            return;
        };
        let mut registers = registers(0);
        registers[REG_COLOR_BUFFER_FORMAT] = 1 << 16; // RGB8
        let mut cpu = CountingMemory::zeroed(0x1_0000);
        let mut gl = CountingMemory::zeroed(0x1_0000);
        rasterize_shaded(
            &registers,
            &mut cpu,
            &mut Resources::default(),
            &cover(viewport, -0.5, FLAT),
        );
        rasterize_shaded(
            &registers,
            &mut gl,
            &mut resources,
            &cover(viewport, -0.5, FLAT),
        );
        let quad = slanted([OTHER, OTHER, OTHER], [1.2, 0.8, 1.0], -0.4);
        rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &quad);
        rasterize_shaded(&registers, &mut gl, &mut resources, &quad);
        resolve(&mut resources, &mut gl, 0);
        assert_rgb8_identical(&cpu, &gl, "RGB8 covers and a slanted triangle");
    }
    // a transparency blend over a base the draw never wrote, the alpha channel
    // of the GL copy staying opaque while the blend reads it as the destination
    {
        let Some(mut resources) = opengl_resources() else {
            return;
        };
        let mut plain = registers(0);
        plain[REG_COLOR_BUFFER_FORMAT] = 1 << 16;
        plain[REG_BLEND_COLOR] = BLEND_CONSTANT;
        let mut blended = plain.clone();
        blended[REG_COLOR_OPERATION] = 0x100;
        // the destination factor reads the copy's own alpha, which stays at the
        // 255 the format decodes like the software path reads back, so a draw
        // whose copy kept anything else there would blend differently
        blended[REG_BLEND_FUNC] = blend_func((0, 6, 8), (0, 1, 0));
        let mut cpu = CountingMemory::zeroed(0x1_0000);
        let mut gl = CountingMemory::zeroed(0x1_0000);
        let base = cover(viewport, -0.5, FLAT);
        rasterize_shaded(&plain, &mut cpu, &mut Resources::default(), &base);
        rasterize_shaded(&plain, &mut gl, &mut resources, &base);
        let quad = slanted([OTHER, OTHER, OTHER], [1.0, 1.0, 1.0], -0.4);
        let placed = placed_triangles(&blended, &quad);
        rasterize_shaded(&blended, &mut cpu, &mut Resources::default(), &quad);
        rasterize_shaded(&blended, &mut gl, &mut resources, &quad);
        resolve(&mut resources, &mut gl, 0);
        assert_rgb8_close(&cpu, &gl, &placed, 1, "an RGB8 transparency blend");
    }
    // a partial color mask keeps the channels it may not change, over the same
    // three bytes the format stores
    {
        let Some(mut resources) = opengl_resources() else {
            return;
        };
        let mut plain = registers(0);
        plain[REG_COLOR_BUFFER_FORMAT] = 1 << 16;
        let mut masked = plain.clone();
        masked[REG_DEPTH_COLOR_MASK] = 0x3 << 8; // red and green only
        let mut cpu = CountingMemory::zeroed(0x1_0000);
        let mut gl = CountingMemory::zeroed(0x1_0000);
        let base = cover(viewport, -0.5, FLAT);
        rasterize_shaded(&plain, &mut cpu, &mut Resources::default(), &base);
        rasterize_shaded(&plain, &mut gl, &mut resources, &base);
        let quad = cover(viewport, -0.4, OTHER);
        rasterize_shaded(&masked, &mut cpu, &mut Resources::default(), &quad);
        rasterize_shaded(&masked, &mut gl, &mut resources, &quad);
        resolve(&mut resources, &mut gl, 0);
        assert_rgb8_identical(&cpu, &gl, "an RGB8 partial color mask");
        // RGB8 stores blue, green, red; the base's FLAT blue is 191 and the
        // masked draw may only change red and green
        assert_eq!(
            rgb8_image(&gl)[8 * BUFFER as usize + 8][0],
            191,
            "the masked RGB8 draw changed blue",
        );
    }
    // what the GL holds resolves into the guest's tiled RGB8 bytes, so a later
    // draw reading the target as a texture sees the same texels both ways
    {
        let Some(mut resources) = opengl_resources() else {
            return;
        };
        let mut registers = registers(0);
        registers[REG_COLOR_BUFFER_FORMAT] = 1 << 16;
        let mut cpu = CountingMemory::zeroed(0x1_0000);
        let mut gl = CountingMemory::zeroed(0x1_0000);
        rasterize_shaded(
            &registers,
            &mut cpu,
            &mut Resources::default(),
            &cover(viewport, -0.5, FLAT),
        );
        rasterize_shaded(
            &registers,
            &mut gl,
            &mut resources,
            &cover(viewport, -0.5, FLAT),
        );
        registers[REG_TEXTURE_CONFIG] = 1;
        registers[TEXTURE_UNIT_BASES[0] + 1] = 16 | (16 << 16);
        registers[TEXTURE_UNIT_BASES[0] + 4] = COLOR >> 3;
        registers[TEXTURE_UNIT_BASES[0] + 13] = 1; // RGB8
        tev_stage(
            &mut registers,
            0,
            [3, 0, 0],
            [3, 0, 0],
            [0; 3],
            [0; 3],
            0,
            0,
            0,
        );
        tev_passthrough(&mut registers, &[0]);
        let quad = uv_quad(viewport, &[(&0, 0.0, 0.0, 1.0, 1.0)]);
        rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &quad);
        rasterize_shaded(&registers, &mut gl, &mut resources, &quad);
        resolve(&mut resources, &mut gl, 0);
        assert_rgb8_identical(&cpu, &gl, "sampling an RGB8 target as a texture");
    }
}

/// the 16 bit color targets draw, blend, mask and sample back like the CPU does,
/// each keeping the RGBA8 copy the GL path holds every target in
#[test]
fn opengl_16bit_targets_match_the_cpu() {
    // the color buffer numbers the format register takes, and the texture
    // format's own, which number 565 and 5551 the other way around
    for (raw, texture_raw, format) in [
        (2u32, 3u32, ColorFormat::Rgb5A1),
        (3, 2, ColorFormat::Rgb565),
        (4, 4, ColorFormat::Rgba4),
    ] {
        let viewport = (0.0, 0.0, 16.0, 16.0);
        // a plain cover, then a slanted triangle over it
        {
            let Some(mut resources) = opengl_resources() else {
                return;
            };
            let mut registers = registers(0);
            registers[REG_COLOR_BUFFER_FORMAT] = raw << 16;
            let mut cpu = CountingMemory::zeroed(0x1_0000);
            let mut gl = CountingMemory::zeroed(0x1_0000);
            rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &cover(viewport, -0.5, FLAT));
            rasterize_shaded(&registers, &mut gl, &mut resources, &cover(viewport, -0.5, FLAT));
            let quad = slanted([OTHER, OTHER, OTHER], [1.2, 0.8, 1.0], -0.4);
            rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &quad);
            rasterize_shaded(&registers, &mut gl, &mut resources, &quad);
            resolve(&mut resources, &mut gl, 0);
            assert_packed_identical(&cpu, &gl, format, &format!("{format:?}: covers and a slanted triangle"));
        }
        // interpolated colors over the whole target, the writeback packing each
        // pixel to the format's own precision on both sides
        {
            let Some(mut resources) = opengl_resources() else {
                return;
            };
            let mut registers = registers(0);
            registers[REG_COLOR_BUFFER_FORMAT] = raw << 16;
            let mut cpu = CountingMemory::zeroed(0x1_0000);
            let mut gl = CountingMemory::zeroed(0x1_0000);
            let quad = slanted(
                [[0.9, 0.1, 0.3, 0.8], [0.2, 0.8, 0.6, 0.3], [0.5, 0.4, 0.9, 0.6]],
                [1.3, 0.7, 1.1],
                -0.4,
            );
            let placed = placed_triangles(&registers, &quad);
            rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &quad);
            rasterize_shaded(&registers, &mut gl, &mut resources, &quad);
            resolve(&mut resources, &mut gl, 0);
            // one step of the widest channel: a byte of float noise crossing a
            // 5/6/4 bit bucket moves the stored value that far
            assert_packed_close(&cpu, &gl, format, &placed, 8, &format!("{format:?}: interpolated colors"));
        }
        // a transparency blend over a base, the factors reading only the source,
        // so the copy's precision never enters the math
        {
            let Some(mut resources) = opengl_resources() else {
                return;
            };
            let mut plain = registers(0);
            plain[REG_COLOR_BUFFER_FORMAT] = raw << 16;
            let mut blended = plain.clone();
            blended[REG_COLOR_OPERATION] = 0x100;
            blended[REG_BLEND_FUNC] = blend_func((0, 6, 7), (0, 1, 0));
            let mut cpu = CountingMemory::zeroed(0x1_0000);
            let mut gl = CountingMemory::zeroed(0x1_0000);
            let base = cover(viewport, -0.5, FLAT);
            rasterize_shaded(&plain, &mut cpu, &mut Resources::default(), &base);
            rasterize_shaded(&plain, &mut gl, &mut resources, &base);
            let quad = slanted([OTHER, OTHER, OTHER], [1.0, 1.0, 1.0], -0.4);
            rasterize_shaded(&blended, &mut cpu, &mut Resources::default(), &quad);
            rasterize_shaded(&blended, &mut gl, &mut resources, &quad);
            resolve(&mut resources, &mut gl, 0);
            assert_packed_identical(&cpu, &gl, format, &format!("{format:?}: a transparency blend"));
        }
        // a blend reading the destination, where the GL copy holds the
        // pipeline's own 8 bits while the software reads the format's expanded
        // bytes back — bounded at the spread between the two plus the encode's
        // own step
        {
            let Some(mut resources) = opengl_resources() else {
                return;
            };
            let mut plain = registers(0);
            plain[REG_COLOR_BUFFER_FORMAT] = raw << 16;
            let mut blended = plain.clone();
            blended[REG_COLOR_OPERATION] = 0x100;
            blended[REG_BLEND_FUNC] = blend_func((0, 2, 4), (0, 1, 0));
            let mut cpu = CountingMemory::zeroed(0x1_0000);
            let mut gl = CountingMemory::zeroed(0x1_0000);
            let base = cover(viewport, -0.5, FLAT);
            rasterize_shaded(&plain, &mut cpu, &mut Resources::default(), &base);
            rasterize_shaded(&plain, &mut gl, &mut resources, &base);
            let quad = slanted([OTHER, OTHER, OTHER], [1.0, 1.0, 1.0], -0.4);
            let placed = placed_triangles(&blended, &quad);
            rasterize_shaded(&blended, &mut cpu, &mut Resources::default(), &quad);
            rasterize_shaded(&blended, &mut gl, &mut resources, &quad);
            resolve(&mut resources, &mut gl, 0);
            assert_packed_close(&cpu, &gl, format, &placed, 6, &format!("{format:?}: a destination blend"));
        }
        // a partial color mask keeps the channels it may not change
        {
            let Some(mut resources) = opengl_resources() else {
                return;
            };
            let mut masked = registers(0);
            masked[REG_COLOR_BUFFER_FORMAT] = raw << 16;
            masked[REG_DEPTH_COLOR_MASK] = 0x7 << 8; // red and green only
            let mut cpu = CountingMemory::zeroed(0x1_0000);
            let mut gl = CountingMemory::zeroed(0x1_0000);
            let base = cover(viewport, -0.5, FLAT);
            rasterize_shaded(&registers(0), &mut cpu, &mut Resources::default(), &base);
            rasterize_shaded(&registers(0), &mut gl, &mut resources, &base);
            let quad = cover(viewport, -0.4, OTHER);
            rasterize_shaded(&masked, &mut cpu, &mut Resources::default(), &quad);
            rasterize_shaded(&masked, &mut gl, &mut resources, &quad);
            resolve(&mut resources, &mut gl, 0);
            assert_packed_identical(&cpu, &gl, format, &format!("{format:?}: a partial color mask"));
        }
        // what the GL holds resolves into the guest's own 2 byte tiles, so a
        // later draw reading the target as a texture sees the same texels
        {
            let Some(mut resources) = opengl_resources() else {
                return;
            };
            let mut registers = registers(0);
            registers[REG_COLOR_BUFFER_FORMAT] = raw << 16;
            let mut cpu = CountingMemory::zeroed(0x1_0000);
            let mut gl = CountingMemory::zeroed(0x1_0000);
            rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &cover(viewport, -0.5, FLAT));
            rasterize_shaded(&registers, &mut gl, &mut resources, &cover(viewport, -0.5, FLAT));
            registers[REG_TEXTURE_CONFIG] = 1;
            registers[TEXTURE_UNIT_BASES[0] + 1] = 16 | (16 << 16);
            registers[TEXTURE_UNIT_BASES[0] + 4] = COLOR >> 3;
            registers[TEXTURE_UNIT_BASES[0] + 13] = texture_raw;
            tev_stage(&mut registers, 0, [3, 0, 0], [3, 0, 0], [0; 3], [0; 3], 0, 0, 0);
            tev_passthrough(&mut registers, &[0]);
            let quad = uv_quad(viewport, &[(&0, 0.0, 0.0, 1.0, 1.0)]);
            rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &quad);
            rasterize_shaded(&registers, &mut gl, &mut resources, &quad);
            resolve(&mut resources, &mut gl, 0);
            assert_packed_identical(&cpu, &gl, format, &format!("{format:?}: sampled back as a texture"));
        }
    }
}

// ---------------------------------------------------------------------------
// fog
// ---------------------------------------------------------------------------

/// a factor of one falling to none across the fog table, one entry at a time
fn fog_ramp() -> Vec<(u32, u32)> {
    (0..128)
        .map(|i| {
            let factor = (127 - i) * 16;
            (i, (factor << 13) | (0x2000 - 16))
        })
        .collect()
}

/// writes the entries into a run's fog table, the index register the way the
/// table writes move it
fn load_fog_table(registers: &mut [u32], table: &mut crate::fog::Table, entries: &[(u32, u32)]) {
    for &(index, value) in entries {
        registers[crate::fog::REG_TABLE_INDEX] = index;
        table.write(registers, value);
    }
}

/// a triangle whose depth slides across the target, so the fog's factor runs
/// over its table instead of sitting at one entry
fn depth_ramp() -> Vec<Vertex> {
    let spots = [(2.0, 2.0), (14.0, 4.0), (4.0, 14.0)];
    let zs = [-0.2f32, -0.5, -0.8];
    let viewport = (0.0, 0.0, BUFFER as f32, BUFFER as f32);
    spots
        .into_iter()
        .zip(zs)
        .map(|(spot, z)| corner_at(viewport, spot.0, spot.1, z, 1.0, FLAT))
        .collect()
}

/// the fog mixes into what the combiners made, by the fragment's depth thru its
/// table and its color register, the alpha test having judged the fragment b4,
/// and the alpha byte it leaves alone. The mix is one rounding step apart at
/// most, the same slack the blends take.
#[test]
fn opengl_fog_matches_the_cpu() {
    let viewport = (0.0, 0.0, 16.0, 16.0);
    // no table ever written: a factor of none, the fog color everywhere
    {
        let Some(mut resources) = opengl_resources() else {
            return;
        };
        let mut registers = registers(0);
        registers[crate::fog::REG_MODE] = 5;
        registers[crate::fog::REG_COLOR] = 0x00_80_40_20;
        let mut cpu = CountingMemory::zeroed(0x1_0000);
        let mut gl = CountingMemory::zeroed(0x1_0000);
        let quad = depth_ramp();
        let placed = placed_triangles(&registers, &quad);
        rasterize_shaded(&registers, &mut cpu, &mut Resources::default(), &quad);
        rasterize_shaded(&registers, &mut gl, &mut resources, &quad);
        resolve(&mut resources, &mut gl, 0);
        assert_images_close(&cpu, &gl, &placed, 1, "an unread fog table");
    }
    // a ramp of factors, by a depth that slides across the triangle, and one
    // flipped to read the table from the far end
    for (name, flip, quad, table) in [
        ("a fog ramp by depth", false, depth_ramp(), fog_ramp()),
        (
            "a flat quad at one depth",
            false,
            cover(viewport, -0.5, FLAT),
            fog_ramp(),
        ),
        (
            "a flipped fog reading from the far end",
            true,
            depth_ramp(),
            fog_ramp(),
        ),
    ] {
        let Some(mut resources) = opengl_resources() else {
            return;
        };
        let mut registers = registers(0);
        registers[crate::fog::REG_MODE] = 5 | ((flip as u32) << 16);
        registers[crate::fog::REG_COLOR] = 0x00_80_40_20;
        let mut cpu_resources = Resources::default();
        load_fog_table(&mut registers, &mut cpu_resources.fog_table, &table);
        load_fog_table(&mut registers, &mut resources.fog_table, &table);
        let mut cpu = CountingMemory::zeroed(0x1_0000);
        let mut gl = CountingMemory::zeroed(0x1_0000);
        let placed = placed_triangles(&registers, &quad);
        rasterize_shaded(&registers, &mut cpu, &mut cpu_resources, &quad);
        rasterize_shaded(&registers, &mut gl, &mut resources, &quad);
        resolve(&mut resources, &mut gl, 0);
        assert_images_close(&cpu, &gl, &placed, 1, name);
    }
    // the alpha test judged the fragment b4 the fog mixed in, over a sliding
    // alpha that splits on the reference, the fogged survivors matching
    {
        let Some(mut resources) = opengl_resources() else {
            return;
        };
        let mut registers = registers(0);
        registers[crate::fog::REG_MODE] = 5;
        registers[crate::fog::REG_COLOR] = 0x00_80_40_20;
        registers[REG_ALPHA_TEST] = 1 | (6 << 4) | (96 << 8); // greater than 96
        let mut cpu_resources = Resources::default();
        load_fog_table(&mut registers, &mut cpu_resources.fog_table, &fog_ramp());
        load_fog_table(&mut registers, &mut resources.fog_table, &fog_ramp());
        let mut cpu = CountingMemory::zeroed(0x1_0000);
        let mut gl = CountingMemory::zeroed(0x1_0000);
        let quad = slanted(
            [
                [0.5, 0.25, 0.75, 0.35],
                [0.5, 0.25, 0.75, 0.42],
                [0.5, 0.25, 0.75, 0.7],
            ],
            [1.0, 1.0, 1.0],
            -0.5,
        );
        let placed = placed_triangles(&registers, &quad);
        rasterize_shaded(&registers, &mut cpu, &mut cpu_resources, &quad);
        rasterize_shaded(&registers, &mut gl, &mut resources, &quad);
        resolve(&mut resources, &mut gl, 0);
        assert_images_close(&cpu, &gl, &placed, 1, "a fog behind an alpha test");
    }
}

// ---------------------------------------------------------------------------
// fragment lighting
// ---------------------------------------------------------------------------

/// where the lighting registers sit, the addresses lighting.rs decodes them from
const LIGHT_ENABLE: usize = 0x08F;
const LIGHT_BLOCK: usize = 0x140; // sixteen words a light
const LIGHT_GLOBAL_AMBIENT: usize = 0x1C0;
const LIGHT_COUNT: usize = 0x1C2;
const LIGHT_CONFIG0: usize = 0x1C3;
const LIGHT_CONFIG1: usize = 0x1C4;
const LIGHT_TABLE_INDEX: usize = 0x1C5;
const LIGHT_TABLE_INPUT: usize = 0x1D1;
const LIGHT_SLOTS: usize = 0x1D9;

/// the words of a light's block, its spec colors, ambient and diffuse, its
/// position and spot direction, its flags and its distance bias and scale
const LIGHT_SPECULAR0: usize = LIGHT_BLOCK;
const LIGHT_SPECULAR1: usize = LIGHT_BLOCK + 1;
const LIGHT_DIFFUSE: usize = LIGHT_BLOCK + 2;
const LIGHT_AMBIENT: usize = LIGHT_BLOCK + 3;
const LIGHT_POSITION_XY: usize = LIGHT_BLOCK + 4;
const LIGHT_POSITION_Z: usize = LIGHT_BLOCK + 5;
const LIGHT_SPOT_XY: usize = LIGHT_BLOCK + 6;
const LIGHT_SPOT_Z: usize = LIGHT_BLOCK + 7;
const LIGHT_FLAGS: usize = LIGHT_BLOCK + 9;
const LIGHT_DISTANCE_BIAS: usize = LIGHT_BLOCK + 10;
const LIGHT_DISTANCE_SCALE: usize = LIGHT_BLOCK + 11;

/// a light color of white, three 10 bit fields of 255
const WHITE10: u32 = 0x3FFF_FFFF;

/// a PICA float w/ a 12 bit mantissa and a 7 bit exponent, the light words'
/// encoding, positive values only, the inverse of lighting's pica_float
fn pica_float20(value: f32) -> u32 {
    if value == 0.0 {
        return 0;
    }
    let bits = value.to_bits();
    let exponent = (((bits >> 23) & 0xFF) as i32 - 64) as u32;
    ((exponent & 0x7F) << 12) | ((bits >> 11) & 0xFFF)
}

/// a PICA half, the position words' encoding, positive values only
fn pica_half(value: f32) -> u32 {
    if value == 0.0 {
        return 0;
    }
    let bits = value.to_bits();
    let exponent = (((bits >> 23) & 0xFF) as i32 - 112) as u32;
    ((exponent & 0x1F) << 10) | ((bits >> 13) & 0x3FF)
}

/// a spotlight direction of -1 along z, a signed 1.11 fixed point
const SPOT_BACKWARD: u32 = 0x2000 - 0x7FF;

/// one lighting table entry, its value and the signed step to the next one
fn table_entry(value: u32, step: i32) -> u32 {
    let magnitude = step.unsigned_abs().min(2047);
    (value & 0xFFF) | (magnitude << 12) | (u32::from(step < 0) << 23)
}

/// a table that climbs from none to nearly one across its entries
fn rising_table(number: usize) -> Vec<(usize, u32)> {
    (0..256)
        .map(|i| (number, table_entry((i * 16) as u32, 16)))
        .collect()
}

/// the same entries, entry for entry, into a run's tables
fn load_lighting_tables(
    registers: &mut [u32],
    tables: &mut crate::lighting::Tables,
    entries: &[(usize, u32)],
) {
    let mut loaded = None;
    for &(number, value) in entries {
        // the table number selects the row; its low byte is the entry, which
        // Tables::write moves on to the next one, so the index is only set when
        // the loader switches rows and each ramp fills its own entries
        if loaded != Some(number) {
            registers[LIGHT_TABLE_INDEX] = (number << 8) as u32;
            loaded = Some(number);
        }
        tables.write(registers, value);
    }
}

/// the lighting-table loader must let the entry index advance through a whole
/// ramp, not rewrite entry zero for every value
#[test]
fn lighting_table_loader_fills_every_entry() {
    let mut registers = vec![0u32; LIGHT_TABLE_INDEX + 1];
    let mut tables = crate::lighting::Tables::default();
    load_lighting_tables(&mut registers, &mut tables, &rising_table(3));
    let entries = tables.entries();
    assert!(entries[3][0][0] < entries[3][128][0], "entry 128 stayed zero");
    assert!(entries[3][128][0] < entries[3][255][0], "entry 255 stayed zero");
}

/// lights the first light for a case that reads none of its tables: the
/// spotlight, the distance attenuation, both distributions, the fresnel and
/// the reflections off, a directional light from +z in diffuse white
fn plain_light(registers: &mut [u32]) {
    registers[LIGHT_ENABLE] = 1;
    registers[LIGHT_COUNT] = 0; // one light
    registers[LIGHT_SLOTS] = 0; // slot 0 carries light 0
    registers[LIGHT_CONFIG0] = 0; // config 0, no shadow, no bump
    registers[LIGHT_CONFIG1] = (1 << 16)
        | (1 << 17)
        | (1 << 19)
        | (1 << 20)
        | (1 << 21)
        | (1 << 22)
        | (1 << 8)
        | (1 << 24);
    registers[LIGHT_SPECULAR0] = 0; // specular 0
    registers[LIGHT_SPECULAR1] = 0; // specular 1
    registers[LIGHT_DIFFUSE] = WHITE10; // diffuse
    registers[LIGHT_AMBIENT] = 0; // ambient
    registers[LIGHT_POSITION_XY] = 0; // position x and y at nothing
    registers[LIGHT_POSITION_Z] = pica_half(1.0); // position z at one
    registers[LIGHT_FLAGS] = 1; // directional
}

/// a triangle whose corners carry their own surface quaternion, view vector,
/// weight and texture coordinates, so the lighting interpolates across it the
/// way a wall of a title would
#[allow(clippy::too_many_arguments)]
fn lit_corners(
    quaternions: [[f32; 4]; 3],
    views: [[f32; 3]; 3],
    ws: [f32; 3],
    z_over_w: f32,
    texcoords: [[[f32; 2]; 3]; 3],
) -> Vec<Vertex> {
    let spots = [(2.0f32, 2.0), (14.0, 4.0), (4.0, 14.0)];
    let viewport = (0.0, 0.0, BUFFER as f32, BUFFER as f32);
    spots
        .into_iter()
        .zip(quaternions)
        .zip(views)
        .zip(ws)
        .zip(texcoords)
        .map(|((((spot, quaternion), view), w), sets)| {
            let mut vertex = corner_at(viewport, spot.0, spot.1, z_over_w, w, FLAT);
            vertex.quaternion = quaternion;
            vertex.view = view;
            vertex.texcoords = sets;
            vertex
        })
        .collect()
}

/// draws and compares one lighting case: the same registers, tables and
/// texture bytes feed the CPU rasterizer and the GL's uber-shader, and the lit
/// bytes come out a rounding step apart at most, the sums and dots the two
/// sides evaluate in different orders wobbling the last step like the
/// combiners already do. Returns the GL's pixels, for cases that compare two
/// runs of the GL against each other
fn run_lit(
    resources: &mut Resources,
    registers: &mut [u32],
    cpu_tables: crate::lighting::Tables,
    texels: Option<&[u8]>,
    quad: &[Vertex],
    name: &str,
) -> Vec<[u8; 4]> {
    let mut cpu_resources = Resources {
        light_tables: cpu_tables,
        ..Default::default()
    };
    let mut cpu = CountingMemory::zeroed(0x1_0000);
    let mut gl = CountingMemory::zeroed(0x1_0000);
    if let Some(texels) = texels {
        lay_texture(&mut cpu, TEXTURE, texels, 0, 8, 8, 1);
        lay_texture(&mut gl, TEXTURE, texels, 0, 8, 8, 1);
    }
    let placed = placed_triangles(registers, quad);
    rasterize_shaded(registers, &mut cpu, &mut cpu_resources, quad);
    rasterize_shaded(registers, &mut gl, resources, quad);
    // the draw reached the GL at all: a silent fallback to the CPU would make
    // the comparison, and anything built on it, prove nothing
    assert!(
        resources
            .opengl
            .as_ref()
            .and_then(|renderer| renderer.surface_dirty(COLOR))
            == Some(true),
        "{name}: lighting draw did not submit to OpenGL",
    );
    resolve(resources, &mut gl, 0);
    assert!(
        resources
            .opengl
            .as_ref()
            .and_then(|renderer| renderer.surface_dirty(COLOR))
            == Some(false),
        "{name}: resolve did not write the resident color target back",
    );
    assert_images_close(&cpu, &gl, &placed, 1, name);
    image(&gl)
}

/// a combiner handing one lighting color straight thru, the stages passing
/// everything else on
fn thru_stage(registers: &mut [u32], source: u32) {
    tev_stage(
        registers,
        0,
        [source, 0, 0],
        [source, 0, 0],
        [0; 3],
        [0; 3],
        0,
        0,
        0,
    );
    tev_passthrough(registers, &[0]);
}

/// every piece of the PICA's fragment lighting, a case at a time
#[test]
fn opengl_fragment_lighting_matches_the_cpu() {
    let Some(mut resources) = opengl_resources() else {
        return;
    };
    let quaternions = [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.5, 0.0, 0.866_025_4],
        [0.0, 0.866_025_4, 0.0, 0.5],
    ];
    // one corner's quaternion off the unit length, so the per fragment
    // normalization has work to do
    let uneven_quaternions = [
        [0.0, 0.0, 0.0, 2.0],
        [0.0, 0.5, 0.0, 0.866_025_4],
        [0.0, 0.866_025_4, 0.0, 0.25],
    ];
    let views = [[0.0, 0.0, -2.0], [0.3, -0.2, -2.5], [-0.4, 0.5, -1.5]];
    let ws = [1.0, 1.3, 0.8];
    let no_texcoords = [[[0.0; 2]; 3]; 3];
    // one span of coordinates over each of the three sets, for the cases that
    // read a texture from any unit
    let full_span = [[[0.0, 0.0]; 3], [[1.0, 0.0]; 3], [[0.0, 1.0]; 3]];
    // the normal map's texels lean red and green about the middle, blue full
    let normal_texels: Vec<u8> = (0..64)
        .flat_map(|texel| {
            [
                96 + (texel * 37 % 64) as u8,
                96 + (texel * 53 % 64) as u8,
                255,
                255,
            ]
        })
        .collect();
    // the shadow's texels run grey, some dim, some bright
    let shadow_texels: Vec<u8> = (0..64)
        .flat_map(|texel| {
            let grey = 96 + (texel * 89 % 128) as u8;
            [grey, grey, grey, 255]
        })
        .collect();

    // a directional light's diffuse over a tilted surface, quaternions of
    // uneven length normalized per fragment, the view vector interpolated
    {
        let mut registers = registers(0);
        plain_light(&mut registers);
        thru_stage(&mut registers, 1);
        let quad = lit_corners(uneven_quaternions, views, ws, -0.5, no_texcoords);
        run_lit(
            &mut resources,
            &mut registers,
            crate::lighting::Tables::default(),
            None,
            &quad,
            "a directional light's diffuse",
        );
    }
    // a specular highlight off the half vector, its distribution table
    {
        let mut registers = registers(0);
        plain_light(&mut registers);
        registers[LIGHT_CONFIG1] &= !(1 << 16); // distribution 0 back on
        registers[LIGHT_SPECULAR0] = WHITE10; // specular 0
        thru_stage(&mut registers, 2);
        let quad = lit_corners(quaternions, views, ws, -0.5, no_texcoords);
        let mut tables = crate::lighting::Tables::default();
        load_lighting_tables(&mut registers, &mut tables, &rising_table(0));
        load_lighting_tables(
            &mut registers,
            &mut resources.light_tables,
            &rising_table(0),
        );
        run_lit(
            &mut resources,
            &mut registers,
            tables,
            None,
            &quad,
            "a specular highlight off the half vector",
        );
    }
    // a spotlight's cone around a positional light
    {
        let mut registers = registers(0);
        plain_light(&mut registers);
        registers[LIGHT_CONFIG1] &= !(1 << 8); // the spotlight back on
        registers[LIGHT_FLAGS] = 0; // positional
        registers[LIGHT_SPOT_XY] = 0; // spot direction x and y at nothing
        registers[LIGHT_SPOT_Z] = SPOT_BACKWARD; // pointing away from the light
        registers[LIGHT_TABLE_INPUT] = 4 << 8; // the light vector against the spot
        thru_stage(&mut registers, 1);
        let quad = lit_corners(quaternions, views, ws, -0.5, no_texcoords);
        let mut tables = crate::lighting::Tables::default();
        load_lighting_tables(&mut registers, &mut tables, &rising_table(8));
        load_lighting_tables(
            &mut registers,
            &mut resources.light_tables,
            &rising_table(8),
        );
        run_lit(
            &mut resources,
            &mut registers,
            tables,
            None,
            &quad,
            "a spotlight's cone",
        );
    }
    // distance attenuation, the light sitting away from the surface so the
    // place in its table slides with the view
    {
        let mut registers = registers(0);
        plain_light(&mut registers);
        registers[LIGHT_CONFIG1] &= !(1 << 24); // the distance back on
        registers[LIGHT_FLAGS] = 0; // positional
        registers[LIGHT_POSITION_XY] = 0;
        registers[LIGHT_POSITION_Z] = pica_half(4.0); // sitting at (0, 0, 4)
        registers[LIGHT_DISTANCE_BIAS] = pica_float20(0.25); // the bias
        registers[LIGHT_DISTANCE_SCALE] = pica_float20(0.25); // the scale
        thru_stage(&mut registers, 1);
        let quad = lit_corners(quaternions, views, ws, -0.5, no_texcoords);
        let mut tables = crate::lighting::Tables::default();
        load_lighting_tables(&mut registers, &mut tables, &rising_table(16));
        load_lighting_tables(
            &mut registers,
            &mut resources.light_tables,
            &rising_table(16),
        );
        run_lit(
            &mut resources,
            &mut registers,
            tables,
            None,
            &quad,
            "a distance attenuation",
        );
    }
    // the fresnel factor into the primary alpha, under a configuration the
    // lighting supports (configuration 0 has no fresnel table at all)
    {
        let mut registers = registers(0);
        plain_light(&mut registers);
        registers[LIGHT_CONFIG1] &= !(1 << 19); // the fresnel back on
        registers[LIGHT_CONFIG0] = (1 << 4) | (1 << 2); // configuration 1, into the primary alpha
        registers[LIGHT_TABLE_INPUT] = 2 << 12; // the normal against the view
        thru_stage(&mut registers, 1);
        let quad = lit_corners(quaternions, views, ws, -0.5, no_texcoords);
        let mut tables = crate::lighting::Tables::default();
        load_lighting_tables(&mut registers, &mut tables, &rising_table(3));
        load_lighting_tables(
            &mut registers,
            &mut resources.light_tables,
            &rising_table(3),
        );
        let with_fresnel = run_lit(
            &mut resources,
            &mut registers,
            tables,
            None,
            &quad,
            "a fresnel factor in the alpha",
        );

        // the same draw with the fresnel table disabled must leave a different
        // alpha, or the case above proved nothing
        registers[LIGHT_CONFIG1] |= 1 << 19;
        let mut tables = crate::lighting::Tables::default();
        load_lighting_tables(&mut registers, &mut tables, &rising_table(3));
        let without_fresnel = run_lit(
            &mut resources,
            &mut registers,
            tables,
            None,
            &quad,
            "the fresnel factor switched off",
        );
        assert!(
            with_fresnel
                .iter()
                .zip(&without_fresnel)
                .any(|(on, off)| on[0] != off[0]),
            "fresnel enabled and disabled left the same alpha",
        );
    }
    // a global ambient the diffuse sums into, an ambient color on the light
    {
        let mut registers = registers(0);
        plain_light(&mut registers);
        registers[LIGHT_GLOBAL_AMBIENT] = WHITE10;
        registers[LIGHT_AMBIENT] = WHITE10; // the light's own ambient
        thru_stage(&mut registers, 1);
        let quad = lit_corners(quaternions, views, ws, -0.5, no_texcoords);
        run_lit(
            &mut resources,
            &mut registers,
            crate::lighting::Tables::default(),
            None,
            &quad,
            "ambient on the light and global",
        );
    }
    // two sided lighting, the geometric factors and the clamped highlights,
    // over a specular the distribution table drives
    {
        let mut registers = registers(0);
        plain_light(&mut registers);
        registers[LIGHT_CONFIG1] &= !(1 << 16); // distribution 0 back on
        registers[LIGHT_CONFIG0] = 1 << 27; // clamp highlights
        registers[LIGHT_SPECULAR0] = WHITE10; // specular 0
        registers[LIGHT_SPECULAR1] = WHITE10; // specular 1
        registers[LIGHT_FLAGS] = 0xE; // two sided, both geometric factors
        thru_stage(&mut registers, 2);
        let quad = lit_corners(quaternions, views, ws, -0.5, no_texcoords);
        let mut tables = crate::lighting::Tables::default();
        load_lighting_tables(&mut registers, &mut tables, &rising_table(0));
        load_lighting_tables(
            &mut registers,
            &mut resources.light_tables,
            &rising_table(0),
        );
        run_lit(
            &mut resources,
            &mut registers,
            tables,
            None,
            &quad,
            "two sided w/ geometric factors and clamped highlights",
        );
    }
    // a shadow texture dimming the light's primary and secondary colors
    {
        let mut registers = registers(0);
        plain_light(&mut registers);
        registers[LIGHT_CONFIG1] &= !(1 << 16); // distribution 0 back on
        registers[LIGHT_CONFIG0] = 1 // shadow on
            | (2 << 24) // reading unit 2
            | (1 << 16) // over the primary
            | (1 << 17); // and the secondary
        registers[LIGHT_SPECULAR0] = WHITE10; // specular 0
        bind_texture_unit(&mut registers, 2, TEXTURE, 8, 8, 0, 0, 0);
        thru_stage(&mut registers, 1);
        let quad = lit_corners(quaternions, views, ws, -0.5, full_span);
        run_lit(
            &mut resources,
            &mut registers,
            crate::lighting::Tables::default(),
            Some(&shadow_texels),
            &quad,
            "a shadow texture on the light",
        );
    }
    // a bump map's normals, rebuilt toward z, lighting the surface
    {
        let mut registers = registers(0);
        plain_light(&mut registers);
        registers[LIGHT_CONFIG0] = 1 << 28; // bump normals off unit 0, rebuilt
        bind_texture_unit(&mut registers, 0, TEXTURE, 8, 8, 0, 0, 0);
        thru_stage(&mut registers, 1);
        let quad = lit_corners(quaternions, views, ws, -0.5, full_span);
        run_lit(
            &mut resources,
            &mut registers,
            crate::lighting::Tables::default(),
            Some(&normal_texels),
            &quad,
            "a bump map's normals",
        );
    }
    // the last configuration's tables, reading the half vector, the view and
    // the tangent a bump map carries, into the specular colors. The light sits
    // off the view axis: a light straight against the view makes a half vector
    // of nearly nothing, whose direction is rounding noise.
    {
        let mut registers = registers(0);
        plain_light(&mut registers);
        registers[LIGHT_CONFIG0] = (8 << 4) // config 8
            | (2 << 28) // bump tangents
            | (1 << 22); // off unit 1
        registers[LIGHT_CONFIG1] &= !((1 << 16) | (1 << 17) | (1 << 20) | (1 << 21) | (1 << 22));
        registers[LIGHT_SPECULAR0] = WHITE10; // specular 0
        registers[LIGHT_SPECULAR1] = WHITE10; // specular 1
        registers[LIGHT_POSITION_XY] = pica_half(1.0); // sitting at (1, 0, 1)
                                                       // the tangent, the half vector, the view and the normal to the tables
        registers[LIGHT_TABLE_INPUT] = 5 | (1 << 20) | (2 << 24);
        bind_texture_unit(&mut registers, 1, TEXTURE, 8, 8, 0, 0, 0);
        thru_stage(&mut registers, 2);
        let quad = lit_corners(quaternions, views, ws, -0.5, full_span);
        let mut tables = crate::lighting::Tables::default();
        let entries = rising_table(0)
            .into_iter()
            .chain(rising_table(1))
            .chain(rising_table(4))
            .chain(rising_table(5))
            .chain(rising_table(6))
            .collect::<Vec<_>>();
        load_lighting_tables(&mut registers, &mut tables, &entries);
        load_lighting_tables(&mut registers, &mut resources.light_tables, &entries);
        run_lit(
            &mut resources,
            &mut registers,
            tables,
            Some(&normal_texels),
            &quad,
            "the tangent tables of the last configuration",
        );
    }
    // the tables a rewrite between draws reaches the GL thru: the distribution
    // climbs, then falls, and the second answer follows the new entries
    {
        let mut registers = registers(0);
        plain_light(&mut registers);
        registers[LIGHT_CONFIG1] &= !(1 << 16); // distribution 0 back on
        registers[LIGHT_SPECULAR0] = WHITE10; // specular 0
        thru_stage(&mut registers, 2);
        let quad = lit_corners(quaternions, views, ws, -0.5, no_texcoords);
        let falling: Vec<(usize, u32)> = (0..256)
            .map(|i| (0, table_entry(((255 - i) * 16) as u32, -16)))
            .collect();
        let mut tables = crate::lighting::Tables::default();
        load_lighting_tables(&mut registers, &mut tables, &rising_table(0));
        load_lighting_tables(
            &mut registers,
            &mut resources.light_tables,
            &rising_table(0),
        );
        run_lit(
            &mut resources,
            &mut registers,
            tables,
            None,
            &quad,
            "a distribution that climbs",
        );
        let mut tables = crate::lighting::Tables::default();
        load_lighting_tables(&mut registers, &mut tables, &falling);
        load_lighting_tables(&mut registers, &mut resources.light_tables, &falling);
        run_lit(
            &mut resources,
            &mut registers,
            tables,
            None,
            &quad,
            "the same distribution rewritten to fall",
        );
    }
    // the lighting off, then on again over the same registers: a draw with no
    // lighting answers the combiner's lighting colors defensively, and the
    // next one lights for real
    {
        let mut registers = registers(0);
        plain_light(&mut registers);
        registers[LIGHT_CONFIG1] &= !(1 << 16); // distribution 0 back on
        registers[LIGHT_SPECULAR0] = WHITE10; // specular 0
        thru_stage(&mut registers, 2);
        let quad = lit_corners(quaternions, views, ws, -0.5, no_texcoords);
        let mut tables = crate::lighting::Tables::default();
        load_lighting_tables(&mut registers, &mut tables, &rising_table(0));
        load_lighting_tables(
            &mut registers,
            &mut resources.light_tables,
            &rising_table(0),
        );
        registers[LIGHT_ENABLE] = 0;
        run_lit(
            &mut resources,
            &mut registers,
            tables,
            None,
            &quad,
            "the lighting switched off",
        );
        registers[LIGHT_ENABLE] = 1;
        let mut tables = crate::lighting::Tables::default();
        load_lighting_tables(&mut registers, &mut tables, &rising_table(0));
        run_lit(
            &mut resources,
            &mut registers,
            tables,
            None,
            &quad,
            "the lighting switched back on",
        );
    }
}
