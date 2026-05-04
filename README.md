# td-librediffusion

A TouchDesigner Custom Operator TOP that hosts
[librediffusion](https://github.com/benjaminben/librediffusion-bb) for
real-time img2img inference. Built around SD-Turbo single-step diffusion,
running fully on the GPU via CUDA texture interop. Tested at 30+ fps at
512x512 on an RTX 4090 (Mobile).

## Requirements

- Windows
- NVIDIA GPU, Turing (sm_75) or newer
- NVIDIA driver supporting CUDA 13 (R570 or newer)
- CUDA Toolkit 13.0+
- TensorRT 10.14+ (development package, with headers and import libraries)
- Visual Studio 2022, version 17.11+ (with the "Desktop development with
  C++" workload)
- TouchDesigner build supporting plugin API v12 (TouchDesigner 2023.30000+)
- A built `librediffusion.dll` and matching engine files. See
  [librediffusion-bb](https://github.com/benjaminben/librediffusion-bb)
  for build instructions and engine compilation.

## Quick start

### 1. Build `librediffusion.dll`

Follow the README at
[github.com/benjaminben/librediffusion-bb](https://github.com/benjaminben/librediffusion-bb)
to clone, build, and produce `librediffusion.dll` plus a folder of `.engine`
files for SD-Turbo at the resolution you want (e.g. `sd-turbo-512`, output
contains `clip.engine`, `unet.engine`, `vae_encoder.engine`,
`vae_decoder.engine`).

### 2. Tell `stage.cmd` where your dependencies live

`stage.cmd` needs three paths to find the runtime DLLs it copies next to
the plugin DLL. Pick whichever method fits your shell.

**Option A — persist for future shells (recommended).** Run once, then every
new cmd or PowerShell window inherits them:

```cmd
setx LIBREDIFF_BUILD "C:\path\to\librediffusion-bb\build"
setx TENSORRT_ROOT   "C:\path\to\TensorRT-10.X.Y.Z"
setx CUDA_BIN        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.X\bin\x64"
```

`setx` writes to your user environment in the registry. **The window you
run it in will not see the new values** — open a fresh shell after.

**Option B — set for the current shell only.**

cmd:
```cmd
set LIBREDIFF_BUILD=C:\path\to\librediffusion-bb\build
set TENSORRT_ROOT=C:\path\to\TensorRT-10.X.Y.Z
set CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.X\bin\x64
```

PowerShell:
```powershell
$env:LIBREDIFF_BUILD = "C:\path\to\librediffusion-bb\build"
$env:TENSORRT_ROOT   = "C:\path\to\TensorRT-10.X.Y.Z"
$env:CUDA_BIN        = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.X\bin\x64"
```

(In PowerShell, plain `set FOO=BAR` does NOT create an environment variable
— it creates a PowerShell-scoped variable that child processes can't see.
Use `$env:FOO = "BAR"` instead.)

**Option C — pass positionally to `stage`.** No env var setup required:

```cmd
stage <librediffusion-build> <tensorrt-root> <cuda-bin>
```

### 3. Build the plugin

```cmd
cd td-librediffusion
build configure   :: cmake configure (first time, or after CMakeLists.txt changes)
build             :: incremental build
stage             :: copy librediffusion.dll + TRT/CUDA DLLs into plugin/
```

`build.cmd` auto-detects MSVC by:
1. Honoring a pre-set `%VCVARS%` if you have one
2. Querying `vswhere.exe` for a VS 2022 (17.x) install
3. Probing well-known disk paths under `C:\Program Files [(x86)]\Microsoft Visual Studio\2022\` (handles cases where vswhere doesn't have the install registered)

You'll need Visual Studio 2022 — VS 2026 / VS 18 ships an MSVC with broken
C++23 ranges support that fails to compile. If you only have VS 2026, install
either VS 2022 BuildTools side-by-side or the legacy v143 toolset inside VS
2026 via the Visual Studio Installer.

### 4. Use in TouchDesigner

1. Drop a **Custom Operator TOP** into your network.
2. Set its **Plugin Path** to `<repo>\plugin\td_librediffusion_top.dll`.
3. Configure on the **LibreDiffusion** parameter page:
   - **Engines Folder**: directory containing the four `.engine` files.
   - **Positive Prompt**: text to condition the output (e.g. `oil painting
     of a viking warrior, dramatic lighting`).
   - **Negative Prompt**: text to push the output away from.
   - **Guidance**: classifier-free guidance scale. `<= 1.0` disables CFG;
     1.2 is a reasonable default with CFG on.
   - **Timestep Index**: 0–49 in the SD-Turbo schedule. Higher index = less
     noise injected, output closer to input. 25 is a good starting point.
   - **Inference Mode**: GPU (native CUDA path) or CPU (round-trip through
     host memory; slower, equivalent output).
4. Wire any RGBA8 source TOP at the engine's resolution into the input.
   Insert a **Resolution TOP** upstream if the source isn't already at
   engine size.

A sample `td/td-librediffusion.toe` is included for reference.

## Plugin inputs

The TOP takes one or two TOP inputs. Both follow the standard
TouchDesigner TOP convention: **uint8 RGBA, bottom-up**.

| Input | Required | Format | Notes |
|---|---|---|---|
| #1 | Yes | uint8 RGBA, bottom-up | Source frame for img2img. Must already match the engine's resolution (see Known limitations). |
| #2 | When ControlNet=On | uint8 RGBA, bottom-up | OpenPose stick-figure (or other ControlNet conditioning image). Same format as input #1; the plugin flips it to top-down internally before handing it to the engine, so do not pre-flip in upstream Script TOPs. Dimensions must match input #1. |

A Script TOP that emits an OpenPose stick figure into input #2 already
complies with this contract by default — no flipping or format
conversion is needed in the Python callback.

## Parameters

| Parameter | Type | Default | Notes |
|---|---|---|---|
| Engines Folder | folder | `""` | Must contain `clip.engine`, `unet.engine`, `vae_encoder.engine`, `vae_decoder.engine` |
| Positive Prompt | string | `""` | Empty = passthrough (no inference) |
| Negative Prompt | string | `""` | |
| Guidance | float | 1.2 | `<= 1.0` disables CFG; values around 1.2–1.5 work well for SD-Turbo |
| Timestep Index | int | 25 | 0 = max noise, 49 = min noise |
| Inference Mode | menu | GPU | Toggle between native CUDA and CPU fallback |

## Architecture

Two layers, deliberately separated:

```
runner/                            Host-agnostic wrapper around librediffusion.
  librediffusion_runner.{hpp,cpp}    Reusable in any host (standalone app, OBS, ...).
  dll_loader.{hpp,cpp}               Forces librediffusion.dll + transitive
                                     deps to load from this DLL's folder via
                                     LoadLibraryEx + LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
                                     (avoids picking up TouchDesigner's
                                     bundled nvinfer_10.dll).
  td_debug_log.hpp                   OutputDebugStringA-based logging
                                     visible in Sysinternals DebugView.

plugin/
  td_top_plugin.cpp                  TouchDesigner-specific glue. Implements
                                     TOP_CPlusPlusBase in CUDA executeMode.
                                     Throwaway when a non-TD host is built.

third_party/derivative/              Vendored TouchDesigner plugin SDK headers.

build.cmd / stage.cmd                Windows build / staging helpers.
```

## Known limitations

- Input texture must already be at the engine's resolution (e.g. 512x512
  for `sd-turbo-512`). Add a Resolution TOP upstream as needed. Internal
  resize is on the roadmap.
- The img2img workflow doesn't add explicit noise to the encoded latent
  before denoising. As a result, low timestep indices (0–10, where the
  model expects heavily-noised input) can produce noisy or unstable output.
  Indices 20–40 typically give clean results.
- Only SD-Turbo / SD 2.1 base models with 1024-dim text encoders are
  currently configured. SD 1.5 (768-dim) and SDXL would need additional
  Config plumbing.

## Roadmap

- Internal resize so any input resolution Just Works
- Multi-engine selection from a menu populated by scanning a root folder
- OSC parameter control
- A standalone host (no TouchDesigner required) reusing `runner/`

## License

MIT, see [LICENSE](LICENSE). The vendored TouchDesigner SDK headers under
`third_party/derivative/` are owned by Derivative Inc. and redistributed
under their Shared Use License — see
[third_party/derivative/README.md](third_party/derivative/README.md).
