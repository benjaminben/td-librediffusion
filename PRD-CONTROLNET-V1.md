# PRD: ControlNet Support (v1) — td-librediffusion

## Scope

Expose ControlNet support in the TouchDesigner custom operator, building on the library-side changes specified in `librediffusion/PRD-CONTROLNET-V1.md`. v1 targets SD-Turbo + Canny ControlNet for img2img inference, single-step, single-batch — the realtime live-visuals configuration.

This PRD covers TD operator parameters, input pin additions, runner integration, and pipeline lifecycle. It does not cover library, engine-export, or validation tooling — see the librediffusion PRD for those.

## Guiding principles

1. **Per-frame inference cost is the primary constraint.** This work targets realtime live concert visuals.
2. **Zero perf cost when ControlNet is not enabled.** Users running today's plain UNet pipeline must see no measurable regression.
3. **Setup ergonomics is secondary** to perf, but failures should be loud and explicit (no silent fallbacks to wrong-but-running behavior).
4. **Live-tunable parameters are first-class.** Strength must be CHOP-drivable so audio-reactive / MIDI-mapped workflows work without code.

## User-visible operator surface

### New parameters

Added to the existing `LibreDiffusionTOP` parameter set alongside the current parameters (`Enginesfolder`, `Positiveprompt`, `Negativeprompt`, `Guidance`, `Timestep`, `Mode`, `Trackmetrics`, `Maxinferencefps`):

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `Controlnet` | menu | `Off` | Values: `Off`, `On`. Toggling triggers pipeline re-init. |
| `Controlnetstrength` | float | `1.0` | Range 0.0–2.0. Live-tunable (CHOP-drivable). Pushed to pipeline per frame via `librediffusion_set_controlnet_strength`. |

The `Controlnet` parameter is intentionally singular (not `Controlnet1`) — multi-controlnet support is a v3+ feature requiring the library's separate-engine path. When that ships, `Controlnet2` and `Controlnet3` parameters will be added alongside the existing `Controlnet` parameter (which becomes "slot 1" semantically). No `.toe` migration required.

### New input pin

Add a second TOP input pin to the operator: **input #2 = control image**.

- Format: TouchDesigner's native RGBA (NHWC uint8)
- Resolution: must match operator's configured engine resolution
- Behavior: forwarded device-pointer to `librediffusion_set_control_image_gpu` per frame, no CPU round-trip

### Behavior matrix

| `Controlnet` | Input #2 connected | Behavior |
|---|---|---|
| `Off` | n/a | Plain UNet engine path. Input #2 ignored. **Identical perf to today.** |
| `On` | yes | Combined engine path. Source video on input #1, control image on input #2 consumed per frame. |
| `On` | no | Operator errors at init (clear message: "ControlNet=On requires a control image on input #2"). Output is the last good cached frame, or transparent on first frame. Inference does not run for that frame, so no perf is wasted. As soon as input #2 is wired, next frame works normally. |

No silent fallback to zero/black control image. The user gets a clear error if the patch is misconfigured.

## Engines folder convention

The operator continues to load engines by convention from the user's configured `Enginesfolder`. ControlNet adds one new file:

| Filename | Required when | Loaded via |
|---|---|---|
| `unet.engine` | `Controlnet = Off` | `librediffusion_config_set_unet_engine` |
| `unet_controlnet.engine` | `Controlnet = On` | `librediffusion_config_set_combined_unet_controlnet_engine` |
| `vae_encoder.engine` | always | unchanged |
| `vae_decoder.engine` | always | unchanged |
| `clip.engine` | always | unchanged |

Users wanting ControlNet support drop a `unet_controlnet.engine` (built via `tools/export_combined_unet_controlnet.py` in the librediffusion repo) into their existing engines folder alongside the existing engines. The plain `unet.engine` stays — it's used when ControlNet is toggled Off.

The plugin does **not** read the engine's `.meta.json` sidecar in v1. Routing is purely convention-based on filename.

## Pipeline lifecycle

### Re-init on `Controlnet` toggle (R1)

Toggling `Controlnet: Off → On` (or vice versa) triggers a full pipeline re-init: the existing pipeline is destroyed and a new one is created with the appropriate engine path set. This blocks the cook for the re-init duration (engine load + buffer reallocation, ~hundreds of ms to a few seconds depending on engine size).

This is acceptable because `Controlnet` is a setup-time decision in live performance — the artist commits to ControlNet-on or ControlNet-off when configuring the patch, not mid-song. Re-init does not happen during a show.

The alternative (R2 — keep both engines loaded simultaneously, switch the binding without re-init) was considered and rejected: live visuals are GPU-VRAM-precious (the artist may be running other GPU tooling — audio FX, video synth, etc.), and the extra ~1–2GB of unused engine sitting in VRAM costs continuously while the toggle savings only matter once during setup.

### Per-frame flow (ControlNet enabled)

In `Runner::process_gpu_rgba8` (or its successor), per frame:

1. NHWC RGBA uint8 source → NCHW RGB fp16 (existing path, unchanged)
2. NHWC RGBA uint8 control image → bind to pipeline via `librediffusion_set_control_image_gpu(control_device_ptr, w, h, stream)`
3. Push current strength: `librediffusion_set_controlnet_strength(strength_value)`
4. `librediffusion_img2img_gpu_half(...)` (existing call, unchanged signature; combined-engine binding handled internally by the library)
5. NCHW fp16 output → NHWC RGBA uint8 (existing path, unchanged)

Steps 2 and 3 are the only additions to the per-frame path. They are no-ops at the library level when combined-engine mode is off, so the per-frame call sequence can be unconditional in the runner — no `if (controlnet_enabled)` branching in the hot loop.

## Implementation changes

### `runner/librediffusion_runner.hpp` / `.cpp`

Extend `Runner::Config` with optional ControlNet fields:

```cpp
struct Config
{
    // ... existing fields ...

    // ControlNet (v1: combined UNet+ControlNet engine)
    std::string combined_unet_controlnet_engine_path;  // empty == ControlNet disabled
};
```

Extend `Runner` with new methods:

```cpp
// Bind a per-frame control image (GPU device pointer, RGBA NHWC uint8).
// No-op if Runner was initialized without combined_unet_controlnet_engine_path.
bool set_control_image_gpu(
    const uint8_t* device_rgba,
    int width, int height,
    runner_cuda_stream_t stream,
    std::string* err_out);

// Set ControlNet conditioning strength (0.0 – 2.0 typical).
// No-op if Runner was initialized without combined_unet_controlnet_engine_path.
void set_controlnet_strength(float strength);
```

Extend `Runner::init` to choose between `librediffusion_config_set_unet_engine` and `librediffusion_config_set_combined_unet_controlnet_engine` based on whether `cfg.combined_unet_controlnet_engine_path` is set.

The runner intentionally does NOT add a separate "process_gpu_rgba8_with_control" entry point. The existing `process_gpu_rgba8` continues to work; control image binding happens via the new setter before the call. This keeps the runner API minimal.

### `plugin/librediffusion_top.hpp` / `librediffusion_top_*.cpp`

Add parameter constants:

```cpp
inline constexpr const char* kParControlnet = "Controlnet";
inline constexpr const char* kParControlnetstrength = "Controlnetstrength";
```

Extend `setupParameters` with the menu and float parameter definitions.

Extend the operator class with cached state to detect `Controlnet` toggle changes:

```cpp
int myLastControlnetMode = 0;  // 0=off, 1=on; -1 = uninitialized
```

In `execute`:

1. Read current `Controlnet` and `Controlnetstrength` parameter values.
2. If `Controlnet` mode changed since last frame, mark pipeline for re-init (existing pattern when `Enginesfolder` or dimensions change).
3. If ControlNet is on:
   - Verify input #2 is connected. If not, set operator error string, output last good frame, return.
   - Acquire input #2's device pointer (per existing TD CUDA texture access pattern).
   - Call `myRunner->set_control_image_gpu(input2_ptr, w, h, stream, &err)`.
   - Call `myRunner->set_controlnet_strength(strength_value)`.
4. Run inference (existing path).

The `tryInit` function is extended to compute `combined_unet_controlnet_engine_path` from `<Enginesfolder>/unet_controlnet.engine` when `Controlnet = On`, and pass it through to `Runner::Config`.

### Operator info / status

Extend `getInfoCHOPChan` and/or `getInfoPopupString` to surface:

- Which engine is currently loaded (`unet.engine` vs `unet_controlnet.engine`)
- Current `Controlnet` mode and strength
- Last error if input #2 is required but missing

## Performance expectations

When ControlNet is **disabled** (`Controlnet = Off`):

- Per-frame inference time matches today's baseline within ±2% (verified by the library-side baseline assertion in the librediffusion PRD).
- No new memory allocated.

When ControlNet is **enabled** (`Controlnet = On`) at SD-Turbo 512²:

- Per-frame UNet engine cost increases by ~30–50% over the plain UNet baseline due to the embedded ControlNet compute. Approximate range: ~7–15 ms per inference on RTX 4090-class hardware vs ~5–10 ms baseline.
- Control image conversion adds ~3–8 μs per frame (negligible relative to UNet).
- Strength scalar update is ~free (1-element host→device transfer).

If measured framerate impact is unacceptable, the escape hatch is library-side: re-export the combined engine using a smaller ControlNet variant (T2I-Adapter, distilled CN). No TD plugin changes required for the swap — just point at a different `unet_controlnet.engine` file.

## Out of scope for v1

- Multi-controlnet (`Controlnet2`, `Controlnet3` slots) — gated on library's (B) separate-engine path
- SDXL ControlNet — gated on library's SDXL combined engine
- `Mode = txt2img` with ControlNet — library supports it; v1 operator does not expose it
- Reading the engine `.meta.json` sidecar for validation/info display
- Per-block ControlNet strength
- ControlNet `start/end` timestep range
- Hot-swap of ControlNet variants without pipeline re-init (R2)

## Acceptance criteria

1. New parameters `Controlnet` (menu Off/On) and `Controlnetstrength` (float 0–2) appear in the operator's parameter dialog.
2. `Controlnetstrength` is CHOP-drivable.
3. Operator accepts a second TOP input pin; input #1 remains the source video.
4. With `Controlnet = Off`: operator behaves identically to today, including frame timing within ±2% of baseline. Input #2 connection state is irrelevant.
5. With `Controlnet = On` and input #2 connected: combined engine is loaded from `<Enginesfolder>/unet_controlnet.engine` and ControlNet conditioning is visible in the output.
6. With `Controlnet = On` and input #2 not connected: operator displays a clear error in its info popup; output is last good frame or transparent on first frame; no inference runs.
7. Toggling `Controlnet` Off↔On triggers pipeline re-init (one-time cook block, no crash).
8. `Controlnetstrength = 0` produces output indistinguishable from `Controlnet = Off` (within fp16 noise).
9. `Controlnetstrength = 2` produces output more strongly constrained by the control image than `Controlnetstrength = 1`.
10. Different control images on input #2 produce visibly different outputs (verifies ControlNet is consuming the input rather than ignoring it).
11. `Runner` API additions (`set_control_image_gpu`, `set_controlnet_strength`, `Config::combined_unet_controlnet_engine_path`) compile and link against the librediffusion library specified by the librediffusion-side PRD.
12. No existing operator parameter or behavior changes — all additions are purely additive.

## Dependencies

- librediffusion library with the ControlNet C API additions (`set_combined_unet_controlnet_engine`, `set_control_image_gpu`, `set_controlnet_strength`, etc.) — see `librediffusion/PRD-CONTROLNET-V1.md`.
- A built `unet_controlnet_canny.engine` in the user's engines folder (produced by `tools/export_combined_unet_controlnet.py` in the librediffusion repo). Renamed to `unet_controlnet.engine` by convention when dropped into the engines folder.
