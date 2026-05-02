// LibreDiffusionRunner: host-agnostic wrapper around the librediffusion C API.
//
// This file deliberately has zero TouchDesigner dependencies so the same
// runner can be linked into the standalone Phase 2 host without modification.
//
// Status: stub. Step 1 of the plan does not exercise this class. Step 2
// fills in init() and Step 3 fills in process() / process_gpu().

#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Forward-declare CUDA's opaque stream tag so the public API has a stable
// type regardless of whether <cuda_runtime.h> has been included by the
// consumer. cudaStream_t is itself a typedef for CUstream_st*, so a TU
// that does include cuda_runtime.h can pass cudaStream_t values to these
// functions without a cast — and the mangled symbol stays the same either
// way, eliminating the include-order ABI footgun.
struct CUstream_st;
using runner_cuda_stream_t = CUstream_st*;

namespace librediff
{

enum class Workflow
{
    SDTURBO_IMG2IMG = 0,
    SDXL_IMG2IMG = 1,
};

class Runner
{
public:
    struct Config
    {
        std::string clip_engine_path;
        std::string unet_engine_path;
        std::string vae_encoder_path;
        std::string vae_decoder_path;
        // ControlNet (v1, combined engine). When non-empty this overrides
        // unet_engine_path and switches the pipeline into combined-engine
        // mode. When empty the runner behaves identically to today.
        std::string combined_unet_controlnet_engine_path;
        int width = 512;
        int height = 512;
        int batch_size = 1;
        std::vector<int> timestep_indices{25};
        Workflow workflow = Workflow::SDTURBO_IMG2IMG;
    };

    Runner();
    ~Runner();

    Runner(const Runner&) = delete;
    Runner& operator=(const Runner&) = delete;

    bool init(const Config& cfg, std::string* err_out);
    bool is_initialized() const;

    bool set_prompt(const std::string& positive, const std::string& negative,
                    std::string* err_out);
    void set_guidance(float scale);
    void set_seed(uint32_t seed);

    // CPU-buffer entry point (Step 3).
    bool process(const uint8_t* rgba_in, uint8_t* rgba_out, int width, int height,
                 std::string* err_out);

    // GPU entry point used by the TD plugin (Step 4).
    // Inputs/outputs are linear RGBA8 NHWC device pointers at engine resolution.
    // Internally: rgba->nchw_half conversion -> img2img_gpu_half -> nchw_half->rgba.
    // All work runs on the supplied stream.
    bool process_gpu_rgba8(
        const uint8_t* rgba_in_device, uint8_t* rgba_out_device, int width, int height,
        runner_cuda_stream_t stream, std::string* err_out);

    // ControlNet (v1) per-frame input. Forwards an RGBA NHWC uint8 device
    // pointer to the library, which performs a fused conversion directly
    // into the engine's persistent control-image buffer. No-op if the
    // runner was initialized without combined_unet_controlnet_engine_path.
    bool set_control_image_gpu(
        const uint8_t* device_rgba, int width, int height,
        runner_cuda_stream_t stream, std::string* err_out);

    // ControlNet (v1) per-frame strength. No-op if combined-engine mode
    // is off.
    void set_controlnet_strength(float strength);

    // True if init() loaded a combined UNet+ControlNet engine.
    bool combined_engine_mode() const;

    // The pipeline's internal CUDA stream. The TD plugin uses this so that
    // its cudaArray copies are ordered with respect to the inference work.
    // Returns nullptr if the runner hasn't been initialized.
    runner_cuda_stream_t cuda_stream() const;

private:
    struct Impl;
    Impl* impl_;
};

} // namespace librediff
