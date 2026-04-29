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

#ifdef __CUDA_RUNTIME_H__
using runner_cuda_stream_t = cudaStream_t;
#else
using runner_cuda_stream_t = void*;
#endif

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

    // The pipeline's internal CUDA stream. The TD plugin uses this so that
    // its cudaArray copies are ordered with respect to the inference work.
    // Returns nullptr if the runner hasn't been initialized.
    runner_cuda_stream_t cuda_stream() const;

private:
    struct Impl;
    Impl* impl_;
};

} // namespace librediff
