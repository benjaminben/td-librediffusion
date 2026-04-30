// GPU-mode inference: run the runner's native CUDA path on `stream` and
// bracket it with cudaEvent records so FpsMeter can report timings.

#include "librediffusion_top.hpp"

#include "td_debug_log.hpp"

#include <cuda_runtime.h>

#include <string>

namespace librediff_td
{

bool LibreDiffusionTOP::runInferenceGpu(int width, int height, cudaStream_t stream)
{
    std::string err;
    myMeter.record_start(stream);
    const bool ok = myRunner->process_gpu_rgba8(
        static_cast<const uint8_t*>(myRgbaInDevice),
        static_cast<uint8_t*>(myRgbaOutDevice), width, height, stream, &err);
    myMeter.record_end(stream);
    if(!ok)
    {
        TDDBG("plugin: process_gpu_rgba8 failed: " << err);
        return false;
    }
    return true;
}

} // namespace librediff_td
