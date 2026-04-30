// CPU-mode inference: device input -> host -> librediffusion_img2img ->
// host -> device. Stream-based event timing isn't meaningful here (the
// real work happens on the host), so this path is not bracketed for
// FpsMeter.

#include "librediffusion_top.hpp"

#include "td_debug_log.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <new>
#include <string>

namespace librediff_td
{

bool LibreDiffusionTOP::ensureHostBuffers(size_t bytes)
{
    if(bytes == myHostBytes && myHostRgbaIn && myHostRgbaOut)
        return true;
    delete[] myHostRgbaIn;
    delete[] myHostRgbaOut;
    myHostRgbaIn = new(std::nothrow) uint8_t[bytes];
    myHostRgbaOut = new(std::nothrow) uint8_t[bytes];
    if(!myHostRgbaIn || !myHostRgbaOut)
    {
        myHostBytes = 0;
        return false;
    }
    myHostBytes = bytes;
    return true;
}

bool LibreDiffusionTOP::runInferenceCpu(int width, int height, size_t bytes,
                                        cudaStream_t stream)
{
    if(!ensureHostBuffers(bytes))
    {
        TDDBG("plugin: host buffer alloc failed");
        return false;
    }
    cudaMemcpyAsync(myHostRgbaIn, myRgbaInDevice, bytes,
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream); // need data on host before the CPU call

    std::string err;
    if(!myRunner->process(myHostRgbaIn, myHostRgbaOut, width, height, &err))
    {
        TDDBG("plugin: process(cpu) failed: " << err);
        return false;
    }
    cudaMemcpyAsync(myRgbaOutDevice, myHostRgbaOut, bytes,
                    cudaMemcpyHostToDevice, stream);
    return true;
}

} // namespace librediff_td
