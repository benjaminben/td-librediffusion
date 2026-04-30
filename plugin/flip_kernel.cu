// In-place vertical flip for RGBA8 images on a CUDA stream. Replaces a
// per-row cudaMemcpy2DToArrayAsync loop in the TOP execute() path.
//
// Each thread owns one pixel in the top half and swaps with its mirror in
// the bottom half. For odd heights the middle row is untouched.

#include <cuda_runtime.h>
#include <cstdint>

namespace librediff_td
{

namespace
{
__global__ void flip_rgba8_inplace_kernel(uint8_t* data, int width,
                                          int half_height, int height,
                                          size_t pitch)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= half_height)
        return;
    const int yMirror = height - 1 - y;
    uint32_t* rowTop = reinterpret_cast<uint32_t*>(
        data + static_cast<size_t>(y) * pitch);
    uint32_t* rowBot = reinterpret_cast<uint32_t*>(
        data + static_cast<size_t>(yMirror) * pitch);
    const uint32_t a = rowTop[x];
    const uint32_t b = rowBot[x];
    rowTop[x] = b;
    rowBot[x] = a;
}
} // namespace

void launch_flip_rgba8_inplace(void* data, int width, int height, size_t pitch,
                               cudaStream_t stream)
{
    if(!data || width <= 0 || height <= 1)
        return;
    const int half = height / 2;
    const dim3 block(16, 16);
    const dim3 grid((width + block.x - 1) / block.x,
                    (half + block.y - 1) / block.y);
    flip_rgba8_inplace_kernel<<<grid, block, 0, stream>>>(
        static_cast<uint8_t*>(data), width, half, height, pitch);
}

} // namespace librediff_td
