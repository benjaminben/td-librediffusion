// LibreDiffusionTOP entry points: lifecycle, execute() orchestration,
// info-CHOP/popup overrides. The GPU and CPU inference branches live in
// _gpu.cpp / _cpu.cpp; parameter setup lives in _params.cpp.

#include "librediffusion_top.hpp"

#include "td_debug_log.hpp"

#include <cuda_runtime.h>

#include <cstdio>

using namespace TD;

namespace librediff_td
{

// Defined in flip_kernel.cu.
void launch_flip_rgba8_inplace(void* data, int width, int height, size_t pitch,
                               cudaStream_t stream);

namespace
{
std::string join_path(const std::string& folder, const char* leaf)
{
    if(folder.empty())
        return {};
    char sep = (folder.back() == '\\' || folder.back() == '/') ? '\0' : '\\';
    return sep ? folder + sep + leaf : folder + leaf;
}

std::string param_string(const OP_Inputs* inputs, const char* name)
{
    const char* str = inputs->getParString(name);
    return str ? str : "";
}

std::string param_path(const OP_Inputs* inputs, const char* name)
{
    const char* path = inputs->getParFilePath(name);
    if(path && *path)
        return path;
    return param_string(inputs, name);
}
} // namespace

LibreDiffusionTOP::LibreDiffusionTOP(const OP_NodeInfo*, TOP_Context* context)
  : myContext{context}
{
}

LibreDiffusionTOP::~LibreDiffusionTOP()
{
    if(myRgbaInDevice)
        cudaFree(myRgbaInDevice);
    if(myRgbaOutDevice)
        cudaFree(myRgbaOutDevice);
    delete[] myHostRgbaIn;
    delete[] myHostRgbaOut;
}

void LibreDiffusionTOP::getGeneralInfo(TOP_GeneralInfo* ginfo, const OP_Inputs*, void*)
{
    ginfo->cookEveryFrameIfAsked = false;
    ginfo->inputSizeIndex = 0;
}

void LibreDiffusionTOP::execute(TOP_Output* output, const OP_Inputs* inputs, void*)
{
    const OP_TOPInput* top = inputs->getInputTOP(0);
    if(!top)
        return;

    const int width = top->textureDesc.width;
    const int height = top->textureDesc.height;

    // Read parameters BEFORE beginCUDAOperations.
    const std::string folder = param_path(inputs, kParEnginesFolder);
    const std::string positive = param_string(inputs, kParPositivePrompt);
    const std::string negative = param_string(inputs, kParNegativePrompt);
    const int timestep = inputs->getParInt(kParTimestep);
    const float guidance = static_cast<float>(inputs->getParDouble(kParGuidance));
    const int mode = inputs->getParInt(kParMode);
    const bool trackMetrics = inputs->getParInt(kParTrackMetrics) != 0;

    myMeter.set_enabled(trackMetrics);
    myMeter.tick();

    if(!folder.empty()
       && (folder != myLastFolder || width != myLastWidth || height != myLastHeight
           || timestep != myLastTimestep))
    {
        tryInit(folder, width, height, timestep);
        myLastFolder = folder;
        myLastWidth = width;
        myLastHeight = height;
        myLastTimestep = timestep;
        myLastPositive.clear();
        myLastNegative.clear();
        myLastGuidance = -1.0f;
    }

    if(myRunner && myRunner->is_initialized()
       && (positive != myLastPositive || negative != myLastNegative))
    {
        std::string err;
        if(myRunner->set_prompt(positive, negative, &err))
        {
            myLastPositive = positive;
            myLastNegative = negative;
        }
        else
        {
            TDDBG("plugin: set_prompt failed: " << err);
        }
    }
    if(myRunner && myRunner->is_initialized() && guidance != myLastGuidance)
    {
        myRunner->set_guidance(guidance);
        myLastGuidance = guidance;
    }

    cudaStream_t stream = myRunner ? myRunner->cuda_stream() : nullptr;

    OP_CUDAAcquireInfo acquire;
    acquire.stream = stream;
    const OP_CUDAArrayInfo* inArray = top->getCUDAArray(acquire, nullptr);
    if(!inArray)
    {
        if(myDiagCount++ < 3)
            TDDBG("plugin: getCUDAArray returned null OP_CUDAArrayInfo*");
        return;
    }

    TOP_CUDAOutputInfo outInfo;
    outInfo.stream = stream;
    outInfo.textureDesc = inArray->textureDesc;
    outInfo.textureDesc.pixelFormat = OP_PixelFormat::RGBA8Fixed;
    outInfo.colorBufferIndex = 0;
    const OP_CUDAArrayInfo* outArray = output->createCUDAArray(outInfo, nullptr);
    if(!outArray)
    {
        if(myDiagCount++ < 3)
            TDDBG("plugin: createCUDAArray returned null");
        return;
    }

    if(!myContext->beginCUDAOperations(nullptr))
    {
        if(myDiagCount++ < 3)
            TDDBG("plugin: beginCUDAOperations failed");
        return;
    }

    bool inferred = false;
    do
    {
        if(!inArray->cudaArray || !outArray->cudaArray)
        {
            if(myDiagCount++ < 3)
                TDDBG("plugin: cudaArray fields still null after begin");
            break;
        }

        if(myDiagCount < 3)
        {
            TDDBG("plugin: execute() in=" << width << "x" << height
                  << " inFmt=" << (int)inArray->textureDesc.pixelFormat
                  << " stream=" << (void*)stream);
        }

        const size_t pitch = static_cast<size_t>(width) * 4;
        const size_t bytes = pitch * static_cast<size_t>(height);
        if(bytes != myRgbaBytes)
        {
            if(myRgbaInDevice)
                cudaFree(myRgbaInDevice);
            if(myRgbaOutDevice)
                cudaFree(myRgbaOutDevice);
            myRgbaInDevice = nullptr;
            myRgbaOutDevice = nullptr;
            if(cudaMalloc(&myRgbaInDevice, bytes) != cudaSuccess
               || cudaMalloc(&myRgbaOutDevice, bytes) != cudaSuccess)
            {
                TDDBG("plugin: cudaMalloc for staging buffers failed");
                break;
            }
            myRgbaBytes = bytes;
        }

        cudaError_t err = cudaMemcpy2DFromArrayAsync(
            myRgbaInDevice, pitch, inArray->cudaArray, 0, 0, pitch, height,
            cudaMemcpyDeviceToDevice, stream);
        if(err != cudaSuccess)
        {
            TDDBG("plugin: cudaMemcpy2DFromArrayAsync failed: "
                  << cudaGetErrorString(err));
            break;
        }

        if(myRunner && myRunner->is_initialized() && !myLastPositive.empty())
        {
            const bool ok = (mode == 1)
                ? runInferenceCpu(width, height, bytes, stream)
                : runInferenceGpu(width, height, stream);
            if(!ok)
                break;
            inferred = true;
        }

        // VAE writes row 0 = top; TD's GL-backed CUDA texture displays row
        // 0 = bottom. V-flip on inferred output; passthrough stays as-is.
        void* src = inferred ? myRgbaOutDevice : myRgbaInDevice;
        if(inferred)
            launch_flip_rgba8_inplace(src, width, height, pitch, stream);

        err = cudaMemcpy2DToArrayAsync(
            outArray->cudaArray, 0, 0, src, pitch, pitch, height,
            cudaMemcpyDeviceToDevice, stream);
        if(err != cudaSuccess)
        {
            TDDBG("plugin: cudaMemcpy2DToArrayAsync failed: "
                  << cudaGetErrorString(err));
        }

        if(myDiagCount < 3)
            TDDBG("plugin: execute() OK inferred=" << inferred);
        ++myDiagCount;
    } while(false);

    myContext->endCUDAOperations(nullptr);
}

int32_t LibreDiffusionTOP::getNumInfoCHOPChans(void*)
{
    return 2;
}

void LibreDiffusionTOP::getInfoCHOPChan(int32_t index, OP_InfoCHOPChan* chan, void*)
{
    if(!chan || !chan->name)
        return;
    switch(index)
    {
    case 0:
        chan->name->setString("inference_fps");
        chan->value = static_cast<float>(myMeter.inference_fps());
        break;
    case 1:
        chan->name->setString("inference_ms");
        chan->value = static_cast<float>(myMeter.inference_ms());
        break;
    default:
        break;
    }
}

void LibreDiffusionTOP::getInfoPopupString(OP_String* info, void*)
{
    if(!info)
        return;
    char buf[160];
    if(!myMeter.enabled())
    {
        std::snprintf(buf, sizeof(buf), "Inference metrics: disabled");
    }
    else
    {
        const double fps = myMeter.inference_fps();
        const double ms = myMeter.inference_ms();
        if(fps <= 0.0 || ms <= 0.0)
            std::snprintf(buf, sizeof(buf), "Inference: idle");
        else
            std::snprintf(buf, sizeof(buf),
                          "Inference: %.1f FPS\nPer-frame: %.2f ms", fps, ms);
    }
    info->setString(buf);
}

void LibreDiffusionTOP::tryInit(const std::string& folder, int width, int height,
                                int timestep)
{
    myRunner.reset();

    librediff::Runner::Config cfg;
    cfg.clip_engine_path = join_path(folder, "clip.engine");
    cfg.unet_engine_path = join_path(folder, "unet.engine");
    cfg.vae_encoder_path = join_path(folder, "vae_encoder.engine");
    cfg.vae_decoder_path = join_path(folder, "vae_decoder.engine");
    cfg.width = width;
    cfg.height = height;
    cfg.batch_size = 1;
    cfg.timestep_indices = {timestep};

    auto runner = std::make_unique<librediff::Runner>();
    std::string err;
    if(runner->init(cfg, &err))
    {
        TDDBG("plugin: runner initialized OK at " << width << "x" << height);
        myRunner = std::move(runner);
    }
    else
    {
        TDDBG("plugin: runner init FAILED: " << err);
    }
}

} // namespace librediff_td
