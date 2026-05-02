// LibreDiffusionTOP entry points: lifecycle, execute() orchestration,
// info-CHOP/popup overrides. The GPU and CPU inference branches live in
// _gpu.cpp / _cpu.cpp; parameter setup lives in _params.cpp.

#include "librediffusion_top.hpp"

#include "td_debug_log.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <filesystem>

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

bool combined_engine_available(const std::string& folder)
{
    if(folder.empty())
        return false;
    std::error_code ec;
    return std::filesystem::exists(join_path(folder, "unet_controlnet.engine"), ec);
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
    if(myControlRgbaDevice)
        cudaFree(myControlRgbaDevice);
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
    // kParMode is also a string menu -- same getParInt caveat as kParControlnet.
    const std::string modeSel = param_string(inputs, kParMode);
    const int mode = (modeSel == "CPU") ? 1 : 0;
    const bool trackMetrics = inputs->getParInt(kParTrackMetrics) != 0;
    const float maxFps = static_cast<float>(inputs->getParDouble(kParMaxInferenceFps));
    // ControlNet parameter is a string menu created via appendMenu. TD's
    // getParInt() does NOT return the menu index for string menus -- it
    // always returns 0 because "Off"/"On" don't parse as integers. Read the
    // selected name via getParString and compare. (Bug history: this read
    // bug silently kept the plugin in plain-UNet mode regardless of the
    // toggle; visible in debugview.log as combined_engine=0 always.)
    const std::string controlnetSel = param_string(inputs, kParControlnet);
    int controlnetMode = (controlnetSel == "On") ? 1 : 0;
    // Grey out ControlNet params + force Off when the selected folder has no
    // unet_controlnet.engine. Keeps the operator usable with plain-UNet-only
    // engine folders even if the user accidentally left the toggle On.
    const bool cnAvailable = combined_engine_available(folder);
    inputs->enablePar(kParControlnet, cnAvailable);
    inputs->enablePar(kParControlnetStrength, cnAvailable);
    if(!cnAvailable)
        controlnetMode = 0;
    const float controlnetStrength
        = static_cast<float>(inputs->getParDouble(kParControlnetStrength));

    myMeter.set_enabled(trackMetrics);
    myMeter.tick();

    // Toggling Controlnet On <-> Off triggers a full pipeline re-init.
    if(!folder.empty()
       && (folder != myLastFolder || width != myLastWidth || height != myLastHeight
           || timestep != myLastTimestep || controlnetMode != myLastControlnetMode))
    {
        tryInit(folder, width, height, timestep, controlnetMode == 1);
        myLastFolder = folder;
        myLastWidth = width;
        myLastHeight = height;
        myLastTimestep = timestep;
        myLastControlnetMode = controlnetMode;
        myLastPositive.clear();
        myLastNegative.clear();
        myLastGuidance = -1.0f;
        myLastControlnetStrength = -1.0f;
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
    // ControlNet strength is CHOP-drivable; push every change.
    if(myRunner && myRunner->is_initialized()
       && myRunner->combined_engine_mode()
       && controlnetStrength != myLastControlnetStrength)
    {
        myRunner->set_controlnet_strength(controlnetStrength);
        myLastControlnetStrength = controlnetStrength;
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

    // Input #2 = ControlNet control image. Required when Controlnet=On.
    const OP_TOPInput* controlTop = inputs->getInputTOP(1);
    const OP_CUDAArrayInfo* controlArray = nullptr;
    if(controlnetMode == 1)
    {
        if(!controlTop)
        {
            myLastErrorMessage
                = "ControlNet=On requires a control image on input #2";
            // Fall through with no inference; output will be passthrough or
            // last good frame, matching the documented behavior.
        }
        else
        {
            const int cw = controlTop->textureDesc.width;
            const int ch = controlTop->textureDesc.height;
            if(cw != width || ch != height)
            {
                myLastErrorMessage = "ControlNet input #2 dimensions must "
                                     "match input #1 (engine resolution)";
                controlTop = nullptr;
            }
            else
            {
                OP_CUDAAcquireInfo controlAcquire;
                controlAcquire.stream = stream;
                controlArray = controlTop->getCUDAArray(controlAcquire, nullptr);
                if(!controlArray)
                {
                    myLastErrorMessage
                        = "ControlNet input #2 getCUDAArray() returned null";
                }
                else
                {
                    myLastErrorMessage.clear();
                }
            }
        }
    }
    else
    {
        myLastErrorMessage.clear();
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
            myHasCachedOutput = false;
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

        // ControlNet (v1): when On + input #2 is connected, copy the
        // control image into our linear staging buffer and forward the
        // device pointer to the runner. Library does the fused conversion
        // directly into the engine's persistent buffer.
        bool controlReady = false;
        if(controlnetMode == 1 && controlArray && controlArray->cudaArray
           && myRunner && myRunner->is_initialized()
           && myRunner->combined_engine_mode())
        {
            if(bytes != myControlRgbaBytes)
            {
                if(myControlRgbaDevice)
                    cudaFree(myControlRgbaDevice);
                myControlRgbaDevice = nullptr;
                if(cudaMalloc(&myControlRgbaDevice, bytes) != cudaSuccess)
                {
                    TDDBG("plugin: cudaMalloc for ControlNet staging failed");
                    myControlRgbaBytes = 0;
                }
                else
                {
                    myControlRgbaBytes = bytes;
                }
            }
            if(myControlRgbaDevice)
            {
                err = cudaMemcpy2DFromArrayAsync(
                    myControlRgbaDevice, pitch, controlArray->cudaArray, 0, 0,
                    pitch, height, cudaMemcpyDeviceToDevice, stream);
                if(err != cudaSuccess)
                {
                    TDDBG("plugin: ControlNet cudaMemcpy2DFromArrayAsync "
                          "failed: " << cudaGetErrorString(err));
                }
                else
                {
                    // Source TOPs are bottom-up like input #1; flip so
                    // the engine sees a top-down control image consistent
                    // with the source.
                    launch_flip_rgba8_inplace(
                        myControlRgbaDevice, width, height, pitch, stream);
                    std::string cnErr;
                    if(myRunner->set_control_image_gpu(
                           static_cast<const uint8_t*>(myControlRgbaDevice),
                           width, height, stream, &cnErr))
                    {
                        controlReady = true;
                    }
                    else
                    {
                        myLastErrorMessage = cnErr;
                    }
                }
            }
        }

        // ControlNet=On but no usable control image -> skip inference for
        // this frame so we don't burn cycles on a bad input. Output falls
        // back to the cached previous frame (if any) or passthrough.
        const bool controlnetBlocking
            = (controlnetMode == 1) && (myRunner && myRunner->combined_engine_mode())
              && !controlReady;

        const bool canInfer =
            myRunner && myRunner->is_initialized() && !myLastPositive.empty()
            && !controlnetBlocking;
        bool throttled = false;
        if(canInfer && maxFps > 0.0f && myHasCachedOutput)
        {
            const auto now = std::chrono::steady_clock::now();
            const double elapsed =
                std::chrono::duration<double>(now - myLastInferenceTime).count();
            if(elapsed < 1.0 / static_cast<double>(maxFps))
                throttled = true;
        }

        if(canInfer && !throttled)
        {
            // TD's CUDA arrays are bottom-up; the VAE encoder expects
            // top-down. Flip the staged input before inference so the
            // encoder, decoder, and our output flip all stay consistent.
            launch_flip_rgba8_inplace(myRgbaInDevice, width, height, pitch, stream);
            myLastInferenceTime = std::chrono::steady_clock::now();
            const bool ok = (mode == 1)
                ? runInferenceCpu(width, height, bytes, stream)
                : runInferenceGpu(width, height, stream);
            if(!ok)
                break;
            inferred = true;
            myHasCachedOutput = true;
        }

        // Output selection: a fresh inference wrote myRgbaOutDevice top-down
        // and needs flipping back; a throttled cook reuses the prior
        // (already-bottom-up) cached result; otherwise passthrough the input.
        void* src;
        bool flipBack = false;
        if(inferred)
        {
            src = myRgbaOutDevice;
            flipBack = true;
        }
        else if(throttled)
        {
            src = myRgbaOutDevice;
        }
        else
        {
            src = myRgbaInDevice;
        }
        if(flipBack)
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
    return 4;
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
    case 2:
        chan->name->setString("controlnet_on");
        chan->value = (myRunner && myRunner->combined_engine_mode()) ? 1.0f : 0.0f;
        break;
    case 3:
        chan->name->setString("controlnet_strength");
        chan->value = myLastControlnetStrength >= 0.0f ? myLastControlnetStrength : 1.0f;
        break;
    default:
        break;
    }
}

void LibreDiffusionTOP::getInfoPopupString(OP_String* info, void*)
{
    if(!info)
        return;
    char buf[512];
    char metricsLine[160];
    if(!myMeter.enabled())
    {
        std::snprintf(metricsLine, sizeof(metricsLine), "Inference metrics: disabled");
    }
    else
    {
        const double fps = myMeter.inference_fps();
        const double ms = myMeter.inference_ms();
        if(fps <= 0.0 || ms <= 0.0)
            std::snprintf(metricsLine, sizeof(metricsLine), "Inference: idle");
        else
            std::snprintf(metricsLine, sizeof(metricsLine),
                          "Inference: %.1f FPS\nPer-frame: %.2f ms", fps, ms);
    }

    const bool cnOn = myRunner && myRunner->combined_engine_mode();
    const char* engineName = cnOn ? "unet_controlnet.engine" : "unet.engine";
    const float strength = myLastControlnetStrength >= 0.0f
                               ? myLastControlnetStrength
                               : 1.0f;

    if(myLastErrorMessage.empty())
    {
        std::snprintf(buf, sizeof(buf),
                      "%s\n"
                      "Engine: %s\n"
                      "ControlNet: %s (strength=%.2f)",
                      metricsLine, engineName,
                      cnOn ? "ON" : "off", strength);
    }
    else
    {
        std::snprintf(buf, sizeof(buf),
                      "%s\n"
                      "Engine: %s\n"
                      "ControlNet: %s (strength=%.2f)\n"
                      "ERROR: %s",
                      metricsLine, engineName,
                      cnOn ? "ON" : "off", strength,
                      myLastErrorMessage.c_str());
    }
    info->setString(buf);
}

void LibreDiffusionTOP::tryInit(const std::string& folder, int width, int height,
                                int timestep, bool controlnet_on)
{
    myRunner.reset();
    myHasCachedOutput = false;

    librediff::Runner::Config cfg;
    cfg.clip_engine_path = join_path(folder, "clip.engine");
    cfg.unet_engine_path = join_path(folder, "unet.engine");
    cfg.vae_encoder_path = join_path(folder, "vae_encoder.engine");
    cfg.vae_decoder_path = join_path(folder, "vae_decoder.engine");
    if(controlnet_on && combined_engine_available(folder))
    {
        cfg.combined_unet_controlnet_engine_path
            = join_path(folder, "unet_controlnet.engine");
    }
    else if(controlnet_on)
    {
        TDDBG("plugin: ControlNet=On but no unet_controlnet.engine in "
              << folder << " -- falling back to plain UNet");
    }
    cfg.width = width;
    cfg.height = height;
    cfg.batch_size = 1;
    cfg.timestep_indices = {timestep};

    auto runner = std::make_unique<librediff::Runner>();
    std::string err;
    if(runner->init(cfg, &err))
    {
        TDDBG("plugin: runner initialized OK at " << width << "x" << height
              << " controlnet=" << (controlnet_on ? "ON" : "OFF"));
        myRunner = std::move(runner);
        myLastErrorMessage.clear();
    }
    else
    {
        TDDBG("plugin: runner init FAILED: " << err);
        myLastErrorMessage = "init failed: " + err;
    }
}

} // namespace librediff_td
