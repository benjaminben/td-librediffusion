// TouchDesigner C++ TOP host for librediffusion.
//
// Step 4: GPU-path img2img using TD's CUDA executeMode. The TOP receives the
// input texture as a cudaArray*, copies it into a linear RGBA8 device buffer,
// hands that to LibreDiffusionRunner::process_gpu_rgba8 (which converts to
// NCHW half, runs img2img_gpu_half, and converts back), then copies the
// output back into TD's output cudaArray. No CPU roundtrip.
//
// Constraint: input texture must be at engine resolution (e.g. 512x512).
// Insert a Resolution TOP upstream if the source differs.

#include "TOP_CPlusPlusBase.h"

#include "librediffusion_runner.hpp"
#include "td_debug_log.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

using namespace TD;

namespace
{
constexpr const char* kParEnginesFolder = "Enginesfolder";
constexpr const char* kParPositivePrompt = "Positiveprompt";
constexpr const char* kParNegativePrompt = "Negativeprompt";
constexpr const char* kParGuidance = "Guidance";
constexpr const char* kParTimestep = "Timestep";
constexpr const char* kParMode = "Mode";  // GPU=0, CPU=1 (Step 3 path)

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

class LibreDiffusionTOP : public TOP_CPlusPlusBase
{
public:
    LibreDiffusionTOP(const OP_NodeInfo*, TOP_Context* context)
      : myContext{context}
    {
    }

    ~LibreDiffusionTOP() override
    {
        if(myRgbaInDevice)
            cudaFree(myRgbaInDevice);
        if(myRgbaOutDevice)
            cudaFree(myRgbaOutDevice);
        delete[] myHostRgbaIn;
        delete[] myHostRgbaOut;
    }

    void getGeneralInfo(TOP_GeneralInfo* ginfo, const OP_Inputs*, void*) override
    {
        ginfo->cookEveryFrameIfAsked = false;
        ginfo->inputSizeIndex = 0;
    }

    void execute(TOP_Output* output, const OP_Inputs* inputs, void*) override
    {
        const OP_TOPInput* top = inputs->getInputTOP(0);
        if(!top)
            return;

        const int width = top->textureDesc.width;
        const int height = top->textureDesc.height;

        // ---- read parameters
        const std::string folder = param_path(inputs, kParEnginesFolder);
        const std::string positive = param_string(inputs, kParPositivePrompt);
        const std::string negative = param_string(inputs, kParNegativePrompt);
        const int timestep = inputs->getParInt(kParTimestep);
        const float guidance = static_cast<float>(inputs->getParDouble(kParGuidance));
        const int mode = inputs->getParInt(kParMode);  // 0=GPU, 1=CPU
                                                       // MUST be read before beginCUDAOperations

        // ---- (re-)init pipeline if engines folder, dims, or timestep changed
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

        // ---- push prompts and guidance to the runner if they changed
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

        // Use the pipeline's stream for everything so the cudaArray copies and
        // the inference work are ordered without explicit synchronization.
        cudaStream_t stream = static_cast<cudaStream_t>(
            myRunner ? myRunner->cuda_stream() : nullptr);

        // 1. Acquire input/output OP_CUDAArrayInfos FIRST. cudaArray fields
        //    will be null at this point; beginCUDAOperations populates them.
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

        // 2. Now begin CUDA operations -- this fills in inArray->cudaArray
        //    and outArray->cudaArray with valid pointers.
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

            // Allocate (or reuse) linear RGBA8 staging buffers sized to the
            // current texture.
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

            // Input cudaArray -> linear device RGBA8.
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
                std::string runErr;
                if(mode == 1)
                {
                    // CPU path: device -> host -> librediffusion_img2img -> host -> device.
                    if(!ensureHostBuffers(bytes))
                    {
                        TDDBG("plugin: host buffer alloc failed");
                        break;
                    }
                    cudaMemcpyAsync(myHostRgbaIn, myRgbaInDevice, bytes,
                                    cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream); // need data in host before CPU call
                    if(!myRunner->process(myHostRgbaIn, myHostRgbaOut, width, height,
                                          &runErr))
                    {
                        TDDBG("plugin: process(cpu) failed: " << runErr);
                        break;
                    }
                    cudaMemcpyAsync(myRgbaOutDevice, myHostRgbaOut, bytes,
                                    cudaMemcpyHostToDevice, stream);
                }
                else
                {
                    if(!myRunner->process_gpu_rgba8(
                           static_cast<const uint8_t*>(myRgbaInDevice),
                           static_cast<uint8_t*>(myRgbaOutDevice), width, height, stream,
                           &runErr))
                    {
                        TDDBG("plugin: process_gpu_rgba8 failed: " << runErr);
                        break;
                    }
                }
                inferred = true;
            }

            // Linear RGBA8 (output if inferred, else input) -> output cudaArray.
            void* src = inferred ? myRgbaOutDevice : myRgbaInDevice;
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

    void setupParameters(OP_ParameterManager* manager, void*) override
    {
        {
            OP_StringParameter sp;
            sp.name = kParEnginesFolder;
            sp.label = "Engines Folder";
            sp.page = "LibreDiffusion";
            sp.defaultValue = "";
            manager->appendFolder(sp);
        }
        {
            OP_StringParameter sp;
            sp.name = kParPositivePrompt;
            sp.label = "Positive Prompt";
            sp.page = "LibreDiffusion";
            sp.defaultValue = "";
            manager->appendString(sp);
        }
        {
            OP_StringParameter sp;
            sp.name = kParNegativePrompt;
            sp.label = "Negative Prompt";
            sp.page = "LibreDiffusion";
            sp.defaultValue = "";
            manager->appendString(sp);
        }
        {
            OP_NumericParameter np;
            np.name = kParGuidance;
            np.label = "Guidance";
            np.page = "LibreDiffusion";
            np.defaultValues[0] = 1.2;
            np.minSliders[0] = 0.0;
            np.maxSliders[0] = 5.0;
            np.minValues[0] = 0.0;
            np.maxValues[0] = 20.0;
            np.clampMins[0] = true;
            np.clampMaxes[0] = true;
            manager->appendFloat(np);
        }
        {
            OP_NumericParameter np;
            np.name = kParTimestep;
            np.label = "Timestep Index";
            np.page = "LibreDiffusion";
            np.defaultValues[0] = 25;
            np.minSliders[0] = 0;
            np.maxSliders[0] = 49;
            np.minValues[0] = 0;
            np.maxValues[0] = 49;
            np.clampMins[0] = true;
            np.clampMaxes[0] = true;
            manager->appendInt(np);
        }
        {
            // GPU = native CUDA path (process_gpu_rgba8). Fast.
            // CPU = round-trip through host memory + librediffusion_img2img.
            //       Same behavior as Step 3, useful as a fallback.
            OP_StringParameter sp;
            sp.name = kParMode;
            sp.label = "Inference Mode";
            sp.page = "LibreDiffusion";
            sp.defaultValue = "GPU";
            const char* labels[] = {"GPU", "CPU"};
            const char* names[] = {"GPU", "CPU"};
            manager->appendMenu(sp, 2, names, labels);
        }
    }

private:
    bool ensureHostBuffers(size_t bytes)
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

    void tryInit(const std::string& folder, int width, int height, int timestep)
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

    TOP_Context* myContext;
    std::unique_ptr<librediff::Runner> myRunner;
    std::string myLastFolder;
    std::string myLastPositive;
    std::string myLastNegative;
    int myLastWidth = 0;
    int myLastHeight = 0;
    int myLastTimestep = -1;
    float myLastGuidance = -1.0f;

    void* myRgbaInDevice = nullptr;
    void* myRgbaOutDevice = nullptr;
    size_t myRgbaBytes = 0;
    uint8_t* myHostRgbaIn = nullptr;
    uint8_t* myHostRgbaOut = nullptr;
    size_t myHostBytes = 0;
    int myDiagCount = 0;
};
} // namespace

extern "C"
{

DLLEXPORT void FillTOPPluginInfo(TOP_PluginInfo* info)
{
    if(!info->setAPIVersion(TOPCPlusPlusAPIVersion))
        return;
    info->executeMode = TOP_ExecuteMode::CUDA;

    OP_CustomOPInfo& customInfo = info->customOPInfo;
    customInfo.opType->setString("Librediffusion");
    customInfo.opLabel->setString("LibreDiffusion");
    customInfo.authorName->setString("github.com/benjaminben");
    customInfo.authorEmail->setString("");

    customInfo.minInputs = 1;
    customInfo.maxInputs = 1;
}

DLLEXPORT TOP_CPlusPlusBase* CreateTOPInstance(const OP_NodeInfo* info, TOP_Context* context)
{
    return new LibreDiffusionTOP(info, context);
}

DLLEXPORT void DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context*)
{
    delete static_cast<LibreDiffusionTOP*>(instance);
}

} // extern "C"
