#pragma once

#include "TOP_CPlusPlusBase.h"
#include "fps_meter.hpp"
#include "librediffusion_runner.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace librediff_td
{

inline constexpr const char* kParEnginesFolder = "Enginesfolder";
inline constexpr const char* kParPositivePrompt = "Positiveprompt";
inline constexpr const char* kParNegativePrompt = "Negativeprompt";
inline constexpr const char* kParGuidance = "Guidance";
inline constexpr const char* kParTimestep = "Timestep";
inline constexpr const char* kParMode = "Mode";              // GPU=0, CPU=1
inline constexpr const char* kParTrackMetrics = "Trackmetrics";

class LibreDiffusionTOP : public TD::TOP_CPlusPlusBase
{
public:
    LibreDiffusionTOP(const TD::OP_NodeInfo*, TD::TOP_Context* context);
    ~LibreDiffusionTOP() override;

    void getGeneralInfo(TD::TOP_GeneralInfo*, const TD::OP_Inputs*, void*) override;
    void execute(TD::TOP_Output*, const TD::OP_Inputs*, void*) override;
    void setupParameters(TD::OP_ParameterManager*, void*) override;

    int32_t getNumInfoCHOPChans(void*) override;
    void getInfoCHOPChan(int32_t index, TD::OP_InfoCHOPChan*, void*) override;
    void getInfoPopupString(TD::OP_String*, void*) override;

private:
    // Defined in librediffusion_top_gpu.cpp / _cpu.cpp.
    bool runInferenceGpu(int width, int height, cudaStream_t stream);
    bool runInferenceCpu(int width, int height, size_t bytes, cudaStream_t stream);
    bool ensureHostBuffers(size_t bytes);

    void tryInit(const std::string& folder, int width, int height, int timestep);

    TD::TOP_Context* myContext;
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

    FpsMeter myMeter;
};

} // namespace librediff_td
