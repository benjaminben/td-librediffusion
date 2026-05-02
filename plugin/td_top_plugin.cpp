// TouchDesigner C++ TOP entry points. The class implementation lives in
// librediffusion_top.cpp; this file is just the C ABI exports TD calls.

#include "TOP_CPlusPlusBase.h"

#include "librediffusion_top.hpp"

using namespace TD;

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
    // Input #1 = source video. Input #2 (optional) = ControlNet control image,
    // consumed when the Controlnet parameter is On.
    customInfo.maxInputs = 2;
}

DLLEXPORT TOP_CPlusPlusBase* CreateTOPInstance(const OP_NodeInfo* info, TOP_Context* context)
{
    return new librediff_td::LibreDiffusionTOP(info, context);
}

DLLEXPORT void DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context*)
{
    delete static_cast<librediff_td::LibreDiffusionTOP*>(instance);
}

} // extern "C"
