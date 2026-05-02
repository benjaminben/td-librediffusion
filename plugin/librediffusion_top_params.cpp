// Parameter setup for the LibreDiffusion TOP. Lives in its own translation
// unit so the main file only deals with cook-time logic.

#include "librediffusion_top.hpp"

using namespace TD;

namespace librediff_td
{

void LibreDiffusionTOP::setupParameters(OP_ParameterManager* manager, void*)
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
        OP_StringParameter sp;
        sp.name = kParMode;
        sp.label = "Inference Mode";
        sp.page = "LibreDiffusion";
        sp.defaultValue = "GPU";
        const char* labels[] = {"GPU", "CPU"};
        const char* names[] = {"GPU", "CPU"};
        manager->appendMenu(sp, 2, names, labels);
    }
    {
        OP_NumericParameter np;
        np.name = kParTrackMetrics;
        np.label = "Track Inference Metrics";
        np.page = "LibreDiffusion";
        np.defaultValues[0] = 1;
        manager->appendToggle(np);
    }
    {
        OP_NumericParameter np;
        np.name = kParMaxInferenceFps;
        np.label = "Max Inference FPS";
        np.page = "LibreDiffusion";
        np.defaultValues[0] = 0.0;  // 0 = unlimited
        np.minSliders[0] = 0.0;
        np.maxSliders[0] = 60.0;
        np.minValues[0] = 0.0;
        np.maxValues[0] = 240.0;
        np.clampMins[0] = true;
        np.clampMaxes[0] = true;
        manager->appendFloat(np);
    }
    {
        // ControlNet (v1). On switches the engines-folder convention from
        // unet.engine to unet_controlnet.engine and triggers a pipeline
        // re-init. Toggling Off reverts to the plain UNet engine.
        OP_StringParameter sp;
        sp.name = kParControlnet;
        sp.label = "ControlNet";
        sp.page = "LibreDiffusion";
        sp.defaultValue = "Off";
        const char* labels[] = {"Off", "On"};
        const char* names[] = {"Off", "On"};
        manager->appendMenu(sp, 2, names, labels);
    }
    {
        OP_NumericParameter np;
        np.name = kParControlnetStrength;
        np.label = "ControlNet Strength";
        np.page = "LibreDiffusion";
        np.defaultValues[0] = 1.0;
        np.minSliders[0] = 0.0;
        np.maxSliders[0] = 2.0;
        np.minValues[0] = 0.0;
        np.maxValues[0] = 2.0;
        np.clampMins[0] = true;
        np.clampMaxes[0] = true;
        manager->appendFloat(np);
    }
}

} // namespace librediff_td
