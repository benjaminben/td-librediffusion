// LibreDiffusionRunner implementation. Step 3 wires up CPU-path img2img:
// init() loads engines + clip, set_prompt() computes embeddings, and
// process() runs librediffusion_img2img.

#include "librediffusion_runner.hpp"
#include "dll_loader.hpp"
#include "td_debug_log.hpp"

#define LIBREDIFFUSION_USE_DLL
#include "librediffusion_c.h"

// sd-turbo.hpp's TIMESTEP_PARAMS literals are written as full double precision
// and assigned to float fields, which fires a wave of truncation warnings.
// Suppress just for this include; the values themselves are fine.
#ifdef _MSC_VER
  #pragma warning(push)
  #pragma warning(disable : 4305) // 'initializing': truncation from 'double' to 'float'
  #pragma warning(disable : 4244) // conversion from 'double' to 'float'
#endif
#include <schedulers/sd-turbo.hpp>
#ifdef _MSC_VER
  #pragma warning(pop)
#endif

#include <vector>

namespace librediff
{

namespace
{
constexpr int kTextSeqLen = 77;
constexpr int kTextHiddenDimSDTurbo = 1024;
constexpr int kClipPadTokenSDTurbo = 0;
} // namespace

struct Runner::Impl
{
    librediffusion_config_handle config = nullptr;
    librediffusion_pipeline_handle pipeline = nullptr;
    librediffusion_clip_handle clip = nullptr;
    bool initialized = false;
    bool prompt_ready = false;
    bool combined_engine = false;
    int width = 0;
    int height = 0;
    int batch_size = 0;
    std::vector<int> timestep_indices;

    // Reusable NCHW half device buffers for process_gpu_rgba8.
    // Allocated lazily on the first call at the configured engine size.
    void* nchw_half_in = nullptr;
    void* nchw_half_out = nullptr;
    size_t nchw_half_bytes = 0;

    ~Impl()
    {
        if(nchw_half_in)
            librediffusion_cuda_free(nchw_half_in);
        if(nchw_half_out)
            librediffusion_cuda_free(nchw_half_out);
        if(clip)
            librediffusion_clip_destroy(clip);
        if(pipeline)
            librediffusion_pipeline_destroy(pipeline);
        if(config)
            librediffusion_config_destroy(config);
    }
};

Runner::Runner() : impl_(new Impl()) {}
Runner::~Runner() { delete impl_; }

namespace
{
bool check(librediffusion_error_t err, const char* what, std::string* err_out)
{
    if(err == LIBREDIFFUSION_SUCCESS)
        return true;
    const char* msg = librediffusion_error_string(err);
    TDDBG("!! " << what << " failed: " << (msg ? msg : "(null)") << " (code " << err << ")");
    if(err_out)
        *err_out = std::string(what) + ": " + (msg ? msg : "unknown error");
    return false;
}

// Look up SD-Turbo scheduler params for each configured timestep index and
// hand them to librediffusion_prepare_scheduler as parallel arrays.
bool prepare_sd_turbo_scheduler(
    librediffusion_pipeline_handle pipeline, const std::vector<int>& indices,
    std::string* err_out)
{
    using namespace streamdiffusion::SCHEDULER_STABILITYAI_SD_TURBO;

    std::vector<float> timesteps, alpha, beta, c_skip, c_out;
    timesteps.reserve(indices.size());
    alpha.reserve(indices.size());
    beta.reserve(indices.size());
    c_skip.reserve(indices.size());
    c_out.reserve(indices.size());

    for(int idx : indices)
    {
        if(idx < 0 || idx >= NUM_TIMESTEPS)
        {
            TDDBG("!! timestep index " << idx << " out of range [0," << NUM_TIMESTEPS << ")");
            if(err_out)
                *err_out = "timestep index out of range";
            return false;
        }
        const auto& p = TIMESTEP_PARAMS[idx];
        timesteps.push_back(static_cast<float>(TIMESTEP_VALUES[idx]));
        alpha.push_back(p.alpha_prod_t_sqrt);
        beta.push_back(p.beta_prod_t_sqrt);
        c_skip.push_back(p.c_skip);
        c_out.push_back(p.c_out);
    }

    return check(
        librediffusion_prepare_scheduler(
            pipeline, timesteps.data(), alpha.data(), beta.data(), c_skip.data(),
            c_out.data(), indices.size()),
        "prepare_scheduler", err_out);
}
} // namespace

bool Runner::init(const Config& cfg, std::string* err_out)
{
    if(impl_->initialized)
    {
        TDDBG("init: already initialized");
        return true;
    }

    if(!ensure_libraries_loaded(err_out))
        return false;

    const bool use_combined = !cfg.combined_unet_controlnet_engine_path.empty();
    TDDBG("init: " << cfg.width << "x" << cfg.height
                   << " batch=" << cfg.batch_size
                   << " timesteps=" << cfg.timestep_indices.size()
                   << " controlnet=" << (use_combined ? "ON" : "OFF"));
    TDDBG("  clip=" << cfg.clip_engine_path);
    if(use_combined)
        TDDBG("  unet+cn=" << cfg.combined_unet_controlnet_engine_path);
    else
        TDDBG("  unet=" << cfg.unet_engine_path);
    TDDBG("  vae_enc=" << cfg.vae_encoder_path);
    TDDBG("  vae_dec=" << cfg.vae_decoder_path);

    librediffusion_config_handle c = nullptr;
    if(!check(librediffusion_config_create(&c), "config_create", err_out))
        return false;
    impl_->config = c;

    if(!check(librediffusion_config_set_device(c, 0),
              "set_device", err_out)) return false;
    if(!check(librediffusion_config_set_model_type(c, MODEL_SD_TURBO),
              "set_model_type", err_out)) return false;
    if(!check(librediffusion_config_set_pipeline_mode(c, MODE_SINGLE_FRAME),
              "set_pipeline_mode", err_out)) return false;
    if(!check(librediffusion_config_set_dimensions(
                  c, cfg.width, cfg.height, cfg.width / 8, cfg.height / 8),
              "set_dimensions", err_out)) return false;
    if(!check(librediffusion_config_set_batch_size(c, cfg.batch_size),
              "set_batch_size", err_out)) return false;
    if(!check(librediffusion_config_set_denoising_steps(
                  c, static_cast<int>(cfg.timestep_indices.size())),
              "set_denoising_steps", err_out)) return false;
    if(!check(librediffusion_config_set_frame_buffer_size(c, 1),
              "set_frame_buffer_size", err_out)) return false;
    // guidance_scale + cfg_type left at librediffusion defaults (1.2 / SELF)
    // which is what Score uses; SD_CFG_NONE disables prompt influence.
    if(!check(librediffusion_config_set_text_config(
                  c, kTextSeqLen, kTextHiddenDimSDTurbo, kClipPadTokenSDTurbo),
              "set_text_config", err_out)) return false;
    if(use_combined)
    {
        if(!check(librediffusion_config_set_combined_unet_controlnet_engine(
                      c, cfg.combined_unet_controlnet_engine_path.c_str()),
                  "set_combined_unet_controlnet_engine", err_out)) return false;
    }
    else
    {
        if(!check(librediffusion_config_set_unet_engine(c, cfg.unet_engine_path.c_str()),
                  "set_unet_engine", err_out)) return false;
    }
    if(!check(librediffusion_config_set_vae_encoder(c, cfg.vae_encoder_path.c_str()),
              "set_vae_encoder", err_out)) return false;
    if(!check(librediffusion_config_set_vae_decoder(c, cfg.vae_decoder_path.c_str()),
              "set_vae_decoder", err_out)) return false;
    if(!check(librediffusion_config_set_timestep_indices(
                  c, cfg.timestep_indices.data(), cfg.timestep_indices.size()),
              "set_timestep_indices", err_out)) return false;

    librediffusion_pipeline_handle p = nullptr;
    if(!check(librediffusion_pipeline_create(c, &p), "pipeline_create", err_out))
        return false;
    impl_->pipeline = p;

    TDDBG("calling pipeline_init_all");
    if(!check(librediffusion_pipeline_init_all(p), "pipeline_init_all", err_out))
        return false;

    librediffusion_clip_handle clip = nullptr;
    if(!check(librediffusion_clip_create(cfg.clip_engine_path.c_str(), &clip),
              "clip_create", err_out))
        return false;
    impl_->clip = clip;

    if(!prepare_sd_turbo_scheduler(p, cfg.timestep_indices, err_out))
        return false;
    TDDBG("scheduler prepared for indices " << cfg.timestep_indices.size() << " step(s)");

    // CRITICAL: reseed populates init_noise_ with random normal values AND
    // zeroes stock_noise_. Without this, stock_noise_ is uninitialized GPU
    // memory and the SELF CFG path (model_pred = uncond + g*(text - uncond))
    // has garbage in `uncond`, drowning out the prompt's contribution.
    if(!check(librediffusion_reseed(p, 42), "reseed", err_out))
        return false;
    TDDBG("pipeline seeded (init_noise + zeroed stock_noise)");

    impl_->width = cfg.width;
    impl_->height = cfg.height;
    impl_->batch_size = cfg.batch_size;
    impl_->timestep_indices = cfg.timestep_indices;
    impl_->combined_engine = use_combined;
    impl_->initialized = true;
    TDDBG("init: complete");
    return true;
}

bool Runner::is_initialized() const { return impl_->initialized; }

bool Runner::set_prompt(
    const std::string& positive, const std::string& negative, std::string* err_out)
{
    if(!impl_->initialized)
    {
        if(err_out)
            *err_out = "set_prompt: runner not initialized";
        return false;
    }

    TDDBG("set_prompt: positive=\"" << positive << "\" negative=\"" << negative << "\"");

    // Run CLIP on the pipeline's stream so prepare_embeds' d2d copy is
    // ordered after CLIP writes its output. Passing nullptr / default stream
    // here lets prepare_embeds read pre-CLIP memory (zero) and the prompt
    // never reaches the UNet.
    librediffusion_stream_t stream = librediffusion_pipeline_get_stream(impl_->pipeline);

    librediffusion_half_t* pos_embeds = nullptr;
    if(!check(librediffusion_clip_compute_embeddings(
                  impl_->clip, positive.c_str(), kClipPadTokenSDTurbo, stream,
                  &pos_embeds),
              "clip_compute_embeddings(positive)", err_out))
        return false;
    bool ok = check(
        librediffusion_prepare_embeds(
            impl_->pipeline, pos_embeds, kTextSeqLen, kTextHiddenDimSDTurbo),
        "prepare_embeds", err_out);
    librediffusion_cuda_free(pos_embeds);
    if(!ok)
        return false;

    librediffusion_half_t* neg_embeds = nullptr;
    if(!check(librediffusion_clip_compute_embeddings(
                  impl_->clip, negative.c_str(), kClipPadTokenSDTurbo, stream,
                  &neg_embeds),
              "clip_compute_embeddings(negative)", err_out))
        return false;
    ok = check(
        librediffusion_prepare_negative_embeds(
            impl_->pipeline, neg_embeds, kTextSeqLen, kTextHiddenDimSDTurbo),
        "prepare_negative_embeds", err_out);
    librediffusion_cuda_free(neg_embeds);
    if(!ok)
        return false;

    impl_->prompt_ready = true;
    return true;
}

void Runner::set_guidance(float scale)
{
    if(impl_->pipeline)
        librediffusion_set_guidance_scale(impl_->pipeline, scale);
}

void Runner::set_seed(uint32_t seed)
{
    if(impl_->pipeline)
        librediffusion_reseed(impl_->pipeline, static_cast<int64_t>(seed));
}

bool Runner::process(
    const uint8_t* rgba_in, uint8_t* rgba_out, int width, int height,
    std::string* err_out)
{
    if(!impl_->initialized || !impl_->prompt_ready)
    {
        if(err_out)
            *err_out = "process: runner not ready (init + set_prompt required)";
        return false;
    }
    if(width != impl_->width || height != impl_->height)
    {
        TDDBG("!! process: dim mismatch input=" << width << "x" << height
                                                << " engine=" << impl_->width << "x"
                                                << impl_->height);
        if(err_out)
            *err_out = "process: input dimensions don't match engine";
        return false;
    }
    return check(
        librediffusion_img2img(impl_->pipeline, rgba_in, rgba_out, width, height),
        "img2img", err_out);
}

runner_cuda_stream_t Runner::cuda_stream() const
{
    if(!impl_->pipeline)
        return nullptr;
    return static_cast<runner_cuda_stream_t>(
        librediffusion_pipeline_get_stream(impl_->pipeline));
}

bool Runner::process_gpu_rgba8(
    const uint8_t* rgba_in_device, uint8_t* rgba_out_device, int width, int height,
    runner_cuda_stream_t stream, std::string* err_out)
{
    if(!impl_->initialized || !impl_->prompt_ready)
    {
        if(err_out)
            *err_out = "process_gpu_rgba8: not ready (init + set_prompt required)";
        return false;
    }
    if(width != impl_->width || height != impl_->height)
    {
        TDDBG("!! process_gpu_rgba8: dim mismatch input=" << width << "x" << height
              << " engine=" << impl_->width << "x" << impl_->height);
        if(err_out)
            *err_out = "process_gpu_rgba8: input dimensions don't match engine";
        return false;
    }

    // 1*3*H*W halves for the NCHW staging buffers.
    const size_t needed_bytes
        = static_cast<size_t>(impl_->batch_size) * 3 * static_cast<size_t>(height)
          * static_cast<size_t>(width) * sizeof(librediffusion_half_t);
    if(needed_bytes != impl_->nchw_half_bytes)
    {
        if(impl_->nchw_half_in)
            librediffusion_cuda_free(impl_->nchw_half_in);
        if(impl_->nchw_half_out)
            librediffusion_cuda_free(impl_->nchw_half_out);
        impl_->nchw_half_in = librediffusion_cuda_malloc(needed_bytes);
        impl_->nchw_half_out = librediffusion_cuda_malloc(needed_bytes);
        if(!impl_->nchw_half_in || !impl_->nchw_half_out)
        {
            impl_->nchw_half_bytes = 0;
            if(err_out)
                *err_out = "process_gpu_rgba8: cuda_malloc for NCHW staging failed";
            return false;
        }
        impl_->nchw_half_bytes = needed_bytes;
    }

    auto* nchw_in = static_cast<librediffusion_half_t*>(impl_->nchw_half_in);
    auto* nchw_out = static_cast<librediffusion_half_t*>(impl_->nchw_half_out);

    if(!check(
           librediffusion_rgba_nhwc_to_nchw_half(
               impl_->pipeline, rgba_in_device, nchw_in, width, height, stream),
           "rgba_nhwc_to_nchw_half", err_out))
        return false;

    if(!check(
           librediffusion_img2img_gpu_half(impl_->pipeline, nchw_in, nchw_out, stream),
           "img2img_gpu_half", err_out))
        return false;

    if(!check(
           librediffusion_nchw_half_to_rgba_nhwc(
               impl_->pipeline, nchw_out, rgba_out_device, width, height, stream),
           "nchw_half_to_rgba_nhwc", err_out))
        return false;

    return true;
}

bool Runner::set_control_image_gpu(
    const uint8_t* device_rgba, int width, int height,
    runner_cuda_stream_t stream, std::string* err_out)
{
    if(!impl_->initialized)
    {
        if(err_out)
            *err_out = "set_control_image_gpu: runner not initialized";
        return false;
    }
    if(!impl_->combined_engine)
        return true; // no-op when ControlNet is disabled
    if(width != impl_->width || height != impl_->height)
    {
        TDDBG("!! set_control_image_gpu: dim mismatch input=" << width << "x" << height
              << " engine=" << impl_->width << "x" << impl_->height);
        if(err_out)
            *err_out = "set_control_image_gpu: input dimensions don't match engine";
        return false;
    }
    return check(
        librediffusion_set_control_image_gpu(
            impl_->pipeline, device_rgba, width, height, stream),
        "set_control_image_gpu", err_out);
}

void Runner::set_controlnet_strength(float strength)
{
    if(!impl_->pipeline || !impl_->combined_engine)
        return;
    librediffusion_set_controlnet_strength(impl_->pipeline, strength);
}

bool Runner::combined_engine_mode() const
{
    return impl_->combined_engine;
}

} // namespace librediff
