// SPDX-FileCopyrightText: Copyright 2026 IFreemz
// SPDX-License-Identifier: GPL-2.0-or-later
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "common/path_util.h"
#include "input/bb_autopilot.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/host_shaders/bb_dlss_composite_comp.h"
#include "video_core/host_shaders/bb_dlss_composite_hdr_comp.h"
#include "video_core/host_shaders/bb_dlss_linearize_comp.h"
#include "video_core/host_shaders/bb_dlss_mirror_r32f_comp.h"
#include "video_core/host_shaders/bb_dlss_mirror_rg16f_comp.h"
#include "video_core/host_shaders/bb_dlss_mirror_rgba16f_comp.h"
#include "video_core/host_shaders/bb_dlss_motion_comp.h"
#include "video_core/host_shaders/bb_dlss_sharpen_hdr_comp.h"
#include "video_core/host_shaders/bb_flicker_meter_comp.h"
#include "video_core/host_shaders/bb_hudless_dim_comp.h"
#include "video_core/renderer_vulkan/vk_bb_resolution.h"
#include "video_core/renderer_vulkan/vk_bb_temporal_dlss.h"
#include "video_core/renderer_vulkan/vk_bb_velocity_mirror.h"
#include "video_core/renderer_vulkan/vk_dlss_ngx.h"
#include "video_core/renderer_vulkan/vk_fsr4_addon.h"
#include "video_core/renderer_vulkan/vk_fsr_upscaler.h"
#include "video_core/renderer_vulkan/vk_hdr_mod.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/texture_cache/texture_cache.h"

namespace Vulkan {

bool BbTemporalDlssRequested() {
    return DlssNgx::Present() || FsrUpscaler::Present() || Fsr4Addon::Present();
}

std::filesystem::path BbDlssSettingsPath() {
    return Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "dlss.ini";
}

namespace {
std::mutex status_mutex;
BbDlssStatus status;
std::string ngx_problem;
std::chrono::steady_clock::time_point last_evaluation;

void ReportFrame(std::string_view reason) {
    std::scoped_lock lock{status_mutex};
    status.reason = reason;
}

std::string DlssProblem() {
    std::scoped_lock lock{status_mutex};
    return ngx_problem;
}
} // namespace

BbDlssStatus BbTemporalDlssStatus() {
    std::scoped_lock lock{status_mutex};
    auto copy = status;
    copy.active = std::chrono::steady_clock::now() - last_evaluation < std::chrono::seconds{1};
    copy.dlss_problem = ngx_problem;
    copy.fsr4_installed = Fsr4Addon::Present();
    if (!BbTemporalDlssRequested())
        copy.reason = "No upscaler files were found next to shadPS4.exe.";
    else if (!copy.active && copy.reason.empty())
        copy.reason = "Waiting for gameplay (menus and loading screens use the normal image).";
    return copy;
}

void BbTemporalDlssReportGpu(std::string_view gpu) {
    std::scoped_lock lock{status_mutex};
    status.gpu = gpu;
}

void BbTemporalDlssReportProblem(std::string_view problem) {
    std::scoped_lock lock{status_mutex};
    ngx_problem = problem;
}

namespace {
constexpr u64 DepthProducer = 0xd3c8bb21;  // samples R32 scene depth at slot 0
constexpr u64 DisplayCopy = 0x38d65b32;    // UI target -> VideoOut, LUT at slot 1
constexpr u64 OpaqueVelocity = 0x34bc187c; // packing scale at binding 0
constexpr u64 AlphaVelocityA = 0x749e4f9e; // packing scale at binding 1
constexpr u64 AlphaVelocityB = 0xb25e4fae;
// Motion blur: two full-screen passes, each blurring the HDR scene (slot 0) along the half-size
// motion buffer (slot 1) that the DepthProducer pass builds. Skipping them keeps that data.
constexpr u64 MotionBlur = 0xe0305cef;
// Copies the HDR scene (slot 0) into the target post-processing runs on. With the Decoupled UI
// patches the scene is smaller than that target and this pass is the game's upscale.
constexpr u64 HdrSceneCopy = 0xccbf44a6;
constexpr u64 CameraBytes = 4096, VelocityOffset = 4096, CoefficientBytes = 4096 + 256;
// HUD coverage counters, one per frame in flight, each in its own aligned slot.
constexpr u64 CoverageSlots = 3, CoverageStride = 256;
constexpr u64 DimmingBytes = 2 * 64 * sizeof(u32); // HUD dimming histogram, GPU only
constexpr vk::ImageSubresourceRange Range{
    .aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1};

bool IsUiShader(u64 hash) {
    return hash == 0x34e8a281 || hash == 0x81d336ce || hash == 0x09957251 || hash == 0x24042a9b ||
           hash == 0xa400228b;
}

template <typename T>
T Make(vk::ResultValue<T> result, const char* what) {
    if (result.result != vk::Result::eSuccess)
        throw std::runtime_error(std::string{what} + ": " + vk::to_string(result.result));
    return std::move(result.value);
}

float Halton(u32 index, u32 base) {
    float f = 1.f, r = 0.f;
    while (index > 0) {
        f /= static_cast<float>(base);
        r += f * static_cast<float>(index % base);
        index /= base;
    }
    return r;
}

struct OwnedImage {
    VideoCore::UniqueImage image;
    vk::UniqueImageView view;
    vk::UniqueImageView rgb_view, bgr_view; // opaque presentation views
    vk::Format format;
    vk::Extent2D extent;
    vk::ImageLayout layout{vk::ImageLayout::eUndefined};
    bool fresh{};
    bool swapped{}; // red and blue swapped, as the game's display copy writes them
    std::optional<FrameGen::FrameInputs> frame_gen; // with a presentable output
    // FSR frame generation: the same frame without the game's HUD, fresh along with it.
    std::unique_ptr<OwnedImage> hudless;
    bool hudless_fresh{};

    // Decoupled UI HUD-less frames: a view in the presenter's format for the game's frame.
    vk::Device device;
    vk::UniqueImageView frame_view;
    vk::Format frame_view_format{};

    OwnedImage(const Instance& instance, vk::Format format_, vk::Extent2D extent_,
               vk::ImageUsageFlags usage, bool presentable = false, vk::ImageCreateFlags flags = {})
        : image{instance.GetDevice(), instance.GetAllocator()}, format{format_}, extent{extent_},
          device{instance.GetDevice()} {
        image.Create(vk::ImageCreateInfo{.flags = flags,
                                         .imageType = vk::ImageType::e2D,
                                         .format = format,
                                         .extent = {extent.width, extent.height, 1},
                                         .mipLevels = 1,
                                         .arrayLayers = 1,
                                         .samples = vk::SampleCountFlagBits::e1,
                                         .tiling = vk::ImageTiling::eOptimal,
                                         .usage = usage});
        const auto device = instance.GetDevice();
        const auto make_view = [&](vk::ComponentMapping mapping) {
            return Make(device.createImageViewUnique(
                            vk::ImageViewCreateInfo{.image = image.image,
                                                    .viewType = vk::ImageViewType::e2D,
                                                    .format = format,
                                                    .components = mapping,
                                                    .subresourceRange = Range}),
                        "temporal DLSS image view");
        };
        view = make_view({});
        if (presentable) {
            using S = vk::ComponentSwizzle;
            rgb_view = make_view({S::eR, S::eG, S::eB, S::eOne});
            bgr_view = make_view({S::eB, S::eG, S::eR, S::eOne});
        }
    }
    // Opaque view as `view_format`, like the presenter's view of the game's own frame (the
    // image needs eMutableFormat).
    vk::ImageView FrameView(vk::Format view_format) {
        if (!frame_view || frame_view_format != view_format) {
            using S = vk::ComponentSwizzle;
            frame_view = Make(device.createImageViewUnique(vk::ImageViewCreateInfo{
                                  .image = image.image,
                                  .viewType = vk::ImageViewType::e2D,
                                  .format = view_format,
                                  .components = {S::eIdentity, S::eIdentity, S::eIdentity, S::eOne},
                                  .subresourceRange = Range}),
                              "temporal DLSS frame view");
            frame_view_format = view_format;
        }
        return *frame_view;
    }
    DlssNgx::Resource Resource() const {
        return {image.image, *view, Range, format, extent};
    }
    void Transit(vk::CommandBuffer command, vk::ImageLayout next, vk::PipelineStageFlags2 stage,
                 vk::AccessFlags2 access) {
        const vk::ImageMemoryBarrier2 barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask =
                layout == vk::ImageLayout::eUndefined
                    ? vk::AccessFlagBits2::eNone
                    : vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = stage,
            .dstAccessMask = access,
            .oldLayout = layout,
            .newLayout = next,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image.image,
            .subresourceRange = Range};
        command.pipelineBarrier2(
            vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
        layout = next;
    }
};

struct ComputePass {
    vk::UniqueDescriptorSetLayout descriptors;
    vk::UniquePipelineLayout layout;
    vk::UniquePipeline pipeline;

    ComputePass(vk::Device device, std::span<const vk::DescriptorType> types,
                std::span<const u32> code, u32 push_bytes) {
        std::vector<vk::DescriptorSetLayoutBinding> bindings;
        for (u32 i = 0; i < types.size(); ++i)
            bindings.push_back({.binding = i,
                                .descriptorType = types[i],
                                .descriptorCount = 1,
                                .stageFlags = vk::ShaderStageFlagBits::eCompute});
        descriptors = Make(device.createDescriptorSetLayoutUnique(
                               {.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptor,
                                .bindingCount = static_cast<u32>(bindings.size()),
                                .pBindings = bindings.data()}),
                           "temporal DLSS descriptor layout");
        const vk::PushConstantRange push{
            .stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = push_bytes};
        layout = Make(device.createPipelineLayoutUnique({.setLayoutCount = 1,
                                                         .pSetLayouts = &descriptors.get(),
                                                         .pushConstantRangeCount = 1,
                                                         .pPushConstantRanges = &push}),
                      "temporal DLSS pipeline layout");
        const auto module = CompileSPV(code, device);
        if (!module)
            throw std::runtime_error("temporal DLSS shader unavailable");
        auto result = device.createComputePipelineUnique(
            {}, vk::ComputePipelineCreateInfo{.stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                                                        .module = module,
                                                        .pName = "main"},
                                              .layout = *layout});
        device.destroyShaderModule(module);
        pipeline = Make(std::move(result), "temporal DLSS pipeline");
    }
};

constexpr char DefaultSettings[] =
    R"(# Upscaler settings for shadPS4. Press F1 in game to change them, or edit this file;
# saved changes apply within a second.

# 1 = upscaling on, 0 = the game's normal image
enabled=1

# auto = DLSS on NVIDIA RTX cards, else FSR 4 when its add-on is installed, else FSR 3.1;
# or dlss, fsr4, fsr
upscaler=auto

# DLSS model: 13 = M, 11 = K, 10 = J, 12 = L, 0 = NVIDIA default
preset=13

# Sharpening after upscaling, 0 (off) to 1
sharpness=0.6

# The game's motion blur: 0 = off (sharper in motion), 1 = on
motion_blur=0

# Full-screen menus (inventory, pause) show the game's own image instead of the upscaled one,
# so the scene does not shimmer behind them: 1 = on, 0 = off
menu_fix=1

# -1 = automatic, 0 = DLAA, 1 = Quality, 2 = Balanced, 3 = Performance, 4 = Ultra Performance
quality=-1

# Output resolution, e.g. output=3840x2160. Empty = the shadPS4 window size.
output=
)";

struct Tune {
    bool enabled = true;
    bool jitter = true;
    float sign_x = 1.f, sign_y = 1.f;
    bool mirror = true;
    bool hud = true;
    bool debug_motion = false;
    bool frame_gen = false; // DLSS frame generation (Streamline loads at launch)
    bool fps_counter = false;
    u32 fg_multiplier = 2; // frames shown per rendered frame with frame generation
    bool fg_reflex = true, fg_reflex_sleep = false; // hidden: Reflex mode and sleep
    bool fsr4_linear = true; // FSR 4 gets linear-light colour instead of sRGB-encoded
    bool camera_snap = true;
    // Camera motion from the game's scene constants instead of its motion-blur matrix:
    // 0 off, 1 the first scene constants of each frame (as first written), 2 the cameras whose
    // reprojection matches the motion-blur matrix. A planar reflection pass can render first
    // with the same constants layout, and its camera moves as a mirror image of the main one.
    int scene_camera = 2;
    float sharpness = 0.6f;
    bool swap = false;
    bool depth_inverted = false;
    u32 preset = 13;       // M: user-tuned default
    vk::Extent2D output{}; // zero: the window size
    int quality = -1;      // -1: from scale
    int upscaler = 0;      // 0 automatic, 1 DLSS, 2 FSR 3.1, 3 FSR 4
    bool motion_blur = false;
    bool menu_fix = true;
    bool auto_exposure = true; // HDR input (decoupled UI): FSR computes the exposure
    // Diagnostic: changing dump to a new non-zero number writes the upscaler's inputs (and
    // with dump_output=1 its output) for the next dump_frames frames into shadPS4\dlss_dump.
    int dump = 0;
    u32 dump_frames = 10;
    bool dump_output = true;
    // Diagnostic experiments, all live:
    //   exp_mirror: 1 mirror the upscaler's inputs left-right (and its output back), 2 top-bottom
    //   exp_reset:  restart the upscaler's history every frame
    //   exp_mv_x/y: multiply the motion vectors
    //   meter:      log the flicker meter (flicker_meter.csv, flicker_grid.csv)
    int exp_mirror = 0;
    bool exp_reset = false;
    float exp_mv_x = 1.f, exp_mv_y = 1.f;
    bool meter = false;
};

enum class Backend { None, Dlss, Fsr, Fsr4 };

const char* BackendName(Backend backend) {
    return backend == Backend::Dlss ? "DLSS" : backend == Backend::Fsr4 ? "FSR 4" : "FSR 3.1";
}
} // namespace

struct BbTemporalDlss::Impl {
    bool requested{}, failed{}, stopped{};
    Tune tune;
    std::filesystem::path tune_path;
    std::filesystem::file_time_type tune_time{};

    // Per-frame (display copy to display copy) state.
    // Jitter frames run from one DLSS evaluation to the next: the game records part of the next
    // frame's geometry before the display copy of the previous one.
    bool ui_phase{}, dlss_ready{}, dlss_pre_hud{}, history_valid{}, depth_learned{};
    bool evaluated_since_copy{};
    std::optional<VideoCore::ImageId> ui_target;
    u32 jittered_before_copy{};
    bool camera_constants{}, velocity_constants{};
    u32 jittered_draws{};
    u64 frame{};
    std::array<float, 2> jitter{};
    std::chrono::steady_clock::time_point last_frame{};
    float frame_ms{1000.f / 60.f};
    struct Tracked {
        VideoCore::ImageId id;
        u64 uid;
    };
    std::optional<Tracked> scene_depth, r32_depth;
    struct Bound {
        VideoCore::ImageId id;
        VideoCore::ImageViewInfo view;
    };
    std::optional<Bound> copy_source, copy_lut, blur_source, hdr_copy_source;
    std::optional<DrawReplacement> replacement;
    std::optional<DisplayCopyReplay> display_copy_replay;
    OwnedImage* replay_output{};
    std::unique_ptr<OwnedImage> pre_lut;         // RenoDX: the composite before the display LUT
    std::unique_ptr<OwnedImage> pre_lut_hudless; // and the same without the HUD
    OwnedImage* replay_hudless{};
    // Decoupled UI + FSR frame generation: the game's UI target just before its first HUD draw,
    // and the game's display copy of it per VideoOut buffer (the frame without the HUD).
    std::optional<VideoCore::ImageId> last_copy_source;
    vk::Format last_copy_format{};
    std::unique_ptr<OwnedImage> hud_snapshot, hud_dimmed;
    bool hud_snapshot_taken{};
    std::unordered_map<VAddr, std::unique_ptr<OwnedImage>> decoupled_hudless;
    // Decoupled UI: the render-size HDR scene image, whose depth attachment is the scene depth.
    std::optional<VideoCore::ImageId> decoupled_scene;
    bool decoupled{}, decoupled_frame{};
    // A Decoupled UI patch the upscaler cannot handle (or that failed): the game scales itself.
    bool decoupled_unsupported{};
    u64 blur_skips{};
    vk::Extent2D render{}, output{};
    u64 evaluations{}, composites{}, fallbacks{};

    std::unique_ptr<VideoCore::Buffer> coefficients;
    // Scene constants (see bbport docs/upscaler.md): far 3000 at [0], render size at [4..5],
    // view 3x4 at [8..19], projection x/y/z/offset at [52, 57, 62, 63], inverse view at [180..191].
    struct Camera {
        std::array<double, 12> view{}, inv_view{};
        std::array<float, 4> proj{};
        bool valid{};
    } camera, previous_camera;
    bool frame_has_camera{}, camera_this_frame{};
    u64 scene_camera_frames{};
    // Every distinct camera of this frame and of the previous one; SelectCamera picks the pair.
    std::array<Camera, 4> candidates{}, previous_candidates{};
    u32 candidate_count{}, previous_candidate_count{};
    // The motion-blur matrix (previous clip -> current clip) and the depth remap it expects,
    // as the CPU saw them at the depth producer: c[0x380..0x38f], c[0], c[1], c[0x26c], c[0x26d].
    std::array<float, 16> blur_rows{};
    std::array<float, 4> blur_depth{};
    bool blur_rows_valid{};
    bool camera_trusted{};
    int chosen_candidate = -1;
    float camera_match_px = -1.f;
    u64 camera_rejects{}, camera_switches{}, camera_unchecked{};
    std::unique_ptr<OwnedImage> snapshot, motion, upscaled;
    std::unique_ptr<OwnedImage> linear_color; // FSR 4 input
    std::unique_ptr<OwnedImage> sharpened;    // decoupled UI: the upscaled scene, sharpened
    // Frame generation reads depth and motion when the frame is presented, after later frames
    // may have overwritten them: each upscaled frame copies them into the next slot.
    struct FrameGenSlot {
        std::unique_ptr<OwnedImage> depth, motion;
    };
    std::array<std::shared_ptr<FrameGenSlot>, 4> frame_gen_slots;
    u32 frame_gen_next{};
    std::optional<FrameGen::FrameInputs> frame_gen_pending;
    // Decoupled UI: frame generation inputs for the game's own frame in each VideoOut buffer.
    std::unordered_map<VAddr, FrameGen::FrameInputs> decoupled_frame_gen;
    bool upscaled_linear{};
    bool upscaled_tone_mapped{}; // RenoDX: reversible tone map, values above 1 kept
    u32 producer_constants{}, producer_depths{}, velocity_draw_constants{};
    std::unordered_map<VAddr, std::unique_ptr<OwnedImage>> outputs;
    std::unique_ptr<ComputePass> mirror_rgba_pass, mirror_r32_pass, mirror_rg16_pass, meter_pass;
    std::unique_ptr<ComputePass> motion_pass, composite_pass, composite_hdr_pass, linearize_pass,
        sharpen_pass, hudless_dim_pass;
    vk::UniqueSampler nearest, linear;
    Backend backend{};
    // Menus cover most of the screen; while one is open the scene is not jittered and the game's
    // own frame is shown, so the render-size scene cannot shimmer through the menu panels.
    std::unique_ptr<VideoCore::Buffer> coverage, dimming;
    bool menu{};
    std::unique_ptr<FsrUpscaler> fsr;
    std::unique_ptr<Fsr4Addon> fsr4;
    std::optional<Fsr4Addon::ContextDesc> fsr4_rejected_size;
    bool fsr_tried{}, fsr4_tried{}, camera_logged{};
    // Diagnostic dump state (see Tune::dump) and the motion pass constants of this frame.
    bool tune_loaded{};
    u32 dump_left{}, dump_index{};
    std::optional<std::string> dump_composite; // file stem for this frame's composite dump
    std::array<float, 22> motion_constants{};
    u32 motion_flags{};

    Impl() {
        requested = BbTemporalDlssRequested();
        if (!requested)
            return;
        tune_path = BbDlssSettingsPath();
        std::error_code error;
        if (!std::filesystem::exists(tune_path, error))
            std::ofstream{tune_path} << DefaultSettings;
        PollTune();
        LOG_INFO(Render_Vulkan, "[DLSS] Settings file {}", tune_path.string());
    }

    void PollTune() {
        std::error_code error;
        const auto time = std::filesystem::last_write_time(tune_path, error);
        if (error || time == tune_time)
            return;
        tune_time = time;
        std::ifstream file{tune_path};
        std::string line;
        auto next = tune;
        while (std::getline(file, line)) {
            const auto eq = line.find('=');
            if (line.empty() || line[0] == '#' || line[0] == ';' || eq == std::string::npos)
                continue;
            const auto key = line.substr(0, eq);
            const auto value = line.substr(eq + 1);
            const int number = std::atoi(value.c_str());
            if (key == "enabled")
                next.enabled = number != 0;
            else if (key == "jitter")
                next.jitter = number != 0;
            else if (key == "jitter_sign_x") // multiplier sent to DLSS (sign and scale)
                next.sign_x = std::clamp(float(std::atof(value.c_str())), -4.f, 4.f);
            else if (key == "jitter_sign_y")
                next.sign_y = std::clamp(float(std::atof(value.c_str())), -4.f, 4.f);
            else if (key == "depth_inverted")
                next.depth_inverted = number != 0;
            else if (key == "preset")
                next.preset = u32(std::clamp(number, 0, 15));
            else if (key == "jitter_swap")
                next.swap = number != 0;
            else if (key == "object_motion")
                next.mirror = number != 0;
            else if (key == "hud")
                next.hud = number != 0;
            else if (key == "sharpness")
                next.sharpness = std::clamp(float(std::atof(value.c_str())), 0.f, 1.f);
            else if (key == "scene_camera")
                next.scene_camera = std::clamp(number, 0, 2);
            else if (key == "camera_snap")
                next.camera_snap = number != 0;
            else if (key == "debug_motion")
                next.debug_motion = number != 0;
            else if (key == "frame_gen")
                next.frame_gen = number != 0;
            else if (key == "fps_counter")
                next.fps_counter = number != 0;
            else if (key == "fg_multiplier")
                next.fg_multiplier = u32(std::clamp(number, 2, 6));
            else if (key == "fg_reflex")
                next.fg_reflex = number != 0;
            else if (key == "fg_reflex_sleep")
                next.fg_reflex_sleep = number != 0;
            else if (key == "fsr4_linear")
                next.fsr4_linear = number != 0;
            else if (key == "auto_exposure")
                next.auto_exposure = number != 0;
            else if (key == "menu_fix")
                next.menu_fix = number != 0;
            else if (key == "motion_blur")
                next.motion_blur = number != 0;
            else if (key == "dump")
                next.dump = number;
            else if (key == "dump_frames")
                next.dump_frames = u32(std::clamp(number, 1, 60));
            else if (key == "dump_output")
                next.dump_output = number != 0;
            else if (key == "exp_mirror")
                next.exp_mirror = std::clamp(number, 0, 3);
            else if (key == "exp_reset")
                next.exp_reset = number != 0;
            else if (key == "exp_mv_x")
                next.exp_mv_x = std::clamp(float(std::atof(value.c_str())), -4.f, 4.f);
            else if (key == "exp_mv_y")
                next.exp_mv_y = std::clamp(float(std::atof(value.c_str())), -4.f, 4.f);
            else if (key == "meter")
                next.meter = number != 0;
            else if (key == "upscaler")
                next.upscaler = value.starts_with("dlss")   ? 1
                                : value.starts_with("fsr4") ? 3
                                : value.starts_with("fsr")  ? 2
                                                            : 0;
            else if (key == "quality")
                next.quality = std::clamp(number, -1, 4);
            else if (key == "output") {
                u32 w{}, h{};
                next.output = std::sscanf(value.c_str(), "%ux%u", &w, &h) == 2 && w && h
                                  ? vk::Extent2D{w, h}
                                  : vk::Extent2D{};
            }
        }
        // A dump starts when the number changes after launch, never from the file as found.
        if (tune_loaded && next.dump != tune.dump && next.dump != 0) {
            dump_left = next.dump_frames;
            dump_index = 0;
            LOG_INFO(Render_Vulkan, "[DLSS-DUMP] Capturing {} frames as dump {}", dump_left,
                     next.dump);
        }
        tune_loaded = true;
        tune = next;
        FrameGen::SetEnabled(tune.frame_gen);
        FrameGen::SetCounterVisible(tune.fps_counter);
        FrameGen::SetMultiplier(tune.fg_multiplier);
        FrameGen::SetReflex(tune.fg_reflex, tune.fg_reflex_sleep);
        LOG_INFO(Render_Vulkan,
                 "[DLSS-TEMPORAL] Settings: enabled={} upscaler={} jitter={} sign=({},{}) "
                 "object_motion={} hud={} debug_motion={} quality={} output={}x{}",
                 tune.enabled, tune.upscaler, tune.jitter, tune.sign_x, tune.sign_y, tune.mirror,
                 tune.hud, tune.debug_motion, tune.quality, tune.output.width, tune.output.height);
    }

    void EnsurePipelines(const Instance& instance) {
        if (motion_pass)
            return;
        const auto device = instance.GetDevice();
        using T = vk::DescriptorType;
        static constexpr std::array motion_types{T::eCombinedImageSampler, T::eCombinedImageSampler,
                                                 T::eStorageImage,         T::eStorageBuffer,
                                                 T::eStorageImage,         T::eStorageBuffer};
        static constexpr std::array composite_types{
            T::eCombinedImageSampler, T::eCombinedImageSampler, T::eCombinedImageSampler,
            T::eCombinedImageSampler, T::eStorageImage,         T::eCombinedImageSampler,
            T::eStorageBuffer,        T::eStorageBuffer};
        motion_pass = std::make_unique<ComputePass>(device, motion_types, BB_DLSS_MOTION_COMP, 112);
        composite_pass =
            std::make_unique<ComputePass>(device, composite_types, BB_DLSS_COMPOSITE_COMP, 8);
        composite_hdr_pass =
            std::make_unique<ComputePass>(device, composite_types, BB_DLSS_COMPOSITE_HDR_COMP, 8);
        static constexpr std::array linearize_types{T::eCombinedImageSampler, T::eStorageImage};
        linearize_pass =
            std::make_unique<ComputePass>(device, linearize_types, BB_DLSS_LINEARIZE_COMP, 4);
        sharpen_pass =
            std::make_unique<ComputePass>(device, linearize_types, BB_DLSS_SHARPEN_HDR_COMP, 4);
        static constexpr std::array hudless_dim_types{T::eCombinedImageSampler,
                                                      T::eCombinedImageSampler, T::eStorageImage,
                                                      T::eStorageBuffer};
        hudless_dim_pass =
            std::make_unique<ComputePass>(device, hudless_dim_types, BB_HUDLESS_DIM_COMP, 4);
        mirror_rgba_pass =
            std::make_unique<ComputePass>(device, linearize_types, BB_DLSS_MIRROR_RGBA16F_COMP, 20);
        mirror_r32_pass =
            std::make_unique<ComputePass>(device, linearize_types, BB_DLSS_MIRROR_R32F_COMP, 20);
        mirror_rg16_pass =
            std::make_unique<ComputePass>(device, linearize_types, BB_DLSS_MIRROR_RG16F_COMP, 20);
        static constexpr std::array meter_types{T::eCombinedImageSampler, T::eCombinedImageSampler,
                                                T::eCombinedImageSampler, T::eStorageBuffer};
        meter_pass = std::make_unique<ComputePass>(device, meter_types, BB_FLICKER_METER_COMP, 20);
        const auto sampler = [&](vk::Filter filter) {
            return Make(device.createSamplerUnique(vk::SamplerCreateInfo{
                            .magFilter = filter,
                            .minFilter = filter,
                            .mipmapMode = vk::SamplerMipmapMode::eNearest,
                            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
                            .addressModeV = vk::SamplerAddressMode::eClampToEdge,
                            .addressModeW = vk::SamplerAddressMode::eClampToEdge,
                            .maxLod = 0}),
                        "temporal DLSS sampler");
        };
        nearest = sampler(vk::Filter::eNearest);
        linear = sampler(vk::Filter::eLinear);
    }

    vk::Extent2D OutputFor(vk::Extent2D in) const {
        if (tune.output.width && tune.output.height)
            return BbUpscaleOutput(in, tune.output, true);
        const auto window = display.load();
        return BbUpscaleOutput(in, {u32(window >> 32), u32(window)});
    }
    std::atomic<u64> display{};

    void AdvanceJitter() {
        const float scale =
            output.width && render.width ? float(output.width) / float(render.width) : 1.5f;
        const u32 phases = std::max<u32>(8, u32(std::ceil(8.f * scale * scale)));
        const u32 index = u32(frame % phases) + 1;
        jitter = {Halton(index, 2) - 0.5f, Halton(index, 3) - 0.5f};
    }

    // After each evaluation attempt: per-frame guide inputs restart and the jitter advances.
    void FrameBoundary(Scheduler& scheduler, BbVelocityMirror& mirror) {
        camera_constants = velocity_constants = depth_learned = false;
        blur_rows_valid = false;
        jittered_draws = jittered_before_copy = 0;
        producer_constants = producer_depths = velocity_draw_constants = 0;
        r32_depth.reset();
        mirror.ConsumeFrame(scheduler.CommandBuffer()); // start a fresh mirror every frame
        ++frame;
        AdvanceJitter();
    }

    // At the game's display copy: presentation bookkeeping only.
    void DisplayCopyDone() {
        decoupled_frame = false;
        if (!evaluated_since_copy)
            history_valid = false;
        evaluated_since_copy = false;
        // Camera frames run display copy to display copy: HUD/post passes after the evaluation
        // may bind the same scene constants again.
        previous_candidates = candidates;
        previous_candidate_count = candidate_count;
        candidate_count = 0;
        if (!frame_has_camera)
            camera.valid = previous_camera.valid = false;
        frame_has_camera = false;
        ui_phase = dlss_ready = dlss_pre_hud = false;
        jittered_before_copy = jittered_draws;
        copy_source.reset();
        copy_lut.reset();
        const auto now = std::chrono::steady_clock::now();
        if (last_frame.time_since_epoch().count())
            frame_ms = std::clamp(
                std::chrono::duration<float, std::milli>(now - last_frame).count(), 1.f, 100.f);
        last_frame = now;
        if (++copies % 30 == 0)
            PollTune();
    }
    u64 copies{};

    Backend ChooseBackend(const Instance& instance, Scheduler& scheduler, vk::Extent2D in,
                          vk::Extent2D out) {
        const auto* ngx = instance.GetDlssNgx();
        const bool dlss = ngx && ngx->IsAvailable();
        const bool rejected_size = fsr4_rejected_size && fsr4_rejected_size->render == in &&
                                   fsr4_rejected_size->output == out;
        const auto fsr_ready = [&] {
            if (!fsr && !fsr_tried) { // load the DLL only when FSR is actually used
                fsr_tried = true;
                fsr = FsrUpscaler::Create(instance);
            }
            return fsr != nullptr;
        };
        const auto fsr4_ready = [&] {
            if (rejected_size)
                return false;
            if (!fsr4 && !fsr4_tried) {
                fsr4_tried = true;
                fsr4 = Fsr4Addon::Create(instance, scheduler);
            }
            return fsr4 != nullptr;
        };
        switch (tune.upscaler) {
        case 1:
            return dlss ? Backend::Dlss : Backend::None;
        case 2:
            return fsr_ready() ? Backend::Fsr : Backend::None;
        case 3:
            if (fsr4_ready())
                return Backend::Fsr4;
            return rejected_size && fsr_ready() ? Backend::Fsr : Backend::None;
        default:
            return dlss           ? Backend::Dlss
                   : fsr4_ready() ? Backend::Fsr4
                   : fsr_ready()  ? Backend::Fsr
                                  : Backend::None;
        }
    }

    std::string NoUpscalerReason() const {
        const auto dlss = DlssProblem();
        const std::string no_fsr =
            FsrUpscaler::Present()
                ? "FSR could not be loaded (amd_fidelityfx_vk.dll is damaged or incompatible)."
                : "FSR needs amd_fidelityfx_vk.dll next to shadPS4.exe.";
        if (tune.upscaler == 1)
            return dlss.empty() ? "DLSS is not available on this system." : dlss;
        if (tune.upscaler == 2)
            return no_fsr;
        if (tune.upscaler == 3) {
            if (fsr4_rejected_size)
                return "FSR 4 is unavailable at this resolution. " + no_fsr;
            return Fsr4Addon::Present()
                       ? "FSR 4 could not start (see the log; it needs a Radeon RX 6000 / RTX "
                         "or newer GPU and the complete fsr4 folder)."
                       : "FSR 4 needs the FSR 4 add-on (the fsr4 folder next to shadPS4.exe).";
        }
        return (dlss.empty() ? std::string{} : dlss + " ") + no_fsr;
    }

    // FSR uses the camera planes and field of view to judge depth differences. Depth is
    // p2 + p3 / z for view distance z, so depth 0 and 1 give the near and far planes.
    FsrUpscaler::Camera FsrCamera() {
        FsrUpscaler::Camera result{0.1f, 3000.0f, 1.0f};
        const auto& p = camera.proj;
        if (camera.valid && p[2] != 0.0f && p[2] != 1.0f && p[3] != 0.0f) {
            const float a = std::abs(-p[3] / p[2]), b = std::abs(p[3] / (1.0f - p[2]));
            if (std::isfinite(a) && std::isfinite(b) && std::min(a, b) > 0.0f) {
                result.near_plane = std::min(a, b);
                result.far_plane = std::max(a, b);
            }
        }
        if (camera.valid && p[1] != 0.0f)
            result.fov_y = 2.0f * std::atan(1.0f / std::abs(p[1]));
        if (!camera_logged && camera.valid) {
            camera_logged = true;
            LOG_INFO(Render_Vulkan, "[FSR] Camera near={} far={} fovY={} (proj {} {} {} {})",
                     result.near_plane, result.far_plane, result.fov_y, p[0], p[1], p[2], p[3]);
        }
        return result;
    }

    // How far apart, in render pixels on average, the previous-frame positions of a few
    // points come out between the scene-camera pair and the game's motion-blur matrix.
    double CameraMismatch(const Camera& prev, const Camera& cur,
                          const std::array<double, 16>& previous_from_current,
                          vk::Extent2D in) const {
        std::array<double, 12> r{};
        const auto& a = prev.view;
        const auto& b = cur.inv_view;
        for (int i = 0; i < 3; ++i)
            for (int col = 0; col < 4; ++col)
                r[i * 4 + col] = a[i * 4 + 0] * b[0 * 4 + col] + a[i * 4 + 1] * b[1 * 4 + col] +
                                 a[i * 4 + 2] * b[2 * 4 + col] + (col == 3 ? a[i * 4 + 3] : 0.0);
        const auto& m = previous_from_current;
        double sum = 0;
        int n = 0;
        for (const double z : {1.5, 6.0, 30.0})
            for (const double y : {-0.7, 0.0, 0.7})
                for (const double x : {-0.7, 0.0, 0.7}) {
                    const double v[4] = {x * z / cur.proj[0], y * z / cur.proj[1], z, 1.0};
                    double pv[3];
                    for (int i = 0; i < 3; ++i)
                        pv[i] = r[i * 4] * v[0] + r[i * 4 + 1] * v[1] + r[i * 4 + 2] * v[2] +
                                r[i * 4 + 3];
                    if (pv[2] <= 1e-4)
                        continue;
                    const double sx = pv[0] * prev.proj[0] / pv[2];
                    const double sy = pv[1] * prev.proj[1] / pv[2];
                    const double raw = cur.proj[2] + cur.proj[3] / z;
                    const double depth =
                        blur_depth[2] * (blur_depth[0] * raw + blur_depth[1]) + blur_depth[3];
                    const double c[4] = {x, y, depth, 1.0};
                    double p[4];
                    for (int i = 0; i < 4; ++i)
                        p[i] = m[i * 4] * c[0] + m[i * 4 + 1] * c[1] + m[i * 4 + 2] * c[2] +
                               m[i * 4 + 3] * c[3];
                    if (std::abs(p[3]) < 1e-9)
                        continue;
                    sum += std::hypot((sx - p[0] / p[3]) * in.width * 0.5,
                                      (sy - p[1] / p[3]) * in.height * 0.5);
                    ++n;
                }
        return n ? sum / n : std::numeric_limits<double>::infinity();
    }

    // Picks this frame's main camera and the previous frame's among the scene constants seen.
    void SelectCamera(vk::Extent2D in) {
        camera.valid = previous_camera.valid = camera_trusted = false;
        camera_match_px = -1.f;
        if (candidate_count == 0 || previous_candidate_count == 0)
            return;
        // As first written: the first scene constants of each frame. Also the fallback for frame
        // generation when the game's matrix was not seen.
        camera = candidates[0];
        previous_camera = previous_candidates[0];
        if (tune.scene_camera == 1) {
            camera_trusted = true;
            return;
        }
        if (!blur_rows_valid) {
            ++camera_unchecked;
            return;
        }
        std::array<double, 16> blur{};
        for (int i = 0; i < 16; ++i)
            blur[i] = blur_rows[i];
        // Rows of the game's matrix take previous clip positions to current ones.
        std::array<double, 16> inverse{};
        {
            auto m = blur;
            for (int i = 0; i < 4; ++i)
                inverse[i * 4 + i] = 1.0;
            for (int c = 0; c < 4; ++c) {
                int pivot = c;
                for (int i = c + 1; i < 4; ++i)
                    if (std::abs(m[i * 4 + c]) > std::abs(m[pivot * 4 + c]))
                        pivot = i;
                if (std::abs(m[pivot * 4 + c]) < 1e-12 || !std::isfinite(m[pivot * 4 + c])) {
                    ++camera_unchecked;
                    return;
                }
                for (int j = 0; j < 4; ++j) {
                    std::swap(m[c * 4 + j], m[pivot * 4 + j]);
                    std::swap(inverse[c * 4 + j], inverse[pivot * 4 + j]);
                }
                const double d = m[c * 4 + c];
                for (int j = 0; j < 4; ++j) {
                    m[c * 4 + j] /= d;
                    inverse[c * 4 + j] /= d;
                }
                for (int i = 0; i < 4; ++i) {
                    if (i == c)
                        continue;
                    const double f = m[i * 4 + c];
                    for (int j = 0; j < 4; ++j) {
                        m[i * 4 + j] -= f * m[c * 4 + j];
                        inverse[i * 4 + j] -= f * inverse[c * 4 + j];
                    }
                }
            }
        }
        double best = std::numeric_limits<double>::infinity();
        u32 best_k = 0, best_j = 0;
        for (u32 k = 0; k < previous_candidate_count; ++k)
            for (u32 j = 0; j < candidate_count; ++j) {
                double e = CameraMismatch(previous_candidates[k], candidates[j], inverse, in);
                // A still camera matches with every self pair: stay with the camera in use.
                if (int(j) == chosen_candidate)
                    e -= 0.05;
                if (e < best) {
                    best = e;
                    best_k = k;
                    best_j = j;
                }
            }
        camera = candidates[best_j];
        previous_camera = previous_candidates[best_k];
        camera_match_px = float(best);
        camera_trusted = best < 2.0;
        if (!camera_trusted) {
            ++camera_rejects;
            return;
        }
        if (int(best_j) != chosen_candidate && chosen_candidate >= 0)
            ++camera_switches;
        chosen_candidate = int(best_j);
    }

    // Copies this frame's depth and motion for frame generation and describes its camera.
    void CaptureFrameGen(const Instance& instance, Runtime& runtime, Scheduler& scheduler,
                         vk::CommandBuffer command, VideoCore::Image& depth, vk::Extent2D in,
                         const DlssNgx::EvalDesc& eval) {
        frame_gen_pending.reset();
        if (!FrameGen::Active() || !frame_has_camera || !camera.valid || !previous_camera.valid ||
            !(depth.backing->image.image_ci.usage & vk::ImageUsageFlagBits::eTransferSrc))
            return;
        auto& owner = frame_gen_slots[frame_gen_next];
        frame_gen_next = (frame_gen_next + 1) % frame_gen_slots.size();
        // A queued present (or a VideoOut buffer) can still own this slot on a slow backend.
        if (!owner || owner.use_count() != 1)
            owner = std::make_shared<FrameGenSlot>();
        auto& slot = *owner;
        // Also retain captures which are dropped before reaching the presenter.
        scheduler.DeferOperation([resources = owner] {});
        using U = vk::ImageUsageFlagBits;
        if (!slot.depth || slot.depth->extent != in) {
            slot.depth =
                std::make_unique<OwnedImage>(instance, vk::Format::eR32Sfloat, in,
                                             U::eSampled | U::eTransferDst | U::eTransferSrc);
            slot.motion =
                std::make_unique<OwnedImage>(instance, vk::Format::eR16G16Sfloat, in,
                                             U::eSampled | U::eTransferDst | U::eTransferSrc);
        }
        const vk::ImageCopy region{
            .srcSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1},
            .dstSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1},
            .extent = {in.width, in.height, 1}};
        runtime.Transit(&depth, vk::ImageLayout::eTransferSrcOptimal,
                        vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead);
        runtime.FlushBarriers();
        motion->Transit(command, vk::ImageLayout::eTransferSrcOptimal,
                        vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead);
        for (auto* image : {slot.depth.get(), slot.motion.get()})
            image->Transit(command, vk::ImageLayout::eTransferDstOptimal,
                           vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite);
        command.copyImage(depth.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                          slot.depth->image.image, vk::ImageLayout::eTransferDstOptimal, region);
        command.copyImage(motion->image.image, vk::ImageLayout::eTransferSrcOptimal,
                          slot.motion->image.image, vk::ImageLayout::eTransferDstOptimal, region);
        for (auto* image : {slot.depth.get(), slot.motion.get()})
            image->Transit(command, vk::ImageLayout::eShaderReadOnlyOptimal,
                           vk::PipelineStageFlagBits2::eAllCommands,
                           vk::AccessFlagBits2::eShaderRead);

        // Column-vector matrices, row by row, in double.
        using Mat = std::array<double, 16>;
        static constexpr Mat Identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        const auto mul = [](const Mat& a, const Mat& b) {
            Mat r{};
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    for (int k = 0; k < 4; ++k)
                        r[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
            return r;
        };
        const auto invert = [](Mat m) {
            Mat r = Identity;
            for (int c = 0; c < 4; ++c) {
                int pivot = c;
                for (int i = c + 1; i < 4; ++i)
                    if (std::abs(m[i * 4 + c]) > std::abs(m[pivot * 4 + c]))
                        pivot = i;
                for (int j = 0; j < 4; ++j) {
                    std::swap(m[c * 4 + j], m[pivot * 4 + j]);
                    std::swap(r[c * 4 + j], r[pivot * 4 + j]);
                }
                const double d = m[c * 4 + c];
                if (d == 0.0)
                    return Identity;
                for (int j = 0; j < 4; ++j) {
                    m[c * 4 + j] /= d;
                    r[c * 4 + j] /= d;
                }
                for (int i = 0; i < 4; ++i) {
                    if (i == c)
                        continue;
                    const double f = m[i * 4 + c];
                    for (int j = 0; j < 4; ++j) {
                        m[i * 4 + j] -= f * m[c * 4 + j];
                        r[i * 4 + j] -= f * r[c * 4 + j];
                    }
                }
            }
            return r;
        };
        // Streamline wants row-vector matrices: the transposes.
        const auto out = [](const Mat& m) {
            std::array<float, 16> r{};
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    r[i * 4 + j] = float(m[j * 4 + i]);
            return r;
        };
        // D3D-style clip space: z_clip = zs * z + zo * w, w_clip = z.
        const auto projection = [](const std::array<float, 4>& p) {
            return Mat{p[0], 0, 0, 0, 0, p[1], 0, 0, 0, 0, p[2], p[3], 0, 0, 1, 0};
        };
        Mat view_to_prev_view{};
        const auto& a = previous_camera.view;
        const auto& b = camera.inv_view;
        for (int r = 0; r < 3; ++r)
            for (int col = 0; col < 4; ++col) {
                double value = a[r * 4 + 0] * b[0 * 4 + col] + a[r * 4 + 1] * b[1 * 4 + col] +
                               a[r * 4 + 2] * b[2 * 4 + col];
                if (col == 3)
                    value += a[r * 4 + 3];
                view_to_prev_view[r * 4 + col] = value;
            }
        view_to_prev_view[15] = 1.0;
        const Mat view_to_clip = projection(camera.proj);
        const Mat clip_to_view = invert(view_to_clip);
        const Mat clip_to_prev_clip =
            mul(mul(projection(previous_camera.proj), view_to_prev_view), clip_to_view);

        FrameGen::FrameInputs inputs{};
        inputs.resources = owner;
        inputs.depth = slot.depth->image.image;
        inputs.depth_view = *slot.depth->view;
        inputs.motion = slot.motion->image.image;
        inputs.motion_view = *slot.motion->view;
        inputs.render = in;
        inputs.view_to_clip = out(view_to_clip);
        inputs.clip_to_view = out(clip_to_view);
        inputs.clip_to_prev_clip = out(clip_to_prev_clip);
        inputs.prev_clip_to_clip = out(invert(clip_to_prev_clip));
        // The inverse view's columns: camera right, up, forward and position in the world.
        for (int i = 0; i < 3; ++i) {
            inputs.right[i] = float(b[i * 4 + 0]);
            inputs.up[i] = float(b[i * 4 + 1]);
            inputs.forward[i] = float(b[i * 4 + 2]);
            inputs.position[i] = float(b[i * 4 + 3]);
        }
        const auto lens = FsrCamera();
        inputs.near_plane = lens.near_plane;
        inputs.far_plane = lens.far_plane;
        inputs.fov_y = lens.fov_y;
        inputs.aspect = camera.proj[1] / camera.proj[0];
        inputs.jitter = {eval.jitter_x, eval.jitter_y};
        inputs.depth_inverted = tune.depth_inverted;
        inputs.reset = eval.reset;
        frame_gen_pending = inputs;
    }

    static u32 DumpBytesPerPixel(vk::Format format) {
        return format == vk::Format::eR16G16B16A16Sfloat ? 8 : 4; // RGBA8, R32, RG16 otherwise
    }

    struct DumpPart {
        std::string name;
        vk::Image image;
        vk::Extent2D extent;
        vk::Format format;
        u64 offset{}, bytes{};
    };

    // Copies the parts (already in eTransferSrcOptimal) into host memory and, once the GPU has
    // finished with them, writes <stem>_<name>_<w>x<h>_<format>.bin and <stem>_<meta_name>.txt
    // from a worker thread, so the frame is not stalled.
    void WriteDump(const Instance& instance, Scheduler& scheduler, vk::CommandBuffer command,
                   std::vector<DumpPart> parts, std::string meta, const std::string& stem,
                   const char* meta_name) {
        u64 total = 0;
        for (auto& part : parts) {
            part.bytes = u64(part.extent.width) * part.extent.height * DumpBytesPerPixel(part.format);
            part.offset = total;
            total += (part.bytes + 255) & ~u64(255);
        }
        auto buffer = std::make_shared<VideoCore::Buffer>(
            instance, 0, total, VideoCore::MemoryType::HostCached, "DLSS dump");
        if (buffer->mapped_data.size() < total)
            throw std::runtime_error("dump buffer is not host visible");
        for (const auto& part : parts) {
            const vk::BufferImageCopy region{
                .bufferOffset = part.offset,
                .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                                     .layerCount = 1},
                .imageExtent = {part.extent.width, part.extent.height, 1}};
            command.copyImageToBuffer(part.image, vk::ImageLayout::eTransferSrcOptimal,
                                      buffer->Handle(), region);
        }
        const vk::BufferMemoryBarrier2 to_host{
            .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eHost,
            .dstAccessMask = vk::AccessFlagBits2::eHostRead,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = buffer->Handle(),
            .offset = 0,
            .size = VK_WHOLE_SIZE};
        command.pipelineBarrier2(
            vk::DependencyInfo{.bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &to_host});
        const auto file_name = [](const std::string& stem, const DumpPart& part) {
            return fmt::format("{}_{}_{}x{}_{}.bin", stem, part.name, part.extent.width,
                               part.extent.height, vk::to_string(part.format));
        };
        for (const auto& part : parts)
            meta += fmt::format("file {} {} {} {} {}\n", part.name, file_name(stem, part),
                                part.extent.width, part.extent.height,
                                vk::to_string(part.format));
        const auto dir = BbDlssSettingsPath().parent_path() / "dlss_dump";
        std::string meta_file = stem + "_" + meta_name + ".txt";
        scheduler.DeferOperation([buffer, parts, meta, dir, stem, meta_file, total, file_name] {
            buffer->Invalidate(0, total);
            auto data = std::make_shared<std::vector<u8>>(buffer->mapped_data.begin(),
                                                          buffer->mapped_data.begin() + total);
            std::thread([data, parts, meta, dir, stem, meta_file, file_name] {
                std::error_code error;
                std::filesystem::create_directories(dir, error);
                for (const auto& part : parts) {
                    std::ofstream file{dir / file_name(stem, part), std::ios::binary};
                    file.write(reinterpret_cast<const char*>(data->data() + part.offset),
                               std::streamsize(part.bytes));
                }
                std::ofstream{dir / meta_file} << meta;
            }).detach();
        });
    }

    // Diagnostic: this frame's upscaler inputs (and output).
    void DumpFrame(const Instance& instance, Runtime& runtime, Scheduler& scheduler,
                   vk::CommandBuffer command, VideoCore::Image& depth, vk::Extent2D in,
                   vk::Extent2D out, const DlssNgx::EvalDesc& eval,
                   const std::array<float, 2>& applied, bool linear) {
        std::vector<DumpPart> parts;
        const bool depth_copyable =
            bool(depth.backing->image.image_ci.usage & vk::ImageUsageFlagBits::eTransferSrc);
        parts.push_back({"color", snapshot->image.image, in, snapshot->format});
        if (depth_copyable)
            parts.push_back({"depth", depth.GetImage(), in, vk::Format::eR32Sfloat});
        parts.push_back({"motion", motion->image.image, in, motion->format});
        OwnedImage* const output_image = tune.dump_output ? upscaled.get() : nullptr;
        if (output_image)
            parts.push_back({"output", output_image->image.image, output_image->extent,
                             output_image->format});

        if (depth_copyable) {
            runtime.Transit(&depth, vk::ImageLayout::eTransferSrcOptimal,
                            vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead);
            runtime.FlushBarriers();
        }
        std::vector<std::pair<OwnedImage*, vk::ImageLayout>> restore;
        for (auto* owned : {snapshot.get(), motion.get(), output_image}) {
            if (!owned)
                continue;
            restore.emplace_back(owned, owned->layout);
            owned->Transit(command, vk::ImageLayout::eTransferSrcOptimal,
                           vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead);
        }

        // Everything needed to redo the motion pass and judge the frame offline.
        const auto join = [](const auto& values) {
            std::string text;
            for (const auto value : values)
                text += fmt::format("{}{:.9g}", text.empty() ? "" : " ", double(value));
            return text;
        };
        std::string meta = fmt::format(
            "dump={} index={} frame={} evaluation={} backend={}\n"
            "render={}x{} output={}x{} color_format={} upscaler_input_linear={}\n"
            "jitter_applied_render_px={:.6f} {:.6f}\njitter_sent={:.6f} {:.6f}\n"
            "reset={} frame_ms={:.3f} jittered_draws={} jittered_before_copy={}\n"
            "motion_flags={:#x} (1 snap, 2 scene camera, 0x100 object motion)\n"
            "tune_jitter={} sign={} {} object_motion={} scene_camera={} camera_snap={} "
            "preset={} quality={} hud={}\n",
            tune.dump, dump_index, frame, evaluations + 1, BackendName(backend), in.width,
            in.height, out.width, out.height, vk::to_string(snapshot->format), linear,
            applied[0], applied[1], eval.jitter_x, eval.jitter_y, eval.reset, frame_ms,
            jittered_draws, jittered_before_copy, motion_flags, tune.jitter, tune.sign_x,
            tune.sign_y, tune.mirror, tune.scene_camera, tune.camera_snap, tune.preset,
            tune.quality, tune.hud);
        meta += "motion_push(reproject12 proj4 prev_proj4 jitter2)=" + join(motion_constants) + "\n";
        meta += fmt::format("camera_valid={} previous_valid={} trusted={} match_px={:.3f} "
                            "candidates={} previous_candidates={} chosen={}\n",
                            camera.valid, previous_camera.valid, camera_trusted, camera_match_px,
                            candidate_count, previous_candidate_count, chosen_candidate);
        for (u32 i = 0; i < candidate_count; ++i)
            meta += fmt::format("candidate{}_view=", i) + join(candidates[i].view) + "\n";
        meta += "blur_rows=" + join(blur_rows) + "\n";
        meta += "blur_depth=" + join(blur_depth) + "\n";
        meta += "camera_view=" + join(camera.view) + "\n";
        meta += "camera_inv_view=" + join(camera.inv_view) + "\n";
        meta += "camera_proj=" + join(camera.proj) + "\n";
        meta += "previous_view=" + join(previous_camera.view) + "\n";
        meta += "previous_inv_view=" + join(previous_camera.inv_view) + "\n";
        meta += "previous_proj=" + join(previous_camera.proj) + "\n";
        const auto stem = fmt::format("d{}_f{:02}", tune.dump, dump_index);
        WriteDump(instance, scheduler, command, std::move(parts), std::move(meta), stem, "meta");
        for (auto& [owned, layout] : restore)
            if (layout != vk::ImageLayout::eUndefined && layout != owned->layout)
                owned->Transit(command, layout, vk::PipelineStageFlagBits2::eAllCommands,
                               vk::AccessFlagBits2::eMemoryRead);
        dump_composite = stem; // the display composite of this frame is dumped too
        ++dump_index;
        if (--dump_left == 0)
            LOG_INFO(Render_Vulkan, "[DLSS-DUMP] Dump {} recorded: {} frames", tune.dump,
                     dump_index);
    }

    // Diagnostic: the composite's other inputs, the game's own frame with its HUD (render size),
    // and what the composite made of it (output size, after the display LUT).
    void DumpComposite(const Instance& instance, Runtime& runtime, Scheduler& scheduler,
                       vk::CommandBuffer command, VideoCore::Image& game_frame,
                       OwnedImage& written, const std::string& stem, u32 flags) {
        std::vector<DumpPart> parts;
        if (game_frame.backing->image.image_ci.usage & vk::ImageUsageFlagBits::eTransferSrc) {
            runtime.Transit(&game_frame, vk::ImageLayout::eTransferSrcOptimal,
                            vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead);
            runtime.FlushBarriers();
            parts.push_back({"final", game_frame.GetImage(),
                             {game_frame.info.size.width, game_frame.info.size.height},
                             game_frame.info.pixel_format});
        }
        written.Transit(command, vk::ImageLayout::eTransferSrcOptimal,
                        vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead);
        parts.push_back({"composite", written.image.image, written.extent, written.format});
        u32 changed{};
        if (coverage && composites >= CoverageSlots)
            std::memcpy(&changed,
                        coverage->mapped_data.data() + (composites % CoverageSlots) * CoverageStride,
                        sizeof(changed));
        std::string meta = fmt::format(
            "composite_flags={:#x} (1 hud, 2 lut, 4 debug, 8 menu, 16 linear, 32 alpha, 128 hdr)\n"
            "menu={} hud_changed_pixels_3_frames_ago={} share={:.4f} final_format={}\n",
            flags, menu, changed, double(changed) / (double(output.width) * output.height),
            vk::to_string(game_frame.info.pixel_format));
        WriteDump(instance, scheduler, command, std::move(parts), std::move(meta), stem,
                  "composite");
    }

    // Diagnostic (exp_mirror): mirrored copies of the upscaler's inputs, and of its output.
    std::unique_ptr<OwnedImage> mirror_color, mirror_depth, mirror_motion, mirror_output;
    // Diagnostic: the game's own motion (motion-blur matrix path) and, per 32x32 cell, the summed
    // difference between the scene-camera motion and it.
    std::unique_ptr<OwnedImage> reference_motion;
    std::unique_ptr<VideoCore::Buffer> motion_error;
    u32 error_grid_w{};
    int mirror_mode_used{};

    void MirrorInto(vk::CommandBuffer command, ComputePass& pass, vk::ImageView source,
                    OwnedImage& target, std::array<float, 4> scale, u32 flip) {
        target.Transit(command, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eComputeShader,
                       vk::AccessFlagBits2::eShaderWrite);
        const std::array images{
            vk::DescriptorImageInfo{*nearest, source, vk::ImageLayout::eShaderReadOnlyOptimal},
            vk::DescriptorImageInfo{{}, *target.view, vk::ImageLayout::eGeneral}};
        const std::array writes{
            vk::WriteDescriptorSet{.dstBinding = 0,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                   .pImageInfo = &images[0]},
            vk::WriteDescriptorSet{.dstBinding = 1,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eStorageImage,
                                   .pImageInfo = &images[1]}};
        struct {
            std::array<float, 4> scale;
            u32 flip;
        } push{scale, flip};
        command.bindPipeline(vk::PipelineBindPoint::eCompute, *pass.pipeline);
        command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *pass.layout, 0, writes);
        command.pushConstants(*pass.layout, vk::ShaderStageFlagBits::eCompute, 0, 20, &push);
        command.dispatch((target.extent.width + 7) / 8, (target.extent.height + 7) / 8, 1);
        target.Transit(command, vk::ImageLayout::eShaderReadOnlyOptimal,
                       vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
    }

    // Diagnostic (meter): per 64x64 cell of the upscaled image, how much it changes from the
    // previous frame once motion is taken out. Read back four frames later and logged per frame
    // (column profile, flicker_meter.csv) and per autopilot step (whole grid, flicker_grid.csv).
    struct MeterSlot {
        std::unique_ptr<VideoCore::Buffer> buffer;
        BbAutopilot::Status pilot{};
        u64 frame{};
        std::string key;
        double t_ms{};
        bool pending{};
    };
    std::array<MeterSlot, 4> meter_slots;
    std::unique_ptr<VideoCore::Buffer> meter_grid;
    std::unique_ptr<OwnedImage> meter_prev;
    bool meter_prev_valid{};
    u64 meter_frames{};
    u32 meter_gw{}, meter_gh{};
    std::ofstream meter_csv, meter_grid_csv;
    std::vector<double> step_sum; // per cell: diff, detail, still, count, motion
    u32 step_frames{};
    std::string step_key;
    const std::chrono::steady_clock::time_point meter_epoch{std::chrono::steady_clock::now()};

    std::string ExperimentKey() const {
        return fmt::format("mirror{} reset{} mv{:g},{:g} jitter{} objmotion{} sign{:g},{:g} "
                           "preset{} camera{} snap{}",
                           tune.exp_mirror, tune.exp_reset ? 1 : 0, tune.exp_mv_x, tune.exp_mv_y,
                           tune.jitter ? 1 : 0, tune.mirror ? 1 : 0, tune.sign_x, tune.sign_y,
                           tune.preset, tune.scene_camera, tune.camera_snap ? 1 : 0);
    }

    void FlushMeterStep() {
        if (step_frames == 0 || step_sum.empty())
            return;
        if (!meter_grid_csv.is_open()) {
            meter_grid_csv.open(BbDlssSettingsPath().parent_path() / "flicker_grid.csv",
                                std::ios::app);
            meter_grid_csv << fmt::format("# session grid {}x{} cells of 64 px; per line: key, "
                                          "frames, then diff per cell, detail per cell, motion "
                                          "per cell (row major; diff/detail x1000, motion px)\n",
                                          meter_gw, meter_gh);
        }
        std::string line = fmt::format("{};{}", step_key, step_frames);
        const size_t cells = size_t(meter_gw) * meter_gh;
        for (int field : {0, 1, 4}) {
            line += ";";
            for (size_t c = 0; c < cells; ++c) {
                const double count = step_sum[c * 5 + 3];
                const double v = count > 0 ? step_sum[c * 5 + field] / count : 0.0;
                line += fmt::format("{}{:.2f}", c ? "," : "",
                                    field == 4 ? v / 64.0 : v * 1000.0 / 4096.0);
            }
        }
        meter_grid_csv << line << '\n';
        meter_grid_csv.flush();
        std::fill(step_sum.begin(), step_sum.end(), 0.0);
        step_frames = 0;
    }

    void ConsumeMeter(MeterSlot& slot) {
        const size_t cells = size_t(meter_gw) * meter_gh;
        const u64 bytes = cells * 5 * sizeof(u32);
        const u64 total_bytes = slot.buffer->SizeBytes();
        slot.buffer->Invalidate(0, total_bytes);
        std::vector<u32> grid(cells * 5);
        std::memcpy(grid.data(), slot.buffer->mapped_data.data(), bytes);
        // Scene-camera vs game motion, per column of 32 render pixels (render px).
        std::vector<double> err_sum(error_grid_w, 0.0), err_count(error_grid_w, 0.0);
        if (error_grid_w && total_bytes > bytes) {
            const size_t err_cells = (total_bytes - bytes) / (2 * sizeof(u32));
            std::vector<u32> errors(err_cells * 2);
            std::memcpy(errors.data(), slot.buffer->mapped_data.data() + bytes,
                        errors.size() * sizeof(u32));
            for (size_t c = 0; c < err_cells; ++c) {
                err_sum[c % error_grid_w] += errors[c * 2];
                err_count[c % error_grid_w] += errors[c * 2 + 1];
            }
        }
        double err_all = 0, err_n = 0;
        for (u32 x = 0; x < error_grid_w; ++x) {
            err_all += err_sum[x];
            err_n += err_count[x];
        }
        if (!meter_csv.is_open()) {
            meter_csv.open(BbDlssSettingsPath().parent_path() / "flicker_meter.csv", std::ios::app);
            meter_csv << fmt::format("# session {} columns of 64 px; t_ms;frame;run;step;active;key;"
                                     "pixels;motion_px;mv_err_px;diff per column x1000;detail per "
                                     "column x1000;still per column x1000;scene-camera motion "
                                     "error per column (render px, -1 none)\n",
                                     meter_gw);
        }
        std::vector<double> col(size_t(meter_gw) * 5, 0.0);
        double total = 0, motion = 0;
        for (u32 y = 0; y < meter_gh; ++y)
            for (u32 x = 0; x < meter_gw; ++x)
                for (int f = 0; f < 5; ++f)
                    col[x * 5 + f] += grid[(size_t(y) * meter_gw + x) * 5 + f];
        for (u32 x = 0; x < meter_gw; ++x) {
            total += col[x * 5 + 3];
            motion += col[x * 5 + 4];
        }
        std::string line = fmt::format("{:.1f};{};{};{};{};{};{:.0f};{:.2f}", slot.t_ms, slot.frame,
                                       slot.pilot.run, slot.pilot.step, slot.pilot.active ? 1 : 0,
                                       slot.key, total, total > 0 ? motion / total / 64.0 : 0.0);
        line += fmt::format(";{:.3f}", err_n > 0 ? err_all / err_n / 64.0 : -1.0);
        for (int field : {0, 1, 2}) {
            line += ";";
            for (u32 x = 0; x < meter_gw; ++x) {
                const double count = col[x * 5 + 3];
                line += fmt::format("{}{:.2f}", x ? "," : "",
                                    count > 0 ? col[x * 5 + field] / count * 1000.0 / 4096.0 : 0.0);
            }
        }
        line += ";";
        for (u32 x = 0; x < error_grid_w; ++x)
            line += fmt::format("{}{:.3f}", x ? "," : "",
                                err_count[x] > 0 ? err_sum[x] / err_count[x] / 64.0 : -1.0);
        meter_csv << line << '\n';
        if (slot.frame % 45 == 0)
            meter_csv.flush();
        // Whole-grid sums per autopilot step and experiment.
        const std::string key = fmt::format("run{} step{} active{} {}", slot.pilot.run,
                                            slot.pilot.step, slot.pilot.active ? 1 : 0, slot.key);
        if (key != step_key) {
            FlushMeterStep();
            step_key = key;
        }
        if (step_sum.size() != cells * 5)
            step_sum.assign(cells * 5, 0.0);
        for (size_t i = 0; i < cells * 5; ++i)
            step_sum[i] += grid[i];
        ++step_frames;
    }

    void RunMeter(const Instance& instance, vk::CommandBuffer command, vk::Extent2D in,
                  vk::Extent2D out) {
        const u32 gw = (out.width + 63) / 64, gh = (out.height + 63) / 64;
        const u64 bytes = u64(gw) * gh * 5 * sizeof(u32);
        const u64 error_bytes = motion_error ? motion_error->SizeBytes() : 0;
        if (!meter_grid || gw != meter_gw || gh != meter_gh ||
            meter_slots[0].buffer->SizeBytes() != bytes + error_bytes) {
            FlushMeterStep();
            meter_grid = std::make_unique<VideoCore::Buffer>(
                instance, 0, bytes, VideoCore::MemoryType::DeviceLocal, "flicker meter");
            for (auto& slot : meter_slots) {
                slot.buffer = std::make_unique<VideoCore::Buffer>(
                    instance, 0, bytes + error_bytes, VideoCore::MemoryType::HostCached,
                    "flicker meter readback");
                slot.pending = false;
            }
            meter_gw = gw;
            meter_gh = gh;
            step_sum.clear();
        }
        if (!meter_prev || meter_prev->extent != out) {
            using U = vk::ImageUsageFlagBits;
            meter_prev = std::make_unique<OwnedImage>(instance, vk::Format::eR16G16B16A16Sfloat, out,
                                                      U::eSampled | U::eTransferDst);
            meter_prev_valid = false;
        }
        auto& slot = meter_slots[meter_frames % meter_slots.size()];
        if (slot.pending) {
            ConsumeMeter(slot);
            slot.pending = false;
        }
        const auto buffer_barrier = [&](vk::Buffer buffer, vk::PipelineStageFlags2 src_stage,
                                        vk::AccessFlags2 src, vk::PipelineStageFlags2 dst_stage,
                                        vk::AccessFlags2 dst) {
            const vk::BufferMemoryBarrier2 barrier{.srcStageMask = src_stage,
                                                   .srcAccessMask = src,
                                                   .dstStageMask = dst_stage,
                                                   .dstAccessMask = dst,
                                                   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                                   .buffer = buffer,
                                                   .offset = 0,
                                                   .size = VK_WHOLE_SIZE};
            command.pipelineBarrier2(
                vk::DependencyInfo{.bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &barrier});
        };
        buffer_barrier(meter_grid->Handle(), vk::PipelineStageFlagBits2::eAllCommands,
                       vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
                       vk::PipelineStageFlagBits2::eClear, vk::AccessFlagBits2::eTransferWrite);
        command.fillBuffer(meter_grid->Handle(), 0, bytes, 0);
        buffer_barrier(meter_grid->Handle(), vk::PipelineStageFlagBits2::eClear,
                       vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eComputeShader,
                       vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
        const auto upscaled_layout = upscaled->layout;
        upscaled->Transit(command, vk::ImageLayout::eShaderReadOnlyOptimal,
                          vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead);
        meter_prev->Transit(command, vk::ImageLayout::eShaderReadOnlyOptimal,
                            vk::PipelineStageFlagBits2::eComputeShader,
                            vk::AccessFlagBits2::eShaderRead);
        const std::array images{
            vk::DescriptorImageInfo{*nearest, *upscaled->view, vk::ImageLayout::eShaderReadOnlyOptimal},
            vk::DescriptorImageInfo{*linear, *meter_prev->view,
                                    vk::ImageLayout::eShaderReadOnlyOptimal},
            vk::DescriptorImageInfo{*nearest, *reference_motion->view,
                                    vk::ImageLayout::eShaderReadOnlyOptimal}};
        const vk::DescriptorBufferInfo grid_info{meter_grid->Handle(), 0, bytes};
        std::array<vk::WriteDescriptorSet, 4> writes;
        for (u32 i = 0; i < 3; ++i)
            writes[i] = {.dstBinding = i,
                         .descriptorCount = 1,
                         .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                         .pImageInfo = &images[i]};
        writes[3] = {.dstBinding = 3,
                     .descriptorCount = 1,
                     .descriptorType = vk::DescriptorType::eStorageBuffer,
                     .pBufferInfo = &grid_info};
        // Follows the game's own motion (reference_motion), whatever the upscaler was given.
        struct {
            float mv_scale[2];
            u32 grid_w, grid_h, have_previous;
        } push{{float(out.width) / float(in.width), float(out.height) / float(in.height)},
               gw,
               gh,
               meter_prev_valid ? 1u : 0u};
        command.bindPipeline(vk::PipelineBindPoint::eCompute, *meter_pass->pipeline);
        command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *meter_pass->layout, 0, writes);
        command.pushConstants(*meter_pass->layout, vk::ShaderStageFlagBits::eCompute, 0, 20, &push);
        command.dispatch((out.width + 7) / 8, (out.height + 7) / 8, 1);
        buffer_barrier(meter_grid->Handle(), vk::PipelineStageFlagBits2::eComputeShader,
                       vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eCopy,
                       vk::AccessFlagBits2::eTransferRead);
        const vk::BufferCopy copy{.srcOffset = 0, .dstOffset = 0, .size = bytes};
        command.copyBuffer(meter_grid->Handle(), slot.buffer->Handle(), copy);
        if (error_bytes) {
            buffer_barrier(motion_error->Handle(), vk::PipelineStageFlagBits2::eComputeShader,
                           vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eCopy,
                           vk::AccessFlagBits2::eTransferRead);
            const vk::BufferCopy error_copy{.srcOffset = 0, .dstOffset = bytes, .size = error_bytes};
            command.copyBuffer(motion_error->Handle(), slot.buffer->Handle(), error_copy);
        }
        buffer_barrier(slot.buffer->Handle(), vk::PipelineStageFlagBits2::eCopy,
                       vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eHost,
                       vk::AccessFlagBits2::eHostRead);
        slot.pilot = BbAutopilot::GetStatus();
        slot.frame = frame;
        slot.key = ExperimentKey();
        slot.t_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                              meter_epoch)
                        .count();
        slot.pending = meter_prev_valid;
        // This frame becomes the next frame's previous one.
        upscaled->Transit(command, vk::ImageLayout::eTransferSrcOptimal,
                          vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead);
        meter_prev->Transit(command, vk::ImageLayout::eTransferDstOptimal,
                            vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite);
        const vk::ImageCopy region{
            .srcSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1},
            .dstSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1},
            .extent = {out.width, out.height, 1}};
        command.copyImage(upscaled->image.image, vk::ImageLayout::eTransferSrcOptimal,
                          meter_prev->image.image, vk::ImageLayout::eTransferDstOptimal, region);
        meter_prev->Transit(command, vk::ImageLayout::eShaderReadOnlyOptimal,
                            vk::PipelineStageFlagBits2::eComputeShader,
                            vk::AccessFlagBits2::eShaderRead);
        if (upscaled_layout != vk::ImageLayout::eUndefined)
            upscaled->Transit(command, upscaled_layout, vk::PipelineStageFlagBits2::eAllCommands,
                              vk::AccessFlagBits2::eMemoryRead);
        meter_prev_valid = true;
        ++meter_frames;
    }

    void InvalidateOutputs() {
        for (auto& [_, image] : outputs)
            image->fresh = false;
    }

    // With `upscale_to`, the decoupled-UI mode: `source` is the HDR scene, upscaled to that size
    // and copied over the game's own upscale instead of being composited at display time.
    bool RunDlss(const Instance& instance, Runtime& runtime, Scheduler& scheduler,
                 VideoCore::TextureCache& cache, BbVelocityMirror& mirror, VideoCore::Image& source,
                 bool pre_hud, std::optional<vk::Extent2D> upscale_to = std::nullopt);
    void SnapshotHud(const Instance& instance, Runtime& runtime, Scheduler& scheduler,
                     VideoCore::Image& target) {
        scheduler.EndRendering();
        EnsurePipelines(instance);
        const vk::Extent2D size{target.info.size.width, target.info.size.height};
        // Float, copied by a shader through the game's own view: RenoDX widens the target and
        // keeps highlights above 1 in it.
        if (!hud_snapshot || hud_snapshot->extent != size) {
            if (hud_snapshot)
                scheduler.Finish();
            using U = vk::ImageUsageFlagBits;
            hud_snapshot = std::make_unique<OwnedImage>(instance, vk::Format::eR16G16B16A16Sfloat,
                                                        size, U::eSampled | U::eStorage);
            hud_dimmed = std::make_unique<OwnedImage>(instance, vk::Format::eR16G16B16A16Sfloat,
                                                      size, U::eSampled | U::eStorage);
        }
        const auto command = scheduler.CommandBuffer(); // after Finish(), see RunDlss
        VideoCore::ImageViewInfo view_info{};
        view_info.format = last_copy_format != vk::Format::eUndefined ? last_copy_format
                                                                      : target.info.pixel_format;
        const auto& view = target.FindView(view_info);
        runtime.Transit(&target, vk::ImageLayout::eShaderReadOnlyOptimal,
                        vk::PipelineStageFlagBits2::eComputeShader,
                        vk::AccessFlagBits2::eShaderRead);
        runtime.FlushBarriers();
        hud_snapshot->Transit(command, vk::ImageLayout::eGeneral,
                              vk::PipelineStageFlagBits2::eComputeShader,
                              vk::AccessFlagBits2::eShaderWrite);
        const std::array images{
            vk::DescriptorImageInfo{*nearest, *view.image_view,
                                    vk::ImageLayout::eShaderReadOnlyOptimal},
            vk::DescriptorImageInfo{{}, *hud_snapshot->view, vk::ImageLayout::eGeneral}};
        const std::array writes{
            vk::WriteDescriptorSet{.dstBinding = 0,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                   .pImageInfo = &images[0]},
            vk::WriteDescriptorSet{.dstBinding = 1,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eStorageImage,
                                   .pImageInfo = &images[1]}};
        const u32 copy_flags = 2; // plain copy
        command.bindPipeline(vk::PipelineBindPoint::eCompute, *linearize_pass->pipeline);
        command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *linearize_pass->layout, 0,
                                     writes);
        command.pushConstants(*linearize_pass->layout, vk::ShaderStageFlagBits::eCompute, 0,
                              sizeof(copy_flags), &copy_flags);
        command.dispatch((size.width + 7) / 8, (size.height + 7) / 8, 1);
        hud_snapshot->Transit(command, vk::ImageLayout::eShaderReadOnlyOptimal,
                              vk::PipelineStageFlagBits2::eFragmentShader,
                              vk::AccessFlagBits2::eShaderRead);
        scheduler.GetDynamicState().Invalidate();
        hud_snapshot_taken = true;
    }

    // At a Decoupled UI display copy: draw it again from the HUD-less snapshot.
    void ReplayHudless(const Instance& instance, Runtime& runtime, Scheduler& scheduler,
                       VideoCore::TextureCache& cache, VideoCore::Image& target) {
        auto& slot = decoupled_hudless[target.info.guest_address];
        const vk::Extent2D size{target.info.size.width, target.info.size.height};
        scheduler.EndRendering();
        // RenoDX's display copy writes values above 1: a float image, presented like the
        // regular path's outputs.
        const bool renodx = RenoDxLoaded();
        const auto format = renodx ? vk::Format::eR16G16B16A16Sfloat : target.info.pixel_format;
        if (!dimming)
            dimming = std::make_unique<VideoCore::Buffer>(instance, 0, DimmingBytes,
                                                          VideoCore::MemoryType::DeviceLocal);
        if (!slot || slot->extent != size || slot->format != format) {
            if (slot)
                scheduler.Finish();
            using U = vk::ImageUsageFlagBits;
            slot = renodx ? std::make_unique<OwnedImage>(instance, format, size,
                                                         U::eSampled | U::eColorAttachment, true)
                          : std::make_unique<OwnedImage>(instance, format, size,
                                                         U::eSampled | U::eColorAttachment, false,
                                                         vk::ImageCreateFlagBits::eMutableFormat);
            slot->swapped = renodx;
        }
        const auto command = scheduler.CommandBuffer();

        // The snapshot with the dimming most of the finished frame shows (fades, dialogs).
        auto& final_image = cache.GetImage(copy_source->id);
        const auto& final_view = final_image.FindView(copy_source->view);
        runtime.Transit(&final_image, vk::ImageLayout::eShaderReadOnlyOptimal,
                        vk::PipelineStageFlagBits2::eComputeShader |
                            vk::PipelineStageFlagBits2::eFragmentShader,
                        vk::AccessFlagBits2::eShaderRead);
        runtime.FlushBarriers();
        hud_dimmed->Transit(command, vk::ImageLayout::eGeneral,
                            vk::PipelineStageFlagBits2::eComputeShader,
                            vk::AccessFlagBits2::eShaderWrite);
        const vk::BufferMemoryBarrier2 clear_barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = dimming->Handle(),
            .offset = 0,
            .size = DimmingBytes};
        command.pipelineBarrier2(vk::DependencyInfo{.bufferMemoryBarrierCount = 1,
                                                    .pBufferMemoryBarriers = &clear_barrier});
        command.fillBuffer(dimming->Handle(), 0, DimmingBytes, 0);
        auto pass_barrier = clear_barrier;
        pass_barrier.srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
        pass_barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
        pass_barrier.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader;
        pass_barrier.dstAccessMask =
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
        command.pipelineBarrier2(vk::DependencyInfo{.bufferMemoryBarrierCount = 1,
                                                    .pBufferMemoryBarriers = &pass_barrier});
        const auto ro = vk::ImageLayout::eShaderReadOnlyOptimal;
        const std::array images{
            vk::DescriptorImageInfo{*nearest, *final_view.image_view, ro},
            vk::DescriptorImageInfo{*nearest, *hud_snapshot->view, ro},
            vk::DescriptorImageInfo{{}, *hud_dimmed->view, vk::ImageLayout::eGeneral}};
        const vk::DescriptorBufferInfo dimming_info{dimming->Handle(), 0, DimmingBytes};
        std::array<vk::WriteDescriptorSet, 4> writes;
        for (u32 i = 0; i < images.size(); ++i)
            writes[i] = {.dstBinding = i,
                         .descriptorCount = 1,
                         .descriptorType = i == 2 ? vk::DescriptorType::eStorageImage
                                                  : vk::DescriptorType::eCombinedImageSampler,
                         .pImageInfo = &images[i]};
        writes[3] = {.dstBinding = 3,
                     .descriptorCount = 1,
                     .descriptorType = vk::DescriptorType::eStorageBuffer,
                     .pBufferInfo = &dimming_info};
        command.bindPipeline(vk::PipelineBindPoint::eCompute, *hudless_dim_pass->pipeline);
        command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *hudless_dim_pass->layout, 0,
                                     writes);
        for (const u32 mode : {0u, 1u}) {
            if (mode == 1) {
                pass_barrier.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
                pass_barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
                command.pipelineBarrier2(vk::DependencyInfo{
                    .bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &pass_barrier});
            }
            command.pushConstants(*hudless_dim_pass->layout, vk::ShaderStageFlagBits::eCompute, 0,
                                  sizeof(mode), &mode);
            command.dispatch((size.width + 7) / 8, (size.height + 7) / 8, 1);
        }
        hud_dimmed->Transit(command, ro, vk::PipelineStageFlagBits2::eFragmentShader,
                            vk::AccessFlagBits2::eShaderRead);

        slot->Transit(command, vk::ImageLayout::eColorAttachmentOptimal,
                      vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                      vk::AccessFlagBits2::eColorAttachmentWrite);
        display_copy_replay = DisplayCopyReplay{.shader_hash = DisplayCopy,
                                                .extent = size,
                                                .hudless_input = *hud_dimmed->view,
                                                .hudless_output = *slot->view};
        replay_hudless = slot.get();
        slot->fresh = true;
        scheduler.GetDynamicState().Invalidate();
    }

    void Composite(const Instance& instance, Runtime& runtime, Scheduler& scheduler,
                   VideoCore::TextureCache& cache, VideoCore::Image& target);
};

bool BbTemporalDlss::Impl::RunDlss(const Instance& instance, Runtime& runtime, Scheduler& scheduler,
                                   VideoCore::TextureCache& cache, BbVelocityMirror& mirror,
                                   VideoCore::Image& source, bool pre_hud,
                                   std::optional<vk::Extent2D> upscale_to) {
    const bool hdr = upscale_to.has_value();
    auto* ngx = instance.GetDlssNgx();
    const vk::Extent2D in{source.info.size.width, source.info.size.height};
    VideoCore::Image* depth =
        r32_depth ? cache.TryGetImage(r32_depth->id, r32_depth->uid) : nullptr;
    const auto reject = [&](const char* reason) {
        ReportFrame(std::string_view{reason} == "no upscaler"
                        ? NoUpscalerReason()
                        : "Waiting for gameplay (menus and loading screens use the normal image).");
        if (++fallbacks <= 8 || fallbacks % 600 == 0)
            LOG_INFO(Render_Vulkan, "[DLSS-TEMPORAL] Stock frame ({}), fallbacks={}", reason,
                     fallbacks);
        return false;
    };
    const auto out = upscale_to ? *upscale_to : OutputFor(in);
    const auto chosen = ChooseBackend(instance, scheduler, in, out);
    if (chosen == Backend::None)
        return reject("no upscaler");
    if (!source.backing || source.backing->image.image_ci.samples != vk::SampleCountFlagBits::e1 ||
        source.backing->image.image_ci.extent != vk::Extent3D{in.width, in.height, 1})
        return reject("unsupported color target");
    if (!depth || !depth->backing || depth->info.size.width != in.width ||
        depth->info.size.height != in.height ||
        depth->backing->image.image_ci.format != vk::Format::eR32Sfloat ||
        depth->backing->image.image_ci.samples != vk::SampleCountFlagBits::e1)
        return reject("scene depth not observed");
    if (!camera_constants || !coefficients)
        return reject("camera constants not observed");

    scheduler.EndRendering();
    EnsurePipelines(instance);
    // RenoDX widens the game's 8-bit targets to floating point and keeps values above 1 there;
    // the scene is copied with a shader into a float image so they survive.
    const bool renodx = RenoDxLoaded() && !hdr;
    const vk::Format color_format =
        renodx ? vk::Format::eR16G16B16A16Sfloat : source.info.pixel_format;
    if (!snapshot || snapshot->extent != in || snapshot->format != color_format || !upscaled ||
        upscaled->extent != out) {
        scheduler.Finish(); // owned images may still be in flight
        using U = vk::ImageUsageFlagBits;
        snapshot = std::make_unique<OwnedImage>(
            instance, color_format, in,
            (renodx ? U::eSampled | U::eStorage : U::eSampled | U::eTransferDst) |
                U::eTransferSrc); // transfer source for the diagnostic dump
        motion = std::make_unique<OwnedImage>(instance, vk::Format::eR16G16Sfloat, in,
                                              U::eSampled | U::eStorage | U::eTransferSrc);
        reference_motion = std::make_unique<OwnedImage>(instance, vk::Format::eR16G16Sfloat, in,
                                                        U::eSampled | U::eStorage);
        motion_error.reset();
        upscaled = std::make_unique<OwnedImage>(instance, vk::Format::eR16G16B16A16Sfloat, out,
                                                U::eSampled | U::eStorage | U::eTransferSrc);
        linear_color = std::make_unique<OwnedImage>(instance, vk::Format::eR16G16B16A16Sfloat, in,
                                                    U::eSampled | U::eStorage);
        sharpened = upscale_to
                        ? std::make_unique<OwnedImage>(instance, vk::Format::eR16G16B16A16Sfloat,
                                                       out, U::eStorage | U::eTransferSrc)
                        : nullptr;
        outputs.clear();
        history_valid = false;
        LOG_INFO(Render_Vulkan, "[DLSS-TEMPORAL] Resources {}x{} -> {}x{}", in.width, in.height,
                 out.width, out.height);
    }
    render = in;
    output = out;
    if (chosen != backend) {
        scheduler.Finish();
        if (backend == Backend::Dlss && ngx)
            ngx->ReleaseFeatureAfterGpuDrain();
        if (backend == Backend::Fsr && fsr)
            fsr->DestroyContextAfterGpuDrain();
        if (backend == Backend::Fsr4 && fsr4)
            fsr4->ReleaseContextAfterGpuDrain();
        backend = chosen;
        history_valid = false;
        LOG_INFO(Render_Vulkan, "[DLSS-TEMPORAL] Upscaler: {}", BackendName(backend));
    }
    if (backend == Backend::Dlss) {
        const DlssNgx::FeatureDesc desc{
            in.width, in.height, out.width, out.height,
            tune.quality >= 0 ? tune.quality
                              : DlssNgx::QualityForScale(float(out.width) / float(in.width)),
            tune.depth_inverted, // UID47/R32 depth measured near < far
            tune.preset,
            // With RenoDX DLSS gets the scene as linear 16-bit float (see `linear` below).
            hdr || RenoDxLoaded()};
        if (!ngx->HasFeature(desc)) {
            scheduler.Finish();
            if (!ngx->CreateFeature(scheduler.CommandBuffer(), desc)) {
                (upscale_to ? decoupled_unsupported : failed) = true;
                LOG_ERROR(Render_Vulkan,
                          "[DLSS-TEMPORAL] Feature creation failed; stock rendering");
                return false;
            }
            history_valid = false;
        }
    } else if (backend == Backend::Fsr4) {
        const Fsr4Addon::ContextDesc desc{in, out};
        if (!fsr4->HasContext(desc)) {
            ReportFrame("Initializing FSR 4...");
            const auto started = std::chrono::steady_clock::now();
            scheduler.Finish();
            if (!fsr4->CreateContext(desc)) {
                fsr4_rejected_size = desc;
                backend = Backend::None;
                ReportFrame("FSR 4 could not initialize; trying FSR 3.1.");
                LOG_WARNING(Render_Vulkan,
                            "[FSR4] Context {}x{} -> {}x{} unavailable; trying FSR 3.1", in.width,
                            in.height, out.width, out.height);
                return false;
            }
            LOG_INFO(Render_Vulkan, "[FSR4] Context ready in {:.0f} ms",
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                               started)
                         .count());
            fsr4_rejected_size.reset();
            history_valid = false;
        }
    } else {
        const FsrUpscaler::ContextDesc desc{in, out, tune.depth_inverted, hdr,
                                            hdr && tune.auto_exposure};
        if (!fsr->HasContext(desc)) {
            scheduler.Finish();
            if (!fsr->CreateContext(desc)) {
                (upscale_to ? decoupled_unsupported : failed) = true;
                LOG_ERROR(Render_Vulkan, "[FSR] Context creation failed; stock rendering");
                return false;
            }
            history_valid = false;
        }
    }

    // Only now: the Finish() calls above submit the command buffer, and recording into a
    // submitted one crashes AMD's driver.
    const auto command = scheduler.CommandBuffer();
    // Snapshot of the scene as DLSS color input (pre-HUD at the first Scaleform draw).
    VideoCore::ImageViewInfo source_info{};
    source_info.format = vk::Format::eR8G8B8A8Unorm;
    const auto& source_view = source.FindView(source_info);
    if (renodx)
        runtime.Transit(&source, vk::ImageLayout::eShaderReadOnlyOptimal,
                        vk::PipelineStageFlagBits2::eComputeShader,
                        vk::AccessFlagBits2::eShaderRead);
    else
        runtime.Transit(&source, vk::ImageLayout::eTransferSrcOptimal,
                        vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead);
    VideoCore::ImageViewInfo depth_info{};
    depth_info.format = vk::Format::eR32Sfloat;
    const auto& depth_view = depth->FindView(depth_info);
    runtime.Transit(depth, vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead);
    runtime.FlushBarriers();
    if (renodx) {
        snapshot->Transit(command, vk::ImageLayout::eGeneral,
                          vk::PipelineStageFlagBits2::eComputeShader,
                          vk::AccessFlagBits2::eShaderWrite);
        const std::array images{
            vk::DescriptorImageInfo{*nearest, *source_view.image_view,
                                    vk::ImageLayout::eShaderReadOnlyOptimal},
            vk::DescriptorImageInfo{{}, *snapshot->view, vk::ImageLayout::eGeneral}};
        const std::array writes{
            vk::WriteDescriptorSet{.dstBinding = 0,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                   .pImageInfo = &images[0]},
            vk::WriteDescriptorSet{.dstBinding = 1,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eStorageImage,
                                   .pImageInfo = &images[1]}};
        const u32 copy_flags = 2; // plain copy
        command.bindPipeline(vk::PipelineBindPoint::eCompute, *linearize_pass->pipeline);
        command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *linearize_pass->layout, 0,
                                     writes);
        command.pushConstants(*linearize_pass->layout, vk::ShaderStageFlagBits::eCompute, 0,
                              sizeof(copy_flags), &copy_flags);
        command.dispatch((in.width + 7) / 8, (in.height + 7) / 8, 1);
    } else {
        snapshot->Transit(command, vk::ImageLayout::eTransferDstOptimal,
                          vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite);
        const vk::ImageCopy region{
            .srcSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1},
            .dstSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1},
            .extent = {in.width, in.height, 1}};
        command.copyImage(source.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                          snapshot->image.image, vk::ImageLayout::eTransferDstOptimal, region);
    }
    snapshot->Transit(command, vk::ImageLayout::eShaderReadOnlyOptimal,
                      vk::PipelineStageFlagBits2::eComputeShader |
                          vk::PipelineStageFlagBits2::eFragmentShader,
                      vk::AccessFlagBits2::eShaderRead);

    // Motion: camera reprojection, replaced by object velocity where the mirror covers.
    SelectCamera(in);
    const auto mirror_frame = mirror.ConsumeFrame(command);
    mirror.SetTargetSize(in);
    const bool use_mirror =
        tune.mirror && velocity_constants && mirror_frame && mirror_frame->extent == in;
    const vk::BufferMemoryBarrier2 constants_barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = coefficients->Handle(),
        .offset = 0,
        .size = CoefficientBytes};
    command.pipelineBarrier2(vk::DependencyInfo{.bufferMemoryBarrierCount = 1,
                                                .pBufferMemoryBarriers = &constants_barrier});
    motion->Transit(command, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eComputeShader,
                    vk::AccessFlagBits2::eShaderWrite);
    reference_motion->Transit(command, vk::ImageLayout::eGeneral,
                              vk::PipelineStageFlagBits2::eComputeShader,
                              vk::AccessFlagBits2::eShaderWrite);
    // Diagnostic: per 32x32 cell, how far the scene-camera motion is from the game's own.
    error_grid_w = (in.width + 31) / 32;
    const u64 error_bytes = u64(error_grid_w) * ((in.height + 31) / 32) * 2 * sizeof(u32);
    if (!motion_error || motion_error->SizeBytes() != error_bytes)
        motion_error = std::make_unique<VideoCore::Buffer>(
            instance, 0, error_bytes, VideoCore::MemoryType::DeviceLocal, "motion error");
    {
        vk::BufferMemoryBarrier2 barrier{.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                                         .srcAccessMask = vk::AccessFlagBits2::eMemoryRead |
                                                          vk::AccessFlagBits2::eMemoryWrite,
                                         .dstStageMask = vk::PipelineStageFlagBits2::eClear,
                                         .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
                                         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                         .buffer = motion_error->Handle(),
                                         .offset = 0,
                                         .size = VK_WHOLE_SIZE};
        command.pipelineBarrier2(
            vk::DependencyInfo{.bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &barrier});
        command.fillBuffer(motion_error->Handle(), 0, error_bytes, 0);
        barrier.srcStageMask = vk::PipelineStageFlagBits2::eClear;
        barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
        barrier.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader;
        barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
        command.pipelineBarrier2(
            vk::DependencyInfo{.bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &barrier});
    }
    const bool jittered = tune.jitter && jittered_draws > 0;
    const std::array<float, 2> applied = jittered ? jitter : std::array<float, 2>{};
    {
        const std::array images{
            vk::DescriptorImageInfo{*nearest, *depth_view.image_view,
                                    vk::ImageLayout::eShaderReadOnlyOptimal},
            vk::DescriptorImageInfo{*nearest, use_mirror ? mirror_frame->view : *snapshot->view,
                                    vk::ImageLayout::eShaderReadOnlyOptimal},
            vk::DescriptorImageInfo{{}, *motion->view, vk::ImageLayout::eGeneral},
            vk::DescriptorImageInfo{{}, *reference_motion->view, vk::ImageLayout::eGeneral}};
        const vk::DescriptorBufferInfo buffer{coefficients->Handle(), 0, CoefficientBytes};
        const vk::DescriptorBufferInfo error_buffer{motion_error->Handle(), 0, VK_WHOLE_SIZE};
        const std::array writes{
            vk::WriteDescriptorSet{.dstBinding = 0,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                   .pImageInfo = &images[0]},
            vk::WriteDescriptorSet{.dstBinding = 1,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                   .pImageInfo = &images[1]},
            vk::WriteDescriptorSet{.dstBinding = 2,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eStorageImage,
                                   .pImageInfo = &images[2]},
            vk::WriteDescriptorSet{.dstBinding = 3,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eStorageBuffer,
                                   .pBufferInfo = &buffer},
            vk::WriteDescriptorSet{.dstBinding = 4,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eStorageImage,
                                   .pImageInfo = &images[3]},
            vk::WriteDescriptorSet{.dstBinding = 5,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eStorageBuffer,
                                   .pBufferInfo = &error_buffer}};
        struct {
            float reproject[12];
            float proj[4], prev_proj[4];
            float jitter[2];
            u32 use_mirror, flags;
            float mv_scale[2];
            u32 err_grid_w, err_enabled;
        } push{};
        const bool scene_camera = tune.scene_camera && frame_has_camera && camera.valid &&
                                  previous_camera.valid && camera_trusted;
        if (scene_camera) {
            // previous view * current inverse view, both affine 3x4, in double precision.
            const auto& a = previous_camera.view;
            const auto& b = camera.inv_view;
            for (int r = 0; r < 3; ++r)
                for (int col = 0; col < 4; ++col) {
                    double value = a[r * 4 + 0] * b[0 * 4 + col] + a[r * 4 + 1] * b[1 * 4 + col] +
                                   a[r * 4 + 2] * b[2 * 4 + col];
                    if (col == 3)
                        value += a[r * 4 + 3];
                    push.reproject[r * 4 + col] = float(value);
                }
            std::copy(camera.proj.begin(), camera.proj.end(), push.proj);
            std::copy(previous_camera.proj.begin(), previous_camera.proj.end(), push.prev_proj);
            ++scene_camera_frames;
        }
        push.jitter[0] = applied[0];
        push.jitter[1] = applied[1];
        push.use_mirror = use_mirror ? 1u : 0u;
        push.flags = (tune.camera_snap ? 1u : 0u) | (scene_camera ? 2u : 0u);
        push.mv_scale[0] = tune.exp_mv_x;
        push.mv_scale[1] = tune.exp_mv_y;
        push.err_grid_w = error_grid_w;
        push.err_enabled = tune.meter ? 1u : 0u;
        std::memcpy(motion_constants.data(), &push, sizeof(motion_constants));
        motion_flags = push.flags | (push.use_mirror << 8);
        command.bindPipeline(vk::PipelineBindPoint::eCompute, *motion_pass->pipeline);
        command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *motion_pass->layout, 0,
                                     writes);
        command.pushConstants(*motion_pass->layout, vk::ShaderStageFlagBits::eCompute, 0,
                              sizeof(push), &push);
        command.dispatch((in.width + 7) / 8, (in.height + 7) / 8, 1);
    }
    runtime.AccessBuffer(coefficients.get(), 0, CoefficientBytes,
                         vk::PipelineStageFlagBits2::eComputeShader,
                         vk::AccessFlagBits2::eShaderRead);
    motion->Transit(command, vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::PipelineStageFlagBits2::eComputeShader |
                        vk::PipelineStageFlagBits2::eFragmentShader,
                    vk::AccessFlagBits2::eShaderRead);
    reference_motion->Transit(command, vk::ImageLayout::eShaderReadOnlyOptimal,
                              vk::PipelineStageFlagBits2::eComputeShader,
                              vk::AccessFlagBits2::eShaderRead);
    upscaled->Transit(command, vk::ImageLayout::eGeneral,
                      vk::PipelineStageFlagBits2::eComputeShader,
                      vk::AccessFlagBits2::eShaderWrite);

    // FSR 4 treats its colour as linear light and has no flag for display-encoded input, so
    // it gets the scene decoded to linear; the composite encodes its output again.
    // With RenoDX loaded DLSS outputs black for the 8-bit display-encoded scene (it works with
    // the 16-bit HDR scene of the Decoupled UI patches), so it gets the scene the same way.
    // With RenoDX every upscaler gets it, reversibly tone mapped so highlights above 1 survive.
    const bool linear = (backend == Backend::Fsr4 && !hdr && tune.fsr4_linear) || renodx;
    if (linear) {
        linear_color->Transit(command, vk::ImageLayout::eGeneral,
                              vk::PipelineStageFlagBits2::eComputeShader,
                              vk::AccessFlagBits2::eShaderWrite);
        const std::array images{
            vk::DescriptorImageInfo{*nearest, *snapshot->view,
                                    vk::ImageLayout::eShaderReadOnlyOptimal},
            vk::DescriptorImageInfo{{}, *linear_color->view, vk::ImageLayout::eGeneral}};
        const std::array writes{
            vk::WriteDescriptorSet{.dstBinding = 0,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                   .pImageInfo = &images[0]},
            vk::WriteDescriptorSet{.dstBinding = 1,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eStorageImage,
                                   .pImageInfo = &images[1]}};
        const u32 linear_flags = renodx ? 1u : 0u;
        command.bindPipeline(vk::PipelineBindPoint::eCompute, *linearize_pass->pipeline);
        command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *linearize_pass->layout, 0,
                                     writes);
        command.pushConstants(*linearize_pass->layout, vk::ShaderStageFlagBits::eCompute, 0,
                              sizeof(linear_flags), &linear_flags);
        command.dispatch((in.width + 7) / 8, (in.height + 7) / 8, 1);
        linear_color->Transit(command, vk::ImageLayout::eShaderReadOnlyOptimal,
                              vk::PipelineStageFlagBits2::eComputeShader,
                              vk::AccessFlagBits2::eShaderRead);
    }
    if (linear != upscaled_linear || renodx != upscaled_tone_mapped)
        history_valid = false; // the history is in the other encoding
    upscaled_linear = linear;
    upscaled_tone_mapped = renodx;
    const DlssNgx::Resource depth_resource{depth->backing->image.image, *depth_view.image_view,
                                           Range, vk::Format::eR32Sfloat, in};
    // Diagnostic: the upscaler sees a mirrored world (and its output is mirrored back).
    const u32 flip = u32(tune.exp_mirror);
    if (int(flip) != mirror_mode_used) {
        history_valid = false;
        mirror_mode_used = int(flip);
        LOG_INFO(Render_Vulkan, "[DLSS-EXP] Mirror mode {}", flip);
    }
    const bool reset = !history_valid || tune.exp_reset;
    float jitter_x = (tune.swap ? applied[1] : applied[0]) * tune.sign_x;
    float jitter_y = (tune.swap ? applied[0] : applied[1]) * tune.sign_y;
    auto color_input = (linear ? linear_color : snapshot)->Resource();
    auto depth_input = depth_resource;
    auto motion_input = motion->Resource();
    auto output_target = upscaled->Resource();
    if (flip != 0) {
        using U = vk::ImageUsageFlagBits;
        const auto make = [&](std::unique_ptr<OwnedImage>& image, vk::Format format,
                              vk::Extent2D size) {
            if (!image || image->extent != size)
                image = std::make_unique<OwnedImage>(instance, format, size,
                                                     U::eSampled | U::eStorage);
        };
        make(mirror_color, vk::Format::eR16G16B16A16Sfloat, in);
        make(mirror_depth, vk::Format::eR32Sfloat, in);
        make(mirror_motion, vk::Format::eR16G16Sfloat, in);
        make(mirror_output, vk::Format::eR16G16B16A16Sfloat, out);
        const float sx = (flip & 1u) ? -1.f : 1.f, sy = (flip & 2u) ? -1.f : 1.f;
        MirrorInto(command, *mirror_rgba_pass, color_input.view, *mirror_color, {1, 1, 1, 1}, flip);
        MirrorInto(command, *mirror_r32_pass, *depth_view.image_view, *mirror_depth, {1, 1, 1, 1},
                   flip);
        MirrorInto(command, *mirror_rg16_pass, *motion->view, *mirror_motion, {sx, sy, 1, 1}, flip);
        mirror_output->Transit(command, vk::ImageLayout::eGeneral,
                               vk::PipelineStageFlagBits2::eComputeShader,
                               vk::AccessFlagBits2::eShaderWrite);
        color_input = mirror_color->Resource();
        depth_input = mirror_depth->Resource();
        motion_input = mirror_motion->Resource();
        output_target = mirror_output->Resource();
        jitter_x *= sx;
        jitter_y *= sy;
    }
    const DlssNgx::EvalDesc eval{jitter_x, jitter_y, reset, frame_ms};
    const bool success =
        backend == Backend::Dlss
            ? ngx->Evaluate(command, color_input, depth_input, motion_input, output_target, eval)
        : backend == Backend::Fsr4
            ? fsr4->Evaluate(command, color_input, depth_input, motion_input, output_target, eval,
                             FsrCamera(), hdr ? (tune.auto_exposure ? 1 : 2) : 0)
            : fsr->Evaluate(command, color_input, depth_input, motion_input, output_target, eval,
                            FsrCamera());
    scheduler.GetDynamicState().Invalidate(); // the upscaler records its own Vulkan state
    if (success && flip != 0) {
        mirror_output->Transit(command, vk::ImageLayout::eShaderReadOnlyOptimal,
                               vk::PipelineStageFlagBits2::eComputeShader,
                               vk::AccessFlagBits2::eShaderRead);
        MirrorInto(command, *mirror_rgba_pass, *mirror_output->view, *upscaled, {1, 1, 1, 1}, flip);
        upscaled->Transit(command, vk::ImageLayout::eGeneral,
                          vk::PipelineStageFlagBits2::eComputeShader,
                          vk::AccessFlagBits2::eShaderWrite);
    }
    if (!success) {
        (upscale_to ? decoupled_unsupported : failed) = true;
        LOG_ERROR(Render_Vulkan, "[DLSS-TEMPORAL] Evaluation failed; stock rendering");
        return false;
    }
    if (upscale_to && tune.sharpness > 0 && sharpened) {
        // Sharpened into its own image: the upscaler keeps the unsharpened one as history.
        upscaled->Transit(command, vk::ImageLayout::eShaderReadOnlyOptimal,
                          vk::PipelineStageFlagBits2::eComputeShader,
                          vk::AccessFlagBits2::eShaderRead);
        sharpened->Transit(command, vk::ImageLayout::eGeneral,
                           vk::PipelineStageFlagBits2::eComputeShader,
                           vk::AccessFlagBits2::eShaderWrite);
        const std::array images{
            vk::DescriptorImageInfo{*nearest, *upscaled->view,
                                    vk::ImageLayout::eShaderReadOnlyOptimal},
            vk::DescriptorImageInfo{{}, *sharpened->view, vk::ImageLayout::eGeneral}};
        const std::array writes{
            vk::WriteDescriptorSet{.dstBinding = 0,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                   .pImageInfo = &images[0]},
            vk::WriteDescriptorSet{.dstBinding = 1,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eStorageImage,
                                   .pImageInfo = &images[1]}};
        command.bindPipeline(vk::PipelineBindPoint::eCompute, *sharpen_pass->pipeline);
        command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *sharpen_pass->layout, 0,
                                     writes);
        command.pushConstants(*sharpen_pass->layout, vk::ShaderStageFlagBits::eCompute, 0,
                              sizeof(tune.sharpness), &tune.sharpness);
        command.dispatch((out.width + 7) / 8, (out.height + 7) / 8, 1);
        sharpened->Transit(command, vk::ImageLayout::eTransferSrcOptimal,
                           vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead);
        replacement = DrawReplacement{.owned = sharpened->image.image};
        decoupled_frame = true;
    } else if (upscale_to) {
        // Replaces the game's own upscale; the game then post-processes and draws its UI.
        upscaled->Transit(command, vk::ImageLayout::eTransferSrcOptimal,
                          vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead);
        replacement = DrawReplacement{.owned = upscaled->image.image};
        decoupled_frame = true;
    } else {
        upscaled->Transit(command, vk::ImageLayout::eShaderReadOnlyOptimal,
                          vk::PipelineStageFlagBits2::eComputeShader,
                          vk::AccessFlagBits2::eShaderRead);
        dlss_ready = true;
    }
    if (tune.meter) {
        try {
            RunMeter(instance, command, in, out);
        } catch (const std::exception& e) {
            tune.meter = false;
            LOG_ERROR(Render_Vulkan, "[DLSS-METER] Stopped: {}", e.what());
        }
    } else if (meter_csv.is_open()) {
        FlushMeterStep();
        meter_csv.flush();
    }
    CaptureFrameGen(instance, runtime, scheduler, command, *depth, in, eval);
    if (dump_left > 0) {
        try {
            DumpFrame(instance, runtime, scheduler, command, *depth, in, out, eval, applied,
                      linear);
        } catch (const std::exception& e) {
            dump_left = 0;
            LOG_ERROR(Render_Vulkan, "[DLSS-DUMP] Stopped: {}", e.what());
        }
    }
    dlss_pre_hud = pre_hud;
    evaluated_since_copy = true;
    history_valid = true;
    {
        std::scoped_lock lock{status_mutex};
        last_evaluation = std::chrono::steady_clock::now();
        status.render_width = in.width;
        status.render_height = in.height;
        status.output_width = out.width;
        status.output_height = out.height;
        status.backend = BackendName(backend);
        status.decoupled = upscale_to.has_value();
        status.reason.clear();
    }
    if (++evaluations <= 3 || evaluations % 1800 == 0)
        LOG_INFO(Render_Vulkan,
                 "[DLSS-TEMPORAL] {} evaluations={} {}x{}->{}x{} preHUD={} "
                 "jitter=({:.3f},{:.3f}) "
                 "jitteredDraws={} (before display copy {}) objectMotion={} reset={} fallbacks={} "
                 "sceneCameraFrames={} cameraRejects={} cameraSwitches={} cameraUnchecked={}",
                 BackendName(backend), evaluations, in.width, in.height, out.width, out.height,
                 pre_hud, applied[0], applied[1], jittered_draws, jittered_before_copy, use_mirror,
                 reset, fallbacks, scene_camera_frames, camera_rejects, camera_switches,
                 camera_unchecked);
    return true;
}

void BbTemporalDlss::Impl::Composite(const Instance& instance, Runtime& runtime,
                                     Scheduler& scheduler, VideoCore::TextureCache& cache,
                                     VideoCore::Image& target) {
    auto& outputs_slot = outputs[target.info.guest_address];
    auto& source = cache.GetImage(copy_source->id);
    auto& lut = cache.GetImage(copy_lut->id);
    if (source.info.size.width != render.width || source.info.size.height != render.height) {
        if (outputs_slot)
            outputs_slot->fresh = false;
        return;
    }
    scheduler.EndRendering();
    // With RenoDX the game's display copy (RenoDX's HDR version of it) is drawn again at the
    // output size into an image like the game's target; the composite stops before the LUT.
    const bool replay = RenoDxLoaded();
    // RenoDX's display copy writes values above 1: the image it is redrawn into is float.
    const auto output_format =
        replay ? vk::Format::eR16G16B16A16Sfloat : vk::Format::eR8G8B8A8Unorm;
    const auto pre_lut_format =
        replay ? vk::Format::eR16G16B16A16Sfloat : vk::Format::eR8G8B8A8Unorm;
    if (!outputs_slot || outputs_slot->extent != output || outputs_slot->format != output_format ||
        (replay && (!pre_lut || pre_lut->extent != output))) {
        if (outputs_slot || pre_lut)
            scheduler.Finish();
        using U = vk::ImageUsageFlagBits;
        // Transfer source: the diagnostic dump copies the composited frame.
        outputs_slot = std::make_unique<OwnedImage>(
            instance, output_format, output,
            (replay ? U::eSampled | U::eColorAttachment : U::eSampled | U::eStorage) |
                U::eTransferSrc,
            true);
        if (replay)
            pre_lut = std::make_unique<OwnedImage>(instance, pre_lut_format, output,
                                                   U::eSampled | U::eStorage | U::eTransferSrc);
    }
    // FSR frame generation moves whatever differs from its HUD-less image as UI instead of
    // interpolating it, so it gets the frame composited a second time without the HUD.
    const bool hudless = frame_gen_pending && dlss_pre_hud && tune.hud &&
                         std::string_view{FrameGen::BackendName()} == "FSR";
    if (hudless && (!outputs_slot->hudless || outputs_slot->hudless->extent != output ||
                    outputs_slot->hudless->format != output_format ||
                    (replay && (!pre_lut_hudless || pre_lut_hudless->extent != output)))) {
        scheduler.Finish();
        using U = vk::ImageUsageFlagBits;
        outputs_slot->hudless = std::make_unique<OwnedImage>(
            instance, output_format, output,
            replay ? U::eSampled | U::eColorAttachment : U::eSampled | U::eStorage, true);
        if (replay)
            pre_lut_hudless = std::make_unique<OwnedImage>(instance, pre_lut_format, output,
                                                           U::eSampled | U::eStorage);
    }
    auto& written = replay ? *pre_lut : *outputs_slot;
    auto* written_hudless = !hudless ? nullptr
                            : replay ? pre_lut_hudless.get()
                                     : outputs_slot->hudless.get();
    const auto command = scheduler.CommandBuffer(); // after Finish(), see RunDlss
    VideoCore::ImageViewInfo source_info{};
    source_info.format = vk::Format::eR8G8B8A8Unorm;
    const auto& source_view = source.FindView(source_info);
    const auto& lut_view = lut.FindView(copy_lut->view);
    runtime.Transit(&source, vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::PipelineStageFlagBits2::eComputeShader |
                        vk::PipelineStageFlagBits2::eFragmentShader,
                    vk::AccessFlagBits2::eShaderRead);
    runtime.Transit(&lut, vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::PipelineStageFlagBits2::eComputeShader |
                        vk::PipelineStageFlagBits2::eFragmentShader,
                    vk::AccessFlagBits2::eShaderRead);
    runtime.FlushBarriers();
    written.Transit(command, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eComputeShader,
                    vk::AccessFlagBits2::eShaderWrite);
    if (written_hudless)
        written_hudless->Transit(command, vk::ImageLayout::eGeneral,
                                 vk::PipelineStageFlagBits2::eComputeShader,
                                 vk::AccessFlagBits2::eShaderWrite);
    const auto ro = vk::ImageLayout::eShaderReadOnlyOptimal;
    if (!coverage)
        coverage = std::make_unique<VideoCore::Buffer>(instance, 0, CoverageSlots * CoverageStride,
                                                       VideoCore::MemoryType::HostCached);
    // The slot written three composites ago holds a finished count by now.
    const u64 slot = (composites % CoverageSlots) * CoverageStride;
    if (composites >= CoverageSlots) {
        u32 changed{};
        std::memcpy(&changed, coverage->mapped_data.data() + slot, sizeof(changed));
        const double share = double(changed) / (double(output.width) * output.height);
        const bool was_menu = menu;
        menu = tune.menu_fix && (menu ? share > 0.25 : share > 0.40);
        if (menu != was_menu)
            LOG_INFO(Render_Vulkan, "[DLSS-TEMPORAL] Menu {} (HUD covers {:.0f}%)",
                     menu ? "open" : "closed", share * 100.0);
    }
    if (!dimming)
        dimming = std::make_unique<VideoCore::Buffer>(instance, 0, DimmingBytes,
                                                      VideoCore::MemoryType::DeviceLocal);
    command.fillBuffer(coverage->Handle(), slot, sizeof(u32), 0);
    command.fillBuffer(dimming->Handle(), 0, DimmingBytes, 0);
    const vk::BufferMemoryBarrier2 coverage_barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = coverage->Handle(),
        .offset = slot,
        .size = sizeof(u32)};
    auto dimming_barrier = coverage_barrier;
    dimming_barrier.buffer = dimming->Handle();
    dimming_barrier.offset = 0;
    dimming_barrier.size = DimmingBytes;
    const std::array clear_barriers{coverage_barrier, dimming_barrier};
    command.pipelineBarrier2(vk::DependencyInfo{.bufferMemoryBarrierCount = 2,
                                                .pBufferMemoryBarriers = clear_barriers.data()});
    const std::array images{vk::DescriptorImageInfo{*nearest, *upscaled->view, ro},
                            vk::DescriptorImageInfo{*linear, *source_view.image_view, ro},
                            vk::DescriptorImageInfo{*linear, *snapshot->view, ro},
                            vk::DescriptorImageInfo{*linear, *lut_view.image_view, ro},
                            vk::DescriptorImageInfo{{}, *written.view, vk::ImageLayout::eGeneral},
                            vk::DescriptorImageInfo{*linear, *motion->view, ro}};
    std::array<vk::WriteDescriptorSet, 8> writes;
    for (u32 i = 0; i < images.size(); ++i)
        writes[i] = {.dstBinding = i,
                     .descriptorCount = 1,
                     .descriptorType = i == 4 ? vk::DescriptorType::eStorageImage
                                              : vk::DescriptorType::eCombinedImageSampler,
                     .pImageInfo = &images[i]};
    const vk::DescriptorBufferInfo coverage_info{coverage->Handle(), slot, sizeof(u32)};
    writes[6] = {.dstBinding = 6,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageBuffer,
                 .pBufferInfo = &coverage_info};
    const vk::DescriptorBufferInfo dimming_info{dimming->Handle(), 0, DimmingBytes};
    writes[7] = {.dstBinding = 7,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageBuffer,
                 .pBufferInfo = &dimming_info};
    const u32 flags = (dlss_pre_hud && tune.hud ? 1u : 0u) | (replay ? 32u : 2u) |
                      (tune.debug_motion ? 4u : 0u) | (menu && dlss_pre_hud ? 8u : 0u) |
                      (upscaled_linear ? 16u : 0u) | (upscaled_tone_mapped ? 128u : 0u);
    const auto& pass = replay ? *composite_hdr_pass : *composite_pass;
    command.bindPipeline(vk::PipelineBindPoint::eCompute, *pass.pipeline);
    command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *pass.layout, 0, writes);
    struct {
        u32 flags;
        float sharpness;
    } push{flags, tune.sharpness};
    command.pushConstants(*pass.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
    command.dispatch((output.width + 7) / 8, (output.height + 7) / 8, 1);
    if (written_hudless) {
        // The HUD-less pass reads the dimming histogram the main pass just wrote.
        auto histogram_barrier = dimming_barrier;
        histogram_barrier.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
        histogram_barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
        command.pipelineBarrier2(vk::DependencyInfo{.bufferMemoryBarrierCount = 1,
                                                    .pBufferMemoryBarriers = &histogram_barrier});
        const vk::DescriptorImageInfo hudless_image{
            {}, *written_hudless->view, vk::ImageLayout::eGeneral};
        writes[4].pImageInfo = &hudless_image;
        command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *pass.layout, 0, writes);
        push.flags = (flags & ~1u) | 64u; // keeps 8: in menus, the game's own scene
        command.pushConstants(*pass.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push),
                              &push);
        command.dispatch((output.width + 7) / 8, (output.height + 7) / 8, 1);
    }
    if (auto stem = std::exchange(dump_composite, std::nullopt)) {
        try {
            DumpComposite(instance, runtime, scheduler, command, source, written, *stem, flags);
        } catch (const std::exception& e) {
            LOG_ERROR(Render_Vulkan, "[DLSS-DUMP] Composite not dumped: {}", e.what());
        }
    }
    outputs_slot->hudless_fresh = written_hudless != nullptr;
    if (replay) {
        pre_lut->Transit(command, ro, vk::PipelineStageFlagBits2::eFragmentShader,
                         vk::AccessFlagBits2::eShaderRead);
        outputs_slot->Transit(command, vk::ImageLayout::eColorAttachmentOptimal,
                              vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                              vk::AccessFlagBits2::eColorAttachmentWrite);
        display_copy_replay =
            DisplayCopyReplay{DisplayCopy, *pre_lut->view, *outputs_slot->view, output};
        if (written_hudless) {
            pre_lut_hudless->Transit(command, ro, vk::PipelineStageFlagBits2::eFragmentShader,
                                     vk::AccessFlagBits2::eShaderRead);
            outputs_slot->hudless->Transit(command, vk::ImageLayout::eColorAttachmentOptimal,
                                           vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                           vk::AccessFlagBits2::eColorAttachmentWrite);
            display_copy_replay->hudless_input = *pre_lut_hudless->view;
            display_copy_replay->hudless_output = *outputs_slot->hudless->view;
            outputs_slot->hudless->swapped = true;
            replay_hudless = outputs_slot->hudless.get();
        }
        replay_output = outputs_slot.get();
        outputs_slot->swapped = true;
    } else {
        outputs_slot->Transit(command, ro, vk::PipelineStageFlagBits2::eFragmentShader,
                              vk::AccessFlagBits2::eShaderRead);
        if (written_hudless)
            written_hudless->Transit(command, ro, vk::PipelineStageFlagBits2::eFragmentShader,
                                     vk::AccessFlagBits2::eShaderRead);
    }
    scheduler.GetDynamicState().Invalidate();
    outputs_slot->fresh = true;
    outputs_slot->frame_gen = std::exchange(frame_gen_pending, std::nullopt);
    if (++composites <= 3 || composites % 1800 == 0)
        LOG_INFO(Render_Vulkan, "[DLSS-TEMPORAL] Display composites={} VideoOut={:#x} {}x{}",
                 composites, target.info.guest_address, output.width, output.height);
}

BbTemporalDlss::BbTemporalDlss() : impl{std::make_unique<Impl>()} {}
BbTemporalDlss::~BbTemporalDlss() = default;

bool BbTemporalDlss::Requested() const {
    return impl->requested;
}

void BbTemporalDlss::ObserveTexture(const VideoCore::Image& image, VideoCore::ImageId id,
                                    const VideoCore::ImageViewInfo& view, u64 hash, u32 slot) {
    if (!impl->requested || impl->failed || impl->stopped)
        return;
    if (hash == DepthProducer && slot == 0 && image.info.pixel_format == vk::Format::eR32Sfloat &&
        ++impl->producer_depths)
        impl->r32_depth = Impl::Tracked{id, image.image_uid};
    else if (hash == DisplayCopy && slot == 0)
        impl->copy_source = Impl::Bound{id, view};
    else if (hash == DisplayCopy && slot == 1)
        impl->copy_lut = Impl::Bound{id, view};
    else if (hash == MotionBlur && slot == 0)
        impl->blur_source = Impl::Bound{id, view};
    else if (hash == HdrSceneCopy && slot == 0)
        impl->hdr_copy_source = Impl::Bound{id, view};
}

void BbTemporalDlss::ObserveSceneConstants(const void* data, u64 size, u64 hash, u32 binding) {
    auto& s = *impl;
    if (!s.requested || s.failed || s.stopped || !data)
        return;
    const auto* f = static_cast<const float*>(data);
    if (hash == DepthProducer && binding == 0 && !s.blur_rows_valid &&
        size >= (0x380 + 16) * sizeof(float) && size >= (0x26d + 1) * sizeof(float)) {
        std::copy_n(f + 0x380, 16, s.blur_rows.begin());
        s.blur_depth = {f[0], f[1], f[0x26c], f[0x26d]};
        s.blur_rows_valid = true;
    }
    if (size < 192 * sizeof(float))
        return;
    if (f[0] != 3000.0f || std::abs(f[1] * f[0] - 1.0f) > 1e-3f || f[4] < 64.0f || f[5] < 64.0f)
        return;
    // Every distinct camera of the frame; the main one is picked at the evaluation.
    Impl::Camera c;
    for (int i = 0; i < 12; ++i) {
        c.view[i] = f[8 + i];
        c.inv_view[i] = f[180 + i];
    }
    c.proj = {f[52], f[57], f[62], f[63]};
    c.valid = f[52] != 0.0f && f[57] != 0.0f && std::isfinite(f[62]) && std::isfinite(f[63]);
    if (!c.valid)
        return;
    for (u32 i = 0; i < s.candidate_count; ++i)
        if (s.candidates[i].view == c.view && s.candidates[i].proj == c.proj)
            return;
    if (s.candidate_count < s.candidates.size())
        s.candidates[s.candidate_count++] = c;
    s.frame_has_camera = true;
}

void BbTemporalDlss::ObserveConstants(Runtime& runtime, const Instance& instance,
                                      const VideoCore::Buffer* source, u64 offset, u64 size,
                                      u64 hash, u32 binding) {
    if (!impl->requested || impl->failed || impl->stopped || !source)
        return;
    impl->producer_constants += hash == DepthProducer && binding == 0;
    impl->velocity_draw_constants +=
        (hash == OpaqueVelocity && binding == 0) ||
        ((hash == AlphaVelocityA || hash == AlphaVelocityB) && binding == 1);
    const bool camera = hash == DepthProducer && binding == 0 && !impl->camera_constants;
    const bool velocity = !impl->velocity_constants &&
                          ((hash == OpaqueVelocity && binding == 0) ||
                           ((hash == AlphaVelocityA || hash == AlphaVelocityB) && binding == 1));
    if (!camera && !velocity)
        return;
    const u64 bytes = camera ? CameraBytes : 8;
    if (size < bytes || offset > source->SizeBytes() || bytes > source->SizeBytes() - offset)
        return;
    try {
        if (!impl->coefficients)
            impl->coefficients = std::make_unique<VideoCore::Buffer>(
                instance, 0, CoefficientBytes, VideoCore::MemoryType::DeviceLocal);
        const vk::BufferCopy copy{
            .srcOffset = offset, .dstOffset = camera ? 0 : VelocityOffset, .size = bytes};
        runtime.CopyBuffer(source, impl->coefficients.get(), std::span{&copy, 1});
        (camera ? impl->camera_constants : impl->velocity_constants) = true;
    } catch (const std::exception& e) {
        impl->failed = true;
        LOG_ERROR(Render_Vulkan, "[DLSS-TEMPORAL] Disabled: {}", e.what());
    }
}

std::array<float, 2> BbTemporalDlss::OnDraw(const Instance& instance, Runtime& runtime,
                                            Scheduler& scheduler, VideoCore::TextureCache& cache,
                                            BbVelocityMirror& mirror, const DrawInfo& draw) {
    auto& s = *impl;
    if (!s.requested || s.failed || s.stopped)
        return {};
    auto* color = draw.color ? &cache.GetImage(draw.color) : nullptr;
    try {
        if (draw.ps_hash == MotionBlur) {
            const auto source = std::exchange(s.blur_source, std::nullopt);
            if (s.tune.motion_blur || !color || !source || source->id == draw.color)
                return {};
            const auto& image = cache.GetImage(source->id);
            if (image.info.size.width != color->info.size.width ||
                image.info.size.height != color->info.size.height ||
                image.info.pixel_format != color->info.pixel_format)
                return {};
            s.replacement = DrawReplacement{.image = source->id};
            if (++s.blur_skips <= 2)
                LOG_INFO(Render_Vulkan, "[DLSS-TEMPORAL] Motion blur pass skipped");
            return {};
        }
        if (draw.ps_hash == HdrSceneCopy && color) {
            const auto source = std::exchange(s.hdr_copy_source, std::nullopt);
            if (!source || source->id == draw.color)
                return {};
            auto& scene = cache.GetImage(source->id);
            const vk::Extent2D target{color->info.size.width, color->info.size.height};
            // Same size: the normal copy before post-processing. Smaller: a Decoupled UI patch,
            // and this draw is the game's upscale of the scene to its UI resolution.
            if (scene.info.size.width == target.width && scene.info.size.height == target.height)
                return {};
            // Upscales of up to 3x only: the variants that render above the UI resolution, or
            // far below it, keep the game's own scaling.
            const bool supported = scene.info.size.width < target.width &&
                                   scene.info.size.height < target.height &&
                                   target.width <= 3 * scene.info.size.width;
            if (!supported || s.decoupled_unsupported) {
                if (!s.decoupled_unsupported)
                    LOG_INFO(Render_Vulkan,
                             "[DLSS-TEMPORAL] Decoupled UI {}x{} -> {}x{} is not upscaled",
                             scene.info.size.width, scene.info.size.height, target.width,
                             target.height);
                s.decoupled_unsupported = true;
                ReportFrame("This Decoupled UI patch is not supported: use one that renders "
                            "between 640x360 and 1600x900, or a regular Resolution Patch.");
                return {};
            }
            if (!s.decoupled)
                LOG_INFO(Render_Vulkan, "[DLSS-TEMPORAL] Decoupled UI: scene {}x{}, UI {}x{}",
                         scene.info.size.width, scene.info.size.height, target.width,
                         target.height);
            s.decoupled = true;
            s.decoupled_scene = source->id;
            if (s.tune.enabled && color->info.pixel_format == vk::Format::eR16G16B16A16Sfloat &&
                scene.info.pixel_format == vk::Format::eR16G16B16A16Sfloat)
                s.RunDlss(instance, runtime, scheduler, cache, mirror, scene, false, target);
            s.FrameBoundary(scheduler, mirror);
            return {};
        }
        if (draw.ps_hash == DisplayCopy && color && s.copy_source && s.copy_lut) {
            // In decoupled-UI mode the scene was upscaled before post-processing already.
            if (s.tune.enabled && !s.dlss_ready && !s.decoupled && !s.decoupled_unsupported) {
                // No Scaleform draw this frame: upscale the finished frame as a whole.
                s.RunDlss(instance, runtime, scheduler, cache, mirror,
                          cache.GetImage(s.copy_source->id), false);
                s.FrameBoundary(scheduler, mirror);
            }
            if (s.tune.enabled && s.dlss_ready && !s.failed)
                s.Composite(instance, runtime, scheduler, cache, *color);
            else if (auto it = s.outputs.find(color->info.guest_address); it != s.outputs.end())
                it->second->fresh = false;
            // Decoupled UI: the game presents its own frame, generated from this frame's inputs.
            auto& hudless = s.decoupled_hudless[color->info.guest_address];
            if (hudless)
                hudless->fresh = false;
            if (auto inputs = std::exchange(s.frame_gen_pending, std::nullopt);
                inputs && s.decoupled && !s.dlss_ready) {
                s.decoupled_frame_gen[color->info.guest_address] = *inputs;
                if (s.hud_snapshot_taken && s.hud_snapshot &&
                    s.hud_snapshot->extent ==
                        vk::Extent2D{color->info.size.width, color->info.size.height})
                    s.ReplayHudless(instance, runtime, scheduler, cache, *color);
            } else {
                s.decoupled_frame_gen.erase(color->info.guest_address);
            }
            s.last_copy_source = s.copy_source->id;
            s.last_copy_format = s.copy_source->view.format;
            s.hud_snapshot_taken = false;
            s.DisplayCopyDone();
            return {};
        }
        if (color && !s.outputs.empty()) {
            // Anything else drawing into a tracked VideoOut buffer makes our output stale.
            if (auto it = s.outputs.find(color->info.guest_address); it != s.outputs.end())
                it->second->fresh = false;
        }
        const auto learn_depth = [&] {
            if (!draw.depth || !color || s.depth_learned)
                return;
            const auto& depth = cache.GetImage(draw.depth);
            if (depth.info.size.width == color->info.size.width &&
                depth.info.size.height == color->info.size.height) {
                s.scene_depth = Impl::Tracked{draw.depth, depth.image_uid};
                s.depth_learned = true;
            }
        };
        // Scaleform also renders movies into offscreen targets; only the scene-sized target
        // (same size as this frame's sampled scene depth) starts the HUD phase.
        const auto scene_sized = [&](const VideoCore::Image& image) {
            const auto* depth =
                s.r32_depth ? cache.TryGetImage(s.r32_depth->id, s.r32_depth->uid) : nullptr;
            return depth && depth->info.size.width == image.info.size.width &&
                   depth->info.size.height == image.info.size.height;
        };
        // Decoupled UI + FSR frame generation: the UI target as it is before the HUD.
        if (s.decoupled && !s.hud_snapshot_taken && s.frame_gen_pending && color &&
            IsUiShader(draw.vs_hash) && s.last_copy_source && draw.color == *s.last_copy_source &&
            std::string_view{FrameGen::BackendName()} == "FSR")
            s.SnapshotHud(instance, runtime, scheduler, *color);
        if (!s.ui_phase && IsUiShader(draw.vs_hash) && color &&
            (color->info.pixel_format == vk::Format::eR8G8B8A8Unorm ||
             color->info.pixel_format == vk::Format::eR8G8B8A8Srgb) &&
            scene_sized(*color)) {
            s.ui_phase = true;
            s.ui_target = draw.color;
            learn_depth();
            if (s.tune.enabled && !s.dlss_ready)
                s.RunDlss(instance, runtime, scheduler, cache, mirror, *color, true);
            s.FrameBoundary(scheduler, mirror);
            return {};
        }
        if (IsUiShader(draw.vs_hash)) {
            if (s.ui_phase)
                learn_depth();
            return {};
        }
        // Decoupled UI: the depth used while drawing the HDR scene is the scene depth.
        if (s.decoupled && s.decoupled_scene && color && draw.color == *s.decoupled_scene &&
            draw.depth) {
            const auto& depth = cache.GetImage(draw.depth);
            if (depth.info.size.width == color->info.size.width &&
                depth.info.size.height == color->info.size.height &&
                (!s.scene_depth || s.scene_depth->id != draw.depth))
                s.scene_depth = Impl::Tracked{draw.depth, depth.image_uid};
        }
        // HUD draws share the scene depth for stencil; anything into the HUD target stays put.
        if (s.ui_target && draw.color == *s.ui_target)
            return {};
        if (s.tune.enabled && s.tune.jitter && !s.menu && s.scene_depth &&
            draw.depth == s.scene_depth->id &&
            cache.GetImage(draw.depth).image_uid == s.scene_depth->uid &&
            (draw.indirect || draw.num_indices > 6 || draw.num_instances > 1)) {
            ++s.jittered_draws;
            return s.jitter;
        }
        return {};
    } catch (const std::exception& e) {
        s.failed = true;
        s.InvalidateOutputs();
        scheduler.GetDynamicState().Invalidate();
        LOG_ERROR(Render_Vulkan, "[DLSS-TEMPORAL] Disabled: {}", e.what());
        return {};
    }
}

std::optional<BbTemporalDlss::DrawReplacement> BbTemporalDlss::TakeDrawReplacement() {
    return std::exchange(impl->replacement, std::nullopt);
}

std::optional<BbTemporalDlss::DisplayCopyReplay> BbTemporalDlss::TakeDisplayCopyReplay() {
    return std::exchange(impl->display_copy_replay, std::nullopt);
}

void BbTemporalDlss::FinishDisplayCopyReplay(vk::CommandBuffer command, bool drawn) {
    if (auto* hudless = std::exchange(impl->replay_hudless, nullptr)) {
        hudless->Transit(command, vk::ImageLayout::eShaderReadOnlyOptimal,
                         vk::PipelineStageFlagBits2::eFragmentShader,
                         vk::AccessFlagBits2::eShaderRead);
        if (!drawn)
            hudless->fresh = false;
    }
    auto* output = std::exchange(impl->replay_output, nullptr);
    if (!output)
        return;
    output->Transit(command, vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderRead);
    if (!drawn) {
        output->fresh = false;
        static bool logged{};
        if (!std::exchange(logged, true))
            LOG_WARNING(Render_Vulkan, "[DLSS-TEMPORAL] RenoDX display copy could not be redrawn");
    }
}

void BbTemporalDlss::SetDisplaySize(u32 width, u32 height) {
    impl->display = (u64{width} << 32) | height;
}

std::optional<BbTemporalDlss::Presentation> BbTemporalDlss::TakePresentation(
    VAddr address, vk::Format frame_view_format) {
    auto& s = *impl;
    if (!s.requested || s.failed || s.stopped || !s.tune.enabled)
        return {};
    const auto it = s.outputs.find(address);
    if (it == s.outputs.end() || !it->second->fresh)
        return {};
    const bool bgr = (frame_view_format == vk::Format::eB8G8R8A8Srgb) != it->second->swapped;
    if (frame_view_format != vk::Format::eB8G8R8A8Srgb &&
        frame_view_format != vk::Format::eR8G8B8A8Srgb)
        return {};
    Presentation presentation{bgr ? *it->second->bgr_view : *it->second->rgb_view,
                              it->second->extent, it->second->frame_gen};
    if (const auto& hudless = it->second->hudless; hudless && it->second->hudless_fresh)
        presentation.hudless = bgr ? *hudless->bgr_view : *hudless->rgb_view;
    return presentation;
}

std::optional<FrameGen::FrameInputs> BbTemporalDlss::TakeFrameGen(VAddr address,
                                                                  vk::Format frame_view_format,
                                                                  vk::ImageView* hudless) {
    auto& s = *impl;
    *hudless = vk::ImageView{};
    const auto it = s.decoupled_frame_gen.find(address);
    if (it == s.decoupled_frame_gen.end())
        return {};
    const auto inputs = it->second;
    s.decoupled_frame_gen.erase(it);
    if (!s.requested || s.failed || s.stopped || !s.tune.enabled)
        return {};
    if (const auto slot = s.decoupled_hudless.find(address);
        slot != s.decoupled_hudless.end() && slot->second && slot->second->fresh) {
        slot->second->fresh = false;
        const auto& image = *slot->second;
        if (image.rgb_view) {
            const bool bgr = (frame_view_format == vk::Format::eB8G8R8A8Srgb) != image.swapped;
            *hudless = bgr ? *image.bgr_view : *image.rgb_view;
        } else {
            *hudless = slot->second->FrameView(frame_view_format);
        }
    }
    return inputs;
}

void BbTemporalDlss::Shutdown(const Instance& instance, Scheduler& scheduler) {
    auto& s = *impl;
    if (!s.requested || s.stopped)
        return;
    s.stopped = true;
    scheduler.EndRendering();
    scheduler.Finish();
    s.fsr.reset();
    s.fsr4.reset();
    if (auto* ngx = instance.GetDlssNgx()) {
        ngx->ReleaseFeatureAfterGpuDrain();
        ngx->Shutdown();
    }
    s.outputs.clear();
    s.snapshot.reset();
    s.motion.reset();
    s.upscaled.reset();
    s.pre_lut.reset();
    s.pre_lut_hudless.reset();
    s.hud_snapshot.reset();
    s.decoupled_hudless.clear();
    s.sharpened.reset();
    s.linear_color.reset();
    s.coefficients.reset();
    s.coverage.reset();
    s.dimming.reset();
    s.motion_pass.reset();
    s.composite_pass.reset();
    s.composite_hdr_pass.reset();
    s.linearize_pass.reset();
    s.sharpen_pass.reset();
    s.hudless_dim_pass.reset();
    LOG_INFO(Render_Vulkan, "[DLSS-TEMPORAL] Teardown: evaluations={} composites={} fallbacks={}",
             s.evaluations, s.composites, s.fallbacks);
}

} // namespace Vulkan
