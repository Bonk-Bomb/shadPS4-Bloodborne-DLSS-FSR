// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// A virtual controller for unattended testing, driven by autopilot.txt in the user folder:
//
//   enabled=1          the autopilot may touch the controller at all
//   keep_connected=1   report a controller even when the real one has gone to sleep
//   boot=1             after launch, press Cross every few seconds until gameplay is upscaled
//                      (title screen -> Continue), never again afterwards
//   run=N              changing N to a new non-zero value plays the steps once more
//   repeat=K           play the step list K times per run
//   step <ms> [lx=..] [ly=..] [rx=..] [ry=..] [buttons...]
//                      sticks from -1 to 1 (x right, y down); buttons: cross circle square
//                      triangle l1 r1 l2 r2 l3 r3 options up down left right touchpad
//
// While a step plays, its sticks replace the real ones and its buttons are added.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "common/logging/log.h"
#include "common/path_util.h"
#include "core/libraries/pad/pad.h"
#include "input/bb_autopilot.h"
#include "video_core/renderer_vulkan/vk_bb_temporal_dlss.h"

namespace BbAutopilot {
namespace {

using Clock = std::chrono::steady_clock;
using Pad = Libraries::Pad::OrbisPadButtonDataOffset;

struct Step {
    u32 ms{};
    float lx{}, ly{}, rx{}, ry{};
    Pad buttons{Pad::None};
};

struct State {
    std::mutex mutex;
    std::filesystem::path path;
    std::filesystem::file_time_type file_time{};
    Clock::time_point last_poll{};
    Clock::time_point launched{Clock::now()};
    bool loaded{};
    bool enabled{}, keep_connected{}, boot{};
    // Boot presses stop for good once gameplay was seen (or after five minutes).
    bool boot_done{};
    bool gameplay{};
    u32 run{}, repeat{1};
    std::vector<Step> steps;
    // The run that is playing.
    bool playing{};
    u32 playing_run{};
    Clock::time_point started{};
    Status status{};
} state;

Pad ButtonByName(const std::string& name) {
    static const std::pair<const char*, Pad> names[] = {
        {"cross", Pad::Cross},   {"circle", Pad::Circle}, {"square", Pad::Square},
        {"triangle", Pad::Triangle}, {"l1", Pad::L1},     {"r1", Pad::R1},
        {"l2", Pad::L2},         {"r2", Pad::R2},         {"l3", Pad::L3},
        {"r3", Pad::R3},         {"options", Pad::Options}, {"up", Pad::Up},
        {"down", Pad::Down},     {"left", Pad::Left},     {"right", Pad::Right},
        {"touchpad", Pad::TouchPad}};
    for (const auto& [n, b] : names)
        if (name == n)
            return b;
    return Pad::None;
}

void Load() {
    std::ifstream file{state.path};
    if (!file)
        return;
    bool enabled = false, keep = false, boot = false;
    u32 run = 0, repeat = 1;
    std::vector<Step> steps;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line.empty() || line[0] == '#' || line[0] == ';')
            continue;
        if (line.rfind("step", 0) == 0) {
            std::istringstream words{line.substr(4)};
            Step step;
            std::string word;
            if (!(words >> step.ms))
                continue;
            while (words >> word) {
                const auto eq = word.find('=');
                if (eq != std::string::npos) {
                    const float v = std::clamp(float(std::atof(word.c_str() + eq + 1)), -1.f, 1.f);
                    const auto key = word.substr(0, eq);
                    if (key == "lx")
                        step.lx = v;
                    else if (key == "ly")
                        step.ly = v;
                    else if (key == "rx")
                        step.rx = v;
                    else if (key == "ry")
                        step.ry = v;
                } else {
                    step.buttons |= ButtonByName(word);
                }
            }
            steps.push_back(step);
            continue;
        }
        const auto eq = line.find('=');
        if (eq == std::string::npos)
            continue;
        const auto key = line.substr(0, eq);
        const long value = std::atol(line.c_str() + eq + 1);
        if (key == "enabled")
            enabled = value != 0;
        else if (key == "keep_connected")
            keep = value != 0;
        else if (key == "boot")
            boot = value != 0;
        else if (key == "run")
            run = u32(std::max<long>(value, 0));
        else if (key == "repeat")
            repeat = u32(std::clamp<long>(value, 1, 1000));
    }
    const bool first = !state.loaded;
    state.loaded = true;
    state.enabled = enabled;
    state.keep_connected = keep;
    state.boot = boot;
    state.repeat = repeat;
    state.steps = std::move(steps);
    // A run starts when its number changes after launch, never from the file as found.
    if (!first && run != 0 && run != state.run) {
        state.playing = enabled && !state.steps.empty();
        state.playing_run = run;
        state.started = Clock::now();
        LOG_INFO(Input, "[AUTOPILOT] Run {}: {} steps x{}", run, state.steps.size(), repeat);
    }
    if (!enabled)
        state.playing = false;
    state.run = run;
}

void Poll(Clock::time_point now) {
    if (now - state.last_poll < std::chrono::milliseconds{250})
        return;
    state.last_poll = now;
    if (state.path.empty())
        state.path = Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "autopilot.txt";
    std::error_code error;
    const auto time = std::filesystem::last_write_time(state.path, error);
    if (!error && time != state.file_time) {
        state.file_time = time;
        Load();
    }
    if (state.boot && !state.boot_done) {
        state.gameplay = Vulkan::BbTemporalDlssStatus().active;
        if (state.gameplay || now - state.launched > std::chrono::minutes{5}) {
            state.boot_done = true;
            LOG_INFO(Input, "[AUTOPILOT] Boot presses finished ({})",
                     state.gameplay ? "gameplay reached" : "gave up after 5 minutes");
        }
    }
}

u8 ToByte(float v) {
    return u8(std::clamp(std::lround(128.0f + v * 127.5f), 0L, 255L));
}

} // namespace

void Apply(Libraries::Pad::OrbisPadData& data) {
    std::scoped_lock lock{state.mutex};
    const auto now = Clock::now();
    Poll(now);
    if (!state.enabled)
        return;
    if (state.boot && !state.boot_done) {
        // From 20 s after launch: Cross held for 150 ms every 4 s.
        const auto since = std::chrono::duration_cast<std::chrono::milliseconds>(
                               now - state.launched)
                               .count();
        if (since > 20000 && (since % 4000) < 150)
            data.buttons |= Pad::Cross;
    }
    if (!state.playing) {
        state.status = {state.playing_run, 0, false};
        return;
    }
    u64 total = 0;
    for (const auto& step : state.steps)
        total += step.ms;
    const u64 elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - state.started).count();
    if (total == 0 || elapsed >= total * state.repeat) {
        state.playing = false;
        state.status = {state.playing_run, 0, false};
        LOG_INFO(Input, "[AUTOPILOT] Run {} finished", state.playing_run);
        return;
    }
    const u64 cycle = elapsed / total;
    u64 at = elapsed % total;
    u32 index = 0;
    while (index + 1 < state.steps.size() && at >= state.steps[index].ms) {
        at -= state.steps[index].ms;
        ++index;
    }
    const auto& step = state.steps[index];
    data.leftStick.x = ToByte(step.lx);
    data.leftStick.y = ToByte(step.ly);
    data.rightStick.x = ToByte(step.rx);
    data.rightStick.y = ToByte(step.ry);
    data.buttons |= step.buttons;
    state.status = {state.playing_run, u32(cycle * state.steps.size() + index + 1), true};
}

bool KeepConnected() {
    std::scoped_lock lock{state.mutex};
    Poll(Clock::now());
    return state.enabled && state.keep_connected;
}

Status GetStatus() {
    std::scoped_lock lock{state.mutex};
    return state.status;
}

} // namespace BbAutopilot
