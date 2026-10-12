// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Libraries::Pad {
struct OrbisPadData;
}

// Unattended testing: a virtual controller driven by autopilot.txt in the user folder.
// The file is re-read within a quarter second of being saved. See bb_autopilot.cpp.
namespace BbAutopilot {

struct Status {
    u32 run;    // the script run that is playing (0: none)
    u32 step;   // index of the step playing, counting repeats
    bool active;
};

// Overlays the virtual controller on pad data the game is about to read.
void Apply(Libraries::Pad::OrbisPadData& data);

// Whether a missing controller should be reported as connected (keep_connected=1).
bool KeepConnected();

Status GetStatus();

} // namespace BbAutopilot
