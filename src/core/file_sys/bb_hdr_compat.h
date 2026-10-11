// SPDX-FileCopyrightText: Copyright 2026 IFreemz
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>
#include <memory>
#include <string_view>

#include "core/file_sys/ifile.h"

namespace Core::FileSys::BbHdrCompat {

void Initialize(bool supported_game, const std::filesystem::path& settings_path);
void Configure(bool supported_game, bool enabled, bool restore_moon);
bool IsCandidate(std::string_view path);
bool ShouldSkipPatch(std::string_view name);
std::unique_ptr<IFile> Open(std::unique_ptr<IFile> source, std::string_view path);

} // namespace Core::FileSys::BbHdrCompat
