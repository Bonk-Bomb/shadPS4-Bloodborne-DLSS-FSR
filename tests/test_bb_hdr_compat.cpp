// SPDX-FileCopyrightText: Copyright 2026 IFreemz
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <gtest/gtest.h>

#include "common/logging/log.h"
#include "core/file_sys/bb_hdr_compat.h"

namespace Common::Log {
std::array<Level, NUM_LOG_CLASSES> g_class_levels{};
void VLog(Class, Level, const char*, int, const char*, fmt::string_view, fmt::format_args) {}
} // namespace Common::Log

namespace {
using namespace Core::FileSys;
constexpr std::string_view Dream = "dvdroot_ps4/param/drawparam/m21_00_0000.gparambnd.dcx";
constexpr std::string_view Moon = "dvdroot_ps4/mtd/allmaterialbnd.mtdbnd.dcx";

class UnrecognizedFile final : public IFile {
public:
    s64 Read(void* dst, u64 size) override {
        ++reads;
        const auto count = position < bytes.size() ? std::min(size, bytes.size() - position) : 0;
        if (count) {
            std::memcpy(dst, bytes.data() + position, count);
            position += count;
        }
        return count;
    }
    s64 Write(const void*, u64) override {
        return -1;
    }
    bool Seek(s64 offset, Common::FS::SeekOrigin origin) override {
        if (origin != Common::FS::SeekOrigin::SetOrigin || offset < 0) {
            return false;
        }
        position = offset;
        return true;
    }
    u64 Tell() const override {
        return position;
    }
    u64 Size() const override {
        return reported_size;
    }
    bool Flush() override {
        return true;
    }
    bool IsOpen() const override {
        return true;
    }
    bool IsReadOnly() const override {
        return read_only;
    }

    std::array<u8, 100> bytes{};
    u64 reported_size{bytes.size()};
    u64 position{7};
    size_t reads{};
    bool read_only{true};
};

TEST(BloodborneHdrCompat, OptionsAreScopedToSupportedGames) {
    BbHdrCompat::Configure(false, true, true);
    EXPECT_FALSE(BbHdrCompat::IsCandidate(Dream));
    EXPECT_FALSE(BbHdrCompat::IsCandidate(Moon));
    EXPECT_FALSE(BbHdrCompat::ShouldSkipPatch("Disable AA"));
    BbHdrCompat::Configure(true, false, true);
    EXPECT_FALSE(BbHdrCompat::IsCandidate(Dream));
    EXPECT_FALSE(BbHdrCompat::IsCandidate(Moon));
    BbHdrCompat::Configure(true, true, false);
    EXPECT_TRUE(BbHdrCompat::IsCandidate(Dream));
    EXPECT_FALSE(BbHdrCompat::IsCandidate(Moon));
    BbHdrCompat::Configure(true, true, true);
    EXPECT_TRUE(BbHdrCompat::IsCandidate(Moon));
    EXPECT_FALSE(BbHdrCompat::IsCandidate("param/drawparam/m21_00_0000.gparambnd.dcx"));
    EXPECT_FALSE(
        BbHdrCompat::IsCandidate("dvdroot_ps4/param/drawparam/../m21_00_0000.gparambnd.dcx"));
}

TEST(BloodborneHdrCompat, UnknownDataPreservesHandleAndPosition) {
    BbHdrCompat::Configure(true, true, false);
    auto file = std::make_unique<UnrecognizedFile>();
    const auto* original = file.get();
    auto result = BbHdrCompat::Open(std::move(file), Dream);
    ASSERT_EQ(result.get(), original);
    EXPECT_EQ(result->Tell(), 7);
    EXPECT_EQ(result->Size(), 100);
    EXPECT_GT(original->reads, 0);
    EXPECT_FALSE(BbHdrCompat::ShouldSkipPatch("Disable AA"));
    EXPECT_FALSE(BbHdrCompat::ShouldSkipPatch("Disable Bloom"));
    EXPECT_FALSE(BbHdrCompat::ShouldSkipPatch("Disable Chromatic Aberration"));
}

TEST(BloodborneHdrCompat, DisabledOrIrrelevantDataIsNeverRead) {
    BbHdrCompat::Configure(true, false, true);
    auto file = std::make_unique<UnrecognizedFile>();
    const auto* original = file.get();
    auto result = BbHdrCompat::Open(std::move(file), Dream);
    EXPECT_EQ(result.get(), original);
    EXPECT_EQ(original->reads, 0);
    BbHdrCompat::Configure(true, true, false);
    result = BbHdrCompat::Open(std::move(result), Moon);
    EXPECT_EQ(result.get(), original);
    EXPECT_EQ(original->reads, 0);
}

TEST(BloodborneHdrCompat, WritableAndOversizedFilesAreNeverRead) {
    BbHdrCompat::Configure(true, true, false);
    auto file = std::make_unique<UnrecognizedFile>();
    file->read_only = false;
    auto* original = file.get();
    auto result = BbHdrCompat::Open(std::move(file), Dream);
    EXPECT_EQ(result.get(), original);
    EXPECT_EQ(original->reads, 0);
    original->read_only = true;
    original->reported_size = std::numeric_limits<u64>::max();
    result = BbHdrCompat::Open(std::move(result), Dream);
    EXPECT_EQ(result.get(), original);
    EXPECT_EQ(original->reads, 0);
    EXPECT_FALSE(BbHdrCompat::Open(nullptr, Dream));
}

TEST(BloodborneHdrCompat, SettingsDefaultToAutomaticWithMoonOff) {
    const auto path =
        std::filesystem::temp_directory_path() /
        ("shadps4-hdr-compat-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".ini");
    BbHdrCompat::Initialize(true, path);
    EXPECT_TRUE(BbHdrCompat::IsCandidate(Dream));
    EXPECT_FALSE(BbHdrCompat::IsCandidate(Moon));
    {
        std::ofstream settings{path};
        settings << "hdr_mod_compat=0\nhdr_moon_restore=1\n";
    }
    BbHdrCompat::Initialize(true, path);
    EXPECT_FALSE(BbHdrCompat::IsCandidate(Dream));
    EXPECT_FALSE(BbHdrCompat::IsCandidate(Moon));
    {
        std::ofstream settings{path};
        settings << "hdr_mod_compat=1\nhdr_moon_restore=1\n";
    }
    BbHdrCompat::Initialize(true, path);
    EXPECT_TRUE(BbHdrCompat::IsCandidate(Dream));
    EXPECT_TRUE(BbHdrCompat::IsCandidate(Moon));
    std::filesystem::remove(path);
}
} // namespace
