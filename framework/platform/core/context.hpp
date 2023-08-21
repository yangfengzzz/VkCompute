//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace vox {
class UnixPlatformContext;
class WindowsPlatformContext;
class AndroidPlatformContext;

/**
 * @brief A platform context contains abstract platform specific operations
 *
 *        A platform can be thought as the physical device and operating system configuration that the application
 *        is running on.
 *
 *        Some platforms can be reused across different hardware configurations, such as Linux and Macos as both
 *        are POSIX compliant. However, some platforms are more specific such as Android and Windows
 */
class PlatformContext {
    // only allow platform contexts to be created by the platform specific implementations
    friend class UnixPlatformContext;
    friend class WindowsPlatformContext;
    friend class AndroidPlatformContext;

public:
    virtual ~PlatformContext() = default;

    virtual const std::vector<std::string> &arguments() const {
        return _arguments;
    }

    virtual const std::string &external_storage_directory() const {
        return _external_storage_directory;
    }

    virtual const std::string &temp_directory() const {
        return _temp_directory;
    }

protected:
    std::vector<std::string> _arguments;
    std::string _external_storage_directory;
    std::string _temp_directory;

    PlatformContext() = default;
};
}// namespace vox