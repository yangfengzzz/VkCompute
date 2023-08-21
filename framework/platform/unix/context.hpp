//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <string>

#include "platform/core/context.hpp"

namespace vox {
/**
 * @brief Unix platform context
 *
 * @warning Use in extreme circumstances with code guarded by the PLATFORM__UNIX define
 */
class UnixPlatformContext final : public PlatformContext {
public:
    UnixPlatformContext(int argc, char **argv);
    ~UnixPlatformContext() override = default;
};
}// namespace vox