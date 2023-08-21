//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "platform/core/entrypoint.hpp"

#include "platform/unix/context.hpp"

std::unique_ptr<vox::PlatformContext> create_platform_context(int argc, char **argv) {
    return std::make_unique<vox::UnixPlatformContext>(argc, argv);
}