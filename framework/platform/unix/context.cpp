//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "platform/unix/context.hpp"

namespace vox {

UnixPlatformContext::UnixPlatformContext(int argc, char **argv) : PlatformContext{} {
    _arguments.reserve(argc);
    for (int i = 1; i < argc; ++i) {
        _arguments.emplace_back(argv[i]);
    }

    const char *env_temp_dir = std::getenv("TMPDIR");
    _temp_directory = env_temp_dir ? std::string(env_temp_dir) + "/" : "/tmp/";
    _external_storage_directory = "";
}
}// namespace vox