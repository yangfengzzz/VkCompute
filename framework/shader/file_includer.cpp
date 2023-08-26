//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "file_includer.h"

#include <mutex>

#include <libshaderc_util/io_shaderc.h>

namespace vox {

shaderc_include_result *MakeErrorIncludeResult(const char *message) {
    return new shaderc_include_result{"", 0, message, strlen(message)};
}

FileIncluder::~FileIncluder() = default;

shaderc_include_result *FileIncluder::GetInclude(
    const char *requested_source, shaderc_include_type include_type,
    const char *requesting_source, size_t) {

    const std::string full_path =
        (include_type == shaderc_include_type_relative) ? file_finder_.FindRelativeReadableFilepath(requesting_source,
                                                                                                    requested_source) :
                                                          file_finder_.FindReadableFilepath(requested_source);

    if (full_path.empty())
        return MakeErrorIncludeResult("Cannot find or open include file.");

    // In principle, several threads could be resolving includes at the same
    // time.  Protect the included_files.

    // Read the file and save its full path and contents into stable addresses.
    auto *new_file_info = new FileInfo{full_path, {}};
    if (!shaderc_util::ReadFile(full_path, &(new_file_info->contents))) {
        return MakeErrorIncludeResult("Cannot read file");
    }

    included_files_.insert(full_path);

    return new shaderc_include_result{
        new_file_info->full_path.data(), new_file_info->full_path.length(),
        new_file_info->contents.data(), new_file_info->contents.size(),
        new_file_info};
}

void FileIncluder::ReleaseInclude(shaderc_include_result *include_result) {
    auto *info = static_cast<FileInfo *>(include_result->user_data);
    delete info;
    delete include_result;
}

}// namespace glslc
