//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "error.h"

#include "helpers.h"

namespace vox {
VulkanException::VulkanException(const VkResult result, const std::string &msg) : result{result},
                                                                                  std::runtime_error{msg} {
    error_message = std::string(std::runtime_error::what()) + std::string{" : "} + to_string(result);
}

const char *VulkanException::what() const noexcept {
    return error_message.c_str();
}
}// namespace vox
