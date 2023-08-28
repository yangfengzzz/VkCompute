//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "core/semaphore.h"
#include "core/device.h"

namespace vox::core {
Semaphore::Semaphore(Device &device, bool is_exported) : VulkanResource{VK_NULL_HANDLE, &device} {
    VkSemaphoreCreateInfo create_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};

    VkExportSemaphoreCreateInfoKHR exportSemaphoreCreateInfo = {};
    exportSemaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;
    exportSemaphoreCreateInfo.pNext = nullptr;
    exportSemaphoreCreateInfo.handleTypes = get_default_semaphore_handle_type();

    if (is_exported) {
        create_info.pNext = &exportSemaphoreCreateInfo;
    }

    if (vkCreateSemaphore(device.get_handle(), &create_info, nullptr, &handle) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create semaphore.");
    }
}

Semaphore::Semaphore(Semaphore &&other) noexcept : VulkanResource{std::move(other)} {
}

Semaphore::~Semaphore() {
    if (handle != VK_NULL_HANDLE) {
        vkDestroySemaphore(device->get_handle(), handle, nullptr);
    }
}

int Semaphore::get_semaphore_handle(VkExternalSemaphoreHandleTypeFlagBits handleType) {
    int fd;

    VkSemaphoreGetFdInfoKHR semaphoreGetFdInfoKHR = {};
    semaphoreGetFdInfoKHR.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    semaphoreGetFdInfoKHR.pNext = nullptr;
    semaphoreGetFdInfoKHR.semaphore = handle;
    semaphoreGetFdInfoKHR.handleType = handleType;

    if (vkGetSemaphoreFdKHR(device->get_handle(), &semaphoreGetFdInfoKHR, &fd) != VK_SUCCESS) {
        LOGE("Failed to retrieve handle for semaphore!");
    }

    return fd;
}

}// namespace vox::core