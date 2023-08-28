//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/semaphore_pool.h"
#include <cuda_runtime_api.h>

namespace vox::compute {
class CudaExternalSemaphore {
public:
    CudaExternalSemaphore(VkDevice device, VkSemaphore semaphore,
                          VkExternalSemaphoreHandleTypeFlagBits handleType = core::SemaphorePool::get_default_semaphore_handle_type());

    ~CudaExternalSemaphore();

private:
    cudaExternalSemaphore_t cuda_semaphore;
};

}// namespace vox::compute