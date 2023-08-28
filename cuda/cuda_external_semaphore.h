//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/semaphore.h"
#include "cuda_stream.h"

namespace vox::compute {
class CudaExternalSemaphore {
public:
    explicit CudaExternalSemaphore(core::Semaphore &semaphore,
                                   VkExternalSemaphoreHandleTypeFlagBits handleType = core::Semaphore::get_default_semaphore_handle_type());

    ~CudaExternalSemaphore();

    void wait(CudaStream &stream);

    void signal(CudaStream &stream);

private:
    cudaExternalSemaphore_t cuda_semaphore{};
};

}// namespace vox::compute