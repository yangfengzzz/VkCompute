//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/buffer.h"
#include <cuda_runtime_api.h>

namespace vox::compute {
class CudaExternalBuffer {
public:
    CudaExternalBuffer(core::Buffer &buffer,
                       VkExternalMemoryHandleTypeFlagBits handleType = core::Buffer::get_default_mem_handle_type());

    ~CudaExternalBuffer();

    inline void *get_cuda_buffer() {
        return cuda_buffer;
    }

private:
    void *cuda_buffer{nullptr};
    cudaExternalMemory_t cuda_mem;
};
}// namespace vox::compute