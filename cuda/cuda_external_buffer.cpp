//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "cuda_external_buffer.h"
#include "utils/helper_cuda.h"

namespace vox::compute {
CudaExternalBuffer::CudaExternalBuffer(core::Buffer &buffer, VkExternalMemoryHandleTypeFlagBits handleType) {
    cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};
    externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    externalMemoryHandleDesc.size = buffer.get_size();
    externalMemoryHandleDesc.handle.fd = buffer.get_memory_handle(handleType);
    checkCudaErrors(cudaImportExternalMemory(&cuda_mem, &externalMemoryHandleDesc));

    cudaExternalMemoryBufferDesc externalMemBufferDesc = {};
    externalMemBufferDesc.offset = 0;
    externalMemBufferDesc.size = buffer.get_size();
    externalMemBufferDesc.flags = 0;
    checkCudaErrors(cudaExternalMemoryGetMappedBuffer(&cuda_buffer, cuda_mem,
                                                      &externalMemBufferDesc));
}

CudaExternalBuffer::~CudaExternalBuffer() {
    if (cuda_buffer) {
        checkCudaErrors(cudaDestroyExternalMemory(cuda_mem));
    }
}

}// namespace vox::compute