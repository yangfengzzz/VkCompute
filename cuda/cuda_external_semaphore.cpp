//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "cuda_external_semaphore.h"
#include "utils/helper_cuda.h"

namespace vox::compute {
CudaExternalSemaphore::CudaExternalSemaphore(core::Semaphore &semaphore, VkExternalSemaphoreHandleTypeFlagBits handleType) {
    cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc = {};
    externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
    externalSemaphoreHandleDesc.handle.fd = semaphore.get_semaphore_handle(handleType);
    externalSemaphoreHandleDesc.flags = 0;
    checkCudaErrors(cudaImportExternalSemaphore(&cuda_semaphore, &externalSemaphoreHandleDesc));
}

CudaExternalSemaphore::~CudaExternalSemaphore() {
    checkCudaErrors(cudaDestroyExternalSemaphore(cuda_semaphore));
}

void CudaExternalSemaphore::wait(CudaStream &stream) {
    cudaExternalSemaphoreWaitParams waitParams = {};
    waitParams.flags = 0;
    waitParams.params.fence.value = 0;
    // Wait for vulkan to complete it's work
    checkCudaErrors(cudaWaitExternalSemaphoresAsync(&cuda_semaphore,
                                                    &waitParams, 1, stream.get_handle()));
}

void CudaExternalSemaphore::signal(CudaStream &stream) {
    cudaExternalSemaphoreSignalParams signalParams = {};
    signalParams.flags = 0;
    signalParams.params.fence.value = 0;
    // Signal vulkan to continue with the updated buffers
    checkCudaErrors(cudaSignalExternalSemaphoresAsync(
        &cuda_semaphore, &signalParams, 1, stream.get_handle()));
}

}// namespace vox::compute