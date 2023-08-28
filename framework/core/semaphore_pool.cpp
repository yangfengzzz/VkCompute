//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "semaphore_pool.h"

#include "core/device.h"

namespace vox::core {
SemaphorePool::SemaphorePool(Device &device) : device{device} {
}

SemaphorePool::~SemaphorePool() {
    reset();
    semaphores.clear();
}

Semaphore SemaphorePool::request_semaphore_with_ownership() {
    // Check if there is an available semaphore, if so, just pilfer one.
    if (active_semaphore_count < semaphores.size()) {
        Semaphore semaphore = std::move(semaphores.back());
        semaphores.pop_back();
        return semaphore;
    }

    return Semaphore(device);
}

void SemaphorePool::release_owned_semaphore(Semaphore semaphore) {
    // We cannot reuse this semaphore until ::reset().
    released_semaphores.push_back(std::move(semaphore));
}

Semaphore &SemaphorePool::request_semaphore() {
    // Check if there is an available semaphore
    if (active_semaphore_count < semaphores.size()) {
        return semaphores[active_semaphore_count++];
    }

    auto semaphore = Semaphore(device);
    semaphores.push_back(std::move(semaphore));

    active_semaphore_count++;

    return semaphores.back();
}

void SemaphorePool::reset() {
    active_semaphore_count = 0;

    // Now we can safely recycle the released semaphores.
    for (auto &sem : released_semaphores) {
        semaphores.push_back(std::move(sem));
    }

    released_semaphores.clear();
}

uint32_t SemaphorePool::get_active_semaphore_count() const {
    return active_semaphore_count;
}

}// namespace vox::core
