//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "common/helpers.h"
#include "common/vk_common.h"
#include "core/vulkan_resource.h"

namespace vox::core {
class Device;
class BufferPool;

struct BufferDesc {
    VkDeviceSize size;
    VkBufferUsageFlags buffer_usage;
    VmaMemoryUsage memory_usage;
};

inline bool operator==(const BufferDesc &x, const BufferDesc &y) {
    return x.size == y.size &&
           x.buffer_usage == y.buffer_usage &&
           x.memory_usage == y.memory_usage;
}

class Buffer : public VulkanResource<VkBuffer, VK_OBJECT_TYPE_BUFFER, const Device> {
public:
    /**
	 * @brief Creates a buffer using VMA
	 * @param device A valid Vulkan device
	 * @param size The size in bytes of the buffer
	 * @param buffer_usage The usage flags for the VkBuffer
	 * @param memory_usage The memory usage of the buffer
	 * @param flags The allocation create flags
	 * @param queue_family_indices optional queue family indices
	 */
    Buffer(Device const &device,
           BufferDesc desc,
           BufferPool *pool = nullptr,
           VmaAllocationCreateFlags flags = VMA_ALLOCATION_CREATE_MAPPED_BIT,
           const std::vector<uint32_t> &queue_family_indices = {});

    /**
     * import external raw_buffer(etc. cuda) into vulkan buffer
     * @param device A valid Vulkan device
     * @param raw_buffer The buffer pointer
     * @param size The size in bytes of the buffer
     * @param usage The usage flags for the VkBuffer
     * @param properties Memory property
     */
    Buffer(Device const &device,
           void *raw_buffer,
           VkDeviceSize size,
           VkBufferUsageFlags usage,
           VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    Buffer(const Buffer &) = delete;

    Buffer(Buffer &&other) noexcept;

    ~Buffer() override;

    Buffer &operator=(const Buffer &) = delete;

    Buffer &operator=(Buffer &&) = delete;

    template<typename T>
    static std::vector<T> copy(std::unordered_map<std::string, Buffer> &buffers, const char *buffer_name) {
        auto iter = buffers.find(buffer_name);
        if (iter == buffers.cend()) {
            return {};
        }
        auto &buffer = iter->second;
        std::vector<T> out;

        const size_t sz = buffer.get_size();
        out.resize(sz / sizeof(T));
        const bool already_mapped = buffer.get_data() != nullptr;
        if (!already_mapped) {
            buffer.map();
        }
        memcpy(&out[0], buffer.get_data(), sz);
        if (!already_mapped) {
            buffer.unmap();
        }
        return out;
    }

    [[nodiscard]] const VkBuffer *get() const;

    [[nodiscard]] VmaAllocation get_allocation() const;

    [[nodiscard]] VkDeviceMemory get_memory() const;

    [[nodiscard]] int get_memory_handle(VkExternalMemoryHandleTypeFlagBits handleType) const;

    /**
	 * @brief Flushes memory if it is HOST_VISIBLE and not HOST_COHERENT
	 */
    void flush() const;

    /**
	 * @brief Maps vulkan memory if it isn't already mapped to an host visible address
	 * @return Pointer to host visible memory
	 */
    uint8_t *map();

    /**
	 * @brief Unmaps vulkan memory from the host visible address
	 */
    void unmap();

    /**
	 * @return The size of the buffer
	 */
    [[nodiscard]] VkDeviceSize get_size() const;

    [[nodiscard]] const uint8_t *get_data() const {
        return mapped_data;
    }

    /**
	 * @brief Copies byte data into the buffer
	 * @param data The data to copy from
	 * @param size The amount of bytes to copy
	 * @param offset The offset to start the copying into the mapped data
	 */
    void update(const uint8_t *data, size_t bytes_size, size_t offset = 0);

    /**
	 * @brief Converts any non byte data into bytes and then updates the buffer
	 * @param data The data to copy from
	 * @param size The amount of bytes to copy
	 * @param offset The offset to start the copying into the mapped data
	 */
    void update(void *data, size_t bytes_size, size_t offset = 0);

    /**
	 * @brief Copies a vector of bytes into the buffer
	 * @param data The data vector to upload
	 * @param offset The offset to start the copying into the mapped data
	 */
    void update(const std::vector<uint8_t> &data, size_t offset = 0);

    /**
	 * @brief Copies an object as byte data into the buffer
	 * @param object The object to convert into byte data
	 * @param offset The offset to start the copying into the mapped data
	 */
    template<class T>
    void convert_and_update(const T &object, size_t offset = 0) {
        update(reinterpret_cast<const uint8_t *>(&object), sizeof(T), offset);
    }

    /**
	 * @return Return the buffer's device address (note: requires that the buffer has been created
     * with the VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT usage fla)
	 */
    uint64_t get_device_address();

    inline BufferDesc get_desc() { return desc; }

public:
    static VkExternalMemoryHandleTypeFlagBits get_default_mem_handle_type() {
        return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    }

private:
    VmaAllocation allocation{VK_NULL_HANDLE};

    VkDeviceMemory memory{VK_NULL_HANDLE};

    BufferDesc desc{};

    uint8_t *mapped_data{nullptr};

    /// Whether the buffer is persistently mapped or not
    bool persistent{false};

    /// Whether the buffer has been mapped with vmaMapMemory
    bool mapped{false};
};

}// namespace vox::core

namespace std {
template<>
struct hash<vox::core::BufferDesc> {
    std::size_t operator()(const vox::core::BufferDesc &desc) const {
        std::size_t result = 0;

        vox::hash_combine(result, desc.size);
        vox::hash_combine(result, desc.memory_usage);
        vox::hash_combine(result, desc.buffer_usage);
        return result;
    }
};
}// namespace std