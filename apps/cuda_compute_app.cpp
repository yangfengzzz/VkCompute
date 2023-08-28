//  Copyright (c) 2022 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "cuda_compute_app.h"

#include "application/components/camera.h"
#include "application/controls/orbit_control.h"
#include "application/material/base_material.h"
#include "application/components/mesh_renderer.h"
#include "application/mesh/primitive_mesh.h"
#include "application/ecs/entity.h"
#include "compute/buffer_utils.h"

namespace vox {
namespace {
class CustomMaterial : public BaseMaterial {
public:
    explicit CustomMaterial(core::Device &device) : BaseMaterial(device, "waveRender") {
        vertex_source_ = ShaderManager::get_singleton().load_shader("base/sinwave.vert", VK_SHADER_STAGE_VERTEX_BIT);
        fragment_source_ = ShaderManager::get_singleton().load_shader("base/sinwave.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
    }
};
}// namespace

bool CudaComputeApp::prepare(const ApplicationOptions &options) {
    ForwardApplication::prepare(options);
    cuda_device = std::make_unique<compute::CudaDevice>(device->get_gpu().get_device_id_properties().deviceUUID, VK_UUID_SIZE);
    cuda_stream = std::make_unique<compute::CudaStream>(*cuda_device);
    cuda_sim = std::make_unique<compute::SineWaveSimulation>((1ULL << 8ULL), (1ULL << 8ULL), *cuda_device);

    VkExportMemoryAllocateInfoKHR vulkanExportMemoryAllocateInfoKHR = {};
    vulkanExportMemoryAllocateInfoKHR.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
    vulkanExportMemoryAllocateInfoKHR.pNext = nullptr;
    vulkanExportMemoryAllocateInfoKHR.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    external_pool = std::make_unique<core::BufferPool>(*device, 1024 * 1024,
                                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                                       &vulkanExportMemoryAllocateInfoKHR);

    const size_t n_verts = cuda_sim->get_width() * cuda_sim->get_height();
    const size_t n_inds = (cuda_sim->get_width() - 1) * (cuda_sim->get_height() - 1) * 6;
    height_buffer = std::make_unique<core::Buffer>(*device, n_verts * sizeof(float),
                                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                                   VMA_MEMORY_USAGE_GPU_ONLY, external_pool.get());
    cuda_height_buffer = std::make_unique<compute::CudaExternalBuffer>(*height_buffer);

    auto index_buffer = core::Buffer(*device, n_inds * sizeof(uint32_t),
                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                     VMA_MEMORY_USAGE_GPU_ONLY);

    xy_buffer = std::make_unique<core::Buffer>(*device, n_verts * sizeof(Vector2F),
                                               VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                               VMA_MEMORY_USAGE_GPU_ONLY);
    init_buffer(index_buffer);

    mesh = std::make_shared<BufferMesh>();
    std::vector<VkVertexInputBindingDescription> bindingDesc;
    std::vector<VkVertexInputAttributeDescription> attribDesc;
    get_vertex_descriptions(bindingDesc, attribDesc);
    mesh->set_vertex_input_state(bindingDesc, attribDesc);
    mesh->set_index_buffer_binding(std::make_unique<IndexBufferBinding>(std::move(index_buffer), VK_INDEX_TYPE_UINT32));
    mesh->set_vertex_buffer_binding(0, height_buffer.get());
    mesh->set_vertex_buffer_binding(1, xy_buffer.get());
    return true;
}

void CudaComputeApp::load_scene() {
    auto scene = scene_manager_->get_current_scene();
    auto root_entity = scene->create_root_entity();

    auto camera_entity = root_entity->create_child();
    camera_entity->transform->set_position(10, 10, 10);
    camera_entity->transform->look_at(Point3F(0, 0, 0));
    main_camera_ = camera_entity->add_component<Camera>();
    camera_entity->add_component<control::OrbitControl>();

    auto cube_entity = root_entity->create_child();
    auto renderer = cube_entity->add_component<MeshRenderer>();
    renderer->set_mesh(mesh);
    material_ = std::make_shared<CustomMaterial>(*device);
    renderer->set_material(material_);

    scene->play();
}

void CudaComputeApp::update_gpu_task(core::CommandBuffer &command_buffer) {
    cuda_sim->step_simulation(1 / 60, static_cast<float *>(cuda_height_buffer->get_cuda_buffer()), *cuda_stream);
}

void CudaComputeApp::get_vertex_descriptions(
    std::vector<VkVertexInputBindingDescription> &bindingDesc,
    std::vector<VkVertexInputAttributeDescription> &attribDesc) {
    bindingDesc.resize(2);
    attribDesc.resize(2);

    bindingDesc[0].binding = 0;
    bindingDesc[0].stride = sizeof(float);
    bindingDesc[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    bindingDesc[1].binding = 1;
    bindingDesc[1].stride = sizeof(Vector2F);
    bindingDesc[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    attribDesc[0].binding = 0;
    attribDesc[0].location = 0;
    attribDesc[0].format = VK_FORMAT_R32_SFLOAT;
    attribDesc[0].offset = 0;

    attribDesc[1].binding = 1;
    attribDesc[1].location = 1;
    attribDesc[1].format = VK_FORMAT_R32G32_SFLOAT;
    attribDesc[1].offset = 0;
}

void CudaComputeApp::init_buffer(core::Buffer &index_buffer) {
    compute::set_device_buffer_via_staging_buffer(
        *device, *height_buffer, height_buffer->get_size(),
        [&](void *ptr, size_t) {
            auto *src_float_buffer = reinterpret_cast<float *>(ptr);
            memset(src_float_buffer, 0, height_buffer->get_size());
        });

    compute::set_device_buffer_via_staging_buffer(
        *device, *xy_buffer, xy_buffer->get_size(),
        [&](void *ptr, size_t) {
            auto *src_float_buffer = reinterpret_cast<Vector2F *>(ptr);
            for (size_t y = 0; y < cuda_sim->get_height(); y++) {
                for (size_t x = 0; x < cuda_sim->get_width(); x++) {
                    src_float_buffer[y * cuda_sim->get_width() + x][0] =
                        (2.0f * float(x)) / float(cuda_sim->get_width() - 1) - 1;
                    src_float_buffer[y * cuda_sim->get_width() + x][1] =
                        (2.0f * float(y)) / float(cuda_sim->get_height() - 1) - 1;
                }
            }
        });

    compute::set_device_buffer_via_staging_buffer(
        *device, index_buffer, index_buffer.get_size(),
        [&](void *ptr, size_t) {
            auto *indices = reinterpret_cast<uint32_t *>(ptr);
            for (size_t y = 0; y < cuda_sim->get_height() - 1; y++) {
                for (size_t x = 0; x < cuda_sim->get_width() - 1; x++) {
                    indices[0] = (uint32_t)((y + 0) * cuda_sim->get_width() + (x + 0));
                    indices[1] = (uint32_t)((y + 1) * cuda_sim->get_width() + (x + 0));
                    indices[2] = (uint32_t)((y + 0) * cuda_sim->get_width() + (x + 1));
                    indices[3] = (uint32_t)((y + 1) * cuda_sim->get_width() + (x + 0));
                    indices[4] = (uint32_t)((y + 1) * cuda_sim->get_width() + (x + 1));
                    indices[5] = (uint32_t)((y + 0) * cuda_sim->get_width() + (x + 1));
                    indices += 6;
                }
            }
        });
}

}// namespace vox