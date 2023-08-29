//  Copyright (c) 2022 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "cuda2vk_app.h"

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
        input_assembly_state_.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        input_assembly_state_.primitive_restart_enable = false;
        rasterization_state_.polygon_mode = VK_POLYGON_MODE_POINT;
        rasterization_state_.cull_mode = VK_CULL_MODE_NONE;
        rasterization_state_.front_face = VK_FRONT_FACE_CLOCKWISE;

        vertex_source_ = ShaderManager::get_singleton().load_shader("base/montecarlo.vert", VK_SHADER_STAGE_VERTEX_BIT);
        fragment_source_ = ShaderManager::get_singleton().load_shader("base/montecarlo.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    void set_frame(float current_frame_idx) {
        shader_data_.set_data("UniformBufferObject", current_frame_idx);
    }
};

class CudaExecuteScript : public Script {
public:
    explicit CudaExecuteScript(Entity *pEntity) : Script(pEntity) {
    }

    void init(Cuda2VkApp *cuda_app) {
        app = cuda_app;
    }

    void on_start() override {
        cuda_sim = &app->get_cuda_sim();
        cuda_stream = std::make_unique<compute::CudaStream>(app->get_cuda_device());
        cuda_wait_semaphore = std::make_unique<compute::CudaExternalSemaphore>(app->get_signal_semaphore());
        cuda_signal_semaphore = std::make_unique<compute::CudaExternalSemaphore>(app->get_wait_semaphore());

        app->get_render_context().external_signal_semaphores = [&](std::vector<VkSemaphore> &signal) {
            // Add this semaphore for vulkan to signal once the vertex buffer is ready
            // for cuda to modify
            signal.push_back(app->get_signal_semaphore().get_handle());
        };
        app->get_render_context().external_wait_semaphores = [&](std::vector<VkSemaphore> &wait, std::vector<VkPipelineStageFlags> &flags) {
            if (frame_count > 1) {
                // Have vulkan wait until cuda is done with the vertex buffer before
                // rendering, We don't do this on the first frame, as the wait semaphore
                // hasn't been initialized yet
                wait.push_back(app->get_wait_semaphore().get_handle());
                // We want to wait until all the pipeline commands are complete before
                // letting cuda work
                flags.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
            }
        };
    }

    void on_update(float delta_time) override {
        auto mat = std::dynamic_pointer_cast<CustomMaterial>(app->get_material());
        mat->set_frame(static_cast<float>(frame_count));
        frame_count++;
        total_time += delta_time;
        cuda_wait_semaphore->wait(*cuda_stream);
        cuda_sim->step_simulation(total_time, *cuda_stream);
        cuda_signal_semaphore->signal(*cuda_stream);
    }

private:
    uint64_t frame_count{0};
    float total_time{};
    Cuda2VkApp *app{};

    std::unique_ptr<compute::CudaStream> cuda_stream{nullptr};
    compute::MonteCarloPiSimulation *cuda_sim{nullptr};

    std::unique_ptr<compute::CudaExternalSemaphore> cuda_wait_semaphore{nullptr};
    std::unique_ptr<compute::CudaExternalSemaphore> cuda_signal_semaphore{nullptr};
};

}// namespace

bool Cuda2VkApp::prepare(const ApplicationOptions &options) {
    add_instance_extension(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
    add_instance_extension(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);

    add_device_extension(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
    add_device_extension(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
    add_device_extension(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    add_device_extension(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);

    ForwardApplication::prepare(options);
    cuda_device = std::make_unique<compute::CudaDevice>(device->get_gpu().get_device_id_properties().deviceUUID, VK_UUID_SIZE);
    cuda_sim = std::make_unique<compute::MonteCarloPiSimulation>(NUM_SIMULATION_POINTS, *cuda_device);

    const size_t n_verts = cuda_sim->get_num_points();
    xy_position_buffer = std::make_unique<core::Buffer>(*device, (void *)(uintptr_t)cuda_sim->get_position_shareable_handle(),
                                                        n_verts * sizeof(Vector2F),
                                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    in_circle_buffer = std::make_unique<core::Buffer>(*device, (void *)(uintptr_t)cuda_sim->get_in_circle_shareable_handle(),
                                                      n_verts * sizeof(float),
                                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

    std::vector<VkVertexInputBindingDescription> bindingDesc;
    std::vector<VkVertexInputAttributeDescription> attribDesc;
    get_vertex_descriptions(bindingDesc, attribDesc);
    mesh->set_vertex_input_state(bindingDesc, attribDesc);
    mesh->set_vertex_buffer_binding(0, in_circle_buffer.get());
    mesh->set_vertex_buffer_binding(1, xy_position_buffer.get());
    mesh->add_sub_mesh(0, n_verts);

    wait_semaphore = std::make_unique<core::Semaphore>(*device, true);
    signal_semaphore = std::make_unique<core::Semaphore>(*device, true);

    return true;
}

void Cuda2VkApp::load_scene() {
    auto scene = scene_manager_->get_current_scene();
    auto root_entity = scene->create_root_entity();

    auto camera_entity = root_entity->create_child();
    camera_entity->transform->set_position(10, 10, 10);
    camera_entity->transform->look_at(Point3F(0, 0, 0));
    main_camera_ = camera_entity->add_component<Camera>();
    main_camera_->enable_frustum_culling_ = false;
    camera_entity->add_component<control::OrbitControl>();

    auto cube_entity = root_entity->create_child();
    auto renderer = cube_entity->add_component<MeshRenderer>();
    mesh = std::make_shared<BufferMesh>();
    renderer->set_mesh(mesh);
    material_ = std::make_shared<CustomMaterial>(*device);
    renderer->set_material(material_);

    auto cuda_execute = cube_entity->add_component<CudaExecuteScript>();
    cuda_execute->init(this);

    scene->play();
}

void Cuda2VkApp::get_vertex_descriptions(
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

}// namespace vox