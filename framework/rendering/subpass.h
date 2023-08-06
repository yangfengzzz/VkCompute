//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "buffer_pool.h"
#include "common/helpers.h"
#include "core/shader_module.h"
#include "rendering/pipeline_state.h"
#include "rendering/render_context.h"
#include "rendering/render_frame.h"

VKBP_DISABLE_WARNINGS()
#include "common/glm_common.h"
VKBP_ENABLE_WARNINGS()

namespace vox {
class CommandBuffer;

struct alignas(16) Light {
    glm::vec4 position; // position.w represents type of light
    glm::vec4 color;    // color.w represents light intensity
    glm::vec4 direction;// direction.w represents range
    glm::vec2 info;     // (only used for spot lights) info.x represents light inner cone angle, info.y represents light outer cone angle
};

struct LightingState {
    std::vector<Light> directional_lights;

    std::vector<Light> point_lights;

    std::vector<Light> spot_lights;

    BufferAllocation light_buffer;
};

/**
 * @brief Calculates the vulkan style projection matrix
 * @param proj The projection matrix
 * @return The vulkan style projection matrix
 */
glm::mat4 vulkan_style_projection(const glm::mat4 &proj);

extern const std::vector<std::string> light_type_definitions;

/**
 * @brief This class defines an interface for subpasses
 *        where they need to implement the draw function.
 *        It is used to construct a RenderPipeline
 */
class Subpass {
public:
    Subpass(RenderContext &render_context, ShaderSource &&vertex_shader, ShaderSource &&fragment_shader);

    Subpass(const Subpass &) = delete;

    Subpass(Subpass &&) = default;

    virtual ~Subpass() = default;

    Subpass &operator=(const Subpass &) = delete;

    Subpass &operator=(Subpass &&) = delete;

    /**
	 * @brief Prepares the shaders and shader variants for a subpass
	 */
    virtual void prepare() = 0;

    /**
	 * @brief Updates the render target attachments with the ones stored in this subpass
	 *        This function is called by the RenderPipeline before beginning the render
	 *        pass and before proceeding with a new subpass.
	 */
    void update_render_target_attachments(RenderTarget &render_target);

    /**
	 * @brief Draw virtual function
	 * @param command_buffer Command buffer to use to record draw commands
	 */
    virtual void draw(CommandBuffer &command_buffer) = 0;

    RenderContext &get_render_context();

    const ShaderSource &get_vertex_shader() const;

    const ShaderSource &get_fragment_shader() const;

    DepthStencilState &get_depth_stencil_state();

    const std::vector<uint32_t> &get_input_attachments() const;

    void set_input_attachments(std::vector<uint32_t> input);

    const std::vector<uint32_t> &get_output_attachments() const;

    void set_output_attachments(std::vector<uint32_t> output);

    void set_sample_count(VkSampleCountFlagBits sample_count);

    const std::vector<uint32_t> &get_color_resolve_attachments() const;

    void set_color_resolve_attachments(std::vector<uint32_t> color_resolve);

    const bool &get_disable_depth_stencil_attachment() const;

    void set_disable_depth_stencil_attachment(bool disable_depth_stencil);

    const uint32_t &get_depth_stencil_resolve_attachment() const;

    void set_depth_stencil_resolve_attachment(uint32_t depth_stencil_resolve);

    const VkResolveModeFlagBits get_depth_stencil_resolve_mode() const;

    void set_depth_stencil_resolve_mode(VkResolveModeFlagBits mode);

    LightingState &get_lighting_state();

    const std::string &get_debug_name() const;

    void set_debug_name(const std::string &name);

protected:
    RenderContext &render_context;

    VkSampleCountFlagBits sample_count{VK_SAMPLE_COUNT_1_BIT};

    // A map of shader resource names and the mode of constant data
    std::unordered_map<std::string, ShaderResourceMode> resource_mode_map;

    /// The structure containing all the requested render-ready lights for the scene
    LightingState lighting_state{};

private:
    std::string debug_name{};

    ShaderSource vertex_shader;

    ShaderSource fragment_shader;

    DepthStencilState depth_stencil_state{};

    /**
	 * @brief When creating the renderpass, pDepthStencilAttachment will
	 *        be set to nullptr, which disables depth testing
	 */
    bool disable_depth_stencil_attachment{false};

    /**
	 * @brief When creating the renderpass, if not None, the resolve
	 *        of the multisampled depth attachment will be enabled,
	 *        with this mode, to depth_stencil_resolve_attachment
	 */
    VkResolveModeFlagBits depth_stencil_resolve_mode{VK_RESOLVE_MODE_NONE};

    /// Default to no input attachments
    std::vector<uint32_t> input_attachments = {};

    /// Default to swapchain output attachment
    std::vector<uint32_t> output_attachments = {0};

    /// Default to no color resolve attachments
    std::vector<uint32_t> color_resolve_attachments = {};

    /// Default to no depth stencil resolve attachment
    uint32_t depth_stencil_resolve_attachment{VK_ATTACHMENT_UNUSED};
};

}// namespace vox
