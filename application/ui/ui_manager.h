//  Copyright (c) 2022 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <string>
#include <unordered_map>

#include "base/singleton.h"
#include "rendering/render_context.h"
#include "imgui.h"

struct GLFWwindow;

namespace vox {
namespace ui {
/**
 * Some styles that you can use to modify the UI looks
 */
enum class Style {
    IM_CLASSIC_STYLE,
    IM_DARK_STYLE,
    IM_LIGHT_STYLE,
    // https://www.unknowncheats.me/forum/direct3d/189635-imgui-style-settings.html
    DUNE_DARK,
    ALTERNATIVE_DARK
};

/**
 * Handle the creation and drawing of the UI
 */
class UiManager : public Singleton<UiManager> {
public:
    static UiManager &get_singleton();

    static UiManager *get_singleton_ptr();

    /**
     * Create the UI manager. Will setup ImGui internally\
     */
    UiManager(GLFWwindow *glfw_window, rendering::RenderContext *context, Style style = Style::IM_DARK_STYLE);

    /**
     * Destroy the UI manager. Will handle_ ImGui destruction internally
     */
    ~UiManager();

    /**
     * Apply a new style to the UI elements
     */
    static void apply_style(Style style);

    /**
     * Load a font (Returns true on success)
     */
    bool load_font(const std::string &id, const std::string &path, float font_size);

    /**
     * Unload a font (Returns true on success)
     */
    bool unload_font(const std::string &id);

    /**
     * Set the given font as the current one (Returns true on success)
     */
    bool use_font(const std::string &id);

    /**
     * Use the default font (ImGui default font)
     */
    void use_default_font();

    /**
     * Allow the user to enable/disable .ini generation to save his editor layout
     */
    void enable_editor_layout_save(bool value);

    /**
     *  Return true if the editor layout save system is on
     */
    [[nodiscard]] static bool is_editor_layout_save_enabled();

    /**
     * Defines a filename for the editor layout save file
     */
    void set_editor_layout_save_filename(const std::string &filename);

    /**
     * Defines a frequency (in seconds) for the auto saving system of the editor layout
     * @param frequency frequency
     */
    static void set_editor_layout_auto_save_frequency(float frequency);

    /**
     * Returns the current frequency (in seconds) for the auto saving system of the editor layout
     */
    static float editor_layout_autosave_frequency();

    /**
     * Reset the UI layout to the given configuration file
     * @param config config
     */
    static void reset_layout(const std::string &config);

    /**
     * @brief Draws the Gui
     * @param command_buffer Command buffer to register draw-commands
     */
    static void draw(core::CommandBuffer &command_buffer);

private:
    std::unordered_map<std::string, ImFont *> fonts_;
    std::string layout_save_filename_ = "imgui.ini";

private:
    rendering::RenderContext *render_context_{nullptr};

    // Global render pass for frame buffer writes
    VkRenderPass render_pass_ = VK_NULL_HANDLE;

    // Descriptor set pool
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;

    /**
     * @brief Updates the Font Texture
     */
    void update_font_texture();

    /**
     * @brief Setup a default render pass
     *        Can be override in derived class to setup a custom render pass (e.g. for MSAA)
     */
    void setup_render_pass();

    void setup_descriptor_pool();
};

}// namespace ui
template<>
inline ui::UiManager *Singleton<ui::UiManager>::ms_singleton{nullptr};
}// namespace vox
