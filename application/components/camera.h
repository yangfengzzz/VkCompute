//  Copyright (c) 2022 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "math/bounding_frustum.h"
#include "math/matrix4x4.h"
#include "math/ray3.h"
#include "framework/shader/shader_data.h"
#include "ecs/component.h"
#include "base/layer.h"
#include "base/update_flag.h"

namespace vox {
class Transform;

/**
 * Camera clear flags enumeration.
 */
enum class CameraClearFlags {
    /* Clear depth and color from background. */
    DEPTH_COLOR,
    /* Clear depth only. */
    DEPTH,
    /* Do nothing. */
    NONE
};

/**
 * Camera component, as the entrance to the three-dimensional world.
 */
class Camera : public Component {
public:
    struct alignas(16) CameraData {
        Matrix4x4F view_mat;
        Matrix4x4F proj_mat;
        Matrix4x4F vp_mat;
        Matrix4x4F view_inv_mat;
        Matrix4x4F proj_inv_mat;
        Point3F camera_pos;
    };

    /** Shader data. */
    ShaderData shader_data_;

    /** Rendering priority - A Camera with higher priority will be rendered on top of a camera with lower priority. */
    int priority_ = 0;

    /** Whether to enable frustum culling, it is enabled by default. */
    bool enable_frustum_culling_ = true;

    /**
     * Determining what to clear when rendering by a Camera.
     * @defaultValue `CameraClearFlags.DepthColor`
     */
    CameraClearFlags clear_flags_ = CameraClearFlags::DEPTH_COLOR;

    /**
     * Culling mask - which layers the camera renders.
     * @remarks Support bit manipulation, corresponding to Entity's layer.
     */
    Layer culling_mask_ = Layer::EVERYTHING;

    /**
     * Create the Camera component.
     * @param entity - Entity
     */
    explicit Camera(Entity *entity);

    [[nodiscard]] const BoundingFrustum &get_frustum() const;

    /**
     * Near clip plane - the closest point to the camera when rendering occurs.
     */
    [[nodiscard]] float get_near_clip_plane() const;

    void set_near_clip_plane(float value);

    /**
     * Far clip plane - the furthest point to the camera when rendering occurs.
     */
    [[nodiscard]] float get_far_clip_plane() const;

    void set_far_clip_plane(float value);

    /**
     * The camera's view angle. activating when camera use perspective projection.
     */
    [[nodiscard]] float get_field_of_view() const;

    void set_field_of_view(float value);

    /**
     * Aspect ratio. The default is automatically calculated by the viewport's aspect ratio. If it is manually set,
     * the manual value will be kept. Call ResetAspectRatio() to restore it.
     */
    [[nodiscard]] float get_aspect_ratio() const;

    void set_aspect_ratio(float value);

    /**
     * Viewport, normalized expression, the upper left corner is (0, 0), and the lower right corner is (1, 1).
     * @remarks Re-assignment is required after modification to ensure that the modification takes effect.
     */
    [[nodiscard]] Vector4F get_viewport() const;

    void set_viewport(const Vector4F &value);

    /**
     * Whether it is orthogonal, the default is false. True will use orthographic projection, false will use perspective
     * projection.
     */
    [[nodiscard]] bool is_orthographic() const;

    void set_is_orthographic(bool value);

    /**
     * Half the size of the camera in orthographic mode.
     */
    [[nodiscard]] float get_orthographic_size() const;

    void set_orthographic_size(float value);

    /**
     * View matrix.
     */
    Matrix4x4F get_view_matrix();

    /**
     * The projection matrix is calculated by the relevant parameters of the camera by default.
     * If it is manually set, the manual value will be maintained. Call ResetProjectionMatrix() to restore it.
     */
    void set_projection_matrix(const Matrix4x4F &value);

    Matrix4x4F get_projection_matrix();

    /**
     * The inverse of the projection matrix.
     */
    Matrix4x4F inverse_projection_matrix();

    /**
     * Whether to enable HDR.
     * @todo When render pipeline modification
     */
    bool is_hdr_enabled();

    void enable_hdr(bool value);

public:
    /**
     * Restore the automatic calculation of projection matrix through FieldOfView, NearClipPlane and FarClipPlane.
     */
    void reset_projection_matrix();

    /**
     * Restore the automatic calculation of the aspect ratio through the viewport aspect ratio.
     */
    void reset_aspect_ratio();

    /**
     * Transform a point from world space to viewport space.
     * @param point - Point in world space
     * @return out - A point in the viewport space, X and Y are the viewport space coordinates,
     * Z is the viewport depth, the near clipping plane is 0, the far clipping plane is 1, and W is the world unit
     * distance from the camera
     */
    Vector4F world_to_viewport_point(const Point3F &point);

    /**
     * Transform a point from viewport space to world space.
     * @param point - Point in viewport space, X and Y are the viewport space coordinates,
     * Z is the viewport depth. The near clipping plane is 0, and the far clipping plane is 1
     * @returns Point in world space
     */
    Point3F viewport_to_world_point(const Vector3F &point);

    /**
     * Generate a ray by a point in viewport.
     * @param point - Point in viewport space, which is represented by normalization
     * @returns Ray
     */
    Ray3F viewport_point_to_ray(const Vector2F &point);

    /**
     * Transform the X and Y coordinates of a point from screen space to viewport space
     * @param point - Point in screen space
     * @returns Point in viewport space
     */
    [[nodiscard]] Vector2F screen_to_viewport_point(const Vector2F &point) const;

    [[nodiscard]] Vector3F screen_to_viewport_point(const Vector3F &point) const;

    /**
     * Transform the X and Y coordinates of a point from viewport space to screen space.
     * @param point - Point in viewport space
     * @returns Point in screen space
     */
    [[nodiscard]] Vector2F viewport_to_screen_point(const Vector2F &point) const;

    [[nodiscard]] Vector3F viewport_to_screen_point(const Vector3F &point) const;

    [[nodiscard]] Vector4F viewport_to_screen_point(const Vector4F &point) const;

    /**
     * Transform a point from world space to screen space.
     * @param point - Point in world space
     * @returns Point of screen space
     */
    Vector4F world_to_screen_point(const Point3F &point);

    /**
     * Transform a point from screen space to world space.
     * @param point - Screen space point
     * @returns Point in world space
     */
    Point3F screen_to_world_point(const Vector3F &point);

    /**
     * Generate a ray by a point in screen.
     * @param point - Point in screen space, the unit is pixel
     * @returns Ray
     */
    Ray3F screen_point_to_ray(const Vector2F &point);

    void resize(uint32_t win_width, uint32_t win_height, uint32_t fb_width, uint32_t fb_height);

    [[nodiscard]] uint32_t get_width() const;

    [[nodiscard]] uint32_t get_height() const;

    [[nodiscard]] uint32_t get_framebuffer_width() const;

    [[nodiscard]] uint32_t get_framebuffer_height() const;

    void update();

public:
    void on_active() override;

    void on_inactive() override;

private:
    void proj_mat_change();

    static Point3F inner_viewport_to_world_point(const Vector3F &point, const Matrix4x4F &inv_view_proj_mat);

    /**
     * The inverse matrix of view projection matrix.
     */
    Matrix4x4F inv_view_proj_mat();

    CameraData camera_data_;
    const std::string camera_property_;

    BoundingFrustum frustum_ = BoundingFrustum();

    bool is_orthographic_ = false;
    bool is_proj_mat_setting_ = false;
    float near_clip_plane_ = 0.1;
    float far_clip_plane_ = 500;
    float field_of_view_ = 45;
    float orthographic_size_ = 10;
    bool is_projection_dirty_ = true;
    bool is_inv_proj_mat_dirty_ = true;
    bool is_frustum_project_dirty_ = true;
    std::optional<float> custom_aspect_ratio_ = std::nullopt;

    std::unique_ptr<UpdateFlag> frustum_view_change_flag_;
    Transform *transform_;
    std::unique_ptr<UpdateFlag> is_view_matrix_dirty_;
    std::unique_ptr<UpdateFlag> is_inv_view_proj_dirty_;
    Matrix4x4F projection_matrix_ = Matrix4x4F();
    Matrix4x4F view_matrix_ = Matrix4x4F();
    Vector4F viewport_ = Vector4F(0, 0, 1, 1);
    Matrix4x4F inverse_projection_matrix_ = Matrix4x4F();
    Vector2F last_aspect_size_ = Vector2F();
    Matrix4x4F inv_view_proj_mat_ = Matrix4x4F();

    uint32_t width_{0};
    uint32_t height_{0};
    uint32_t fb_width_{0};
    uint32_t fb_height_{0};
};

}// namespace vox
