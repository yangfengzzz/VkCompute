#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 5) uniform cameraData {
    mat4 view_mat;
    mat4 proj_mat;
    mat4 vp_mat;
    mat4 view_inv_mat;
    mat4 proj_inv_mat;
    vec3 camera_pos;
} camera_data;

layout(set = 0, binding = 6) uniform rendererData {
    mat4 local_mat;
    mat4 model_mat;
    mat4 normal_mat;
} renderer_data;

layout(location = 0) in float height;
layout(location = 1) in vec2 xyPos;

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = camera_data.vp_mat * renderer_data.model_mat * vec4(xyPos.xy, height, 1.0f);
    fragColor = vec3(0.0f, (height + 0.5f), 0.0f);
}
