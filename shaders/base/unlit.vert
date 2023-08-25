#version 450

#include "base/common.h"
#include "base/constants.h"

layout(location = Position) in vec3 POSITION;
layout(location = UV_0) in vec2 TEXCOORD_0;
layout(location = Color_0) in vec4 COLOR_0;
layout(location = Normal) in vec3 NORMAL;
layout(location = Tangent) in vec4 TANGENT;

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

layout(set = 0, binding = 7) uniform tilingOffset {
    vec4 value;
} tiling_offset;

//----------------------------------------------------------------------------------------------------------------------
layout (location = 0) out vec2 v_uv;

void main() {
    vec4 position = vec4(POSITION, 1.0);

    //------------------------------------------------------------------------------------------------------------------
    if (HAS_UV) {
        v_uv = TEXCOORD_0;
    } else {
        // may need this calculate normal
        v_uv = vec2(0., 0.);
    }

    if (NEED_TILINGOFFSET) {
        v_uv = v_uv * tiling_offset.value.xy + tiling_offset.value.zw;
    }

    //------------------------------------------------------------------------------------------------------------------
    gl_Position = camera_data.vp_mat * renderer_data.model_mat * position;
}