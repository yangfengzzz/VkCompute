#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    float frame;
} ubo;

layout(location = 0) in float pointInsideCircle;
layout(location = 1) in vec2 xyPos;
 
layout(location = 0) out vec3 fragColor;
 
const float PI = 3.1415926;
 
out gl_PerVertex
{
    vec4 gl_Position;
    float gl_PointSize;
};
 
void main() {
    gl_PointSize = 1.0;
    gl_Position = vec4(xyPos.xy, 0.0f, 1.0f);
    float color_r = 1.0f + 0.5f * sin(ubo.frame / 100.0f);
    float color_g = 1.0f + 0.5f * sin((ubo.frame / 100.0f) + (2.0f*PI/3.0f));
    float color_b = 1.0f;
    fragColor = vec3(pointInsideCircle.x * color_r, pointInsideCircle.x * color_g, color_b);
}
