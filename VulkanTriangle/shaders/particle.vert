#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec4 inGradientPos;

layout (location = 0) out vec4 outColor;
layout (location = 1) out float outGradientPos;

layout(binding = 2) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

out gl_PerVertex
{
	vec4 gl_Position;
	float gl_PointSize;
};

void main () 
{
  gl_PointSize = 8.0;
  outColor = vec4(0.035);
  outGradientPos = inGradientPos.x;
  gl_Position = ubo.proj*ubo.view*ubo.model*vec4(inPos.xyz, 1.0);
}