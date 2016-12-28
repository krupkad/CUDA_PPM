#version 330

layout(location = 0) in vec3 Position;
layout(location = 1) in vec3 Normal;
layout(location = 2) in vec2 UV;

uniform vec3 CamDir;
uniform vec3 uColor;
out vec3 vColor;
uniform mat4 model;
uniform mat4 invTrModel;
uniform bool nShade;

out vec3 vPosition;
out vec3 vNormal;
out vec2 vUV;

void main() {
  gl_Position = model*vec4(Position,1.0);
  vColor = uColor;
  vPosition = Position;
  vNormal = normalize(invTrModel * vec4(normalize(Normal), 0.0f)).xyz;
  vUV = UV;
}
