#version 330

in vec3 Position;
in vec3 Normal;
in vec2 UV;
in vec3 CamDir;

uniform vec3 uColor;
out vec3 vColor;
uniform mat4 model;
uniform mat4 invTrModel;
uniform bool nShade;

out vec3 vPosition;
out vec3 vNormal;
out vec2 vUV;
out vec3 vCamDir;

void main() {
  gl_Position = model*vec4(Position,1.0);
  vColor = uColor;
  vPosition = Position;
  vNormal = normalize(invTrModel * vec4(normalize(Normal), 0.0f)).xyz;
  vUV = UV;
  vCamDir = normalize(CamDir);
}
