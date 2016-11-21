#version 330

in vec3 Position;
uniform vec3 uColor;
out vec3 vColor;
uniform mat4 model;

void main() {
  gl_Position = model*vec4(Position,1.0);
  vColor = uColor;
}
