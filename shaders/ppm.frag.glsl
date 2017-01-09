#version 330

in vec3 vPosition;
in vec3 vNormal;
in vec2 vUV;

out vec3 color;

uniform vec3 CamDir;
uniform vec3 uColor;
uniform bool nShade;
uniform bool dbgNormals;

void main() {
	if (nShade) {
    if (dbgNormals)
      color = abs(vNormal);
    else {
      vec3 lPos = vec3(2,-2,-1);
      vec3 lDir = normalize(lPos-vPosition);
      vec3 hDir = normalize(CamDir + lDir);
      float hDot = max(dot(hDir, vNormal),0.0);
      float spec = pow(hDot, 5);
      color = 0.3 + 1.0*vec3(spec)*uColor + 5.0*clamp(dot(vNormal, lDir)/dot(lPos-vPosition,lPos-vPosition),0,1)*uColor;
	  }
  }
	else
		color = uColor;
}
