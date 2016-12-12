#version 330

in vec3 vPosition;
in vec3 vNormal;
in vec2 vUV;
in vec3 vCamDir;

out vec3 color;
uniform vec3 uColor;
uniform bool nShade;

void main() {
	if (nShade) {
		vec3 lDir = -normalize(vec3(1,-4,4));
		vec3 hDir = normalize(vCamDir + lDir);
		float hDot = max(dot(hDir, vNormal),0.0);
		float spec = pow(hDot, 5);
		color = 0.3 + 1.0*vec3(spec)*uColor + clamp(dot(vNormal, lDir),0,1)*uColor;
	}
	else
		color = uColor;
}
