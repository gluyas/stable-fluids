#version 460
#line 3

in vec3 a_pos;

out vec3 v_pos;

out vec4 gl_Position;

void main() {
    v_pos = a_pos;
    gl_Position = u_camera * vec4(a_pos, 1.0);
}
