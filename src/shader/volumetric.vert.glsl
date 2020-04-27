#version 460
#line 3

in vec3 a_pos;

out vec3 v_pos;

out vec4 gl_Position;

void main() {
    v_pos = a_pos;
    if ((u_debug_render_flags & DEBUG_RENDER_FLAG_CLIP_BOUNDS) != 0) {
        v_pos = max(v_pos, u_debug_render_clip_bounds[0]);
        v_pos = min(v_pos, u_debug_render_clip_bounds[1]);
    }
    gl_Position = u_camera * vec4(v_pos, 1.0);
}
