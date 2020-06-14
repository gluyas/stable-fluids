#line 1002

uniform double u_time;

uniform sampler3D u_density_field;
uniform mat2x3    u_density_field_bounds;

uniform mat4 u_camera;
uniform vec3 u_camera_pos;

// TODO: move debug visualization into another shader program
uniform sampler3D u_debug_data_volume;

uniform int   u_debug_render_mode_and_flags     = DEBUG_RENDER_MODE_DEFAULT | DEBUG_RENDER_FLAG_NONE;
uniform float u_debug_render_density_factor     = 1.0;
uniform float u_debug_render_velocity_threshold = 0.0;
uniform ivec3 u_debug_render_clip_bounds[2]     = ivec3[2](ivec3(0), ivec3(SIM_GRID_SIZE));

vec3 to_texture_space(vec3 p) {
    return (p-u_density_field_bounds[0]) / (u_density_field_bounds[1]-u_density_field_bounds[0]);
}

vec3 to_grid_space(vec3 p) {
    return to_texture_space(p) * SIM_GRID_SIZE;
}
ivec3 grid_space_to_index(vec3 p) {
    return ivec3(floor(p));
}
vec3 grid_index_to_world_space(ivec3 p) {
    return u_density_field_bounds[0] + vec3(p)/SIM_GRID_SIZE*(u_density_field_bounds[1]-u_density_field_bounds[0]);
}
ivec3 to_grid_index(vec3 p) {
    return grid_space_to_index(to_grid_space(p));
}
ivec3 to_grid_index_clamped(vec3 p) {
    if ((u_debug_render_mode_and_flags & DEBUG_RENDER_FLAG_CLIP_BOUNDS) != 0) {
        return min(max(to_grid_index(p), u_debug_render_clip_bounds[0]), u_debug_render_clip_bounds[1]-1);
    } else {
        return min(max(to_grid_index(p), 0), SIM_GRID_SIZE-1);
    }
}
bool in_grid_index_bounds(ivec3 p) {
    if ((u_debug_render_mode_and_flags & DEBUG_RENDER_FLAG_CLIP_BOUNDS) != 0) {
        return all(greaterThanEqual(p, u_debug_render_clip_bounds[0]))
            && all(lessThan(        p, u_debug_render_clip_bounds[1]));
    } else {
        return all(greaterThanEqual(p, ivec3(0)))
            && all(lessThan(        p, ivec3(SIM_GRID_SIZE)));
    }
}

bool in_bounds(vec3 p) {
    if ((u_debug_render_mode_and_flags & DEBUG_RENDER_FLAG_CLIP_BOUNDS) != 0) {
        return in_grid_index_bounds(to_grid_index(p));
    } else {
        return all(greaterThanEqual(p, u_density_field_bounds[0]))
            && all(lessThanEqual(   p, u_density_field_bounds[1]));
    }
}
