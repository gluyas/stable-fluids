#version 460
#line 3

#define INTEGRATOR_MAX_ITERATIONS 512
#define INTEGRATOR_STEP_SIZE 0.008

in vec3 v_pos;

out vec4 gl_FragColor;

void debug_render_velocities();

void main() {
    switch(u_debug_render_mode_and_flags & DEBUG_RENDER_MODE_MASK) {
    case DEBUG_RENDER_MODE_VELOCITIES:
        debug_render_velocities();
        return;
    case DEBUG_RENDER_MODE_AXIS_COLOR:
        gl_FragColor = vec4((v_pos + 1.0)/2.0, 1.0);
        return;
    }

    vec3 sample_point = v_pos;
    vec3 sample_step  = normalize(v_pos - u_camera_pos) * INTEGRATOR_STEP_SIZE;

    // TODO: use a better integrator
    float integral = 0.0;
    int i;
    for (i = 0; i < INTEGRATOR_MAX_ITERATIONS; i++) {
        vec3 sample_point_normalized = (sample_point-u_density_field_bounds[0]) / (u_density_field_bounds[1]-u_density_field_bounds[0]);
        integral += texture(u_density_field, sample_point_normalized).x*INTEGRATOR_STEP_SIZE;
        if (integral >= 1.0) {
            integral = 1.0;
            break;
        }

        sample_point += sample_step;
        if (!in_bounds(sample_point)) break;
    }
    if (i < INTEGRATOR_MAX_ITERATIONS) gl_FragColor = vec4(vec3(1), integral);
    else                               gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
}

// http://www.cse.chalmers.se/edu/year/2010/course/TDA361/grid.pdf
void debug_render_velocities() {
    vec3  u  = to_grid_space(v_pos);
    ivec3 ud = to_grid_index_clamped(v_pos);

    const vec3  v  = u - to_grid_space(u_camera_pos);
    const ivec3 vd = ivec3(sign(v));

    while (in_grid_index_bounds(ud)) {
        // test intersection
        vec4 debug_data = texelFetch(u_debug_data_volume, ud, 0);
        if (debug_data.w > u_debug_render_velocity_threshold) {
            gl_FragColor = vec4((debug_data.xyz + 1.0)/2.0, 1.0);
            return;
        }
        // step to next grid index
        vec3 tmax = (vec3(ud + max(vd, 0)) - u) / v;
        float t = abs(min(min(tmax.x, tmax.y), tmax.z)); // abs prevents a rare infinite loop
        u  += v*t;
        ud += vd*ivec3(step(-t, -tmax));
    }
    discard;
}
