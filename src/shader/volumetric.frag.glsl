#version 460
#line 3

#define INTEGRATOR_MAX_ITERATIONS 512
#define INTEGRATOR_STEP_SIZE 0.008

in vec3 v_pos;

out vec4 gl_FragColor;

void main() {
    vec3 sample_point = v_pos;
    vec3 sample_step  = normalize(v_pos - u_camera_pos) * INTEGRATOR_STEP_SIZE;
    //vec3 sample_step = vec3(INTEGRATOR_STEP_SIZE, 0.0, 0.0);

    // TODO: use a better integrator
    float integral = 0.0;
    int i;
    for (i = 0; i < INTEGRATOR_MAX_ITERATIONS; i++) {
        vec3 sample_point_normalized = (sample_point-u_density_field_bounds[0]) / (u_density_field_bounds[1]-u_density_field_bounds[0]);
        // integral += texture(u_density_field, sample_point_normalized).x*INTEGRATOR_STEP_SIZE;
        float v = texture(u_density_field, sample_point_normalized).x;
        if (v >= 0.5) {
            integral = 1.0;
            break;
        }

        sample_point += sample_step;
        if (any(greaterThan(sample_point, u_density_field_bounds[1]))
        ||  any(lessThan(   sample_point, u_density_field_bounds[0]))
        ) {
            break;
        }
    }
    integral *= pow(1.0 - float(i) / float(INTEGRATOR_MAX_ITERATIONS), 4);
    if (i < INTEGRATOR_MAX_ITERATIONS) gl_FragColor = vec4(vec3(integral), 1.0);
    else                               gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
    // gl_FragColor = vec4(v_pos, 1.0);
}
