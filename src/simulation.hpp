void sim_init(GLenum density_texture_unit, GLenum debug_data_texture_unit);

void sim_terminate();

void sim_update(double dt);

enum SimDebugDataMode {
    None                           = 0,
    NormalizedVelocityAndMagnitude = 1,
};
extern SimDebugDataMode sim_debug_data_mode;

void sim_debug_reset_density_field(float max_density, double t);
void sim_debug_reset_velocity_field(float max_velocity, double tx, double ty, double tz);
