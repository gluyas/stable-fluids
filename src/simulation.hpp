void sim_init(GLenum density_texture_unit, GLenum debug_data_texture_unit);
void sim_terminate();

void sim_begin_frame();
void sim_end_frame();

extern int sim_pressure_project_iterations;

void sim_update(double dt);

void sim_add_velocity_and_density(
    int x, int y, int z,
    float xf, float yf, float zf, float df,
    float* velocity, float* density, int pitch, bool nan_is_mask,
    int xlen, int ylen, int zlen
);
void sim_add_velocity_and_density_along_ray(
    float rpx, float rpy, float rpz,
    float rvx, float rvy, float rvz,
    float radius, float hardness,
    float vx, float vy, float vz, float d
);

enum SimDebugDataMode {
    None                           = 0,
    NormalizedVelocityAndMagnitude = 1,
};
extern SimDebugDataMode sim_debug_data_mode;

extern bool sim_debug_use_basic_advection;

void sim_update_debug_data();

void sim_debug_reset_density_field(float max_density, double t);
void sim_debug_reset_velocity_field(float max_velocity, double tx, double ty, double tz);
