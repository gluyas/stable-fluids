void sim_init(GLenum density_texture_unit, GLenum debug_data_texture_unit);
void sim_terminate();

void sim_begin_frame();
void sim_end_frame();

extern int sim_pressure_project_iterations;

extern double sim_vorticity_confinement;

void sim_update(double dt);

void sim_reset_velocity_and_density();

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
    None                            = 0,
    NormalizedVelocityAndMagnitude  = 1,
    NormalizedVorticityAndMagnitude = 2,
};
extern SimDebugDataMode sim_debug_data_mode;

extern bool sim_debug_use_basic_advection;

void sim_update_debug_data();
