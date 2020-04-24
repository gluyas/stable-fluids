#define GRID_SIZE 256
#define GRID_ELEMENTS (GRID_SIZE*GRID_SIZE*GRID_SIZE)
#define BLOCK_SIZE 8

#define DENSITY_TEXTURE_SIZE (GRID_ELEMENTS*sizeof(float))

void sim_init(GLuint* texture);

void sim_end();

void sim_update(double dt);

void sim_debug_reset_density_field(double t);
void sim_debug_reset_velocity_field(double tx, double ty, double tz);
