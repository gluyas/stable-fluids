#define GRID_SIZE 128
#define BLOCK_SIZE 8

#define GRID_FOOTPRINT GRID_SIZE*GRID_SIZE*GRID_SIZE*sizeof(float)

void init_accelerated_simulation();

void end_accelerated_simulation();

void update_density_field_accelerated(float *h_density_field, double time);
