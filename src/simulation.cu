#define TAU  6.28318530717958647692
#define TAUf 6.28318530717958647692f

#include "simulation.hpp"

float* d_density_field;

__global__
void update_density_field_kernel(float* density_field, double time) {
    int x = blockIdx.x*BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y*BLOCK_SIZE + threadIdx.y;
    int z = blockIdx.z*BLOCK_SIZE + threadIdx.z;

    int index = x + y*GRID_SIZE + z*GRID_SIZE*GRID_SIZE;

    float px = ((float) x / (float) (GRID_SIZE-1))*2.0 - 1.0;
    float py = ((float) y / (float) (GRID_SIZE-1))*2.0 - 1.0;
    float pz = ((float) z / (float) (GRID_SIZE-1))*2.0 - 1.0;

    const float fade_radius = 0.8f;

    // sphere fade
    float d = fmaxf(0.0, (sqrt(px*px + py*py + pz*pz) - fade_radius) / (1-fade_radius));
    float fade = fmin(1.0, 1.0 - sqrt(d));

    float t = 0.3*time;
    px -= 0.25;
    pz -= 0.25;
    float l = sqrt(px*px + pz*pz);
    float wave = sinf(TAU*(py + 1.5*sinf(TAU*(0.5*l + 0.3*t)) + 0.15*(1.0+0.2*sin(TAU*0.4*t))*0.2*t));

    density_field[index] = 0.7 * fade * (fmaxf(0.0, wave));
}

__host__
void init_accelerated_simulation() {
    cudaMalloc(&d_density_field, GRID_FOOTPRINT);
}

__host__
void end_accelerated_simulation() {
    cudaFree(d_density_field);
}

// run the update in a CUDA kernel and copy to array
__host__
void update_density_field_accelerated(float *h_density_field, double time) {
    cudaMemcpy(d_density_field, h_density_field, GRID_FOOTPRINT, cudaMemcpyHostToDevice);

    dim3 blocks  = dim3(GRID_SIZE/BLOCK_SIZE, GRID_SIZE/BLOCK_SIZE, GRID_SIZE/BLOCK_SIZE);
    dim3 threads = dim3(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);

    update_density_field_kernel<<<blocks, threads>>>(d_density_field, time);

    cudaMemcpy(h_density_field, d_density_field, GRID_FOOTPRINT, cudaMemcpyDeviceToHost);
}
