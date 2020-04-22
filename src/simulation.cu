#define TAU  6.28318530717958647692
#define TAUf 6.28318530717958647692f

#include <cstdio>

#include "glad/glad.h"
#include <cuda_gl_interop.h>

#include "simulation.hpp"

// https://stackoverflow.com/a/14038590/11617929
#define cudaCheckErrors(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"CUDA Assert %s(%d): %s\n", file, line, cudaGetErrorString(code));
      if (abort) exit(code);
   }
}

// CUDA + GL 3D TEXTURE INTEROP STRATEGY
//
// Start of program:
// 1. Init texture the normal OpenGL way
// 2. Register texture with CUDA as a graphics resource
//  - Need to specify surface read/write flag
//
// Every frame:
// 3. Map the texture resource to CUDA
//  - Must not be accessed by OpenGL while mapped
// 4. Get a CUDA array pointer from the resource
// 5. Bind array to surface reference
// 6. Write to surface in kernel
// 7. Unmap the texture resource
//  - OpenGL can access the texture again
//
// End of program:
// 8. Unregister the texture

// texture as initialized by GL
GLuint                           gl_density_texture;
// CUDA's resource specifier of the texture
cudaGraphicsResource_t           gl_density_texture_resource;
// CUDA's surface reference
surface<void, cudaSurfaceType3D> d_density_texture_surface;

__host__
void init_accelerated_simulation(GLuint* density_texture) {
    // TODO: cuda-hosted velocity field

    // OPENGL-HOSTED DENSITY TEXTURE (kernel output)
    glGenTextures(1, &gl_density_texture);
    if (density_texture) *density_texture = gl_density_texture;
    glBindTexture(GL_TEXTURE_3D, gl_density_texture);

    // texture parameters
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAX_LEVEL, 0);

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // allocate texture storage
    glTexImage3D(
        GL_TEXTURE_3D, 0, GL_R32F,
        GRID_SIZE, GRID_SIZE, GRID_SIZE, 0,
        GL_RED, GL_FLOAT, NULL
    );

    // get a handle to the texture for CUDA
    cudaCheckErrors(cudaGraphicsGLRegisterImage(
        &gl_density_texture_resource, gl_density_texture,
        GL_TEXTURE_3D,
        // flag to enable surface writing to arrays
        cudaGraphicsRegisterFlagsSurfaceLoadStore
    ));
}

__host__
void end_accelerated_simulation() {
    cudaCheckErrors(cudaGraphicsUnregisterResource(gl_density_texture_resource));
    glDeleteTextures(1, &gl_density_texture);
}

__global__
void update_density_field_kernel(double time) {
    int x = blockIdx.x*BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y*BLOCK_SIZE + threadIdx.y;
    int z = blockIdx.z*BLOCK_SIZE + threadIdx.z;

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
    float wave = sinf(TAU*(py + 1.5*sinf(TAU*(0.5*l + 0.3*t)) + 0.15*(1.0+0.2*sinf(TAU*0.4*t))*0.2*t));

    float density = 0.7 * fade * (fmaxf(0.0, wave));
    surf3Dwrite(
        density, d_density_texture_surface,
        x*sizeof(float), y, z,
        cudaBoundaryModeTrap
    );
}

// run the update in a CUDA kernel and copy to array
__host__
void update_density_field_accelerated(double time) {
    // borrow resource from GL and bind to surface reference
    cudaCheckErrors(cudaGraphicsMapResources(1, &gl_density_texture_resource, 0));
    cudaArray_t d_density_texture_array;
    cudaCheckErrors(cudaGraphicsSubResourceGetMappedArray(
        &d_density_texture_array, gl_density_texture_resource,
        0, 0
    ));
    // bind the texture's array to the surface reference, enabling writing
    cudaCheckErrors(cudaBindSurfaceToArray(
        d_density_texture_surface, d_density_texture_array
    ));

    // run kernel
    dim3 blocks  = dim3(GRID_SIZE/BLOCK_SIZE, GRID_SIZE/BLOCK_SIZE, GRID_SIZE/BLOCK_SIZE);
    dim3 threads = dim3(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    update_density_field_kernel<<<blocks, threads>>>(time);

    // return the resource to GL
    cudaCheckErrors(cudaGraphicsUnmapResources(1, &gl_density_texture_resource, 0));
}
