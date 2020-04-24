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

long sim_frame_counter = 0;
// tick on even frames: read from tick buffer, write to tock buffer
__host__
inline bool sim_frame_is_tick() {
    return sim_frame_counter % 2 == 0;
}
// tock on odd frames:  read from tock buffer, write to tick buffer
__host__
inline bool sim_frame_is_tock() {
    return sim_frame_counter % 2 != 0;
}

#define TEXTURE_ADDRESS_NORMALIZED true
#define TEXTURE_ADDRESS_MODE       cudaAddressModeBorder

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

// DENSITY FIELD
// OpenGL interop
GLuint                 gl_density_texture;
cudaGraphicsResource_t gl_density_texture_resource;

// kernel read/write targets
surface<void,  cudaSurfaceType3D>                          d_density_write_surface;
texture<float, cudaTextureType3D, cudaReadModeElementType> d_density_read_texture;

// array to map gl_density_texture onto
cudaArray_t d_density_texture_mapped_array;
// backing array for d_density_read_texture
cudaArray_t d_density_read_array; // TODO: experiment with copy-free (double-buffer) implementation

// VELOCITY FIELD
// backing velocity field double buffers
cudaArray_t d_velocity_tick_array;
cudaArray_t d_velocity_tock_array;

// kernel read/write targets
// backing arrays are swapped between tick and tock
surface<void,   cudaSurfaceType3D>                          d_velocity_write_surface;
texture<float4, cudaTextureType3D, cudaReadModeElementType> d_velocity_read_texture;
// TODO: find a use for the 4th velocity component

// TODO: OpenGL texture for rendering velocity field

// INITIALIZATION

__host__
void sim_init(GLuint* density_texture) {
    // OPENGL-HOSTED DENSITY TEXTURE (kernel output)
    glGenTextures(1, &gl_density_texture);
    if (density_texture) *density_texture = gl_density_texture;
    glBindTexture(GL_TEXTURE_3D, gl_density_texture);

    // texture parameters
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);

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

    // initialize CUDA's helper buffer d_density_read_array
    cudaChannelFormatDesc density_format = cudaCreateChannelDesc<float>();
    cudaExtent density_extent = make_cudaExtent(GRID_SIZE, GRID_SIZE, GRID_SIZE);
    cudaCheckErrors(cudaMalloc3DArray(
        &d_density_read_array, &density_format, density_extent,
        cudaArraySurfaceLoadStore
    ));
    cudaCheckErrors(cudaBindTextureToArray(
        d_density_read_texture, d_density_read_array
    ));
    d_density_read_texture.normalized     = TEXTURE_ADDRESS_NORMALIZED;
    d_density_read_texture.addressMode[0] = TEXTURE_ADDRESS_MODE;
    d_density_read_texture.addressMode[1] = TEXTURE_ADDRESS_MODE;
    d_density_read_texture.addressMode[2] = TEXTURE_ADDRESS_MODE;

    // CUDA-HOSTED VELOCITY FIELD BUFFERS
    cudaChannelFormatDesc velocity_format = cudaCreateChannelDesc<float4>();
    cudaExtent velocity_extent = make_cudaExtent(GRID_SIZE, GRID_SIZE, GRID_SIZE);
    cudaCheckErrors(cudaMalloc3DArray(
        &d_velocity_tick_array, &velocity_format, velocity_extent,
        cudaArraySurfaceLoadStore
    ));
    cudaCheckErrors(cudaMalloc3DArray(
        &d_velocity_tock_array, &velocity_format, velocity_extent,
        cudaArraySurfaceLoadStore
    ));
    d_velocity_read_texture.normalized     = TEXTURE_ADDRESS_NORMALIZED;
    d_velocity_read_texture.addressMode[0] = TEXTURE_ADDRESS_MODE;
    d_velocity_read_texture.addressMode[1] = TEXTURE_ADDRESS_MODE;
    d_velocity_read_texture.addressMode[2] = TEXTURE_ADDRESS_MODE;
}

__host__
void sim_end() {
    cudaCheckErrors(cudaGraphicsUnregisterResource(gl_density_texture_resource));
    glDeleteTextures(1, &gl_density_texture);
}

// RESOURCE MAP & UNMAP PROCEDURES

__host__
void sim_map_gl_density() {
    // borrow resource from GL and bind to surface reference
    cudaCheckErrors(cudaGraphicsMapResources(1, &gl_density_texture_resource, 0));
    cudaCheckErrors(cudaGraphicsSubResourceGetMappedArray(
        &d_density_texture_mapped_array, gl_density_texture_resource,
        0, 0
    ));
    // bind the texture's array to the surface reference, enabling writing
    cudaCheckErrors(cudaBindSurfaceToArray(
        d_density_write_surface, d_density_texture_mapped_array
    ));
}

__host__
void sim_unmap_gl_density_without_updating_read_array() {
    cudaCheckErrors(cudaGraphicsUnmapResources(1, &gl_density_texture_resource, 0));
}

__host__
void sim_unmap_gl_density() {
    // TODO: experiment with copy direction and async
    cudaMemcpy3DParms copy = {0};
    copy.srcArray = d_density_texture_mapped_array;
    copy.dstArray = d_density_read_array;
    copy.extent   = make_cudaExtent(GRID_SIZE, GRID_SIZE, GRID_SIZE);
    copy.kind     = cudaMemcpyDeviceToDevice;
    cudaCheckErrors(cudaMemcpy3D(&copy));
    sim_unmap_gl_density_without_updating_read_array();
}

// API FUNCTIONS & KERNELS

__global__
void sim_update_kernel(double dt) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    float px = ((float) x / (float) (GRID_SIZE-1));
    float py = ((float) y / (float) (GRID_SIZE-1));
    float pz = ((float) z / (float) (GRID_SIZE-1));
    float4 v = tex3D(
        d_velocity_read_texture,
        px, py, pz
    );
    v.x *= dt / (float) (GRID_SIZE-1);
    v.y *= dt / (float) (GRID_SIZE-1);
    v.z *= dt / (float) (GRID_SIZE-1);

    // TODO: find q using a real path integrator
    float qx = px - v.x;
    float qy = py - v.y;
    float qz = pz - v.z;
    // advection
    float4 w = tex3D(
        d_velocity_read_texture,
        qx, qy, qz
    );
    surf3Dwrite(
        w, d_velocity_write_surface,
        x*sizeof(float4), y, z,
        cudaBoundaryModeTrap
    );

    // substance transport
    float density = tex3D(
        d_density_read_texture,
        qx, qy, qz
    );
    // float density = v.x;
    surf3Dwrite(
        density, d_density_write_surface,
        x*sizeof(float), y, z,
        cudaBoundaryModeTrap
    );
}

// TODO: constant dt
__host__
void sim_update(double dt) {
    sim_map_gl_density();

    // set read and write buffers
    if (sim_frame_is_tick()) {
        cudaCheckErrors(cudaBindTextureToArray(d_velocity_read_texture,  d_velocity_tick_array));
        cudaCheckErrors(cudaBindSurfaceToArray(d_velocity_write_surface, d_velocity_tock_array));
    } else {
        cudaCheckErrors(cudaBindTextureToArray(d_velocity_read_texture,  d_velocity_tock_array));
        cudaCheckErrors(cudaBindSurfaceToArray(d_velocity_write_surface, d_velocity_tick_array));
    }

    dim3 blocks  = dim3(GRID_SIZE/BLOCK_SIZE, GRID_SIZE/BLOCK_SIZE, GRID_SIZE/BLOCK_SIZE);
    dim3 threads = dim3(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    sim_update_kernel<<<blocks, threads>>>(dt);

    sim_unmap_gl_density();
    sim_frame_counter += 1;
}

// DEBUG FUNCTIONS

__global__
void sim_debug_reset_velocity_field_kernel(float3 time) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    float px = ((float) x / (float) (GRID_SIZE-1))*2.0 - 1.0;
    float py = ((float) y / (float) (GRID_SIZE-1))*2.0 - 1.0;
    float pz = ((float) z / (float) (GRID_SIZE-1))*2.0 - 1.0;

    const float fade_radius = 0.8f;

    float ts[3] = {time.x, time.y, time.z};
    float v[3];
    for (int i = 0; i < 3; i++) {
        // sphere fade
        float d = fmaxf(0.0, (sqrt(px*px + py*py + pz*pz) - fade_radius) / (1-fade_radius));
        float fade = fmin(1.0, 1.0 - sqrt(d));

        float t = 0.3f*ts[i];
        px -= 0.25;
        pz -= 0.25;
        float l = sqrt(px*px + pz*pz);
        float wave = sinf(0.15*TAU*(py + 1.5*sinf(TAU*(0.5*l + 0.3*t)) + 0.15*(1.0+0.2*sinf(TAU*0.4*t))*0.2*t));

        v[i] = (float) (15.0 * wave);
    }
    float4 velocity = make_float4(v[0], v[1], v[2], 0.0);
    // float4 velocity = make_float4(90.0, 0.0, 0.0, 0.0);
    surf3Dwrite(
        velocity, d_velocity_write_surface,
        x*sizeof(float4), y, z,
        cudaBoundaryModeTrap
    );
}

__host__
void sim_debug_reset_velocity_field(double tx, double ty, double tz) {
    // TODO: decide correct buffer to write into
    if (sim_frame_is_tick()) {
        cudaCheckErrors(cudaBindSurfaceToArray(d_velocity_write_surface, d_velocity_tick_array));
    } else {
        cudaCheckErrors(cudaBindSurfaceToArray(d_velocity_write_surface, d_velocity_tock_array));
    }

    dim3 blocks  = dim3(GRID_SIZE/BLOCK_SIZE, GRID_SIZE/BLOCK_SIZE, GRID_SIZE/BLOCK_SIZE);
    dim3 threads = dim3(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    sim_debug_reset_velocity_field_kernel<<<blocks, threads>>>(make_float3(tx, ty, tz));
}

__global__
void sim_debug_reset_density_field_kernel(double time) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

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
        density, d_density_write_surface,
        x*sizeof(float), y, z,
        cudaBoundaryModeTrap
    );
}

__host__
void sim_debug_reset_density_field(double t) {
    sim_map_gl_density();

    dim3 blocks  = dim3(GRID_SIZE/BLOCK_SIZE, GRID_SIZE/BLOCK_SIZE, GRID_SIZE/BLOCK_SIZE);
    dim3 threads = dim3(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    sim_debug_reset_density_field_kernel<<<blocks, threads>>>(t);

    sim_unmap_gl_density();
}
