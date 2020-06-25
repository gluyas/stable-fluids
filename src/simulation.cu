#include <cstdio>

#include "glad/glad.h"
#include <cuda_gl_interop.h>

#include "global_defines.h"
#include "simulation.hpp"

#define GRID_SIZE SIM_GRID_SIZE
#define GRID_ELEMENTS (GRID_SIZE*GRID_SIZE*GRID_SIZE)

// operate kernels on 8x8 sheets
static_assert(GRID_SIZE % 8 == 0, "SIM_GRID_SIZE must be a multiple of 8");
#define BLOCK_DIM  dim3(GRID_SIZE/8, GRID_SIZE/8, GRID_SIZE)
#define THREAD_DIM dim3(8, 8, 1)

#define DENSITY_TEXTURE_SIZE (GRID_ELEMENTS*sizeof(float))

// https://stackoverflow.com/a/14038590/11617929
#define cudaCheckErrors(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Assert %s(%d): %s\n", file, line, cudaGetErrorString(code));
        if (abort) exit(code);
    }
}

void set_volume_texture_parameters(textureReference* texture) {
    texture->normalized = true;
    texture->filterMode = cudaFilterModeLinear;
    texture->addressMode[0] = texture->addressMode[1] = texture->addressMode[2] = cudaAddressModeClamp; // TODO: border or clamp?
    texture->minMipmapLevelClamp = texture->maxMipmapLevelClamp = 0.0;
    texture->mipmapFilterMode = cudaFilterModePoint;
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

// DENSITY FIELD
// OpenGL interop
GLuint                 gl_density_texture;
cudaGraphicsResource_t gl_density_texture_resource;

// kernel read/write targets
surface<void,  cudaSurfaceType3D>                          d_density_write_surface;
texture<float, cudaTextureType3D, cudaReadModeElementType> d_density_read_texture;
surface<void,  cudaSurfaceType3D>                          d_density_copy_surface;

// array to map gl_density_texture onto
cudaArray_t d_density_texture_mapped_array;
// backing array for d_density_read_texture
cudaArray_t d_density_read_array; // TODO: experiment with copy-free (double-buffer) implementation

// VELOCITY FIELD
// backing velocity field double buffers
cudaArray_t d_velocity_tick_array;
cudaArray_t d_velocity_tock_array;

// kernel read/write targets
surface<void,   cudaSurfaceType3D>                          d_velocity_write_surface;
texture<float4, cudaTextureType3D, cudaReadModeElementType> d_velocity_read_texture;
surface<void,   cudaSurfaceType3D>                          d_velocity_copy_surface;

// DEBUG DATA FIELD
// OpenGL interop
GLuint                 gl_debug_data_texture;
cudaGraphicsResource_t gl_debug_data_texture_resource;

// surface reference
surface<void, cudaSurfaceType3D> d_debug_data_write_surface;

// output parameters
SimDebugDataMode sim_debug_data_mode = None;

// SIMULATION PARAMETERS

int sim_pressure_project_iterations = 16;

double sim_vorticity_confinement = 4.0;

// INITIALIZATION

// call BEFORE update double-buffered state
__host__
void sim_swap_buffers() {
    static bool tick = false;
    tick = !tick;

    if (tick) {
        cudaCheckErrors(cudaBindTextureToArray(d_velocity_read_texture,  d_velocity_tick_array));
        cudaCheckErrors(cudaBindSurfaceToArray(d_velocity_write_surface, d_velocity_tock_array));
    } else {
        cudaCheckErrors(cudaBindTextureToArray(d_velocity_read_texture,  d_velocity_tock_array));
        cudaCheckErrors(cudaBindSurfaceToArray(d_velocity_write_surface, d_velocity_tick_array));
    }
}

__host__
void sim_init(GLenum density_texture_unit, GLenum debug_data_texture_unit) {
    // OPENGL-HOSTED DENSITY TEXTURE
    glActiveTexture(density_texture_unit);
    glGenTextures(1, &gl_density_texture);
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
    glActiveTexture(0);

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
    set_volume_texture_parameters(&d_density_read_texture);

    // OPENGL-HOSTED DEBUG TEXTURE
    glActiveTexture(debug_data_texture_unit);
    glGenTextures(1, &gl_debug_data_texture);
    glBindTexture(GL_TEXTURE_3D, gl_debug_data_texture);

    // texture parameters
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAX_LEVEL, 0);

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // allocate texture storage
    glTexImage3D(
        GL_TEXTURE_3D, 0, GL_RGBA32F,
        GRID_SIZE, GRID_SIZE, GRID_SIZE, 0,
        GL_RGBA, GL_FLOAT, NULL
    );

    // get a handle to the texture for CUDA
    cudaCheckErrors(cudaGraphicsGLRegisterImage(
        &gl_debug_data_texture_resource, gl_debug_data_texture,
        GL_TEXTURE_3D,
        // flag to enable surface writing to arrays
        cudaGraphicsRegisterFlagsSurfaceLoadStore
    ));
    glActiveTexture(0);

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
    set_volume_texture_parameters(&d_velocity_read_texture);

    // initialize double buffers by calling sim_swap_buffers
    sim_swap_buffers();
}

__host__
void sim_terminate() {
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

__host__
void sim_map_gl_debug_data() {
    // borrow resource from GL and bind to surface reference
    cudaCheckErrors(cudaGraphicsMapResources(1, &gl_debug_data_texture_resource, 0));
    cudaArray_t d_debug_data_texture_mapped_array;
    cudaCheckErrors(cudaGraphicsSubResourceGetMappedArray(
        &d_debug_data_texture_mapped_array, gl_debug_data_texture_resource,
        0, 0
    ));
    // bind the texture's array to the surface reference, enabling writing
    cudaCheckErrors(cudaBindSurfaceToArray(
        d_debug_data_write_surface, d_debug_data_texture_mapped_array
    ));
}

__host__
void sim_unmap_gl_debug_data() {
    cudaCheckErrors(cudaGraphicsUnmapResources(1, &gl_debug_data_texture_resource, 0));
}

// SIMULATION FUNCTIONS

__global__
void sim_pressure_project_jacobi_iteration_kernel() {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    const float d = 1.0 / GRID_SIZE;
    float px = ((float) x + 0.5) * d;
    float py = ((float) y + 0.5) * d;
    float pz = ((float) z + 0.5) * d;

    // TODO: compare performance of texture fetch and surface read
    float4 vc = tex3D(d_velocity_read_texture, px,   py,   pz);

    float4 vl = make_float4(0,0,0, vc.w); if (x != 0)           vl = tex3D(d_velocity_read_texture, px-d, py,   pz);
    float4 vr = make_float4(0,0,0, vc.w); if (x != GRID_SIZE-1) vr = tex3D(d_velocity_read_texture, px+d, py,   pz);

    float4 vb = make_float4(0,0,0, vc.w); if (y != 0)           vb = tex3D(d_velocity_read_texture, px,   py-d, pz);
    float4 vf = make_float4(0,0,0, vc.w); if (y != GRID_SIZE-1) vf = tex3D(d_velocity_read_texture, px,   py+d, pz);

    float4 vd = make_float4(0,0,0, vc.w); if (z != 0)           vd = tex3D(d_velocity_read_texture, px,   py,   pz-d);
    float4 vu = make_float4(0,0,0, vc.w); if (z != GRID_SIZE-1) vu = tex3D(d_velocity_read_texture, px,   py,   pz+d);

    float  u_div  = (vr.x-vl.x + vf.y-vb.y + vu.z-vd.z) * 0.5;
    float  p_out  = (vl.w+vr.w+vb.w+vf.w+vd.w+vu.w - u_div) * (1.0/6.0);

    float3 p_grad = make_float3((vr.w-vl.w)*0.5, (vf.w-vb.w)*0.5, (vu.w-vd.w)*0.5);

    float4 v_out  = make_float4(vc.x-p_grad.x, vc.y-p_grad.y, vc.z-p_grad.z, p_out);
    surf3Dwrite(v_out, d_velocity_write_surface, sizeof(float4)*x, y, z, cudaBoundaryModeTrap);
}

bool sim_debug_use_basic_advection = false;
__global__
void sim_advection_first_order_kernel(double dt);

__global__
void sim_advection_bfecc_kernel(double dt) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    float px = ((float) x + 0.5) / GRID_SIZE;
    float py = ((float) y + 0.5) / GRID_SIZE;
    float pz = ((float) z + 0.5) / GRID_SIZE;
    float4 pv = tex3D(d_velocity_read_texture, px, py, pz);

    // ADVECTION
    // implements BFECC method described in Selle et al.

    // forward advection
    float qx = px - pv.x*dt;
    float qy = py - pv.y*dt;
    float qz = pz - pv.z*dt;
    float4 qv = tex3D(d_velocity_read_texture, qx, qy, qz);

    // reverse advection
    float rx = qx + qv.x*dt;
    float ry = qy + qv.y*dt;
    float rz = qz + qv.z*dt;

    // error correction
    float sx = 0.5*(3*px - rx);
    float sy = 0.5*(3*py - ry);
    float sz = 0.5*(3*pz - rz);
    float4 sv = tex3D(d_velocity_read_texture, sx, sy, sz);

    // final advection
    float tx = sx - sv.x*dt;
    float ty = sy - sv.y*dt;
    float tz = sz - sv.z*dt;

    float4 w = tex3D(d_velocity_read_texture, tx, ty, tz);
    w.w = 0.0;
    surf3Dwrite(
        w, d_velocity_write_surface,
        x*sizeof(float4), y, z,
        cudaBoundaryModeTrap
    );

    // SUBSTANCE TRANSPORT
    float d = tex3D(d_density_read_texture, tx, ty, tz);
    surf3Dwrite(
        d, d_density_write_surface,
        x*sizeof(float), y, z,
        cudaBoundaryModeTrap
    );
}

__device__
inline float3 discrete_curl(float4 l, float4 r, float4 b, float4 f, float4 d, float4 u) {
    float3 curl;
    curl.x = (f.z - b.z) - (u.y - d.y);
    curl.y = (u.x - d.x) - (r.z - l.z);
    curl.z = (r.y - l.y) - (f.x - b.x);
    return curl;
}

__global__
void sim_vorticity_confinement_kernel(double dt_epsilon) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    const float d = 1.0 / GRID_SIZE;
    const float dd = d+d;
    float px = ((float) x + 0.5) / GRID_SIZE;
    float py = ((float) y + 0.5) / GRID_SIZE;
    float pz = ((float) z + 0.5) / GRID_SIZE;

    // TOOD: verify units
    // TODO: combine with advection kernel
    // TODO: make use of shared memory

    // collect primary and secondary neighbours' velocities
    // TODO: use only forward or backwards diferences, change orientation each frame
    float4 vc  = tex3D(d_velocity_read_texture, px,   py,   pz);

    float4 vl  = tex3D(d_velocity_read_texture, px-d, py,   pz);
    float4 vlb = tex3D(d_velocity_read_texture, px-d, py-d, pz);
    float4 vlf = tex3D(d_velocity_read_texture, px-d, py+d, pz);
    float4 vll = tex3D(d_velocity_read_texture, px-dd,py,  pz);

    float4 vr  = tex3D(d_velocity_read_texture, px+d, py,   pz);
    float4 vrb = tex3D(d_velocity_read_texture, px+d, py-d, pz);
    float4 vrf = tex3D(d_velocity_read_texture, px+d, py+d, pz);
    float4 vrr = tex3D(d_velocity_read_texture, px+dd,py, pz);

    float4 vb  = tex3D(d_velocity_read_texture, px,   py-d, pz);
    float4 vbd = tex3D(d_velocity_read_texture, px,   py-d, pz-d);
    float4 vbu = tex3D(d_velocity_read_texture, px,   py-d, pz+d);
    float4 vbb = tex3D(d_velocity_read_texture, px,   py-dd,pz);

    float4 vf  = tex3D(d_velocity_read_texture, px,   py+d, pz);
    float4 vfd = tex3D(d_velocity_read_texture, px,   py+d, pz-d);
    float4 vfu = tex3D(d_velocity_read_texture, px,   py+d, pz+d);
    float4 vff = tex3D(d_velocity_read_texture, px,   py+dd,pz);

    float4 vd  = tex3D(d_velocity_read_texture, px,   py,   pz-d);
    float4 vdl = tex3D(d_velocity_read_texture, px-d, py,   pz-d);
    float4 vdr = tex3D(d_velocity_read_texture, px+d, py,   pz-d);
    float4 vdd = tex3D(d_velocity_read_texture, px+d, py,   pz-dd);

    float4 vu  = tex3D(d_velocity_read_texture, px,   py,   pz+d);
    float4 vul = tex3D(d_velocity_read_texture, px-d, py,   pz+d);
    float4 vur = tex3D(d_velocity_read_texture, px+d, py,   pz+d);
    float4 vuu = tex3D(d_velocity_read_texture, px+d, py,   pz+dd);

    // calculate vorticities (curl of velocity field) and magnitudes of primary neighbours
    float3 eta; // gradient of the magnitude of the vorticity

    float3 cl = discrete_curl(vll, vc, vlb, vlf, vdl, vul);
    float3 cr = discrete_curl(vc, vrr, vrb, vrf, vdr, vur);
    eta.x = norm3df(cr.x, cr.y, cr.z) - norm3df(cl.x, cl.y, cl.z);

    float3 cb = discrete_curl(vlb, vrb, vbb, vc, vbd, vbu);
    float3 cf = discrete_curl(vlf, vrf, vc, vff, vfd, vfu);
    eta.y = norm3df(cf.x, cf.y, cf.z) - norm3df(cb.x, cb.y, cb.z);

    float3 cd = discrete_curl(vdl, vdr, vbd, vfd, vdd, vc);
    float3 cu = discrete_curl(vul, vur, vbu, vfu, vc, vuu);
    eta.z = norm3df(cu.x, cu.y, cu.z) - norm3df(cd.x, cd.y, cd.z);

    // calculate vorticity confinement
    float eta_norm = 1.0/norm3df(eta.x, eta.y, eta.z);
    if (isfinite(eta_norm)) {
        eta.x *= eta_norm; eta.y *= eta_norm; eta.z *= eta_norm;

        float3 cc = discrete_curl(vl, vr, vb, vf, vd, vu);

        // apply vorticity confinement (cross of eta and cc)
        vc.x += (eta.y*cc.z - eta.z*cc.y)*dt_epsilon;
        vc.y += (eta.z*cc.x - eta.x*cc.z)*dt_epsilon;
        vc.z += (eta.x*cc.y - eta.y*cc.x)*dt_epsilon;
    }
    surf3Dwrite(
        vc, d_velocity_write_surface,
        x*sizeof(float4), y, z,
        cudaBoundaryModeTrap
    );
}

__global__
void sim_update_debug_data_kernel(SimDebugDataMode debug_data_mode) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    const float d = 1.0 / GRID_SIZE;
    float px = ((float) x + 0.5) / GRID_SIZE;
    float py = ((float) y + 0.5) / GRID_SIZE;
    float pz = ((float) z + 0.5) / GRID_SIZE;
    float4 v = tex3D(d_velocity_read_texture,
        px, py, pz
    );

    float4 debug_data;
    switch (debug_data_mode) {
    case NormalizedVelocityAndMagnitude: {
        float l = norm3df(v.x, v.y, v.z);
        float n = 0.0; if (l > 0.0) n = 1.0 / l;
        debug_data = make_float4(v.x*n, v.y*n, v.z*n, l);
        break;
    }
    case NormalizedVorticityAndMagnitude: {
        float4 vl  = tex3D(d_velocity_read_texture, px-d, py,   pz);
        float4 vr  = tex3D(d_velocity_read_texture, px+d, py,   pz);
        float4 vb  = tex3D(d_velocity_read_texture, px,   py-d, pz);
        float4 vf  = tex3D(d_velocity_read_texture, px,   py+d, pz);
        float4 vd  = tex3D(d_velocity_read_texture, px,   py,   pz-d);
        float4 vu  = tex3D(d_velocity_read_texture, px,   py,   pz+d);

        float3 cc = discrete_curl(vl, vr, vb, vf, vd, vu);
        float l = norm3df(cc.x, cc.y, cc.z);
        float n = 0.0; if (l > 0.0) n = 1.0 / l;
        debug_data = make_float4(cc.x*n, cc.y*n, cc.z*n, l);

        break;
    }
    default:
        debug_data = make_float4(NAN, NAN, NAN, NAN);
        break;
    }
    surf3Dwrite(
        debug_data, d_debug_data_write_surface,
        x*sizeof(float4), y, z,
        cudaBoundaryModeTrap
    );
}

__host__
void sim_update_debug_data() {
    sim_map_gl_debug_data();
    sim_update_debug_data_kernel<<<BLOCK_DIM, THREAD_DIM>>>(sim_debug_data_mode);
    sim_unmap_gl_debug_data();
}

// TODO: constant dt
__host__
void sim_update(double dt) {
    sim_map_gl_density();

    if (sim_vorticity_confinement != 0.0) {
        sim_vorticity_confinement_kernel<<<BLOCK_DIM, THREAD_DIM>>>(dt*sim_vorticity_confinement);
        sim_swap_buffers();
    }

    if (sim_debug_use_basic_advection) sim_advection_first_order_kernel<<<BLOCK_DIM, THREAD_DIM>>>(dt);
    else                               sim_advection_bfecc_kernel<<<BLOCK_DIM, THREAD_DIM>>>(dt);
    sim_swap_buffers();

    for (int i = 0; i < sim_pressure_project_iterations; i++) {
        sim_pressure_project_jacobi_iteration_kernel<<<BLOCK_DIM, THREAD_DIM>>>();
        sim_swap_buffers();
    }

    if (sim_debug_data_mode) {
        sim_update_debug_data();
    }

    sim_unmap_gl_density();
}

// DATA TRANSFER FUNCTIONS

__global__
void sim_reset_velocity_and_density_kernel() {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    surf3Dwrite(make_float4(0,0,0,0), d_velocity_write_surface, x*sizeof(float4), y, z);
    surf3Dwrite(0,                    d_density_write_surface,  x*sizeof(float),  y, z);
}

__host__
void sim_reset_velocity_and_density() {
    sim_map_gl_density();
    sim_reset_velocity_and_density_kernel<<<BLOCK_DIM, THREAD_DIM>>>();
    sim_swap_buffers();
    sim_unmap_gl_density();
}

__global__
void sim_add_velocity_and_density_kernel(
    int x, int y, int z,
    float xf, float yf, float zf, float df,
    bool velocity, bool density, bool nan_is_mask,
    int xlen, int ylen, int zlen
) {
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;
    if (ix > xlen || iy > ylen || iz > zlen) return;

    if (velocity) {
        float4 v = surf3Dread<float4>(d_velocity_copy_surface, ix*(int)sizeof(float4), iy, iz, cudaBoundaryModeTrap);
        if (!nan_is_mask || !isnan(v.w)) {
            float4 w = surf3Dread<float4>(d_velocity_write_surface, (x + ix)*sizeof(float4), y + iy, z + iz, cudaBoundaryModeTrap);
            v.x *= xf; v.x += w.x;
            v.y *= yf; v.y += w.y;
            v.z *= zf; v.z += w.z;
            surf3Dwrite(
                v, d_velocity_write_surface,
                (x + ix)*sizeof(float4), y + iy, z + iz,
                cudaBoundaryModeTrap
            );
        }
    }
    if (density) {
        float d = surf3Dread<float>(d_density_copy_surface, ix*(int)sizeof(float), iy, iz, cudaBoundaryModeTrap);
        if (!nan_is_mask || !isnan(d)) {
            d *= df;
            d += surf3Dread<float>(d_density_write_surface, (x + ix)*sizeof(float), y + iy, z + iz, cudaBoundaryModeTrap);
            surf3Dwrite(
                d, d_density_write_surface,
                (x + ix)*sizeof(float), y + iy, z + iz,
                cudaBoundaryModeTrap
            );
        }
    }
}

__host__
void sim_add_velocity_and_density(
    int x, int y, int z,
    float xf, float yf, float zf, float df,
    float* velocity, float* density, int pitch_elems, bool nan_is_mask,
    int xlen, int ylen, int zlen
) {
    if (x >= GRID_SIZE || x < -xlen || xlen < 1
    ||  y >= GRID_SIZE || y < -ylen || ylen < 1
    ||  z >= GRID_SIZE || z < -zlen || zlen < 1
    ) {
        return;
    }

    // TODO: fix the weird buffer swapping here
    sim_swap_buffers();
    if (density) sim_map_gl_density();

    // ensure copy buffer capacity
    static cudaExtent extent = make_cudaExtent(0, 0, 0);
    static cudaArray_t d_velocity_copy_array;
    static cudaArray_t d_density_copy_array;

    if (xlen > extent.width || ylen > extent.height || zlen > extent.depth) {
        // realloc copy-buffers
        if (extent.width != 0) {
            cudaCheckErrors(cudaFreeArray(d_velocity_copy_array));
            cudaCheckErrors(cudaFreeArray(d_density_copy_array));
        }
        extent.width  = max(extent.width,  (size_t) xlen);
        extent.height = max(extent.height, (size_t) ylen);
        extent.depth  = max(extent.depth,  (size_t) zlen);

        cudaChannelFormatDesc velocity_format = cudaCreateChannelDesc<float4>();
        cudaCheckErrors(cudaMalloc3DArray(
            &d_velocity_copy_array, &velocity_format, extent,
            cudaArraySurfaceLoadStore
        ));
        cudaCheckErrors(cudaBindSurfaceToArray(d_velocity_copy_surface, d_velocity_copy_array));

        cudaChannelFormatDesc density_format = cudaCreateChannelDesc<float>();
        cudaCheckErrors(cudaMalloc3DArray(
            &d_density_copy_array, &density_format, extent,
            cudaArraySurfaceLoadStore
        ));
        cudaCheckErrors(cudaBindSurfaceToArray(d_density_copy_surface, d_density_copy_array));
    }

    // copy arrays to copy-buffers
    cudaMemcpy3DParms copy = {0};
    copy.kind   = cudaMemcpyHostToDevice;

    copy.srcPtr.pitch = pitch_elems != 0? pitch_elems : xlen; // multiply by elem size later
    copy.srcPtr.xsize = xlen;
    copy.srcPtr.ysize = ylen;

    // clamp origin
    if (x < 0) { xlen -= -x; copy.srcPos.x = -x; x = 0; }
    if (y < 0) { ylen -= -y; copy.srcPos.y = -y; y = 0; }
    if (z < 0) { zlen -= -z; copy.srcPos.z = -z; z = 0; }

    // clamp extents
    xlen = min(xlen, GRID_SIZE - x);
    ylen = min(ylen, GRID_SIZE - y);
    zlen = min(zlen, GRID_SIZE - z);
    copy.extent = make_cudaExtent(xlen, ylen, zlen);

    if (velocity) {
        cudaMemcpy3DParms copy_velocity = copy;
        copy_velocity.dstArray      = d_velocity_copy_array;
        copy_velocity.srcPtr.ptr    = velocity;
        copy_velocity.srcPtr.pitch *= sizeof(float4);
        copy_velocity.srcPos.x     *= sizeof(float4);
        cudaCheckErrors(cudaMemcpy3D(&copy_velocity));
    }
    if (density) {
        cudaMemcpy3DParms copy_density = copy;
        copy_density.dstArray      = d_density_copy_array;
        copy_density.srcPtr.ptr    = density;
        copy_density.srcPtr.pitch *= sizeof(float);
        copy_density.srcPos.x     *= sizeof(float);
        cudaCheckErrors(cudaMemcpy3D(&copy_density));
    }

    // launch copy kernel
    dim3 block_dim = dim3(
        (xlen - 1) / THREAD_DIM.x + 1,
        (ylen - 1) / THREAD_DIM.y + 1,
        (zlen - 1) / THREAD_DIM.z + 1
    );
    sim_add_velocity_and_density_kernel<<<block_dim, THREAD_DIM>>>(
        x, y, z,
        xf, yf, zf, df,
        velocity != NULL, density != NULL, nan_is_mask,
        xlen, ylen, zlen
    );

    if (density) sim_unmap_gl_density();
    sim_swap_buffers();
}

__global__
void sim_add_velocity_and_density_along_ray_kernel(
    float3 rp, float3 rv,
    float radius2, float hardness,
    float4 v, float d
) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    float px = ((float) x + 0.5) / GRID_SIZE;
    float py = ((float) y + 0.5) / GRID_SIZE;
    float pz = ((float) z + 0.5) / GRID_SIZE;

    float dot = (px-rp.x)*rv.x + (py-rp.y)*rv.y + (pz-rp.z)*rv.z;
    px -= dot*rv.x + rp.x;
    py -= dot*rv.y + rp.y;
    pz -= dot*rv.z + rp.z;
    float distance2 = px*px + py*py + pz*pz;
    if (distance2 > radius2) return;

    if (hardness != 1.0) {
        float t = min(1.0, 1.0 - (distance2/radius2 - hardness)/(1.0 - hardness));
        v.x *= t;
        v.y *= t;
        v.z *= t;
        d   *= t;
    }

    if (v.w != 0) {
        float4 w = surf3Dread<float4>(d_velocity_write_surface, x*sizeof(float4), y, z, cudaBoundaryModeTrap);
        v.x += w.x;
        v.y += w.y;
        v.z += w.z;
        surf3Dwrite(
            v, d_velocity_write_surface,
            x*sizeof(float4), y, z,
            cudaBoundaryModeTrap
        );
    }
    if (d != 0) {
        d += surf3Dread<float>(d_density_write_surface, x*sizeof(float), y, z, cudaBoundaryModeTrap);
        surf3Dwrite(
            d, d_density_write_surface,
            x*sizeof(float), y, z,
            cudaBoundaryModeTrap
        );
    }
}

__host__
void sim_add_velocity_and_density_along_ray(
    float rpx, float rpy, float rpz,
    float rvx, float rvy, float rvz,
    float radius, float hardness,
    float vx, float vy, float vz, float d
) {
    // TODO: fix the weird buffer swapping here
    sim_swap_buffers();
    if (d != 0) sim_map_gl_density();

    float3 rp = make_float3(rpx, rpy, rpz);
    float3 rv = make_float3(rvx, rvy, rvz);
    float4 v = make_float4(vx, vy, vz, abs(vx)+abs(vy)+abs(vz));
    sim_add_velocity_and_density_along_ray_kernel<<<BLOCK_DIM, THREAD_DIM>>>(rp, rv, radius*radius, hardness, v, d);

    if (d != 0) sim_unmap_gl_density();
    sim_swap_buffers();
}

// DEBUG FUNCTIONS

__global__
void sim_debug_reset_velocity_field_kernel(float max_component, float3 time) {
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

        v[i] = max_component * wave;
    }
    float4 velocity = make_float4(v[0], v[1], v[2], 0.0);
    surf3Dwrite(
        velocity, d_velocity_write_surface,
        x*sizeof(float4), y, z,
        cudaBoundaryModeTrap
    );
}

__host__
void sim_debug_reset_velocity_field(float max_velocity, double tx, double ty, double tz) {
    float max_component = max_velocity/sqrt(3);

    sim_debug_reset_velocity_field_kernel<<<BLOCK_DIM, THREAD_DIM>>>(max_component, make_float3(tx, ty, tz));
    sim_swap_buffers();
    // TODO: pressure project?
}

__global__
void sim_debug_reset_density_field_kernel(float max_density, double time) {
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

    float density = max_density * fade * (fmaxf(0.0, wave)*fmaxf(0.0, wave));
    surf3Dwrite(
        density, d_density_write_surface,
        x*sizeof(float), y, z,
        cudaBoundaryModeTrap
    );
}

__host__
void sim_debug_reset_density_field(float max_density, double t) {
    sim_map_gl_density();
    sim_debug_reset_density_field_kernel<<<BLOCK_DIM, THREAD_DIM>>>(max_density, t);
    sim_swap_buffers();
    sim_unmap_gl_density();
}

// LEGACY CODE

__global__
void sim_advection_first_order_kernel(double dt) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    float px = ((float) x + 0.5) / GRID_SIZE;
    float py = ((float) y + 0.5) / GRID_SIZE;
    float pz = ((float) z + 0.5) / GRID_SIZE;
    float4 v = tex3D(
        d_velocity_read_texture,
        px, py, pz
    );
    // TODO: higher order path integrator to find q
    // TODO: store velocities as per-frame deltas with a fixed dt
    float qx = px - v.x*dt;
    float qy = py - v.y*dt;
    float qz = pz - v.z*dt;

    // ADVECTION
    float4 w = tex3D(
        d_velocity_read_texture,
        qx, qy, qz
    );
    w.w = 0.0; // TODO: correct handling of pressure field?
    surf3Dwrite(
        w, d_velocity_write_surface,
        x*sizeof(float4), y, z,
        cudaBoundaryModeTrap
    );

    // SUBSTANCE TRANSPORT
    // TODO: transport after pressure projection
    float density = tex3D(
        d_density_read_texture,
        qx, qy, qz
    );
    surf3Dwrite(
        density, d_density_write_surface,
        x*sizeof(float), y, z,
        cudaBoundaryModeTrap
    );
}

__global__
void sim_advection_maccormack_kernel(double dt) {
    // TODO: unstable at high velocity or dt

    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    float px = ((float) x + 0.5) / GRID_SIZE;
    float py = ((float) y + 0.5) / GRID_SIZE;
    float pz = ((float) z + 0.5) / GRID_SIZE;
    float4 pv = tex3D(d_velocity_read_texture, px, py, pz);

    // ADVECTION
    // implements Selle et al. MacCormack method

    // forward advection
    float qx = px - pv.x*dt;
    float qy = py - pv.y*dt;
    float qz = pz - pv.z*dt;
    float4 qv = tex3D(d_velocity_read_texture, qx, qy, qz);

    // reverse advection
    float rx = qx + qv.x*dt;
    float ry = qy + qv.y*dt;
    float rz = qz + qv.z*dt;
    float4 rv = tex3D(d_velocity_read_texture, rx, ry, rz);

    // error-compensated final advection
    float4 w = make_float4(
        qv.x + 0.5*(pv.x - rv.x),
        qv.y + 0.5*(pv.y - rv.y),
        qv.z + 0.5*(pv.z - rv.z),
        0.0
    );
    surf3Dwrite(
        w, d_velocity_write_surface,
        x*sizeof(float4), y, z,
        cudaBoundaryModeTrap
    );

    // SUBSTANCE TRANSPORT
    // also uses MacCormack method
    float pd = tex3D(d_density_read_texture, px, py, pz);
    float qd = tex3D(d_density_read_texture, qx, qy, qz);
    float rd = tex3D(d_density_read_texture, rx, ry, rz);
    float d  = qd + 0.5*(pd - rd);
    surf3Dwrite(
        d, d_density_write_surface,
        x*sizeof(float), y, z,
        cudaBoundaryModeTrap
    );
}
