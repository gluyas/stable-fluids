// this file needs to be compatible with C++ and GLSL

#define TAU  6.28318530717958647692
#define TAUf 6.28318530717958647692f

#define DEGREES  (TAU  / 360.0)
#define DEGREESf (TAUf / 360.0f)

#define SIM_GRID_SIZE  128

// lower byte encodes render mode
#define DEBUG_RENDER_MODE_MASK        0x00FF
#define DEBUG_RENDER_MODE_DEFAULT     0x0000
#define DEBUG_RENDER_MODE_VELOCITIES  0x0001
#define DEBUG_RENDER_MODE_AXIS_COLOR  0x0002
// upper byte encodes render flags
#define DEBUG_RENDER_FLAG_NONE        0x0000
#define DEBUG_RENDER_FLAG_CLIP_BOUNDS 0x0100

