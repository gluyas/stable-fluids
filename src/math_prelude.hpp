#include <cmath>

#define TAU  6.28318530717958647692
#define TAUf 6.28318530717958647692f

#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"
#include "glm/mat3x3.hpp"
#include "glm/mat4x4.hpp"
#include "glm/geometric.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/rotate_vector.hpp"
using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::mat3;
using glm::mat4;
using glm::dot;
using glm::cross;
using glm::length;
using glm::normalize;
using glm::transpose;
using glm::inverse;
using glm::abs;
using glm::max;
using glm::min;

#define VEC3_0 vec3{0.0f, 0.0f, 0.0f}
#define VEC3_1 vec3{1.0f, 1.0f, 1.0f}
#define VEC3_X vec3{1.0f, 0.0f, 0.0f}
#define VEC3_Y vec3{0.0f, 1.0f, 0.0f}
#define VEC3_Z vec3{0.0f, 0.0f, 1.0f}
#define VEC3_NAN vec3{NAN, NAN, NAN}

#define VEC4_0 vec4{0.0f, 0.0f, 0.0f, 0.0f}
#define VEC4_1 vec4{1.0f, 1.0f, 1.0f, 0.0f}
#define VEC4_X vec4{1.0f, 0.0f, 0.0f, 0.0f}
#define VEC4_Y vec4{0.0f, 1.0f, 0.0f, 0.0f}
#define VEC4_Z vec4{0.0f, 0.0f, 1.0f, 0.0f}
#define VEC4_W vec4{0.0f, 0.0f, 0.0f, 1.0f}
#define VEC4_NAN vec4{NAN, NAN, NAN, NAN}


