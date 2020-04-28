#include <cstdlib>
#include <cstdio>
#include <cstring>

#include "glad/glad.h"

#define GLFW_DLL
#include "GLFW/glfw3.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "global_defines.h"
#include "math_prelude.hpp"
#include "simulation.hpp"

extern "C" {
_declspec(dllexport) uint64_t NvOptimusEnablement = 0x00000001;
}

double g_time = 0.0;
double g_delta_time;

float g_camera_elevation = 0.0;
float g_camera_azimuth   = 0.0;
float g_camera_distance  = 3.0;

const vec3 CUBE_VERTS[8] = {
    vec3( 1.0,  1.0,  1.0),
    vec3(-1.0,  1.0,  1.0),
    vec3( 1.0, -1.0,  1.0),
    vec3(-1.0, -1.0,  1.0),
    vec3( 1.0,  1.0, -1.0),
    vec3(-1.0,  1.0, -1.0),
    vec3( 1.0, -1.0, -1.0),
    vec3(-1.0, -1.0, -1.0),
};

const uint8_t CUBE_INDICES[6*2*3] = {
    0, 1, 3,  3, 2, 0,
    0, 2, 6,  6, 4, 0,
    0, 4, 5,  5, 1, 0,
    7, 3, 1,  1, 5, 7,
    7, 5, 4,  4, 6, 7,
    7, 6, 2,  2, 3, 7
};

void glfw_error_callback(int error, const char* description) {
    fprintf(stderr, "GLFW error: %s\n", description);
}

const float debug_max_velocity = 1.0;
const float debug_render_velocity_threshold_base = debug_max_velocity / 2.0;
const float debug_render_velocity_threshold_incr = debug_render_velocity_threshold_base / 10.0;

// TODO: this is nasty and will scale badly
int   input_set_sim_debug_data_mode = 0;
float input_mod_debug_render_velocity_threshold = debug_render_velocity_threshold_base;
bool  input_do_debug_reset_velocity_field = true;
bool  input_do_debug_reset_density_field = true;

void glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    switch (key) {
    case GLFW_KEY_MINUS:
        if (action == GLFW_PRESS) input_mod_debug_render_velocity_threshold -= debug_render_velocity_threshold_incr;
        break;
    case GLFW_KEY_EQUAL:
        if (action == GLFW_PRESS) input_mod_debug_render_velocity_threshold += debug_render_velocity_threshold_incr;
        break;
    case GLFW_KEY_SPACE:
        if (action == GLFW_PRESS) {
            float k = TAU*100;
            if (mods & GLFW_MOD_CONTROL) input_do_debug_reset_velocity_field = true;
            input_do_debug_reset_density_field = true;
        }
        break;
    case GLFW_KEY_GRAVE_ACCENT:
        if (action == GLFW_PRESS) {
            if (sim_debug_data_mode) input_set_sim_debug_data_mode = -1;
            else                     input_set_sim_debug_data_mode =  1;
        }
        break;
    case GLFW_KEY_ESCAPE:
        if (action == GLFW_PRESS) glfwSetWindowShouldClose(window, GLFW_TRUE);
        break;
    }
}

void glfw_cursor_pos_callback(GLFWwindow* window, double xpos, double ypos) {
    int width, height;
    glfwGetWindowSize(window, &width, &height);
    vec2 normalized = vec2(xpos / float(width-1), ypos / float(height-1));
    g_camera_azimuth   = TAU * normalized.x;
    g_camera_elevation = TAU / 4.0 * 2.0*(normalized.y - 0.5);
}

void gl_delete_program_and_attached_shaders(GLuint program) {
    GLint shaders_count;
    glGetProgramiv(program, GL_ATTACHED_SHADERS, &shaders_count);
    GLuint* shaders = (GLuint*) malloc(shaders_count*sizeof(GLuint));
    glGetAttachedShaders(program, shaders_count, NULL, shaders);

    for (int i = 0; i < shaders_count; i++) {
        glDetachShader(program, shaders[i]);
        glDeleteShader(shaders[i]);
    }
    glDeleteProgram(program);
    free(shaders);
}

char* read_file_to_string(char* filename, size_t* len) {
    FILE* file = fopen(filename, "rb");
    fseek(file, 0, SEEK_END);
    size_t filesize = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* buffer = (char*) malloc(filesize+1);
    fread(buffer, 1, filesize, file);
    fclose(file);

    buffer[filesize] = '\0';
    if (len) {
        *len = filesize;
    }
    return buffer;
}

#define UNIFORM(PROGRAM, NAME) GLint (NAME) = glGetUniformLocation((PROGRAM), #NAME)
#define ATTRIBUTE(PROGRAM, NAME) GLuint (NAME) = glGetAttribLocation((PROGRAM), #NAME)

GLuint gl_compile_and_link_shaders(
    int headers_count, GLchar** header_srcs, GLint* header_lens,
    int shaders_count, GLchar** shader_srcs, GLint* shader_lens, GLenum* shader_types
) {
    bool error = false;

    GLchar** srcs = (GLchar**) malloc((2 + headers_count)*sizeof(GLchar**));
    memcpy(srcs+1, header_srcs, headers_count*sizeof(GLchar*));
    GLint*   lens = (GLint*)   malloc((2 + headers_count)*sizeof(GLint*));
    memcpy(lens+1, header_lens, headers_count*sizeof(GLint));

    GLuint* shaders = (GLuint*) malloc(shaders_count*sizeof(GLuint));
    GLuint program = glCreateProgram();
    assert(program != 0);

    for (int i = 0; i < shaders_count; i++) {
        // PARSE VERSION DIRECTIVE
        const char* version_string       = "#version";
        const int   version_string_len   = strlen(version_string);
        int         version_string_index = 0;

        for (int j = 0; j < shader_lens[i]; j++) {
            if (version_string_index < version_string_len) {
                // match version specifier
                if (version_string[version_string_index] == shader_srcs[i][j]) version_string_index += 1;
                else                                                           version_string_index = 0;
            } else if (shader_srcs[i][j] == '\n') {
                // find end of line
                version_string_index = j+1;
                break;
            }
        }
        if (version_string_index < version_string_len) {
            fprintf(stderr, "shader %d parse error: missing #version directive\n", i);
            error = true;
            continue;
        }

        // COMPOSE SHADER AND HEADERS
        srcs[0] = shader_srcs[i];
        lens[0] = version_string_index;
        srcs[headers_count+1] = shader_srcs[i] + version_string_index;
        lens[headers_count+1] = shader_lens[i] - version_string_index;

        // COMPILE SHADER
        shaders[i] = glCreateShader(shader_types[i]);
        assert(shaders[i] != 0);
        glShaderSource(shaders[i], headers_count+2, srcs, lens);
        glCompileShader(shaders[i]);

        // check errors
        GLint compile_status = GL_FALSE;
        glGetShaderiv(shaders[i], GL_COMPILE_STATUS, &compile_status);
        if (compile_status == GL_FALSE) {
            GLint error_len = 0;
            glGetShaderiv(shaders[i], GL_INFO_LOG_LENGTH, &error_len);
            char* error_msg = (char*) malloc(error_len*sizeof(char));
            glGetShaderInfoLog(shaders[i], error_len, NULL, error_msg);
            glDeleteShader(shaders[i]);

            fprintf(stderr, "shader %d compile error:\n%s\n", i, error_msg);
            error = true;
            free(error_msg);
            continue;
        }
        glAttachShader(program, shaders[i]);
    }
    if (!error) {
        glLinkProgram(program);

        // check errors
        GLint link_status = GL_FALSE;
        glGetProgramiv(program, GL_LINK_STATUS, &link_status);
        if (link_status == GL_FALSE) {
            GLint error_len = 0;
            glGetProgramiv(program, GL_INFO_LOG_LENGTH, &error_len);
            char* error_msg = (char*) malloc(error_len*sizeof(char));
            glGetProgramInfoLog(program, error_len, NULL, error_msg);

            fprintf(stderr, "program link error:\n%s\n", error_msg);
            error = true;
            free(error_msg);
        }
    }
    if (error) {
        gl_delete_program_and_attached_shaders(program);
        program = 0;
    }

    free(srcs); free(lens); free(shaders);
    return program;
}

void main() {
    // WINDOWING & CONTEXT SETUP
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        abort();
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Stable Fluids", NULL, NULL);
    glfwMakeContextCurrent(window);

    glfwSetKeyCallback(window, glfw_key_callback);
    glfwSetCursorPosCallback(window, glfw_cursor_pos_callback);

    gladLoadGLLoader((GLADloadproc) glfwGetProcAddress);

    // SHADER PROGRAM SETUP

    const int num_headers = 2;
    const int num_shaders = 2;
    GLchar* header_srcs[num_headers]; GLint header_lens[num_headers];
    GLchar* shader_srcs[num_shaders]; GLint shader_lens[num_shaders]; GLenum shader_types[num_shaders];
    size_t len;

    // global_defines.h has special treatment to prevent mismatch between shaders and compiled code
    header_srcs[num_headers-2] = read_file_to_string("out/global_defines.h", &len);
    header_lens[num_headers-2] = (GLint) len;

    header_srcs[num_headers-1] = read_file_to_string("src/shader/volumetric.head.glsl", &len);
    header_lens[num_headers-1] = (GLint) len;

    shader_types[0] = GL_VERTEX_SHADER;
    shader_srcs[0] = read_file_to_string("src/shader/volumetric.vert.glsl", &len);
    shader_lens[0] = (GLint) len;

    shader_types[1] = GL_FRAGMENT_SHADER;
    shader_srcs[1] = read_file_to_string("src/shader/volumetric.frag.glsl", &len);
    shader_lens[1] = (GLint) len;

    GLuint program = gl_compile_and_link_shaders(
        num_headers, header_srcs, header_lens,
        num_shaders, shader_srcs, shader_lens, shader_types
    );
    assert(program);
    free(header_srcs[0]); free(shader_srcs[0]); free(shader_srcs[1]);
    glUseProgram(program);

    UNIFORM(program, u_density_field);
    glUniform1i(u_density_field, 0);

    UNIFORM(program, u_density_field_bounds);
    vec3 density_field_bounds[2] = {-VEC3_1, VEC3_1};
    glUniformMatrix2x3fv(u_density_field_bounds, 1, GL_FALSE, (GLfloat*) &density_field_bounds);

    UNIFORM(program, u_camera);
    mat4 camera = mat4(1.0);
    glUniformMatrix4fv(u_camera, 1, GL_FALSE, (GLfloat*) &camera);

    UNIFORM(program, u_camera_pos);
    glUniform3f(u_camera_pos, 0.0, 0.0, 1.0);

    UNIFORM(program, u_time);

    UNIFORM(program, u_debug_data_volume);
    glUniform1i(u_debug_data_volume, 1);

    // TODO: add GUI controls for debug visuals
    UNIFORM(program, u_debug_render_flags);
    UNIFORM(program, u_debug_render_velocity_threshold);
    UNIFORM(program, u_debug_render_clip_bounds);

    ATTRIBUTE(program, a_pos);

    // VERTEX ARRAYS

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, 8*sizeof(vec3), CUBE_VERTS, GL_STATIC_DRAW);
    glEnableVertexAttribArray(a_pos);
    glVertexAttribPointer(a_pos, 3, GL_FLOAT, GL_FALSE, 0, 0);

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    // INIT IMGUI
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // INIT COMPUTE
    sim_init(GL_TEXTURE0, GL_TEXTURE1);

    // MAIN LOOP
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        { // UPDATE TIME
            double time = glfwGetTime();
            g_delta_time = time - g_time;
            g_time = time;
            glUniform1d(u_time, g_time);
        }

        // UPDATE DEBUG VISUALIZATIONS
        if (input_do_debug_reset_density_field) {
            sim_debug_reset_density_field(1.0, 100*g_time);
            input_do_debug_reset_density_field = false;
        }
        if (input_do_debug_reset_velocity_field) {
            float k = TAU*100;
            sim_debug_reset_velocity_field(debug_max_velocity, k*(g_time+1), k*(g_time+2), k*(g_time+3));
            input_do_debug_reset_velocity_field = false;
        }
        if (input_set_sim_debug_data_mode > 0) {
            sim_debug_data_mode = NormalizedVelocityAndMagnitude;
            glUniform1i(u_debug_render_flags, DEBUG_RENDER_FLAG_VELOCITIES | DEBUG_RENDER_FLAG_CLIP_BOUNDS);
        } else if (input_set_sim_debug_data_mode < 0) {
            sim_debug_data_mode = None;
            glUniform1i(u_debug_render_flags, DEBUG_RENDER_FLAG_NONE);
        }
        if (input_mod_debug_render_velocity_threshold) {
            static float threshold = 0.0;
            threshold = fmax(threshold + input_mod_debug_render_velocity_threshold, 0.0);
            glUniform1f(u_debug_render_velocity_threshold, threshold);
            input_mod_debug_render_velocity_threshold = 0.0;
        }

        vec3 camera_pos = rotateZ(rotateX(-g_camera_distance*VEC3_Y, g_camera_elevation), g_camera_azimuth);
        mat4 camera = glm::perspective(1.0f, 1280.0f/720.0f, 0.01f, 1000.0f) * glm::lookAt(camera_pos, VEC3_0, VEC3_Z);
        glUniform3fv(u_camera_pos, 1, (GLfloat*) &camera_pos);
        glUniformMatrix4fv(u_camera, 1, false, (GLfloat*) &camera);

        sim_update(0.0333333);

        // RENDER

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDrawElements(GL_TRIANGLES, 3*6*2, GL_UNSIGNED_BYTE, CUBE_INDICES);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // CLEANUP
    sim_terminate();
    glfwTerminate();
}
