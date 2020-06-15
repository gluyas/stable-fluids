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

double g_time = 0.0;
double g_delta_time;

float g_camera_elevation = 0.0;
float g_camera_azimuth   = 0.0;
float g_camera_distance  = 3.0;

const float camera_near_clip    = 0.01f;
const float camera_min_distance = sqrt(3) + 2*camera_near_clip;

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

const uint8_t CUBE_FACE_INDICES[6*2*3] = {
    0, 1, 3,  3, 2, 0,
    0, 2, 6,  6, 4, 0,
    0, 4, 5,  5, 1, 0,
    7, 3, 1,  1, 5, 7,
    7, 5, 4,  4, 6, 7,
    7, 6, 2,  2, 3, 7
};

const uint8_t CUBE_OUTLINE_INDICES[16] = {
    0, 1, 3, 2, 0, 4, 5, 1, 5, 7, 3, 7, 6, 2, 6, 4
};

void glfw_error_callback(int error, const char* description) {
    fprintf(stderr, "GLFW error: %s\n", description);
}

bool  debug_menu = true;
float debug_delta_time   = 1.0/30.0;
float debug_render_velocity_threshold = 0.5;
float debug_render_density_factor = 1.0;
bool  debug_render_boundaries = false;

// KEYBOARD INPUT

bool input_mouse_drag_is_brush = true;
bool input_do_debug_reset_velocity_field = true;
bool input_do_debug_reset_density_field = true;

SimDebugDataMode input_set_sim_debug_data_mode = sim_debug_data_mode;

void glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureKeyboard) return;

    switch (key) {
    case GLFW_KEY_1:
        if (action == GLFW_PRESS) input_set_sim_debug_data_mode = None;
        break;
    case GLFW_KEY_2:
        if (action == GLFW_PRESS) input_set_sim_debug_data_mode = NormalizedVelocityAndMagnitude;
        break;
    case GLFW_KEY_B:
        if (action == GLFW_PRESS) debug_render_boundaries = !debug_render_boundaries;
        break;
    case GLFW_KEY_R:
        if (action == GLFW_PRESS) {
            input_do_debug_reset_velocity_field = true;
            input_do_debug_reset_density_field = true;
        }
        break;
    case GLFW_KEY_SPACE: {
        static GLFWcursor* hand = NULL;
        if (hand == NULL) {
            hand = glfwCreateStandardCursor(GLFW_HAND_CURSOR);
        }
        if (action == GLFW_PRESS) {
            input_mouse_drag_is_brush = false;
            glfwSetCursor(window, hand);
        } else if (action == GLFW_RELEASE) {
            input_mouse_drag_is_brush = true;
            glfwSetCursor(window, NULL);
        }
        break;
    }
    case GLFW_KEY_GRAVE_ACCENT:
        if (action == GLFW_PRESS) debug_menu = !debug_menu;
        break;
    case GLFW_KEY_ESCAPE:
        if (action == GLFW_PRESS) glfwSetWindowShouldClose(window, GLFW_TRUE);
        break;
    }
}

// MOUSE INPUT
// TODO: clean up input state variables

double input_cursor_xpos = 0;
double input_cursor_ypos = 0;
double input_cursor_xpos_delta = 0;
double input_cursor_ypos_delta = 0;

vec2 input_cursor_screen_pos = vec3(0.0);

bool input_mouse_shoot_emitter   = false;
bool input_mouse_dragging_camera = false;
bool input_mouse_dragging_brush  = false;

void glfw_cursor_pos_callback(GLFWwindow* window, double xpos, double ypos) {
    input_cursor_xpos_delta = xpos - input_cursor_xpos;
    input_cursor_ypos_delta = ypos - input_cursor_ypos;
    input_cursor_xpos = xpos;
    input_cursor_ypos = ypos;

    int width, height;
    glfwGetWindowSize(window, &width, &height);
    input_cursor_screen_pos = vec2(xpos/float(width-1), 1.0-ypos/float(height-1))*2.0f - vec2(1.0);

    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;

    if (input_mouse_dragging_camera) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        vec2 normalized = vec2(input_cursor_xpos_delta / float(width-1), input_cursor_ypos_delta / float(height-1));
        g_camera_azimuth   -= TAU * normalized.x;
        g_camera_elevation += TAU / 4.0 * 2.0*(normalized.y);
        g_camera_elevation = fmin(fmax(g_camera_elevation, -85*DEGREES), 85*DEGREES);
    }
}

void glfw_mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    ImGuiIO& io = ImGui::GetIO();

    if (button == GLFW_MOUSE_BUTTON_2 || button == GLFW_MOUSE_BUTTON_1 && (mods & GLFW_MOD_CONTROL) != 0) {
        if (action == GLFW_PRESS && !io.WantCaptureMouse) input_mouse_shoot_emitter = true;
        else if (action == GLFW_RELEASE) input_mouse_shoot_emitter = false;
    } else if (button == GLFW_MOUSE_BUTTON_1) {
        if (action == GLFW_PRESS && !io.WantCaptureMouse) {
            if (input_mouse_drag_is_brush) input_mouse_dragging_brush  = true;
            else                           input_mouse_dragging_camera = true;
        } else if (action == GLFW_RELEASE) {
            input_mouse_dragging_brush  = false;
            input_mouse_dragging_camera = false;
            g_camera_azimuth = glm::mod(g_camera_azimuth + TAU/2, TAU) - TAU/2;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
        input_mouse_shoot_emitter = false;
    }
}

void glfw_scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;

    g_camera_distance *= pow(1.1, -yoffset);
    g_camera_distance = max(camera_min_distance, g_camera_distance);
}

int window_width  = 1920;
int window_height = 1080;
void glfw_window_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    window_width = width;
    window_height = height;
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
    GLFWwindow* window = glfwCreateWindow(window_width, window_height, "Stable Fluids", NULL, NULL);
    glfwMakeContextCurrent(window);

    glfwSetKeyCallback(window, glfw_key_callback);
    glfwSetCursorPosCallback(window, glfw_cursor_pos_callback);
    glfwSetMouseButtonCallback(window, glfw_mouse_button_callback);
    glfwSetScrollCallback(window, glfw_scroll_callback);
    glfwSetWindowSizeCallback(window, glfw_window_size_callback);

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

    UNIFORM(program, u_debug_render_mode_and_flags);

    UNIFORM(program, u_debug_render_velocity_threshold);
    glUniform1f(u_debug_render_velocity_threshold, debug_render_velocity_threshold);

    UNIFORM(program, u_debug_render_density_factor);
    glUniform1f(u_debug_render_density_factor, debug_render_density_factor);

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

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // INIT IMGUI
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // INIT COMPUTE
    sim_init(GL_TEXTURE0, GL_TEXTURE1);

    const int sphere_size = 16;
    const float sphere_fade_radius = 0.1;
    vec4  sphere_velocity[sphere_size][sphere_size][sphere_size];
    float sphere_density[sphere_size][sphere_size][sphere_size];
    for (int i = 0; i < sphere_size; i++) {
        for (int j = 0; j < sphere_size; j++) {
            for (int k = 0; k < sphere_size; k++) {
                vec3 p = 2.0f*vec3((float) i, (float) j, (float) k) / (float) (sphere_size-1) - vec3(1.0);
                float l = length(p);
                float w = min(max((1.0 - l)/sphere_fade_radius, 0.0), 1.0);
                float v;
                if (l > 1.0) v = NAN;
                else         v = w;
                sphere_density[i][j][k]  = v;
                sphere_velocity[i][j][k] = vec4(w, w, w, v);
            }
        }
    }

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

        static unsigned int debug_render_mode_and_flags = DEBUG_RENDER_FLAG_NONE;
        bool         update_debug_render_mode_and_flags = false;

        static float debug_emitter_time     = 0.0;
        static bool  debug_emitter_aim      = false;
        static float debug_emitter_speed    = 3.0;
        static float debug_emitter_freq     = 0.2*TAU;
        static float debug_emitter_offset   = 0.0;
        static float debug_emitter_density  = 1.0;
        static float debug_emitter_wobble   = TAU*0.05;
        static float debug_emitter_altitude = 0.0;
        float debug_emitter_vertical_speed = 0.0;

        static float debug_brush_strength = 2.0;
        static float debug_brush_radius   = 0.2;
        static float debug_brush_hardness = 0.5;

        static float debug_camera_auto_rotate = 0.0;

        if (debug_menu) {
            // ImGui::ShowDemoWindow();

            ImGui::Begin("Tools Menu");

            ImGui::Text("Camera");
            ImGui::SliderAngle("elevation", &g_camera_elevation, -85, 85);
            bool set_azimuth = ImGui::SliderAngle("azimuth", &g_camera_azimuth);
            // TODO: allow going inside the rendered volume
            ImGui::SliderFloat("distace", &g_camera_distance, camera_min_distance, 10.0, "%.3f", 3.0);
            ImGui::SliderAngle("auto rotation", &debug_camera_auto_rotate, -180, 180, "%.0f deg/s");
            if (set_azimuth) debug_camera_auto_rotate = 0.0;
            ImGui::Separator();

            ImGui::Text("Emitter");
            ImGui::SameLine();
            ImGui::Checkbox("manual aim", &debug_emitter_aim);
            ImGui::SliderFloat("force", &debug_emitter_speed, 0.0, 100.0, "%.3f m/s/s", 2.0);
            ImGui::SliderFloat("volume", &debug_emitter_density, 0.0, 10.0, "%.3f", 2.0);
            static float debug_emitter_freq_temp = debug_emitter_freq;
            if (!debug_emitter_aim) {
                if (ImGui::SliderAngle("rotation speed", &debug_emitter_freq_temp, -360.0, 360.0, "%.0f deg/s")) {
                    // magic math to keep the orientation stable
                    if (debug_emitter_freq_temp != 0.0) debug_emitter_offset = debug_emitter_freq*(debug_emitter_time + debug_emitter_offset) / debug_emitter_freq_temp - debug_emitter_time;
                    debug_emitter_freq = debug_emitter_freq_temp;
                }
                ImGui::SliderAngle("wobble", &debug_emitter_wobble, 0.0, 90.0);
            }
            static float debug_emitter_altitude_temp = debug_emitter_altitude;
            if (ImGui::SliderFloat("altitude", &debug_emitter_altitude_temp, -1.0, 1.0)) {
                debug_emitter_vertical_speed = debug_emitter_altitude_temp - debug_emitter_altitude;
                debug_emitter_altitude = debug_emitter_altitude_temp;
            }
            ImGui::Separator();

            ImGui::Text("Brush");
            ImGui::SliderFloat("strength##brush", &debug_brush_strength, 0.0, 10.0);
            ImGui::SliderFloat("radius##brush", &debug_brush_radius, 0.0, 0.5);
            ImGui::SliderFloat("hardness##brush", &debug_brush_hardness, 0.0, 1.0);
            ImGui::Separator();

            ImGui::Text("Simulation");
            ImGui::SameLine();
            if (ImGui::Button("reset##simulation")) {
                input_do_debug_reset_velocity_field = true;
                input_do_debug_reset_density_field = true;
            }
            static float debug_delta_time_ms = debug_delta_time * 1000.0;
            if (ImGui::SliderFloat("delta time", &debug_delta_time_ms, 0.0f, 1000.0/7.5, "%.3f ms", 1.0)) {
                debug_delta_time = debug_delta_time_ms / 1000.0;
            }
            ImGui::SliderInt("pressure projection iterations", &sim_pressure_project_iterations, 0, 128);
            ImGui::Checkbox("first-order advection", &sim_debug_use_basic_advection);
            ImGui::Separator();

            ImGui::Text("Rendering");

            // velocity field
            if (ImGui::RadioButton("default", input_set_sim_debug_data_mode == None)) {
                input_set_sim_debug_data_mode = None;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("velocity field", input_set_sim_debug_data_mode == NormalizedVelocityAndMagnitude)) {
                input_set_sim_debug_data_mode = NormalizedVelocityAndMagnitude;
            }
            if (input_set_sim_debug_data_mode == None) {
                if (ImGui::SliderFloat("density factor", &debug_render_density_factor, 0.0, 1.0)) {
                    glUniform1f(u_debug_render_density_factor, debug_render_density_factor);
                }
            } else if (input_set_sim_debug_data_mode == NormalizedVelocityAndMagnitude) {
                if (ImGui::DragFloat("render threshold", &debug_render_velocity_threshold, 0.002f, 0.0, INFINITY, "%.3f m/s", 2.0)) {
                    glUniform1f(u_debug_render_velocity_threshold, debug_render_velocity_threshold);
                }
            }

            // clip bounds
            ImGui::Checkbox("show boundaries", &debug_render_boundaries);
            static int debug_render_clip_bounds_interleaved[6] = {0,SIM_GRID_SIZE, 0,SIM_GRID_SIZE, 0,SIM_GRID_SIZE};
            if (ImGui::CheckboxFlags("enable clip bounds", &debug_render_mode_and_flags, DEBUG_RENDER_FLAG_CLIP_BOUNDS)) update_debug_render_mode_and_flags = true;
            if (debug_render_mode_and_flags & DEBUG_RENDER_FLAG_CLIP_BOUNDS) {
                bool update_debug_render_clip_bounds = false;
                if (ImGui::Button("reset##x clip")) {
                    debug_render_clip_bounds_interleaved[0] = 0; debug_render_clip_bounds_interleaved[1] = SIM_GRID_SIZE;
                    update_debug_render_clip_bounds = true;
                } ImGui::SameLine();
                if (ImGui::SliderInt2("x clip", &debug_render_clip_bounds_interleaved[0], 0, SIM_GRID_SIZE)) update_debug_render_clip_bounds = true;

                if (ImGui::Button("reset##y clip")) {
                    debug_render_clip_bounds_interleaved[2] = 0; debug_render_clip_bounds_interleaved[3] = SIM_GRID_SIZE;
                    update_debug_render_clip_bounds = true;
                } ImGui::SameLine();
                if (ImGui::SliderInt2("y clip", &debug_render_clip_bounds_interleaved[2], 0, SIM_GRID_SIZE)) update_debug_render_clip_bounds = true;

                if (ImGui::Button("reset##z clip")) {
                    debug_render_clip_bounds_interleaved[4] = 0; debug_render_clip_bounds_interleaved[5] = SIM_GRID_SIZE;
                    update_debug_render_clip_bounds = true;
                } ImGui::SameLine();
                if (ImGui::SliderInt2("z clip", &debug_render_clip_bounds_interleaved[4], 0, SIM_GRID_SIZE)) update_debug_render_clip_bounds = true;

                if (update_debug_render_clip_bounds) {
                    int debug_render_clip_bounds[6] = {
                        debug_render_clip_bounds_interleaved[0], debug_render_clip_bounds_interleaved[2], debug_render_clip_bounds_interleaved[4],
                        debug_render_clip_bounds_interleaved[1], debug_render_clip_bounds_interleaved[3], debug_render_clip_bounds_interleaved[5],
                    };
                    glUniform3iv(u_debug_render_clip_bounds, 2, debug_render_clip_bounds);
                }
            }
            ImGui::Separator();

            ImGui::Text("prev. frame %.3f ms (%.1f fps)", 1000.0  * g_delta_time, 1.0 / g_delta_time);
            ImGui::Text("avg.  frame %.3f ms (%.1f fps)", 1000.0f / io.Framerate, io.Framerate);

            ImGui::End();
        }

        if (debug_camera_auto_rotate) {
            g_camera_azimuth += debug_camera_auto_rotate * debug_delta_time;
            g_camera_azimuth = glm::mod(g_camera_azimuth + TAU/2, TAU) - TAU/2;
        }
        vec3 camera_pos = rotateZ(rotateX(-g_camera_distance*VEC3_Y, -g_camera_elevation), g_camera_azimuth);
        vec3 camera_target = VEC3_0;
        mat4 camera = glm::perspective(1.0f, (float) window_width / (float) window_height, camera_near_clip, 1000.0f) * glm::lookAt(camera_pos, camera_target, VEC3_Z);
        glUniform3fv(u_camera_pos, 1, (GLfloat*) &camera_pos);
        glUniformMatrix4fv(u_camera, 1, false, (GLfloat*) &camera);

        // calculate user dragging effects
        mat4 camera_inverse = inverse(camera);
        vec4 mouse_ray_p = camera_inverse*vec4(input_cursor_screen_pos, 0.0, 1.0);
        mouse_ray_p /= mouse_ray_p.w;
        vec4 mouse_ray_q = camera_inverse*vec4(input_cursor_screen_pos, 1.0, 1.0);
        mouse_ray_q /= mouse_ray_q.w;
        vec3 mouse_ray_v = normalize(vec3(mouse_ray_p) - vec3(mouse_ray_q));

        static vec3 mouse_ray_cube_hit = VEC3_1;
        if (input_mouse_shoot_emitter) {
            float min_distance = INFINITY;
            for (int i = 0; i < 3; i++) {
                vec3 face_normal = VEC3_0;
                float face_distance = (1.0 + abs(mouse_ray_p[i])) / abs(mouse_ray_v[i]);
                if (face_distance < min_distance) {
                    mouse_ray_cube_hit = vec3(mouse_ray_p) - vec3(mouse_ray_v)*face_distance;
                    min_distance = face_distance;
                }
            }
        }

        static vec3 mouse_ray_billboard_hit_prev = vec3(NAN);
        if (input_mouse_dragging_brush) {
            // calculate intersection with billboarded plane at origin
            vec3 billboard_n = camera_pos - camera_target;
            vec3 mouse_ray_billboard_hit = vec3(mouse_ray_p) - vec3(mouse_ray_v)*dot(billboard_n, vec3(mouse_ray_p))/dot(billboard_n, vec3(mouse_ray_v));

            vec3 mouse_ray_d = mouse_ray_billboard_hit - mouse_ray_billboard_hit_prev;
            mouse_ray_billboard_hit_prev = mouse_ray_billboard_hit;

            if (!isnan(mouse_ray_d.x)) {
                mouse_ray_p += vec4(1.0);
                mouse_ray_p *= 0.5;
                mouse_ray_d *= 0.5 * debug_brush_strength;

                sim_add_velocity_and_density_along_ray(
                    mouse_ray_p.x, mouse_ray_p.y, mouse_ray_p.z,
                    mouse_ray_v.x, mouse_ray_v.y, mouse_ray_v.z,
                    debug_brush_radius * 0.5, debug_brush_hardness,
                    mouse_ray_d.x, mouse_ray_d.y, mouse_ray_d.z, 0.0
                );
            }
        } else {
            mouse_ray_billboard_hit_prev = vec3(NAN);
        }

        if (input_do_debug_reset_density_field) {
            sim_debug_reset_density_field(0, 0);
            input_do_debug_reset_density_field = false;

            debug_render_density_factor = 1.0;
            glUniform1f(u_debug_render_density_factor, debug_render_density_factor);
        }
        if (input_do_debug_reset_velocity_field) {
            double k = TAU*100;
            sim_debug_reset_velocity_field(0, 0, 0, 0);
            input_do_debug_reset_velocity_field = false;
        }
        if (input_set_sim_debug_data_mode != sim_debug_data_mode) {
            debug_render_mode_and_flags &= ~DEBUG_RENDER_MODE_MASK;
            switch (input_set_sim_debug_data_mode) {
            case None:
                debug_render_mode_and_flags |= DEBUG_RENDER_MODE_DEFAULT;
                break;
            case NormalizedVelocityAndMagnitude:
                debug_render_mode_and_flags |= DEBUG_RENDER_MODE_VELOCITIES;
                break;
            }
            sim_debug_data_mode = input_set_sim_debug_data_mode;
            update_debug_render_mode_and_flags = true;
        }
        if (update_debug_render_mode_and_flags) {
            glUniform1i(u_debug_render_mode_and_flags, debug_render_mode_and_flags);
            update_debug_render_mode_and_flags = false;
        }

        float sphere_time = -debug_emitter_freq*(debug_emitter_time + debug_emitter_offset);
        vec3 sphere_pos = vec3(0.0, 0.0, debug_emitter_altitude);
        vec3 sphere_vel;
        if (input_mouse_shoot_emitter) {
            sphere_vel = normalize(mouse_ray_cube_hit) * debug_emitter_speed * debug_delta_time;
        } else if (!debug_emitter_aim) {
            sphere_vel = -vec3(-cos(sphere_time), sin(sphere_time), 0.0);
            float wobble_time = debug_emitter_wobble*sin(0.8*1.61803*TAU*debug_emitter_time);
            sphere_vel *= cos(wobble_time);
            sphere_vel.z = sin(wobble_time);
            sphere_vel.z += debug_emitter_vertical_speed;
            sphere_vel *= debug_emitter_speed;
            sphere_vel *= debug_delta_time;
        }
        if (!debug_emitter_aim || input_mouse_shoot_emitter) {
            sim_add_velocity_and_density(
                (int) ((sphere_pos.x+1) * (float) (SIM_GRID_SIZE-1)/2)-sphere_size/2,
                (int) ((sphere_pos.y+1) * (float) (SIM_GRID_SIZE-1)/2)-sphere_size/2,
                (int) ((sphere_pos.z+1) * (float) (SIM_GRID_SIZE-1)/2)-sphere_size/2,
                // (SIM_GRID_SIZE-sphere_size)/2, (SIM_GRID_SIZE-sphere_size)/2, (SIM_GRID_SIZE-sphere_size)/2,
                sphere_vel.x, sphere_vel.y, sphere_vel.z, debug_emitter_density*debug_emitter_speed*debug_delta_time,
                (float*) sphere_velocity, (float*) sphere_density, 0, true,
                sphere_size, sphere_size, sphere_size
            );
            debug_emitter_time += debug_delta_time;
        }

        // SIM UPDATE
        sim_update(debug_delta_time);

        // RENDER
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // volume outline
        if (debug_render_boundaries) {
            glUniform1i(u_debug_render_mode_and_flags, debug_render_mode_and_flags & ~DEBUG_RENDER_MODE_MASK | DEBUG_RENDER_MODE_AXIS_COLOR);
            glDrawElements(GL_LINE_STRIP, 16, GL_UNSIGNED_BYTE, CUBE_OUTLINE_INDICES);
            glUniform1i(u_debug_render_mode_and_flags, debug_render_mode_and_flags);
        }

        // simulation
        glDrawElements(GL_TRIANGLES, 3*6*2, GL_UNSIGNED_BYTE, CUBE_FACE_INDICES);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // CLEANUP
    sim_terminate();
    glfwTerminate();
}
