@echo off

mkdir out

rem GLAD
cl /Foout/ /Ilib/ lib/glad/glad.c /c

rem imgui
cl /Foout/ /Ilib/ /Ilib/imgui ^
    lib/imgui/imgui.cpp ^
    lib/imgui/imgui_widgets.cpp ^
    lib/imgui/imgui_draw.cpp ^
    lib/imgui/imgui_demo.cpp ^
    lib/imgui/imgui_impl_glfw.cpp ^
    lib/imgui/imgui_impl_opengl3.cpp ^
    /c
