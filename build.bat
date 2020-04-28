@echo off

nvcc ^
    src/main.cpp ^
    src/simulation.cu ^
    lib/glfw/glfw3dll.lib ^
    out/glad.obj ^
        out/imgui.obj out/imgui_widgets.obj out/imgui_draw.obj out/imgui_demo.obj ^
        out/imgui_impl_glfw.obj out/imgui_impl_opengl3.obj ^
        --include-path lib/imgui/ ^
    --include-path lib/ ^
    --output-file out/stable_fluids ^
    %*

if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
copy /Y .\src\global_defines.h .\out\global_defines.h > NUL
