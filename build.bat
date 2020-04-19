@echo off

nvcc src/main.cpp src/simulation.cu out/glad.obj lib/glfw/glfw3dll.lib --include-path lib/ --output-file out/stable_fluids %*
