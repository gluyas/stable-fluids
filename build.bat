@echo off

cl /Zi /Fdout/ /Foout/ /Ilib/ /Isrc/ src/main.cpp out/glad.obj lib/glfw/glfw3dll.lib /link /OUT:out/stable_fluids.exe
