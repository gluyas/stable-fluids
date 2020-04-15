@echo off

mkdir out

rem GLAD
cl /Foout/ /Ilib/ lib/glad/glad.c /c
