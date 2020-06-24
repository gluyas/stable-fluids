CONTROLS:
SPACE + MOUSE1: orbit camera
SCROLLWHEEL:    zoom camera
R:              reset simulation

MOUSE1:         disturb fluid flow
MOUSE2:         aim emitter (works best with emitter aim only checkbox enabled)

TILDE/GRAVE:    toggle ui
1, 2:           set render mode


BUILD FROM SOURCE:
in project root directory from cmd with cl and nvcc in path:

first:
$ build_lib
    builds the external library sources and puts them in the /out/ directory
    these will be linked later by the other build commands
    only needs to be run once

then:
$ run
    builds and runs the program from source with default configuration

$ build [nvcc compiler options (eg: --debug, -O3)]
    builds the program to out/stable_fluids.exe

$ rerun
    runs the program if it has already been compiled (same as $ .\out\stable_fluids)
