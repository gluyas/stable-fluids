@echo off
setlocal

call ./build
if %ERRORLEVEL% neq 0 (
    echo. & echo build failed
    exit /b
)

echo.
.\out\stable_fluids

endlocal
