@echo off
setlocal
cd /d "%~dp0"
rem ============================================================
rem Build the 2D solver (sph_core2.exe) and 3D solver (sph_core3.exe).
rem Requires g++ with OpenMP support (MSYS2/MinGW; see README.md).
rem ============================================================
g++ -O3 -march=native -fopenmp -std=c++17 sph_core2.cpp -o sph_core2.exe
if %errorlevel% neq 0 goto fail
g++ -O3 -march=native -fopenmp -std=c++17 -DDIM=3 sph_core2.cpp -o sph_core3.exe
if %errorlevel% neq 0 goto fail
echo.
echo Build succeeded: sph_core2.exe (2D) and sph_core3.exe (3D).
echo Start the GUI with: python gui2.py  (or double-click run_gui.bat)
pause
exit /b 0
:fail
echo.
echo BUILD FAILED. Check that g++ is installed and available in PATH:
echo    g++ --version
pause
exit /b 1
endlocal
