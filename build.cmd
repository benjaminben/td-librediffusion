@echo off
REM Build helper for td-librediffusion.
REM
REM Usage:
REM   build           - incremental rebuild
REM   build configure - run cmake configure (first time, or after CMakeLists changes)
REM   build clean     - wipe build dir, then configure + build
REM
REM MSVC environment is auto-detected via vswhere.exe (ships with VS 2017+).
REM Override by setting %VCVARS% to a vcvars64.bat path before running, e.g.:
REM   set VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat

setlocal EnableDelayedExpansion

set HERE=%~dp0
if "%HERE:~-1%"=="\" set HERE=%HERE:~0,-1%
set BUILD=%HERE%\build

set CMD=%1
if "%CMD%"=="" set CMD=build
if /i "%CMD%"=="help" goto :usage
if /i "%CMD%"=="-h" goto :usage

REM Already inside a developer command prompt? Skip vcvars discovery.
if defined VSCMD_VER goto :have_msvc

REM Caller-provided override?
if defined VCVARS goto :try_vcvars

REM Auto-detect via vswhere. Restrict to VS 2022 (17.x) since newer
REM previews (e.g. VS 18 Insiders MSVC 14.50) ship an incomplete
REM <xutility> that breaks our C++23 ranges usage.
set VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe
if not exist "!VSWHERE!" goto :probe_known_paths

for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -version "[17.0^,18.0)" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set VS_INSTALL=%%i
)
if defined VS_INSTALL (
    set VCVARS=!VS_INSTALL!\VC\Auxiliary\Build\vcvars64.bat
    goto :try_vcvars
)

REM vswhere came up empty -- VS 2022 may still be on disk but unregistered.
REM Probe the standard VS 2022 install paths directly.
:probe_known_paths
call :check_path "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call :check_path "C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
call :check_path "C:\Program Files (x86)\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
call :check_path "C:\Program Files (x86)\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
call :check_path "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call :check_path "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
call :check_path "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
call :check_path "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
if defined VCVARS goto :try_vcvars
goto :no_vs_instance

:check_path
if defined VCVARS goto :eof
if exist %1 set VCVARS=%~1
goto :eof

:try_vcvars
if not exist "!VCVARS!" goto :no_vcvars
echo Loading MSVC environment from !VCVARS!...
call "!VCVARS!" >nul
goto :have_msvc

:no_vs_instance
echo ERROR: no Visual Studio 2022 (17.x) installation found.
echo   - vswhere did not return a 17.x install.
echo   - The 8 standard VS 2022 paths under "Program Files" / "Program Files (x86)"
echo     were probed and none contained vcvars64.bat.
echo.
echo Install Visual Studio 2022 with the "Desktop development with C++" workload,
echo or set %%VCVARS%% manually to your vcvars64.bat path before running this script.
echo.
echo Note: VS 18 / VS 2026 ships an MSVC with broken C++23 ranges support that
echo fails to compile this project. Use VS 2022 17.x.
exit /b 1

:no_vcvars
echo ERROR: vcvars64.bat not found at "!VCVARS!"
echo Set %%VCVARS%% to a valid vcvars64.bat path, or run from
echo "x64 Native Tools Command Prompt for VS 2022".
exit /b 1

:have_msvc
if /i "%CMD%"=="clean" (
    if exist "!BUILD!" rmdir /S /Q "!BUILD!"
    set CMD=configure-then-build
)

if /i "%CMD%"=="configure" goto :do_configure
if /i "%CMD%"=="configure-then-build" goto :do_configure
if /i "%CMD%"=="build" goto :do_build

echo Unknown command: %1
goto :usage

:do_configure
echo Configuring with cmake...
cmake -S "!HERE!" -B "!BUILD!" -G Ninja -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1
if /i "%CMD%"=="configure" exit /b
goto :do_build

:do_build
if not exist "!BUILD!\build.ninja" (
    echo build dir missing -- running configure first.
    goto :do_configure
)
echo Building...
cmake --build "!BUILD!"
if errorlevel 1 exit /b 1
echo.
echo Plugin DLL: !HERE!\plugin\td_librediffusion_top.dll
exit /b

:usage
echo Usage: build [configure^|build^|clean]
echo.
echo   configure  cmake configure into .\build\
echo   build      incremental cmake --build (default)
echo   clean      wipe .\build\ and reconfigure + build
exit /b 1
