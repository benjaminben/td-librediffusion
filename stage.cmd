@echo off
REM Stage librediffusion.dll + TRT/CUDA runtime DLLs into plugin/ so they
REM resolve from the same directory as td_librediffusion_top.dll when TD
REM loads the plugin via LoadLibraryEx + LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR.
REM
REM Required environment variables (set them once in your shell or in a
REM personal `env.cmd` you source before running this):
REM
REM   LIBREDIFF_BUILD  Path to the librediffusion build directory
REM                    (the folder containing librediffusion.dll).
REM                    Example: C:\src\librediffusion-bb\build
REM
REM   TENSORRT_ROOT    Path to the TensorRT install root.
REM                    The bin\ subfolder must contain nvinfer_10.dll etc.
REM                    Example: C:\src\TensorRT-10.16.1.11
REM
REM   CUDA_BIN         Path to the directory containing the CUDA runtime DLLs
REM                    (cudart64_13.dll, cublas64_13.dll, ...). On CUDA 13+
REM                    these live under bin\x64, not bin\ directly.
REM                    Example: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64
REM
REM Optional override: pass any of the three as a command-line argument:
REM   stage <build-dir> <tensorrt-root> <cuda-bin>

setlocal

set HERE=%~dp0
if "%HERE:~-1%"=="\" set HERE=%HERE:~0,-1%
set TARGET=%HERE%\plugin

if not "%~1"=="" set LIBREDIFF_BUILD=%~1
if not "%~2"=="" set TENSORRT_ROOT=%~2
if not "%~3"=="" set CUDA_BIN=%~3

set MISSING=
if "%LIBREDIFF_BUILD%"=="" set MISSING=%MISSING% LIBREDIFF_BUILD
if "%TENSORRT_ROOT%"=="" set MISSING=%MISSING% TENSORRT_ROOT
if "%CUDA_BIN%"==""      set MISSING=%MISSING% CUDA_BIN

if not "%MISSING%"=="" (
    echo.
    echo ERROR: required environment variables not set:%MISSING%
    echo.
    echo Set them in your shell, e.g.:
    echo   set LIBREDIFF_BUILD=C:\src\librediffusion-bb\build
    echo   set TENSORRT_ROOT=C:\src\TensorRT-10.16.1.11
    echo   set CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64
    echo.
    echo Or pass them positionally: stage ^<build^> ^<trt-root^> ^<cuda-bin^>
    exit /b 1
)

echo === Sources ===
echo LIBREDIFF_BUILD = %LIBREDIFF_BUILD%
echo TENSORRT_ROOT   = %TENSORRT_ROOT%
echo CUDA_BIN        = %CUDA_BIN%
echo TARGET          = %TARGET%
echo.

if not exist "%TARGET%" (
    echo ERROR: plugin folder does not exist: %TARGET%
    echo Run `build` first.
    exit /b 1
)

echo --- librediffusion.dll ---
if exist "%LIBREDIFF_BUILD%\librediffusion.dll" (
    copy /Y "%LIBREDIFF_BUILD%\librediffusion.dll" "%TARGET%\" >nul && echo OK     librediffusion.dll
) else (
    echo MISSING: %LIBREDIFF_BUILD%\librediffusion.dll  -- build librediffusion first
)

echo.
echo --- TensorRT *.dll from %TENSORRT_ROOT%\bin ---
if exist "%TENSORRT_ROOT%\bin" (
    copy /Y "%TENSORRT_ROOT%\bin\*.dll" "%TARGET%\" >nul && echo OK     TensorRT bin\*.dll
) else (
    echo MISSING: %TENSORRT_ROOT%\bin
)

echo.
echo --- CUDA runtime DLLs ---
call :copyone "%CUDA_BIN%\cudart64_13.dll"
call :copyone "%CUDA_BIN%\curand64_10.dll"
call :copyone "%CUDA_BIN%\cublas64_13.dll"
call :copyone "%CUDA_BIN%\cublasLt64_13.dll"
call :copyone "%CUDA_BIN%\nppig64_13.dll"
call :copyone "%CUDA_BIN%\nppc64_13.dll"

echo.
echo === Plugin folder DLLs ===
dir /B "%TARGET%\*.dll"
exit /b

:copyone
if exist %1 (
    copy /Y %1 "%TARGET%\" >nul && echo OK     %~nx1 || echo FAILED %~nx1
) else (
    echo MISSING %~nx1   (looked at %~1)
)
goto :eof
