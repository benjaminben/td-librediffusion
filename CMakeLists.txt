cmake_minimum_required(VERSION 3.18)
project(td_librediffusion CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---- TD plugin output goes into ./plugin/ next to td_top_plugin.cpp.
# TouchDesigner loads the .dll from wherever its Plugin Path parameter points.
set(PLUGIN_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/plugin)
foreach(cfg "" _DEBUG _RELEASE _MINSIZEREL _RELWITHDEBINFO)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY${cfg} ${PLUGIN_OUTPUT_DIR})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY${cfg} ${PLUGIN_OUTPUT_DIR})
endforeach()

# ---- librediffusion location. Override at configure time if needed.
set(LIBREDIFFUSION_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../librediffusion"
    CACHE PATH "Path to librediffusion source root")
set(LIBREDIFFUSION_BUILD "${LIBREDIFFUSION_ROOT}/build"
    CACHE PATH "Path to librediffusion build dir (where .lib lives)")

set(LIBREDIFFUSION_IMPORT_LIB "${LIBREDIFFUSION_BUILD}/librediffusion.lib"
    CACHE FILEPATH "Import library for librediffusion.dll")
set(LIBREDIFFUSION_HEADERS "${LIBREDIFFUSION_ROOT}/src"
    CACHE PATH "Directory containing librediffusion_c.h")

if(NOT EXISTS ${LIBREDIFFUSION_IMPORT_LIB})
    message(FATAL_ERROR
        "librediffusion.lib not found at ${LIBREDIFFUSION_IMPORT_LIB}.\n"
        "Build librediffusion first (see https://github.com/benjaminben/librediffusion-bb),\n"
        "or pass -DLIBREDIFFUSION_ROOT=<path> / -DLIBREDIFFUSION_IMPORT_LIB=<path>.")
endif()

# ---- CUDA runtime (needed by the TD plugin to copy cudaArrays in CUDA mode).
find_package(CUDAToolkit REQUIRED)

# ---- Runner (host-agnostic). Links librediffusion.lib so any consumer
# (TD plugin now, standalone host later) just depends on this target.
add_library(librediffusion_runner STATIC
    runner/librediffusion_runner.cpp
    runner/librediffusion_runner.hpp
    runner/dll_loader.cpp
    runner/dll_loader.hpp
    runner/td_debug_log.hpp
)
target_include_directories(librediffusion_runner PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/runner
)
target_include_directories(librediffusion_runner PRIVATE
    ${LIBREDIFFUSION_HEADERS}
)
# schedulers/*.hpp is header-only, used to build prepare_scheduler param arrays.
target_link_libraries(librediffusion_runner PUBLIC
    ${LIBREDIFFUSION_IMPORT_LIB}
)

# ---- TD plugin (throwaway glue).
add_library(td_librediffusion_top MODULE
    plugin/td_top_plugin.cpp
)
set_target_properties(td_librediffusion_top PROPERTIES
    PREFIX ""
    SUFFIX ".dll"
)
target_include_directories(td_librediffusion_top PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/derivative
)
target_link_libraries(td_librediffusion_top PRIVATE
    librediffusion_runner
    CUDA::cudart
)

# Delay-load librediffusion.dll so its imports (nvinfer, cudart, ...) are not
# resolved at plugin load time. ensure_libraries_loaded() in the runner pre-
# loads librediffusion.dll with LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR so its deps
# resolve from our plugin folder, NOT from TouchDesigner.exe's directory.
if(MSVC)
    target_link_options(td_librediffusion_top PRIVATE
        "/DELAYLOAD:librediffusion.dll"
    )
    target_link_libraries(td_librediffusion_top PRIVATE delayimp.lib)

    target_compile_options(td_librediffusion_top PRIVATE /W3 /permissive-)
    target_compile_options(librediffusion_runner PRIVATE /W3 /permissive-)
endif()

message(STATUS "td-librediffusion will build: ${PLUGIN_OUTPUT_DIR}/td_librediffusion_top.dll")
message(STATUS "  linking against: ${LIBREDIFFUSION_IMPORT_LIB}")
