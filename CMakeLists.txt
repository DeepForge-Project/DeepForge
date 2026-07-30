cmake_minimum_required(VERSION 3.27)
project(DeepForge VERSION 0.1.0 LANGUAGES CXX C)

set(DEEPFORGE_REQUIRED_LLVM_VERSION "22.1.8")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Prevent in-source builds
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR
        "In-source builds are not allowed. Please use a build directory:\n"
        "  mkdir build && cd build && cmake ..")
endif()

# ---- Options ----
option(DEEPFORGE_BUILD_TESTS "Build DeepForge tests" OFF)
option(DEEPFORGE_BUILD_TOOLS "Build DeepForge command-line tools" ON)
option(DEEPFORGE_BUILD_IMPORTER "Build the CPU-only serialized Graph importer" ON)
option(DEEPFORGE_ENABLE_MLIR "Enable the MLIR-dependent compiler components" ON)

set(DEEPFORGE_CUDNN_FRONTEND_ROOT
    "$ENV{DEEPFORGE_CUDNN_FRONTEND_ROOT}"
    CACHE PATH "Root of the pinned cudnn-frontend source checkout")

# ---- Find MLIR (optional for the importer-only build) ----
if(DEEPFORGE_ENABLE_MLIR)
    find_package(MLIR REQUIRED CONFIG)
    if(NOT LLVM_PACKAGE_VERSION VERSION_EQUAL DEEPFORGE_REQUIRED_LLVM_VERSION)
        message(FATAL_ERROR
            "DeepForge requires LLVM/MLIR ${DEEPFORGE_REQUIRED_LLVM_VERSION}, "
            "but MLIR_DIR provides ${LLVM_PACKAGE_VERSION}")
    endif()
    message(STATUS "Using MLIR from: ${MLIR_DIR}")
    message(STATUS "LLVM/MLIR version: ${LLVM_PACKAGE_VERSION}")
    message(STATUS "LLVM binaries at: ${LLVM_TOOLS_BINARY_DIR}")

    # ---- Add MLIR infrastructure ----
    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
    include(AddMLIR)
endif()

# ---- Subdirectories ----
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/CMakeLists.txt")
    add_subdirectory(include)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/CMakeLists.txt")
    add_subdirectory(lib)
endif()

if(DEEPFORGE_BUILD_TOOLS AND NOT DEEPFORGE_ENABLE_MLIR)
    message(FATAL_ERROR "DEEPFORGE_BUILD_TOOLS requires DEEPFORGE_ENABLE_MLIR=ON")
endif()

if(DEEPFORGE_BUILD_TOOLS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/CMakeLists.txt")
    add_subdirectory(tools)
endif()

if(DEEPFORGE_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/CMakeLists.txt")
    if(NOT DEEPFORGE_BUILD_IMPORTER)
        message(FATAL_ERROR "DEEPFORGE_BUILD_TESTS currently requires DEEPFORGE_BUILD_IMPORTER=ON")
    endif()
    enable_testing()
    add_subdirectory(test)
endif()
