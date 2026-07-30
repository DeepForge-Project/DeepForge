find_path(DEEPFORGE_CUDNN_FRONTEND_INCLUDE_DIR
    NAMES cudnn_frontend_version.h
    PATH_SUFFIXES include
    DOC "Directory containing pinned cudnn-frontend v1.24.0 headers"
)

if(NOT DEEPFORGE_CUDNN_FRONTEND_INCLUDE_DIR)
    message(FATAL_ERROR
        "Cannot find cudnn-frontend headers. Add the v1.24.0 checkout to "
        "CMAKE_PREFIX_PATH, or set DEEPFORGE_CUDNN_FRONTEND_INCLUDE_DIR.")
endif()

set(DEEPFORGE_NLOHMANN_JSON_HEADER
    "${DEEPFORGE_CUDNN_FRONTEND_INCLUDE_DIR}/cudnn_frontend/thirdparty/nlohmann/json.hpp")
set(DEEPFORGE_CUDNN_FRONTEND_VERSION_HEADER
    "${DEEPFORGE_CUDNN_FRONTEND_INCLUDE_DIR}/cudnn_frontend_version.h")
if(NOT EXISTS "${DEEPFORGE_NLOHMANN_JSON_HEADER}")
    message(FATAL_ERROR
        "Cannot find the pinned nlohmann/json header at "
        "${DEEPFORGE_NLOHMANN_JSON_HEADER}")
endif()
if(NOT EXISTS "${DEEPFORGE_CUDNN_FRONTEND_VERSION_HEADER}")
    message(FATAL_ERROR
        "Cannot find cuDNN Frontend version header at "
        "${DEEPFORGE_CUDNN_FRONTEND_VERSION_HEADER}")
endif()

file(STRINGS "${DEEPFORGE_CUDNN_FRONTEND_VERSION_HEADER}"
    DEEPFORGE_FRONTEND_MAJOR_LINE
    REGEX "^#define CUDNN_FRONTEND_MAJOR_VERSION[ \t]+1$")
file(STRINGS "${DEEPFORGE_CUDNN_FRONTEND_VERSION_HEADER}"
    DEEPFORGE_FRONTEND_MINOR_LINE
    REGEX "^#define CUDNN_FRONTEND_MINOR_VERSION[ \t]+24$")
file(STRINGS "${DEEPFORGE_CUDNN_FRONTEND_VERSION_HEADER}"
    DEEPFORGE_FRONTEND_PATCH_LINE
    REGEX "^#define CUDNN_FRONTEND_PATCH_VERSION[ \t]+0$")
if(NOT DEEPFORGE_FRONTEND_MAJOR_LINE OR
   NOT DEEPFORGE_FRONTEND_MINOR_LINE OR
   NOT DEEPFORGE_FRONTEND_PATCH_LINE)
    message(FATAL_ERROR
        "DeepForge requires cudnn-frontend v1.24.0; the selected checkout has a different version")
endif()

file(STRINGS "${DEEPFORGE_NLOHMANN_JSON_HEADER}"
    DEEPFORGE_JSON_MAJOR_LINE
    REGEX "^#define NLOHMANN_JSON_VERSION_MAJOR[ \t]+3")
file(STRINGS "${DEEPFORGE_NLOHMANN_JSON_HEADER}"
    DEEPFORGE_JSON_MINOR_LINE
    REGEX "^#define NLOHMANN_JSON_VERSION_MINOR[ \t]+11")
file(STRINGS "${DEEPFORGE_NLOHMANN_JSON_HEADER}"
    DEEPFORGE_JSON_PATCH_LINE
    REGEX "^#define NLOHMANN_JSON_VERSION_PATCH[ \t]+3")
if(NOT DEEPFORGE_JSON_MAJOR_LINE OR
   NOT DEEPFORGE_JSON_MINOR_LINE OR
   NOT DEEPFORGE_JSON_PATCH_LINE)
    message(FATAL_ERROR
        "DeepForge requires the cudnn-frontend vendored nlohmann/json 3.11.3 header")
endif()

message(STATUS "Using cuDNN Frontend headers: ${DEEPFORGE_CUDNN_FRONTEND_INCLUDE_DIR} (v1.24.0)")
message(STATUS "Using vendored nlohmann/json: 3.11.3")

add_library(deepforge_import STATIC
    SerializedGraphImporter.cpp
)
add_library(DeepForge::Import ALIAS deepforge_import)

target_compile_features(deepforge_import PUBLIC cxx_std_20)
target_include_directories(deepforge_import
    PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)
target_include_directories(deepforge_import SYSTEM
    PRIVATE
        "${DEEPFORGE_CUDNN_FRONTEND_INCLUDE_DIR}"
)

if(MSVC)
    target_compile_options(deepforge_import PRIVATE /W4)
else()
    target_compile_options(deepforge_import PRIVATE -Wall -Wextra -Wpedantic)
endif()
