add_library(deepforge_transforms STATIC
    BufferizationPipeline.cpp
    WorkspacePlanner.cpp
)
add_library(DeepForge::Transforms ALIAS deepforge_transforms)

target_compile_features(deepforge_transforms PUBLIC cxx_std_20)
target_include_directories(deepforge_transforms
    PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)
target_include_directories(deepforge_transforms SYSTEM PUBLIC
    "${MLIR_INCLUDE_DIRS}"
    "${LLVM_INCLUDE_DIRS}"
)
target_link_libraries(deepforge_transforms
    PUBLIC
        DeepForge::MLIRImport
        MLIRArithTransforms
        MLIRBufferizationDialect
        MLIRBufferizationToMemRef
        MLIRBufferizationTransforms
        MLIRLinalgTransforms
        MLIRMemRefDialect
        MLIRPass
        MLIRTensorTransforms
        MLIRTransforms
)

if(MSVC)
    target_compile_options(deepforge_transforms PRIVATE /W4)
else()
    target_compile_options(deepforge_transforms PRIVATE -Wall -Wextra -Wpedantic)
endif()
