add_executable(deepforge_mlir_import_test
    Conv2DImporterTest.cpp
)

target_link_libraries(deepforge_mlir_import_test PRIVATE DeepForge::MLIRImport)
target_include_directories(deepforge_mlir_import_test SYSTEM
    PRIVATE
        "${DEEPFORGE_CUDNN_FRONTEND_ROOT}/include"
)

if(MSVC)
    target_compile_options(deepforge_mlir_import_test PRIVATE /W4)
else()
    target_compile_options(deepforge_mlir_import_test PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(
    NAME deepforge-mlir-import
    COMMAND deepforge_mlir_import_test
            "${PROJECT_SOURCE_DIR}/test/fixtures/conv2d_f32.json"
)

add_test(
    NAME deepforge-mlir-one-shot-bufferize
    COMMAND "${LLVM_TOOLS_BINARY_DIR}/mlir-opt"
            "${PROJECT_SOURCE_DIR}/test/fixtures/conv2d_f32.mlir"
            "--pass-pipeline=builtin.module(one-shot-bufferize)"
)
