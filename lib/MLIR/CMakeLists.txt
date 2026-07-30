add_library(deepforge_mlir_import STATIC
    Conv2DImporter.cpp
)
add_library(DeepForge::MLIRImport ALIAS deepforge_mlir_import)

target_compile_features(deepforge_mlir_import PUBLIC cxx_std_20)
target_include_directories(deepforge_mlir_import
    PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)
target_include_directories(deepforge_mlir_import SYSTEM PUBLIC
        "${MLIR_INCLUDE_DIRS}"
        "${LLVM_INCLUDE_DIRS}"
)
target_link_libraries(deepforge_mlir_import
    PUBLIC
        DeepForge::Import
        MLIRIR
        MLIRParser
        MLIRArithDialect
        MLIRFuncDialect
        MLIRLinalgDialect
        MLIRTensorDialect
)

if(MSVC)
    target_compile_options(deepforge_mlir_import PRIVATE /W4)
else()
    target_compile_options(deepforge_mlir_import PRIVATE -Wall -Wextra -Wpedantic)
endif()
