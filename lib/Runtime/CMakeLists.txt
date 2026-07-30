add_library(deepforge_runtime STATIC
    Executable.cpp
)
add_library(DeepForge::Runtime ALIAS deepforge_runtime)

target_compile_features(deepforge_runtime PUBLIC cxx_std_20)
target_include_directories(deepforge_runtime
    PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_include_directories(deepforge_runtime SYSTEM PUBLIC
    "${MLIR_INCLUDE_DIRS}"
    "${LLVM_INCLUDE_DIRS}"
)
target_link_libraries(deepforge_runtime
    PUBLIC
        MLIRExecutionEngine
        MLIRLLVMToLLVMIRTranslation
        LLVMJITLink
        LLVMOrcJIT
)

if(MSVC)
    target_compile_options(deepforge_runtime PRIVATE /W4)
else()
    target_compile_options(deepforge_runtime PRIVATE -Wall -Wextra -Wpedantic)
endif()
