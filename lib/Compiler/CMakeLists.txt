add_library(deepforge_compiler STATIC
    Artifact.cpp
    Compiler.cpp
    FoundationalGraph.cpp
    NormalizationGraph.cpp
    ScalarLowering.cpp
    TrainingGraph.cpp
)
add_library(DeepForge::Compiler ALIAS deepforge_compiler)

target_compile_features(deepforge_compiler PUBLIC cxx_std_20)
target_include_directories(deepforge_compiler
    PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_include_directories(deepforge_compiler SYSTEM PUBLIC
    "${MLIR_INCLUDE_DIRS}"
    "${LLVM_INCLUDE_DIRS}"
)
target_link_libraries(deepforge_compiler
    PUBLIC
        DeepForge::Transforms
        DeepForge::Runtime
        MLIRAffineDialect
        MLIRAffineToStandard
        MLIRArithToLLVM
        MLIRControlFlowDialect
        MLIRControlFlowToLLVM
        MLIRFuncToLLVM
        MLIRIndexDialect
        MLIRIndexToLLVM
        MLIRLinalgTransforms
        MLIRMathDialect
        MLIRMathToLLVM
        MLIRMathToLibm
        MLIRMemRefToLLVM
        MLIRMemRefTransforms
        MLIRReconcileUnrealizedCasts
        MLIRSCFToControlFlow
        MLIRTargetLLVMIRExport
        MLIRToLLVMIRTranslationRegistration
        MLIRVectorDialect
        MLIRVectorToLLVM
        MLIRVectorToLLVMPass
        LLVMCore
        LLVMCodeGen
        LLVMExecutionEngine
        LLVMMC
        LLVMObject
        LLVMTarget
        LLVMX86CodeGen
        LLVMX86Desc
        LLVMX86Info
)

if(MSVC)
    target_compile_options(deepforge_compiler PRIVATE /W4)
else()
    target_compile_options(deepforge_compiler PRIVATE -Wall -Wextra -Wpedantic)
endif()
