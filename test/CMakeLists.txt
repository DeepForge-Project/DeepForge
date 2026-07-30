add_subdirectory(Import)

if(DEEPFORGE_ENABLE_MLIR)
    add_subdirectory(MLIR)
    add_subdirectory(Transforms)
    add_subdirectory(Compiler)
endif()
