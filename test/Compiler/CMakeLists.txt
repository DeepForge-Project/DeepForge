add_executable(deepforge_compiler_test
    CompilerTest.cpp
)

add_executable(deepforge_runtime_header_test RuntimeHeaderTest.cpp)
target_include_directories(deepforge_runtime_header_test
    PRIVATE "${PROJECT_SOURCE_DIR}/include"
)
if(MSVC)
    target_compile_options(deepforge_runtime_header_test PRIVATE /W4)
else()
    target_compile_options(deepforge_runtime_header_test PRIVATE
        -Wall -Wextra -Wpedantic)
endif()
add_test(NAME deepforge-runtime-public-header
         COMMAND deepforge_runtime_header_test)

find_package(Threads REQUIRED)
target_link_libraries(deepforge_compiler_test
    PRIVATE
        DeepForge::Compiler
        Threads::Threads
)
target_include_directories(deepforge_compiler_test SYSTEM
    PRIVATE
        "${DEEPFORGE_CUDNN_FRONTEND_ROOT}/include"
)

if(MSVC)
    target_compile_options(deepforge_compiler_test PRIVATE /W4)
else()
    target_compile_options(deepforge_compiler_test PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(
    NAME deepforge-compiler
    COMMAND deepforge_compiler_test
            "${PROJECT_SOURCE_DIR}/test/fixtures/conv2d_f32.json"
)

add_test(
    NAME deepforge-compiler-c17
    COMMAND deepforge_compiler_test
            "${PROJECT_SOURCE_DIR}/test/fixtures/conv2d_f32_c17.json"
)

add_executable(deepforge_artifact_test
    ArtifactTest.cpp
)

target_link_libraries(deepforge_artifact_test
    PRIVATE
        DeepForge::Compiler
        Threads::Threads
)
target_include_directories(deepforge_artifact_test SYSTEM
    PRIVATE
        "${DEEPFORGE_CUDNN_FRONTEND_ROOT}/include"
)

if(MSVC)
    target_compile_options(deepforge_artifact_test PRIVATE /W4)
else()
    target_compile_options(deepforge_artifact_test PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(
    NAME deepforge-artifact
    COMMAND deepforge_artifact_test
            "${PROJECT_SOURCE_DIR}/test/fixtures/conv2d_f32_c17.json"
            "${CMAKE_CURRENT_BINARY_DIR}/artifact-scratch"
)

add_test(
    NAME deepforge-artifact-no-workspace
    COMMAND deepforge_artifact_test
            "${PROJECT_SOURCE_DIR}/test/fixtures/conv2d_f32_no_padding.json"
            "${CMAKE_CURRENT_BINARY_DIR}/artifact-no-workspace-scratch"
)

add_test(
    NAME deepforge-isa-inspection
    COMMAND "${CMAKE_COMMAND}"
            "-DLLVM_OBJDUMP=${LLVM_TOOLS_BINARY_DIR}/llvm-objdump"
            "-DLLVM_READELF=${LLVM_TOOLS_BINARY_DIR}/llvm-readelf"
            "-DSCALAR_OBJECT=${CMAKE_CURRENT_BINARY_DIR}/artifact-scratch/scalar.o"
            "-DAVX2_OBJECT=${CMAKE_CURRENT_BINARY_DIR}/artifact-scratch/avx2.o"
            "-DAVX512_OBJECT=${CMAKE_CURRENT_BINARY_DIR}/artifact-scratch/avx512.o"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckIsa.cmake"
)
set_tests_properties(deepforge-isa-inspection PROPERTIES
    DEPENDS deepforge-artifact
)

add_executable(deepforge_variant_matrix_test
    VariantMatrixTest.cpp
)
target_link_libraries(deepforge_variant_matrix_test PRIVATE DeepForge::Compiler)
if(MSVC)
    target_compile_options(deepforge_variant_matrix_test PRIVATE /W4)
else()
    target_compile_options(deepforge_variant_matrix_test PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_test(NAME deepforge-variant-matrix COMMAND deepforge_variant_matrix_test)

if(TARGET deepforge-compile)
    set(DEEPFORGE_CLI_SCRATCH "${CMAKE_CURRENT_BINARY_DIR}/cli-scratch")
    file(MAKE_DIRECTORY "${DEEPFORGE_CLI_SCRATCH}")

    add_test(
        NAME deepforge-cli-artifact
        COMMAND deepforge-compile
                "${PROJECT_SOURCE_DIR}/test/fixtures/conv2d_f32.json"
                --input-format=json
                --target=x86-64
                --emit=object
                --dump-ir=imported:${DEEPFORGE_CLI_SCRATCH}/imported.mlir
                --dump-ir=bufferized:${DEEPFORGE_CLI_SCRATCH}/bufferized.mlir
                --dump-ir=llvm:${DEEPFORGE_CLI_SCRATCH}/llvm.mlir
                -o "${DEEPFORGE_CLI_SCRATCH}/conv2d.dfo"
    )
    add_test(
        NAME deepforge-cli-inspect
        COMMAND deepforge-compile
                --inspect "${DEEPFORGE_CLI_SCRATCH}/conv2d.dfo"
    )
    set_tests_properties(deepforge-cli-inspect PROPERTIES
        DEPENDS deepforge-cli-artifact
    )

    add_test(
        NAME deepforge-cli-inspect-invalid-options
        COMMAND deepforge-compile
                --inspect "${DEEPFORGE_CLI_SCRATCH}/conv2d.dfo"
                --variant=scalar
    )
    set_tests_properties(deepforge-cli-inspect-invalid-options PROPERTIES
        WILL_FAIL TRUE
    )

    add_test(
        NAME deepforge-cli-llvm-ir
        COMMAND deepforge-compile
                "${PROJECT_SOURCE_DIR}/test/fixtures/conv2d_f32_c17.json"
                --input-format=auto
                --emit=llvm-ir
                --variant=avx2
                -o "${DEEPFORGE_CLI_SCRATCH}/conv2d-avx2.ll"
    )

    add_test(
        NAME deepforge-cli-ubjson
        COMMAND deepforge-compile
                "${CMAKE_CURRENT_BINARY_DIR}/artifact-scratch/conv2d.ubjson"
                --input-format=ubjson
                --target=x86-64
                --emit=object
                -o "${DEEPFORGE_CLI_SCRATCH}/conv2d-ubjson.dfo"
    )
    set_tests_properties(deepforge-cli-ubjson PROPERTIES
        DEPENDS deepforge-artifact
    )

    add_test(
        NAME deepforge-cli-invalid-target
        COMMAND deepforge-compile
                "${PROJECT_SOURCE_DIR}/test/fixtures/conv2d_f32.json"
                --target=aarch64
                -o "${DEEPFORGE_CLI_SCRATCH}/invalid.dfo"
    )
    set_tests_properties(deepforge-cli-invalid-target PROPERTIES WILL_FAIL TRUE)

    add_test(
        NAME deepforge-cli-empty-output
        COMMAND deepforge-compile
                "${PROJECT_SOURCE_DIR}/test/fixtures/conv2d_f32.json"
                --output=
    )
    set_tests_properties(deepforge-cli-empty-output PROPERTIES WILL_FAIL TRUE)
endif()

if(TARGET deepforge-benchmark)
    add_test(
        NAME deepforge-benchmark-smoke
        COMMAND deepforge-benchmark --profile=small --iterations=1
    )
    set_tests_properties(deepforge-benchmark-smoke PROPERTIES
        LABELS benchmark
        TIMEOUT 60
    )
endif()
