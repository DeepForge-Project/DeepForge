add_executable(deepforge_bufferization_test
    BufferizationTest.cpp
)

target_link_libraries(deepforge_bufferization_test PRIVATE DeepForge::Transforms)
target_include_directories(deepforge_bufferization_test SYSTEM
    PRIVATE
        "${DEEPFORGE_CUDNN_FRONTEND_ROOT}/include"
)

if(MSVC)
    target_compile_options(deepforge_bufferization_test PRIVATE /W4)
else()
    target_compile_options(deepforge_bufferization_test PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(
    NAME deepforge-bufferization
    COMMAND deepforge_bufferization_test
            "${PROJECT_SOURCE_DIR}/test/fixtures/conv2d_f32.json"
)
