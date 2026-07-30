add_executable(deepforge_importer_test
    SerializedGraphImporterTest.cpp
)

target_link_libraries(deepforge_importer_test PRIVATE DeepForge::Import)
target_include_directories(deepforge_importer_test SYSTEM
    PRIVATE
        "${DEEPFORGE_CUDNN_FRONTEND_ROOT}/include"
)

if(MSVC)
    target_compile_options(deepforge_importer_test PRIVATE /W4)
else()
    target_compile_options(deepforge_importer_test PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(
    NAME deepforge-importer
    COMMAND deepforge_importer_test
            "${PROJECT_SOURCE_DIR}/test/fixtures/conv2d_f32.json"
)
