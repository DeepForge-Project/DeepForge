add_executable(deepforge-compile
    deepforge-compile.cpp
)

target_link_libraries(deepforge-compile PRIVATE DeepForge::Compiler)
target_compile_features(deepforge-compile PRIVATE cxx_std_20)

if(MSVC)
    target_compile_options(deepforge-compile PRIVATE /W4)
else()
    target_compile_options(deepforge-compile PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS deepforge-compile RUNTIME DESTINATION bin)

add_executable(deepforge-benchmark
    deepforge-benchmark.cpp
)
target_link_libraries(deepforge-benchmark PRIVATE DeepForge::Compiler)
target_compile_features(deepforge-benchmark PRIVATE cxx_std_20)
if(MSVC)
    target_compile_options(deepforge-benchmark PRIVATE /W4)
else()
    target_compile_options(deepforge-benchmark PRIVATE -Wall -Wextra -Wpedantic)
endif()
install(TARGETS deepforge-benchmark RUNTIME DESTINATION bin)
