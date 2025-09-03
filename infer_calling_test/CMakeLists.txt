cmake_minimum_required(VERSION 3.10)

# Add the executable
add_executable(infer_test main.cpp ${CMAKE_SOURCE_DIR}/examples/main/infer.cpp)
install(TARGETS infer_test RUNTIME)

# Specify the location of the infer library
target_link_libraries(infer_test PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(infer_test PRIVATE cxx_std_11)
