# TODO: should not use this
if (WIN32)
    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()

#
# libraries
#

# llama

list(APPEND LLAMA_SOURCE_FILES
    ../include/llama.h
    llama.cpp
    llama-vocab.cpp
    llama-grammar.cpp
    llama-sampling.cpp
    unicode.h
    unicode.cpp
    unicode-data.cpp
    prefetch.h
    prefetch.cpp
    io.h
    io-frontend.h
    io-frontend.cpp
    pipeline.h
    pipeline.cpp
    layer-sched.cpp
    io-stage.cpp
    decrypt-stage.cpp
)

if (LLAMA_CHCORE_API)
    list(APPEND LLAMA_SOURCE_FILES alloc-stage-chcore.cpp)
else()
    list(APPEND LLAMA_SOURCE_FILES alloc-stage.cpp io-backend.cpp)
endif()

add_library(llama ${LLAMA_SOURCE_FILES})

target_include_directories(llama PUBLIC . ../include)
target_compile_features   (llama PUBLIC cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

# find_package(OpenSSL REQUIRED)
# target_include_directories(llama PRIVATE ${OPENSSL_INCLUDE_DIR})
# target_link_libraries(llama PRIVATE OpenSSL::SSL OpenSSL::Crypto)
# target_link_libraries(llama PUBLIC aio)

list(APPEND LLAMA_INCLUDE_DIR
    /home/vectorxj/openharmony/third_party/openssl/include
    /home/vectorxj/chcore/opentrustee_llm/libaio-0.3.113/install/include
)
list(APPEND LLAMA_LINK_DIR
)

if (LLAMA_CHCORE_API)
list(APPEND LLAMA_INCLUDE_DIR
    /home/vectorxj/openharmony/base/tee/tee_os_kernel/user/chcore-libc/musl-libc/include
)
list(APPEND LLAMA_LINK_DIR
    /home/vectorxj/openssl/openssl-3.4.1/install/lib/libcrypto.so.3
    /home/vectorxj/openssl/openssl-3.4.1/install/lib/libssl.so.3
)
else()
list(APPEND LLAMA_INCLUDE_DIR
    /home/vectorxj/chcore/opentrustee_llm/rknn-toolkit2-2.3.0/rknpu2/runtime/Linux/librknn_api/include
    /home/vectorxj/chcore/opentrustee_llm/libaio-0.3.113/install-glibc/include
)
list(APPEND LLAMA_LINK_DIR
    /home/vectorxj/chcore/opentrustee_llm/rknn-toolkit2-2.3.0/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so
    /home/vectorxj/openharmony/third_party/openssl/install-glibc/lib/libcrypto.so.3
    /home/vectorxj/openharmony/third_party/openssl/install-glibc/lib/libssl.so.3
    /home/vectorxj/chcore/opentrustee_llm/libaio-0.3.113/install-glibc/lib/libaio.so.1.0.2
)
endif()

target_include_directories(llama PUBLIC ${LLAMA_INCLUDE_DIR})
target_link_libraries(llama PUBLIC ${LLAMA_LINK_DIR})

if (NOT LLAMA_CHCORE_API)
add_library(
    remoting_backend
    interface.h
    io.h
    io-backend.cpp
    ca-backend.cpp
)
target_compile_features(remoting_backend PUBLIC cxx_std_17)
target_include_directories(
    remoting_backend
    PUBLIC
    /home/vectorxj/chcore/opentrustee_llm/libaio-0.3.113/install-glibc/include
    /home/vectorxj/chcore/opentrustee_llm/rknn-toolkit2-2.3.0/rknpu2/runtime/Linux/librknn_api/include
)
target_link_libraries(
    remoting_backend
    PUBLIC
    /home/vectorxj/chcore/opentrustee_llm/libaio-0.3.113/install-glibc/lib/libaio.so.1.0.2
)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_SHARED LLAMA_BUILD)
endif()
