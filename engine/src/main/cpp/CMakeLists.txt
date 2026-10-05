cmake_minimum_required(VERSION 3.18)
project(mossnano_jni CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

# google/sentencepiece, built as a static library (bundles its own protobuf-lite).
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
set(SPM_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_NFKC_COMPILE OFF CACHE BOOL "" FORCE)
set(SPM_USE_BUILTIN_PROTOBUF ON CACHE BOOL "" FORCE)
FetchContent_Declare(
  sentencepiece
  GIT_REPOSITORY https://github.com/google/sentencepiece.git
  GIT_TAG 17d7580d6407802f85855d2cc9190634e2c95624 # v0.2.0
)
FetchContent_GetProperties(sentencepiece)
if(NOT sentencepiece_POPULATED)
  FetchContent_Populate(sentencepiece)
  add_subdirectory(${sentencepiece_SOURCE_DIR} ${sentencepiece_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

add_library(mossnano_jni SHARED spm_jni.cpp)
target_include_directories(mossnano_jni PRIVATE ${sentencepiece_SOURCE_DIR}/src)
target_link_libraries(mossnano_jni PRIVATE sentencepiece-static)
if(ANDROID)
  target_link_libraries(mossnano_jni PRIVATE log)
  target_compile_options(mossnano_jni PRIVATE -O2 -fvisibility=hidden)
  target_link_options(mossnano_jni PRIVATE -Wl,--gc-sections -Wl,-z,max-page-size=16384)
else()
  find_package(JNI REQUIRED)
  target_include_directories(mossnano_jni PRIVATE ${JNI_INCLUDE_DIRS})
endif()
