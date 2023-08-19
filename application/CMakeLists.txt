# Copyright (c) 2019-2023, Arm Limited and Contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 the "License";
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.16)

project(application LANGUAGES C CXX)

set(COMMON_FILES
        # Header Files
        application.h
        application.cpp
        debug_info.h
        debug_info.cpp
        graphics_application.h
        graphics_application.cpp
        )

set(TEXTURE_FILES
        texture/texture.h
        texture/texture.cpp
        texture/stb.h
        texture/stb.cpp
        texture/ktx.h
        texture/ktx.cpp
        texture/astc.h
        texture/astc.cpp)

source_group("\\" FILES ${FRAMEWORK_FILES})
source_group("common\\" FILES ${COMMON_FILES})

set(PROJECT_FILES
        ${COMMON_FILES}
        ${TEXTURE_FILES}
        )

# mask out the min/max macros from minwindef.h
if (MSVC)
    add_definitions(-DNOMINMAX)
endif ()

#NB: switch this to shared library and things stop working. (there is likely two copies of volk somewhere.
add_library(${PROJECT_NAME} OBJECT ${PROJECT_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

# compiler flags based on compiler type
if (NOT MSVC)
    target_compile_options(${PROJECT_NAME} PUBLIC -fexceptions)
endif ()

if (MSVC)
    target_compile_options(${PROJECT_NAME} PUBLIC /MP)
endif ()

if (${VKB_VALIDATION_LAYERS})
    target_compile_definitions(${PROJECT_NAME} PUBLIC VKB_VALIDATION_LAYERS)
endif ()

# GPU assisted validation layers are not available on macOS.
if (${VKB_VALIDATION_LAYERS_GPU_ASSISTED})
    if (APPLE)
        message(WARNING "GPU assisted validation layers are not currently available on macOS.")
    else ()
        target_compile_definitions(${PROJECT_NAME} PUBLIC VKB_VALIDATION_LAYERS)
        target_compile_definitions(${PROJECT_NAME} PUBLIC VKB_VALIDATION_LAYERS_GPU_ASSISTED)
    endif ()
endif ()

if (${VKB_VALIDATION_LAYERS_BEST_PRACTICES})
    target_compile_definitions(${PROJECT_NAME} PUBLIC VKB_VALIDATION_LAYERS)
    target_compile_definitions(${PROJECT_NAME} PUBLIC VKB_VALIDATION_LAYERS_BEST_PRACTICES)
endif ()

if (${VKB_VALIDATION_LAYERS_SYNCHRONIZATION})
    target_compile_definitions(${PROJECT_NAME} PUBLIC VKB_VALIDATION_LAYERS)
    target_compile_definitions(${PROJECT_NAME} PUBLIC VKB_VALIDATION_LAYERS_SYNCHRONIZATION)
endif ()

if (${VKB_VULKAN_DEBUG})
    target_compile_definitions(${PROJECT_NAME} PUBLIC VKB_VULKAN_DEBUG)
endif ()

if (${VKB_ENABLE_PORTABILITY})
    message(STATUS "Vulkan Portability extension is enabled")
    target_compile_definitions(${PROJECT_NAME} PUBLIC VKB_ENABLE_PORTABILITY)
endif ()

if (${VKB_WARNINGS_AS_ERRORS})
    message(STATUS "Warnings as Errors Enabled")
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        target_compile_options(${PROJECT_NAME} PRIVATE -Werror)
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
        target_compile_options(${PROJECT_NAME} PRIVATE /W3 /WX)
    endif ()
endif ()

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ../)

# Link third party libraries
target_link_libraries(${PROJECT_NAME} PUBLIC
        vkb__core
        math
        framework
        volk
        glslang
        SPIRV
        vma
        hwcpipe
        spirv-cross-glsl
        glslang-default-resource-limits
        spdlog
        stb
        astc
        ktx
        )

if (${NEED_LINK_ATOMIC})
    target_link_libraries(${PROJECT_NAME} PUBLIC atomic)
endif ()

# Link platform specific libraries
if (ANDROID)
    # Import game-activity static lib inside the game-activity_static prefab module.
    find_package(game-activity REQUIRED CONFIG)
    target_link_libraries(${PROJECT_NAME} PUBLIC log android game-activity::game-activity_static)
else ()
    if (DIRECT_TO_DISPLAY)
        target_link_libraries(${PROJECT_NAME} PRIVATE dl)
    else ()
        target_link_libraries(${PROJECT_NAME} PRIVATE glfw)
    endif ()
endif ()

if (VKB_DO_CLANG_TIDY)
    set_target_properties(framework PROPERTIES CXX_CLANG_TIDY "${VKB_DO_CLANG_TIDY}")
endif ()