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

project(framework LANGUAGES C CXX)

# Platform settings
add_subdirectory(platform/core)

if(ANDROID)
    add_subdirectory(platform/android)
elseif(WIN32)
    add_subdirectory(platform/windows)
elseif(APPLE OR UNIX)
    add_subdirectory(platform/unix)
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

set(COMMON_FILES
        # Header Files
        common/error.h
        common/helpers.h
        common/logging.h
        common/strings.h
        common/tags.h
        common/timer.h
        common/utils.h
        common/vk_common.h
        common/vk_initializers.h
        common/filesystem.h
        # Source Files
        common/error.cpp
        common/timer.cpp
        common/utils.cpp
        common/vk_common.cpp
        common/filesystem.cpp
        )

set(SHADER_FILES
        # Header Files
        shader/glsl_compiler.h
        shader/spirv_reflection.h
        shader/shader_module.h
        shader/shader_data.h
        shader/shader_variant.h
        # Source Files
        shader/glsl_compiler.cpp
        shader/spirv_reflection.cpp
        shader/shader_module.cpp
        shader/shader_data.cpp
        shader/shader_variant.cpp
        )

set(COMPUTE_FILES
        compute/compute_context.h
        compute/compute_context.cpp)

set(RENDERING_FILES
        # Header files
        rendering/postprocessing_pipeline.h
        rendering/postprocessing_pass.h
        rendering/postprocessing_renderpass.h
        rendering/postprocessing_computepass.h
        rendering/render_context.h
        rendering/render_frame.h
        rendering/render_pipeline.h
        rendering/render_target.h
        rendering/subpass.h
        rendering/framebuffer.h
        rendering/swapchain.h
        # Source files
        rendering/postprocessing_pipeline.cpp
        rendering/postprocessing_pass.cpp
        rendering/postprocessing_renderpass.cpp
        rendering/postprocessing_computepass.cpp
        rendering/render_context.cpp
        rendering/render_frame.cpp
        rendering/render_pipeline.cpp
        rendering/render_target.cpp
        rendering/subpass.cpp
        rendering/framebuffer.cpp
        rendering/swapchain.cpp
        )

set(STATS_FILES
        # Header Files
        stats/stats.h
        stats/stats_common.h
        stats/stats_provider.h
        stats/frame_time_stats_provider.h
        stats/hwcpipe_stats_provider.h
        stats/vulkan_stats_provider.h

        # Source Files
        stats/stats.cpp
        stats/stats_provider.cpp
        stats/frame_time_stats_provider.cpp
        stats/hwcpipe_stats_provider.cpp
        stats/vulkan_stats_provider.cpp)

set(CORE_FILES
        # Header Files
        core/instance.h
        core/physical_device.h
        core/device.h
        core/debug.h
        core/pipeline_layout.h
        core/pipeline.h
        core/descriptor_set_layout.h
        core/descriptor_pool.h
        core/descriptor_set.h
        core/queue.h
        core/command_pool.h
        core/command_buffer.h
        core/buffer.h
        core/buffer_pool.h
        core/image.h
        core/image_view.h
        core/instance.h
        core/sampler.h
        core/render_pass.h
        core/query_pool.h
        core/scratch_buffer.h
        core/acceleration_structure.h
        core/shader_binding_table.h
        core/vulkan_resource.h
        core/resource_binding_state.h
        core/pipeline_state.h
        core/fence_pool.h
        core/resource_record.h
        core/resource_replay.h
        core/semaphore_pool.h
        core/resource_cache.h
        core/frame_resource.h
        core/sampled_image.h
        # Source Files
        core/instance.cpp
        core/physical_device.cpp
        core/device.cpp
        core/debug.cpp
        core/image.cpp
        core/pipeline_layout.cpp
        core/pipeline.cpp
        core/descriptor_set_layout.cpp
        core/descriptor_pool.cpp
        core/descriptor_set.cpp
        core/queue.cpp
        core/command_pool.cpp
        core/command_buffer.cpp
        core/buffer.cpp
        core/buffer_pool.cpp
        core/image.cpp
        core/image_view.cpp
        core/instance.cpp
        core/sampler.cpp
        core/render_pass.cpp
        core/query_pool.cpp
        core/scratch_buffer.cpp
        core/acceleration_structure.cpp
        core/shader_binding_table.cpp
        core/vulkan_resource.cpp
        core/resource_binding_state.cpp
        core/pipeline_state.cpp
        core/fence_pool.cpp
        core/resource_record.cpp
        core/resource_replay.cpp
        core/semaphore_pool.cpp
        core/resource_cache.cpp
        core/frame_resource.cpp
        core/sampled_image.cpp
        )

set(PLATFORM_FILES
        # Header Files
        platform/platform.h
        platform/window.h
        platform/input_events.h
        platform/headless_window.h

        # Source Files
        platform/headless_window.cpp
        platform/platform.cpp
        platform/window.cpp
        platform/input_events.cpp
        )

set(GLFW_FILES
        # Header Files
        platform/glfw_window.h
        # Source Files
        platform/glfw_window.cpp
        )

set(PARSER_FILES
        # Header Files
        platform/parsers/CLI11.h
        platform/parsers/help_formatter.h
        # Source Files
        platform/parsers/CLI11.cpp
        platform/parsers/help_formatter.cpp
        )

set(ANDROID_FILES
        # Header Files
        platform/android/android_platform.h
        platform/android/android_window.h
        # Source Files
        platform/android/android_platform.cpp
        platform/android/android_window.cpp)

set(WINDOWS_FILES
        # Header Files
        platform/windows/windows_platform.h
        # Source Files
        platform/windows/windows_platform.cpp)

set(UNIX_FILES
        # Header Files
        platform/unix/unix_platform.h
        # Source Files
        platform/unix/unix_platform.cpp)

set(LINUX_D2D_FILES
        # Header Files
        platform/unix/unix_d2d_platform.h
        platform/unix/direct_window.h
        # Source Files
        platform/unix/unix_d2d_platform.cpp
        platform/unix/direct_window.cpp)

source_group("\\" FILES ${FRAMEWORK_FILES})
source_group("common\\" FILES ${COMMON_FILES})
source_group("platform\\" FILES ${PLATFORM_FILES})
source_group("platform\\" FILES ${GLFW_FILES})
source_group("platform\\parsers" FILES ${PARSER_FILES})
source_group("platform\\windows" FILES ${WINDOWS_FILES})
source_group("platform\\android" FILES ${ANDROID_FILES})
source_group("platform\\unix" FILES ${UNIX_FILES})
source_group("platform\\unix" FILES ${LINUX_D2D_FILES})
source_group("core\\" FILES ${CORE_FILES})
source_group("rendering\\" FILES ${RENDERING_FILES})
source_group("stats\\" FILES ${STATS_FILES})

set(PROJECT_FILES
        ${COMMON_FILES}
        ${PLATFORM_FILES}
        ${CORE_FILES}
        ${RENDERING_FILES}
        ${COMPUTE_FILES}
        ${STATS_FILES}
        ${SHADER_FILES}
        )

# Add files based on platform
if (ANDROID)
    list(APPEND PROJECT_FILES ${ANDROID_FILES})
    message(STATUS "Android platform detected")
else ()
    if (DIRECT_TO_DISPLAY)
        list(APPEND PROJECT_FILES ${LINUX_D2D_FILES})
        message(STATUS "Unix d2d platform detected")
    else ()
        list(APPEND PROJECT_FILES ${GLFW_FILES})
        if (WIN32)
            list(APPEND PROJECT_FILES ${WINDOWS_FILES})
            message(STATUS "Windows platform detected")
        else ()
            list(APPEND PROJECT_FILES ${UNIX_FILES})
            message(STATUS "Unix platform detected")
        endif ()
    endif ()
endif ()

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

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link third party libraries
target_link_libraries(${PROJECT_NAME} PUBLIC
        vkb__core
        volk
        glslang
        SPIRV
        vma
        hwcpipe
        spirv-cross-glsl
        glslang-default-resource-limits
        spdlog
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