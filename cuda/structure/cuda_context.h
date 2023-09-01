//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "math/math_core.h"

// this is the core runtime API exposed on the DLL level
extern "C" {
WP_API int init();
//WP_API void shutdown();

// whether Warp was compiled with CUDA support
WP_API int is_cuda_enabled();
// whether Warp was compiled with enhanced CUDA compatibility
WP_API int is_cuda_compatibility_enabled();
// whether Warp was compiled with CUTLASS support
WP_API int is_cutlass_enabled();
// whether Warp was compiled with debug support
WP_API int is_debug_enabled();

WP_API uint16_t float_to_half_bits(float x);
WP_API float half_bits_to_float(uint16_t u);

WP_API void *alloc_host(size_t s);
WP_API void *alloc_pinned(size_t s);
WP_API void *alloc_device(void *context, size_t s);

WP_API void free_host(void *ptr);
WP_API void free_pinned(void *ptr);
WP_API void free_device(void *context, void *ptr);

// all memcpys are performed asynchronously
WP_API void memcpy_h2h(void *dest, void *src, size_t n);
WP_API void memcpy_h2d(void *context, void *dest, void *src, size_t n);
WP_API void memcpy_d2h(void *context, void *dest, void *src, size_t n);
WP_API void memcpy_d2d(void *context, void *dest, void *src, size_t n);
WP_API void memcpy_peer(void *context, void *dest, void *src, size_t n);

// all memsets are performed asynchronously
WP_API void memset_host(void *dest, int value, size_t n);
WP_API void memset_device(void *context, void *dest, int value, size_t n);

// takes srcsize bytes starting at src and repeats them n times at dst (writes srcsize * n bytes in total):
WP_API void memtile_host(void *dest, const void *src, size_t srcsize, size_t n);
WP_API void memtile_device(void *context, void *dest, const void *src, size_t srcsize, size_t n);

WP_API int cuda_driver_version(); // CUDA driver version
WP_API int cuda_toolkit_version();// CUDA Toolkit version used to build Warp

WP_API int nvrtc_supported_arch_count();
WP_API void nvrtc_supported_archs(int *archs);

WP_API int cuda_device_get_count();
WP_API void *cuda_device_primary_context_retain(int ordinal);
WP_API void cuda_device_primary_context_release(int ordinal);
WP_API const char *cuda_device_get_name(int ordinal);
WP_API int cuda_device_get_arch(int ordinal);
WP_API int cuda_device_is_uva(int ordinal);

WP_API void *cuda_context_get_current();
WP_API void cuda_context_set_current(void *context);
WP_API void cuda_context_push_current(void *context);
WP_API void cuda_context_pop_current();
WP_API void *cuda_context_create(int device_ordinal);
WP_API void cuda_context_destroy(void *context);
WP_API int cuda_context_get_device_ordinal(void *context);
WP_API int cuda_context_is_primary(void *context);
WP_API void *cuda_context_get_stream(void *context);
WP_API void cuda_context_set_stream(void *context, void *stream);
WP_API int cuda_context_can_access_peer(void *context, void *peer_context);
WP_API int cuda_context_enable_peer_access(void *context, void *peer_context);

// ensures all device side operations have completed in the current context
WP_API void cuda_context_synchronize(void *context);

// return cudaError_t code
WP_API uint64_t cuda_context_check(void *context);

WP_API void *cuda_stream_create(void *context);
WP_API void cuda_stream_destroy(void *context, void *stream);
WP_API void cuda_stream_synchronize(void *context, void *stream);
WP_API void *cuda_stream_get_current();
WP_API void cuda_stream_wait_event(void *context, void *stream, void *event);
WP_API void cuda_stream_wait_stream(void *context, void *stream, void *other_stream, void *event);

WP_API void *cuda_event_create(void *context, unsigned flags);
WP_API void cuda_event_destroy(void *context, void *event);
WP_API void cuda_event_record(void *context, void *event, void *stream);

WP_API void cuda_graph_begin_capture(void *context);
WP_API void *cuda_graph_end_capture(void *context);
WP_API void cuda_graph_launch(void *context, void *graph);
WP_API void cuda_graph_destroy(void *context, void *graph);

WP_API size_t cuda_compile_program(const char *cuda_src, int arch, const char *include_dir, bool debug, bool verbose, bool verify_fp, bool fast_math, const char *output_file);

WP_API void *cuda_load_module(void *context, const char *ptx);
WP_API void cuda_unload_module(void *context, void *module);
WP_API void *cuda_get_kernel(void *context, void *module, const char *name);
WP_API size_t cuda_launch_kernel(void *context, void *kernel, size_t dim, void **args);

WP_API void cuda_set_context_restore_policy(bool always_restore);
WP_API int cuda_get_context_restore_policy();

WP_API void cuda_graphics_map(void *context, void *resource);
WP_API void cuda_graphics_unmap(void *context, void *resource);
WP_API void cuda_graphics_device_ptr_and_size(void *context, void *resource, uint64_t *ptr, size_t *size);
WP_API void *cuda_graphics_register_gl_buffer(void *context, uint32_t gl_buffer, unsigned int flags);
WP_API void cuda_graphics_unregister_resource(void *context, void *resource);

}// extern "C"
