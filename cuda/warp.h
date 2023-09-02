////  Copyright (c) 2023 Feng Yang
////
////  I am making my contributions/submissions to this project solely in my
////  personal capacity and am not conveying any rights to any intellectual
////  property of any third parties.
//
//#pragma once
//
//// defines all crt + builtin types
//#include "builtin.h"
//
//// this is the core runtime API exposed on the DLL level
//extern "C" {
// uint64_t hash_grid_create_host(int dim_x, int dim_y, int dim_z);
// void hash_grid_reserve_host(uint64_t id, int num_points);
// void hash_grid_destroy_host(uint64_t id);
// void hash_grid_update_host(uint64_t id, float cell_width, const wp::vec3 *positions, int num_points);
//
// uint64_t hash_grid_create_device(void *context, int dim_x, int dim_y, int dim_z);
// void hash_grid_reserve_device(uint64_t id, int num_points);
// void hash_grid_destroy_device(uint64_t id);
// void hash_grid_update_device(uint64_t id, float cell_width, const wp::vec3 *positions, int num_points);
//
// bool cutlass_gemm(int compute_capability, int m, int n, int k, const char *datatype,
//                         const void *a, const void *b, const void *c, void *d, float alpha, float beta,
//                         bool row_major_a, bool row_major_b, bool allow_tf32x3_arith, int batch_count);
//
// uint64_t volume_create_host(void *buf, uint64_t size);
// void volume_get_buffer_info_host(uint64_t id, void **buf, uint64_t *size);
// void volume_get_tiles_host(uint64_t id, void **buf, uint64_t *size);
// void volume_destroy_host(uint64_t id);
//
// uint64_t volume_create_device(void *context, void *buf, uint64_t size);
// uint64_t volume_f_from_tiles_device(void *context, void *points, int num_points, float voxel_size, float bg_value, float tx, float ty, float tz, bool points_in_world_space);
// uint64_t volume_v_from_tiles_device(void *context, void *points, int num_points, float voxel_size, float bg_value_x, float bg_value_y, float bg_value_z, float tx, float ty, float tz, bool points_in_world_space);
// uint64_t volume_i_from_tiles_device(void *context, void *points, int num_points, float voxel_size, int bg_value, float tx, float ty, float tz, bool points_in_world_space);
// void volume_get_buffer_info_device(uint64_t id, void **buf, uint64_t *size);
// void volume_get_tiles_device(uint64_t id, void **buf, uint64_t *size);
// void volume_destroy_device(uint64_t id);
//
// void volume_get_voxel_size(uint64_t id, float *dx, float *dy, float *dz);
//
// uint64_t marching_cubes_create_device(void *context);
// void marching_cubes_destroy_device(uint64_t id);
// int marching_cubes_surface_device(uint64_t id, const float *field, int nx, int ny, int nz, float threshold, wp::vec3 *verts, int *triangles, int max_verts, int max_tris, int *out_num_verts, int *out_num_tris);
//
//// generic copy supporting non-contiguous arrays
// size_t array_copy_host(void *dst, void *src, int dst_type, int src_type, int elem_size);
// size_t array_copy_device(void *context, void *dst, void *src, int dst_type, int src_type, int elem_size);
//
//// generic fill for non-contiguous arrays
// void array_fill_host(void *arr, int arr_type, const void *value, int value_size);
// void array_fill_device(void *context, void *arr, int arr_type, const void *value, int value_size);
//
// void array_inner_float_host(uint64_t a, uint64_t b, uint64_t out, int count, int stride_a, int stride_b, int type_len);
// void array_inner_double_host(uint64_t a, uint64_t b, uint64_t out, int count, int stride_a, int stride_b, int type_len);
// void array_inner_float_device(uint64_t a, uint64_t b, uint64_t out, int count, int stride_a, int stride_b, int type_len);
// void array_inner_double_device(uint64_t a, uint64_t b, uint64_t out, int count, int stride_a, int stride_b, int type_len);
//
// void array_sum_float_device(uint64_t a, uint64_t out, int count, int stride, int type_len);
// void array_sum_float_host(uint64_t a, uint64_t out, int count, int stride, int type_len);
// void array_sum_double_host(uint64_t a, uint64_t out, int count, int stride, int type_len);
// void array_sum_double_device(uint64_t a, uint64_t out, int count, int stride, int type_len);
//
// void array_scan_int_host(uint64_t in, uint64_t out, int len, bool inclusive);
// void array_scan_float_host(uint64_t in, uint64_t out, int len, bool inclusive);
//
// void array_scan_int_device(uint64_t in, uint64_t out, int len, bool inclusive);
// void array_scan_float_device(uint64_t in, uint64_t out, int len, bool inclusive);
//
// void radix_sort_pairs_int_host(uint64_t keys, uint64_t values, int n);
// void radix_sort_pairs_int_device(uint64_t keys, uint64_t values, int n);
//
// void runlength_encode_int_host(uint64_t values, uint64_t run_values, uint64_t run_lengths, uint64_t run_count, int n);
// void runlength_encode_int_device(uint64_t values, uint64_t run_values, uint64_t run_lengths, uint64_t run_count, int n);
//
// int bsr_matrix_from_triplets_float_host(
//    int rows_per_block,
//    int cols_per_block,
//    int row_count,
//    int nnz,
//    uint64_t tpl_rows,
//    uint64_t tpl_columns,
//    uint64_t tpl_values,
//    uint64_t bsr_offsets,
//    uint64_t bsr_columns,
//    uint64_t bsr_values);
// int bsr_matrix_from_triplets_double_host(
//    int rows_per_block,
//    int cols_per_block,
//    int row_count,
//    int nnz,
//    uint64_t tpl_rows,
//    uint64_t tpl_columns,
//    uint64_t tpl_values,
//    uint64_t bsr_offsets,
//    uint64_t bsr_columns,
//    uint64_t bsr_values);
//
// int bsr_matrix_from_triplets_float_device(
//    int rows_per_block,
//    int cols_per_block,
//    int row_count,
//    int nnz,
//    uint64_t tpl_rows,
//    uint64_t tpl_columns,
//    uint64_t tpl_values,
//    uint64_t bsr_offsets,
//    uint64_t bsr_columns,
//    uint64_t bsr_values);
// int bsr_matrix_from_triplets_double_device(
//    int rows_per_block,
//    int cols_per_block,
//    int row_count,
//    int nnz,
//    uint64_t tpl_rows,
//    uint64_t tpl_columns,
//    uint64_t tpl_values,
//    uint64_t bsr_offsets,
//    uint64_t bsr_columns,
//    uint64_t bsr_values);
//
// void bsr_transpose_float_host(int rows_per_block, int cols_per_block,
//                                     int row_count, int col_count, int nnz,
//                                     uint64_t bsr_offsets, uint64_t bsr_columns,
//                                     uint64_t bsr_values,
//                                     uint64_t transposed_bsr_offsets,
//                                     uint64_t transposed_bsr_columns,
//                                     uint64_t transposed_bsr_values);
// void bsr_transpose_double_host(int rows_per_block, int cols_per_block,
//                                      int row_count, int col_count, int nnz,
//                                      uint64_t bsr_offsets, uint64_t bsr_columns,
//                                      uint64_t bsr_values,
//                                      uint64_t transposed_bsr_offsets,
//                                      uint64_t transposed_bsr_columns,
//                                      uint64_t transposed_bsr_values);
//
// void bsr_transpose_float_device(int rows_per_block, int cols_per_block,
//                                       int row_count, int col_count, int nnz,
//                                       uint64_t bsr_offsets, uint64_t bsr_columns,
//                                       uint64_t bsr_values,
//                                       uint64_t transposed_bsr_offsets,
//                                       uint64_t transposed_bsr_columns,
//                                       uint64_t transposed_bsr_values);
// void bsr_transpose_double_device(int rows_per_block, int cols_per_block,
//                                        int row_count, int col_count, int nnz,
//                                        uint64_t bsr_offsets, uint64_t bsr_columns,
//                                        uint64_t bsr_values,
//                                        uint64_t transposed_bsr_offsets,
//                                        uint64_t transposed_bsr_columns,
//                                        uint64_t transposed_bsr_values);
//}
