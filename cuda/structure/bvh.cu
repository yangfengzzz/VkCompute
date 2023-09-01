//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "cuda_context.h"
#include "cuda_util.h"
#include "bvh.h"

#include <vector>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace wp {

__global__ void bvh_refit_kernel(int n, const int *__restrict__ parents, int *__restrict__ child_count, BVHPackedNodeHalf *__restrict__ lowers, BVHPackedNodeHalf *__restrict__ uppers, const bounds3 *bounds) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < n) {
        bool leaf = lowers[index].b;

        if (leaf) {
            // update the leaf node
            const int leaf_index = lowers[index].i;
            const bounds3 &b = bounds[leaf_index];

            make_node(lowers + index, b.lower, leaf_index, true);
            make_node(uppers + index, b.upper, 0, false);
        } else {
            // only keep leaf threads
            return;
        }

        // update hierarchy
        for (;;) {
            int parent = parents[index];

            // reached root
            if (parent == -1)
                return;

            // ensure all writes are visible
            __threadfence();

            int finished = atomicAdd(&child_count[parent], 1);

            // if we have are the last thread (such that the parent node is now complete)
            // then update its bounds and move onto the the next parent in the hierarchy
            if (finished == 1) {
                const int left_child = lowers[parent].i;
                const int right_child = uppers[parent].i;

                vec3 left_lower = vec3(lowers[left_child].x,
                                       lowers[left_child].y,
                                       lowers[left_child].z);

                vec3 left_upper = vec3(uppers[left_child].x,
                                       uppers[left_child].y,
                                       uppers[left_child].z);

                vec3 right_lower = vec3(lowers[right_child].x,
                                        lowers[right_child].y,
                                        lowers[right_child].z);

                vec3 right_upper = vec3(uppers[right_child].x,
                                        uppers[right_child].y,
                                        uppers[right_child].z);

                // union of child bounds
                vec3 lower = min(left_lower, right_lower);
                vec3 upper = max(left_upper, right_upper);

                // write new BVH nodes
                make_node(lowers + parent, lower, left_child, false);
                make_node(uppers + parent, upper, right_child, false);

                // move onto processing the parent
                index = parent;
            } else {
                // parent not ready (we are the first child), terminate thread
                break;
            }
        }
    }
}

void bvh_refit_device(BVH &bvh, const bounds3 *b) {
    ContextGuard guard(bvh.context);

    // clear child counters
    memset_device(WP_CURRENT_CONTEXT, bvh.node_counts, 0, sizeof(int) * bvh.max_nodes);

    wp_launch_device(WP_CURRENT_CONTEXT, bvh_refit_kernel, bvh.max_nodes, (bvh.max_nodes, bvh.node_parents, bvh.node_counts, bvh.node_lowers, bvh.node_uppers, b));
}

__global__ void set_bounds_from_lowers_and_uppers(int n, bounds3 *b, const vec3 *lowers, const vec3 *uppers) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        b[tid] = bounds3(lowers[tid], uppers[tid]);
    }
}

}// namespace wp

// refit to data stored in the bvh

void bvh_refit_device(uint64_t id) {
    wp::BVH bvh;
    if (bvh_get_descriptor(id, bvh)) {
        ContextGuard guard(bvh.context);
        wp_launch_device(WP_CURRENT_CONTEXT, wp::set_bounds_from_lowers_and_uppers, bvh.num_bounds, (bvh.num_bounds, bvh.bounds, bvh.lowers, bvh.uppers));

        bvh_refit_device(bvh, bvh.bounds);
    }
}
