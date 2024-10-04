%%cuda_group_save -g "source" -n "gpu_solution.cu"
/**
 * The file in which you will implement your GPU solutions!
 */

#include "algorithm_choices.h"

#include <chrono>    // for timing
#include <iostream>  // std::cout, std::endl
#include <limits> // for +infinity value
#include <sstream>

#include "cuda_common.h"

    namespace csc485b {
    namespace a1 {
        namespace gpu {

            /**
             * The CPU baseline benefits from warm caches because the data was generated on
             * the CPU. Run the data through the GPU once with some arbitrary logic to
             * ensure that the GPU cache is warm too and the comparison is more fair.
             */
            __global__
                void warm_the_gpu(element_t* data, std::size_t invert_at_pos, std::size_t num_elements)
            {
                int const th_id = blockIdx.x * blockDim.x + threadIdx.x;

                // We know this will never be true, because of the data generator logic,
                // but I doubt that the compiler will figure it out. Thus every element
                // should be read, but none of them should be modified.
                if (th_id < num_elements && data[th_id] > num_elements * 100)
                {
                    ++data[th_id]; // should not be possible.
                }
            }

            // Used this StackOverflow post for algorithm to find next power of 2.
            // https://stackoverflow.com/questions/364985/algorithm-for-finding-the-smallest-power-of-two-thats-greater-or-equal-to-a-giv
            // Commented out since we do not have to worry about input not being power of two.
            //std::size_t next_power_of_two(element_t x) {
            //    --x;
            //    x |= x >> 1;
            //    x |= x >> 2;
            //    x |= x >> 4;
            //    x |= x >> 8;
            //    x |= x >> 16;
            //    return x + 1;
            //}

            /**
             * Your solution. Should match the CPU output.
             */
            __global__
                void opposing_sort(element_t* data, std::size_t invert_at_pos, std::size_t num_elements)
            {
                element_t const th_id = blockIdx.x * blockDim.x + threadIdx.x;
                extern __shared__ element_t s_data[];

                if (th_id < num_elements)
                {
                    // Load the data from global memory to shared memory
                    // s_data[threadIdx.x] = data[th_id];

                    // Perform Bitonic Sort but DIFFERENT
                    for (std::size_t stage = 2; stage <= num_elements; stage <<= 1) {
                        for (std::size_t step = stage >> 1; step > 0; step >>= 1) {
                            // Determine the index of the element to compare with
                            std::size_t partner = th_id ^ step;

                            if (partner > th_id && partner < num_elements) {
                                // Ascending if th_id and partner are in the same stage group (because stage shifts by 1 bit each time, this works nicely)
                                bool ascending = (th_id & stage) == 0;

                                // Compare and swap based on direction
                                if (ascending) {
                                    if (data[th_id] > data[partner]) {
                                        element_t temp = data[th_id];
                                        data[th_id] = data[partner];
                                        data[partner] = temp;
                                    }
                                }
                                else {
                                    // Descending order
                                    if (data[th_id] < data[partner]) {
                                        element_t temp = data[th_id];
                                        data[th_id] = data[partner];
                                        data[partner] = temp;
                                    }
                                }
                            }
                            __syncthreads();
                        }
                    }

                    // Copy the sorted data back to global memory for this thread
                    // data[th_id] = s_data[threadIdx.x];

                    // Reverses array at invert position onwards.
                    if (th_id >= invert_at_pos) {
                        s_data[num_elements - 1 - th_id] = data[th_id];
                        __syncthreads();
                        int outOffset = blockDim.x * (blockIdx.x);
                        int out = outOffset + th_id;
                        data[out] = s_data[th_id - invert_at_pos];
                    }
                }
            }

            __global__
            void opposing_sort_step( element_t * data, std::size_t num_elements, int j, int k, bool direction )
            {
                int const th_id = blockIdx.x * blockDim.x + threadIdx.x;
                // Determine the index of the element to compare with
                int partner = th_id ^ j;

                if (partner > th_id && partner < num_elements) {
                    // Ascending if th_id and partner are in the same stage group (because stage shifts by 1 bit each time, this works nicely)
                    // If direction is false, the boolean value for ascending is flipped.
                    bool ascending = ((th_id & k) == 0) ^ !direction;

                    // Compare and swap based on direction
                    if (ascending) {
                        if (data[th_id] > data[partner]) {
                            element_t temp = data[th_id];
                            data[th_id] = data[partner];
                            data[partner] = temp;
                        }
                    } else {
                        if (data[th_id] < data[partner]) {
                            element_t temp = data[th_id];
                            data[th_id] = data[partner];
                            data[partner] = temp;
                        }
                    }
                }
            }


            /**
             * Performs all the logic of allocating device vectors and copying host/input
             * vectors to the device. Times the opposing_sort() kernel with wall time,
             * but excludes set up and tear down costs such as mallocs, frees, and memcpies.
             */
            void run_gpu_soln(std::vector< element_t > data, std::size_t switch_at, std::size_t n)
            {

                // Check if input is not a power of 2, and then pad input to a power of 2.
                // Code has been commented out as test cases will only be powers of 2.
                // if (n & (n-1)) {
                //    std::size_t next_power_of_2 = next_power_of_two(n);
                //    for (std::size_t i = 0; i < (next_power_of_2 - n); ++i) {
                //        data.push_back(std::numeric_limits<element_t>::max()-1);
                //    }
                //    n = next_power_of_2;
                //}
                std::size_t const threads_per_block = 1024;
                std::size_t const num_blocks = (n + threads_per_block - 1) / threads_per_block;

                // Allocate arrays on the device/GPU
                element_t* d_data;
                cudaMalloc((void**)&d_data, sizeof(element_t) * n);
                CHECK_ERROR("Allocating input array on device");

                // Copy the input from the host to the device/GPU
                cudaMemcpy(d_data, data.data(), sizeof(element_t) * n, cudaMemcpyHostToDevice);
                CHECK_ERROR("Copying input array to device");

                // Warm the cache on the GPU for a more fair comparison
                warm_the_gpu << < num_blocks, threads_per_block >> > (d_data, switch_at, n);

                // Time the execution of the kernel that you implemented
                auto const kernel_start = std::chrono::high_resolution_clock::now();
                auto const smem_size = threads_per_block * sizeof(element_t);

                // Run with a single thread block if input size allows
                if (n <= 1024) {
                    opposing_sort << < num_blocks, threads_per_block, smem_size >> > (d_data, switch_at, n);
                }
                else {
                    // Otherwise do inter-block synchronization in CPU code
                    int j, k;
                    // Major step
                    for (k = 2; k <= n; k <<= 1) {
                        // Minor step
                        for(j = k >> 1; j > 0; j >>= 1) {
                        opposing_sort_step<<< num_blocks, threads_per_block>>>( d_data, n, j, k, true);
                        }
                    }
                    // Re-sort last quarter in reverse order
                    for (k = 2; k <= n; k <<= 1) {
                        // Minor step
                        for(j = k >> 1; j > 0; j >>= 1) {
                        opposing_sort_step<<< num_blocks, threads_per_block>>>( d_data + switch_at, n - switch_at, j, k, false);
                        }
                    }
                }
                // Remove positive infinities from padded input if original input was not a power of two.
                // Commented out since input is guaranteed to be a power of two in the test cases.
                 //std::vector<element_t> result;
                 //for (std::size_t i = 0; i < n; ++i) {
                 //    if (data[i] != std::numeric_limits<element_t>::max() - 1) {
                 //        result.push_back(data[i]);
                 //    }
                 //}

                auto const kernel_end = std::chrono::high_resolution_clock::now();
                //for (auto const x : result) std::cout << x << " "; std::cout << std::endl;
                CHECK_ERROR("Executing kernel on device");

                // After the timer ends, copy the result back, free the device vector,
                // and echo out the timings and the results.
                cudaMemcpy(data.data(), d_data, sizeof(element_t) * n, cudaMemcpyDeviceToHost);
                CHECK_ERROR("Transferring result back to host");
                cudaFree(d_data);
                CHECK_ERROR("Freeing device memory");

                std::cout << "GPU Solution time: "
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(kernel_end - kernel_start).count()
                    << " ns" << std::endl;

                //for (auto const x : data) std::cout << x << " "; std::cout << std::endl;
                std::ostringstream buffer;

                // Simulate writing many strings to the buffer
                for (std::size_t i = 0; i < n; ++i) {
                    buffer << data[i] << " ";
                }
                buffer << '\n';

                // Print the entire buffer content to console with a single std::cout call
                std::cout << buffer.str();
            }

        } // namespace gpu
    } // namespace a1
} // namespace csc485b