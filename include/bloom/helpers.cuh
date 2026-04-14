#pragma once

#include <cuda_runtime.h>

#include <cuda/std/bit>
#include <cuda/std/concepts>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace bloom {

/**
 * @brief Exception thrown on CUDA runtime errors.
 */
class CudaError : public std::runtime_error {
   public:
    CudaError(cudaError_t code, const char* file, int line)
        : std::runtime_error(
              std::string(file) + ":" + std::to_string(line) + " " + cudaGetErrorString(code)
          ),
          code_(code) {
    }

    [[nodiscard]] cudaError_t code() const noexcept {
        return code_;
    }

   private:
    cudaError_t code_;
};

}  // namespace bloom

namespace bloom::detail {

/**
 * @brief Integer ceiling division.
 */
template <cuda::std::integral Integer>
constexpr auto divUp(Integer x, Integer y) {
    return (x + y - 1) / y;
}

#if __CUDA_ARCH__ >= 1000

/**
 * @brief Loads 256 bits from global memory using the non-coherent cache path.
 *
 * This function uses inline PTX for 256-bit vectorized loads.
 * For uint64_t: loads 4 values (v4.u64)
 * For uint32_t: loads 8 values (v8.u32)
 *
 * @note Only available on sm_100+ architectures with PTX 8.8.
 *       Use __CUDA_ARCH__ >= 1000 guard at call sites.
 *
 * @tparam T Element type (uint32_t or uint64_t)
 * @param ptr Source pointer (must be 32-byte aligned)
 * @param out Output array (4 elements for uint64_t, 8 for uint32_t)
 */
template <typename T>
__device__ __forceinline__ void load256BitGlobalNC(const T* ptr, T* out) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8, "T must be uint32_t or uint64_t");

    if constexpr (sizeof(T) == 8) {
        asm volatile("ld.global.nc.v4.u64 {%0, %1, %2, %3}, [%4];"
                     : "=l"(out[0]), "=l"(out[1]), "=l"(out[2]), "=l"(out[3])
                     : "l"(ptr));
    } else {
        asm volatile("ld.global.nc.v8.u32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                     : "=r"(out[0]),
                       "=r"(out[1]),
                       "=r"(out[2]),
                       "=r"(out[3]),
                       "=r"(out[4]),
                       "=r"(out[5]),
                       "=r"(out[6]),
                       "=r"(out[7])
                     : "l"(ptr));
    }
}

__device__ __forceinline__ void load256BitGlobalNC(
    const uint64_t* ptr,
    uint64_t& out0,
    uint64_t& out1,
    uint64_t& out2,
    uint64_t& out3
) {
    asm volatile("ld.global.nc.v4.u64 {%0, %1, %2, %3}, [%4];"
                 : "=l"(out0), "=l"(out1), "=l"(out2), "=l"(out3)
                 : "l"(ptr));
}

#endif

/**
 * @brief Macro for checking CUDA errors.
 * Throws bloom::CudaError on failure.
 */
#define CUDA_CALL(err)                                    \
    do {                                                  \
        cudaError_t err_ = (err);                         \
        if (err_ == cudaSuccess) [[likely]] {             \
            break;                                        \
        }                                                 \
        throw bloom::CudaError(err_, __FILE__, __LINE__); \
    } while (0)

/**
 * @brief Calculates the maximum occupancy grid size for a kernel.
 *
 * @tparam Kernel Type of the kernel function.
 * @param blockSize Block size (threads per block).
 * @param kernel The kernel function.
 * @param dynamicSMemSize Dynamic shared memory size per block.
 * @return uint64_t The calculated grid size (number of blocks).
 */
template <typename Kernel>
uint64_t maxOccupancyGridSize(int32_t blockSize, Kernel kernel, uint64_t dynamicSMemSize) {
    int device = 0;
    cudaGetDevice(&device);

    int numSM = -1;
    cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, device);

    int maxActiveBlocksPerSM{};
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocksPerSM, kernel, blockSize, dynamicSMemSize
    );

    return maxActiveBlocksPerSM * numSM;
}

}  // namespace bloom::detail
