#pragma once

#include <cuda/std/span>

#include <cstddef>

namespace bloom {

/**
 * @brief A span that is assumed to point to device-accessible memory.
 *
 * Layout-identical to cuda::std::span<T> so it can be passed to kernels
 * by value.
 *
 * The distinct type prevents accidentally mixing host and device
 * pointers at compile time.
 */
template <typename T>
struct device_span : cuda::std::span<T> {
    using cuda::std::span<T>::span;

    /// Implicit widening from device_span<U> where U* converts to T*
    /// (e.g. device_span<int> -> device_span<const int>).
    template <typename U>
        requires std::is_convertible_v<U (*)[], T (*)[]>
    constexpr explicit device_span(device_span<U> other) noexcept
        : cuda::std::span<T>(other.data(), other.size()) {
    }
};

}  // namespace bloom
