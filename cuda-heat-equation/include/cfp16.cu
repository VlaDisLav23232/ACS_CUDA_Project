#include <cstdint>
#include <cuda_runtime.h>

using cfp16_t = uint16_t;

__host__ __device__ inline uint32_t float_as_uint(float x) {
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = x;
    return bits.u;
}

__host__ __device__ inline float uint_as_float(uint32_t x) {
    union {
        uint32_t u;
        float f;
    } bits;
    bits.u = x;
    return bits.f;
}

__host__ __device__ inline float cfp16_to_float(cfp16_t x) {
    const uint32_t e = (static_cast<uint32_t>(x) & 0x7800u) >> 11;
    const uint32_t m = (static_cast<uint32_t>(x) & 0x07FFu) << 12;
    const uint32_t v = float_as_uint(static_cast<float>(m)) >> 23;
    return uint_as_float(
        (static_cast<uint32_t>(x & 0x8000u)) << 16
        | (e != 0u) * (((e + 112u) << 23) | m)
        | ((e == 0u) & (m != 0u)) * (((v - 37u) << 23) | ((m << (150u - v)) & 0x007FF000u))
    );
}

__host__ __device__ inline cfp16_t float_to_cfp16(float x) {
    const uint32_t b = float_as_uint(x) + 0x00000800u;
    const uint32_t e = (b & 0x7F800000u) >> 23;
    const uint32_t m = b & 0x007FFFFFu;
    return static_cast<cfp16_t>(
        ((b & 0x80000000u) >> 16)
        | (e > 112u) * ((((e - 112u) << 11) & 0x7800u) | (m >> 12))
        | ((e < 113u) & (e > 100u)) * (((((0x7FF800u + m) >> (124u - e)) + 1u) >> 1))
    );
}