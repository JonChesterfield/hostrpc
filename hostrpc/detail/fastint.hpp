#ifndef FASTINT_HPP_INCLUDED
#define FASTINT_HPP_INCLUDED

#include <stdint.h>

namespace hostrpc
{
namespace fastint
{

template <uint64_t T>
struct bits
{
  enum : uint8_t
  {
    value = T <= UINT8_MAX    ? 8
            : T <= UINT16_MAX ? 16
            : T <= UINT32_MAX ? 32
                              : 64,
  };
};

template <uint8_t bits>
struct dispatch;
template <>
struct dispatch<8>
{
  using type = uint8_t;
};
template <>
struct dispatch<16>
{
  using type = uint16_t;
};
template <>
struct dispatch<32>
{
  using type = uint32_t;
};
template <>
struct dispatch<64>
{
  using type = uint64_t;
};

template <uint64_t V>
struct sufficientType
{
  using type = typename dispatch<bits<V>::value>::type;
};

}  // namespace fastint
}  // namespace hostrpc

#endif
