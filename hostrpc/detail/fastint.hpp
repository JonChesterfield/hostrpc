#ifndef FASTINT_HPP_INCLUDED
#define FASTINT_HPP_INCLUDED

#include "../platform/detect.hpp"
#include <stdint.h>

namespace hostrpc
{
template <uint64_t lhs, uint64_t rhs>
HOSTRPC_ANNOTATE constexpr bool static_equal()
{
  static_assert(lhs == rhs, "");
  return lhs == rhs;
}

template <uint64_t lhs, uint64_t rhs>
HOSTRPC_ANNOTATE constexpr bool static_lessthan_equal()
{
  static_assert(lhs <= rhs, "");
  return lhs <= rhs;
}

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

template <typename T>
struct fastint_runtime
{
  using type = T;
  using selfType = fastint_runtime<type>;

 private:
  T SZ;

  template <uint64_t Y>
  HOSTRPC_ANNOTATE constexpr static typename fastint::sufficientType<Y>::type
  retype()
  {
    // Reduce uint64_t to the smallest type that can hold it
    // static error if the value would not fit in the runtime instance
    static_assert(
        static_lessthan_equal<sizeof(typename fastint::sufficientType<Y>::type),
                              sizeof(type)>(),
        "TODO");
    return Y;
  }

 public:
  HOSTRPC_ANNOTATE constexpr fastint_runtime(type N) : SZ(N) {}

  HOSTRPC_ANNOTATE constexpr fastint_runtime() : SZ(0) {}

  HOSTRPC_ANNOTATE constexpr type value() const { return SZ; }
  HOSTRPC_ANNOTATE constexpr operator type() const { return value(); }

  HOSTRPC_ANNOTATE constexpr selfType popcount() const
  {
    return __builtin_popcountl(value());
  }

  HOSTRPC_ANNOTATE constexpr selfType findFirstSet() const
  {
    return __builtin_ffsl(value());
  }

  // implicit type narrowing hazards here, for now assert on mismatch
  template <uint64_t Y>
  HOSTRPC_ANNOTATE constexpr selfType bitwiseOr()
  {
    constexpr auto n = retype<Y>();
    static_assert(static_lessthan_equal<sizeof(n), sizeof(type)>(), "");
    return value() | n;
  }

  template <uint64_t Y>
  HOSTRPC_ANNOTATE constexpr selfType bitwiseAnd()
  {
    constexpr auto n = retype<Y>();
    static_assert(static_lessthan_equal<sizeof(n), sizeof(type)>(), "");
    return value() & n;
  }
  template <uint64_t Y>
  HOSTRPC_ANNOTATE constexpr selfType subtract()
  {
    constexpr auto n = retype<Y>();
    static_assert(static_lessthan_equal<sizeof(n), sizeof(type)>(), "");
    return value() - n;
  }
};

template <uint64_t V>
struct fastint_compiletime
{
  using type = typename fastint::sufficientType<V>::type;

 private:
  HOSTRPC_ANNOTATE constexpr static fastint_runtime<type> rt()
  {
    return fastint_runtime<type>(value());
  }

  template <uint64_t Y>
  HOSTRPC_ANNOTATE constexpr static typename fastint::sufficientType<Y>::type
  retype()
  {
    static_assert(
        static_equal<sizeof(typename fastint::sufficientType<Y>::type),
                     sizeof(type)>(),
        "TODO");
    return Y;
  }

 public:
  HOSTRPC_ANNOTATE constexpr fastint_compiletime() {}

  HOSTRPC_ANNOTATE constexpr static type value() { return V; }
  HOSTRPC_ANNOTATE constexpr operator type() const { return value(); }

  HOSTRPC_ANNOTATE constexpr auto popcount() const
  {
    return fastint_compiletime<rt().popcount()>();
  }
  HOSTRPC_ANNOTATE constexpr auto findFirstSet() const
  {
    return fastint_compiletime<rt().findFirstSet()>();
  }

  template <uint64_t Y>
  HOSTRPC_ANNOTATE constexpr auto bitwiseOr() const
  {
    constexpr auto n = retype<Y>();
    static_assert(sizeof(n) == sizeof(type), "TODO");
    return fastint_compiletime<rt().bitwiseOr(n)>();
  }

  template <uint64_t Y>
  HOSTRPC_ANNOTATE constexpr auto bitwiseAnd() const
  {
    constexpr auto n = retype<Y>();
    static_assert(sizeof(n) == sizeof(type), "TODO");
    return fastint_compiletime<rt().bitwiseAnd(n)>();
  }

  template <uint64_t Y>
  HOSTRPC_ANNOTATE constexpr auto subtract() const
  {
    constexpr auto n = retype<Y>();
    static_assert(sizeof(n) == sizeof(type), "TODO");
    return fastint_compiletime<rt().template subtract<n>()>();
  }
};

}  // namespace hostrpc

#endif
