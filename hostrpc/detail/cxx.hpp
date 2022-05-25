#ifndef HOSTRPC_CXX_HPP_INCLUDED
#define HOSTRPC_CXX_HPP_INCLUDED

// Minimal part of lib c++, reimplemented here for use from freestanding code
// Derived from libc++ where the implementation is not straightforward

#include <stddef.h> // size_t, consider dropping

namespace hostrpc
{
namespace cxx
{
// std::move reimplemented
template <class T>
struct remove_reference
{
  typedef T type;
};
template <class T>
struct remove_reference<T &>
{
  typedef T type;
};
template <class T>
struct remove_reference<T &&>
{
  typedef T type;
};

template <class T>
constexpr T &&forward(typename remove_reference<T>::type &t) noexcept
{
  return static_cast<T &&>(t);
}

template <class T>
constexpr T &&forward(typename remove_reference<T>::type &&t) noexcept
{
  return static_cast<T &&>(t);
}

template <class T>
inline constexpr typename remove_reference<T>::type &&move(T &&x)
{
  typedef typename remove_reference<T>::type U;
  return static_cast<U &&>(x);
}

template <class T>
T &&cxx_declval(int);
template <class T>
T cxx_declval(long);
template <class T>
decltype(cxx_declval<T>(0)) declval() noexcept;

template <class T, T v>
struct integral_constant
{
  static constexpr const T value = v;
  typedef T value_type;
  typedef integral_constant type;
  constexpr operator value_type() const noexcept { return value; }
  constexpr value_type operator()() const noexcept { return value; }
};

template <class T, T... Is>
struct integer_sequence
{
  typedef T value_type;
  // TODO: static_assert( is_integral<T>::value)
  static constexpr size_t size() noexcept { return sizeof...(Is); }
};

template <size_t... Is>
using index_sequence = integer_sequence<size_t, Is...>;

#if __has_builtin(__make_integer_seq)
template <class _Tp, _Tp _Np>
using make_integer_sequence = __make_integer_seq<integer_sequence, _Tp, _Np>;
#else
#error "Expected to build with a clang that has __make_integer_seq"
#endif

template <size_t _Np>
using make_index_sequence = make_integer_sequence<size_t, _Np>;

template <class T>
struct is_trivially_copyable
    : public integral_constant<bool, __is_trivially_copyable(T)>
{
};

// Only implementing the small subset of array currently used
template <class _Tp, size_t _Size>
struct array
{
  typedef _Tp value_type;
  typedef const value_type &const_reference;

  typedef size_t size_type;

  _Tp __elems_[_Size];

  constexpr const_reference operator[](size_type __n) const noexcept
  {
    return __elems_[__n];
  }

  constexpr size_type size() const { return _Size; }
};

}  // namespace cxx
}  // namespace hostrpc

#endif
