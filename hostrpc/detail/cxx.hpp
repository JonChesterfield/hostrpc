#ifndef HOSTRPC_CXX_HPP_INCLUDED
#define HOSTRPC_CXX_HPP_INCLUDED

// Minimal part of lib c++, reimplemented here for use from freestanding code
// Derived from libc++ where the implementation is not straightforward

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

template <class T>
struct is_trivially_copyable
    : public integral_constant<bool, __is_trivially_copyable(T)>
{
};

template <bool B>
using bool_constant = integral_constant<bool, B>;

typedef bool_constant<true> true_type;
typedef bool_constant<false> false_type;
  
}  // namespace cxx
}  // namespace hostrpc

#endif
