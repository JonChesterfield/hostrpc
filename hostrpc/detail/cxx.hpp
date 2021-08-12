#ifndef HOSTRPC_CXX_HPP_INCLUDED
#define HOSTRPC_CXX_HPP_INCLUDED

// Minimal part of lib c++, reimplemented here for use from freestanding code

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
inline constexpr typename remove_reference<T>::type &&move(T &&x)
{
  typedef typename remove_reference<T>::type U;
  return static_cast<U &&>(x);
}
}  // namespace cxx
}  // namespace hostrpc

#endif
