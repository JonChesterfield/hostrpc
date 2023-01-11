#ifndef HOSTRPC_CXX_TUPLE_HPP_INCLUDED
#define HOSTRPC_CXX_TUPLE_HPP_INCLUDED

#include <stddef.h>

#include "cxx.hpp"

#include "../platform/detect.hpp"

namespace hostrpc
{
namespace cxx
{
template <typename T, typename... Ts>
struct tuple;

namespace detail
{
template <size_t I, typename T, typename... Ts>
struct nthAccessor
{
  static_assert(I <= sizeof...(Ts), "");
  using type = typename nthAccessor<I - 1, Ts...>::type;
  HOSTRPC_ANNOTATE
  static type get(tuple<T, Ts...> const& t)
  {
    return nthAccessor<I - 1, Ts...>::get(t.rest);
  }
};

template <typename T, typename... Ts>
struct nthAccessor<0, T, Ts...>
{
  using type = T;
  HOSTRPC_ANNOTATE
  static type get(tuple<T, Ts...> const& t) { return t.value; }
};

template <size_t S, size_t E>
struct equalImpl
{
  static_assert(S < E, "");
  template <typename T, typename U>
  HOSTRPC_ANNOTATE static bool equal(T const& t, U const& u)
  {
    bool StartEqual = t.template get<S>() == u.template get<S>();
    bool RestEqual = equalImpl<S + 1, E>::equal(t, u);
    return StartEqual && RestEqual;
  }
};

template <size_t SE>
struct equalImpl<SE, SE>
{
  template <typename T, typename U>
  static bool equal(T const&, U const&)
  {
    return true;
  }
};
}  // namespace detail

template <typename T>
struct tuple<T>
{
  HOSTRPC_ANNOTATE
  tuple() : value() {}
  HOSTRPC_ANNOTATE
  tuple(T t) : value(t) {}

  static_assert(is_trivially_copyable<T>() /*::value*/,
                "Require trivially copyable type");

  using type = tuple<T>;

  HOSTRPC_ANNOTATE
  static constexpr size_t size() { return 1u; }
  HOSTRPC_ANNOTATE
  static constexpr size_t count_bytes() { return sizeof(T); }

  HOSTRPC_ANNOTATE
  void into_bytes(unsigned char* b) const
  {
    __builtin_memcpy(b, &value, sizeof(T));
  }
  HOSTRPC_ANNOTATE
  void from_bytes(unsigned char* b) { __builtin_memcpy(&value, b, sizeof(T)); }

  template <size_t I>
  HOSTRPC_ANNOTATE typename detail::nthAccessor<I, T>::type get() const
  {
    static_assert(I <= size(), "");
    return detail::nthAccessor<I, T>::get(*this);
  }

  T value;
};

template <typename T, typename... Ts>
struct tuple
{
  HOSTRPC_ANNOTATE
  tuple() : value(), rest() {}
  HOSTRPC_ANNOTATE
  tuple(T t, Ts... ts) : value(t), rest(ts...) {}

  using type = tuple<T, Ts...>;
  HOSTRPC_ANNOTATE
  static constexpr size_t size() { return 1u + tuple<Ts...>::size(); }
  HOSTRPC_ANNOTATE
  static constexpr size_t count_bytes()
  {
    return sizeof(T) + tuple<Ts...>::count_bytes();
  }

  HOSTRPC_ANNOTATE
  void into_bytes(unsigned char* b) const
  {
    __builtin_memcpy(b, &value, sizeof(T));
    b += sizeof(T);
    return rest.into_bytes(b);  // todo: musttail where available
  }
  HOSTRPC_ANNOTATE
  void from_bytes(unsigned char* b)
  {
    __builtin_memcpy(&value, b, sizeof(T));
    b += sizeof(T);
    return rest.from_bytes(b);
  }

  template <size_t I>
  HOSTRPC_ANNOTATE typename detail::nthAccessor<I, T, Ts...>::type get() const
  {
    static_assert(I <= size(), "");
    return detail::nthAccessor<I, T, Ts...>::get(*this);
  }

  T value;
  tuple<Ts...> rest;
};

template <typename... Ts>
HOSTRPC_ANNOTATE bool operator==(tuple<Ts...> const& x, tuple<Ts...> const& y)
{
  static_assert(sizeof...(Ts) == tuple<Ts...>::size(), "");
  return detail::equalImpl<0, sizeof...(Ts)>::equal(x, y);
}

template <typename... Ts>
HOSTRPC_ANNOTATE bool operator!=(tuple<Ts...> const& x, tuple<Ts...> const& y)
{
  return !(x == y);
}

template <typename... Ts>
HOSTRPC_ANNOTATE tuple<Ts...> make_tuple(Ts... ts)
{
  return tuple<Ts...>(ts...);
}

}  // namespace cxx
}  // namespace hostrpc

#endif
