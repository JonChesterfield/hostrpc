#include "EvilUnit.h"
#include "thirdparty/make_function.h"
#include <assert.h>

#include "detail/cxx.hpp"
#include "detail/tuple.hpp"

#include <stdint.h>

template <typename... T>
__attribute__((used)) bool round_trip(hostrpc::cxx::tuple<T...> const &t)
{
  hostrpc::cxx::tuple<T...> cp;
  unsigned char bytes[t.count_bytes()];
  t.into_bytes(bytes);
  cp.from_bytes(bytes);
  return t == cp;
}

#if 1

struct TargetF
{
  // no static operator(), don't want a TargetF{}()
  __attribute__((noinline)) static void call(unsigned ID, unsigned char *bytes)
  {
    asm volatile("" : "+r"(bytes), "+r"(ID)::"memory");
  }
};

// https://stackoverflow.com/questions/687490/how-do-i-expand-a-tuple-into-variadic-template-functions-arguments,
// modified to work with local libc++ functions might want get to be a free
// function instead of a member of tuple (matches std, seems to be because
// writing .template everywhere is annoying)

namespace hostrpc
{
namespace detail
{
namespace type
{
// Book keeping is by type specialization

template <size_t S>
struct count
{
  static constexpr size_t N = count<S - 1>::N;
};

template <>
struct count<0>
{
  static constexpr size_t N = 0;
};

// Get a constexpr size_t ID, starts at zero
#define HOSTRPC_CURRENT_IDENTIFIER() \
  (::hostrpc::detail::type::count<__LINE__>::N)

// Increment value returned by HOSTRPC_CURRENT_IDENTIFIER
#define HOSTRPC_INCREMENT_IDENTIFIER()                       \
  template <>                                                \
  struct ::hostrpc::detail::type::count<__LINE__>            \
  {                                                          \
    static constexpr size_t N =                              \
        1 + ::hostrpc::detail::type::count<__LINE__ - 1>::N; \
  }

template <size_t N>
struct function_pointer;

}  // namespace type

template <typename F>
struct FunctionToTuple;
template <typename R, typename... Ts>
struct FunctionToTuple<R (*)(Ts...)>
{
  using Type = hostrpc::cxx::tuple<R, Ts...>;
};

template <size_t N>
struct Apply
{
  template <typename F, typename T, typename... A>
  static inline auto apply(F &&f, T &&t, A &&...a)
  {
    return Apply<N - 1>::apply(
        hostrpc::cxx::forward<F>(f), hostrpc::cxx::forward<T>(t),
        hostrpc::cxx::forward<T>(t).template get<N - 1>(),
        hostrpc::cxx::forward<A>(a)...);
  }
};

template <>
struct Apply<0>
{
  template <typename F, typename T, typename... A>
  static inline auto apply(F &&f, T &&, A &&...a)
  {
    return hostrpc::cxx::forward<F>(f)(hostrpc::cxx::forward<A>(a)...);
  }
};

template <typename F, typename T>
inline auto apply(F &&f, T &&t)
{
  return Apply<hostrpc::cxx::remove_reference<T>::type::size()>::apply(
      hostrpc::cxx::forward<F>(f), hostrpc::cxx::forward<T>(t));
}

}  // namespace detail

}  // namespace hostrpc

// Each instantiation of the macro allocates a new integer and associates
// it with the current symbol
#define HOSTRPC_SYNTAX(SYMBOL, ARITY)                                        \
  HOSTRPC_INCREMENT_IDENTIFIER();                                            \
  namespace hostrpc                                                          \
  {                                                                          \
  namespace detail                                                           \
  {                                                                          \
  struct SYMBOL##_Trait                                                      \
  {                                                                          \
    enum                                                                     \
    {                                                                        \
      ID = HOSTRPC_CURRENT_IDENTIFIER() - 1,                                 \
    };                                                                       \
    using T = llvm::make_function::trait<decltype(&SYMBOL)>;                 \
    static bool call_by_integer(unsigned reqID, unsigned char *b)            \
    {                                                                        \
      if (reqID != ID) return false;                                         \
      ::hostrpc::detail::FunctionToTuple<decltype(&SYMBOL)>::Type tmp;       \
      tmp.from_bytes(b);                                                     \
      T::ReturnType res = ::hostrpc::detail::apply(&SYMBOL, tmp.rest);       \
      tmp.value = res;                                                       \
      tmp.into_bytes(b);                                                     \
      return true;                                                           \
    }                                                                        \
    /*todo: specify argument types directly? */                              \
    template <typename TargetFunctor, typename... Ts>                        \
    static T::ReturnType this_shimfunction(Ts... t)                          \
    {                                                                        \
      ::hostrpc::detail::FunctionToTuple<decltype(&SYMBOL)>::Type tup(       \
          T::ReturnType{}, t...);                                            \
      unsigned char bytes[tup.count_bytes()];                                \
      tup.into_bytes(bytes);                                                 \
      TargetFunctor::call(ID, bytes);                                        \
      tup.from_bytes(bytes);                                                 \
      return tup.template get<0>();                                          \
    }                                                                        \
  };                                                                         \
  template <>                                                                \
  struct ::hostrpc::detail::type::function_pointer<                          \
      HOSTRPC_CURRENT_IDENTIFIER() - 1>                                      \
  {                                                                          \
    static constexpr bool (*call())(unsigned, unsigned char *)               \
    {                                                                        \
      return SYMBOL##_Trait::call_by_integer;                                \
    }                                                                        \
  };                                                                         \
  }                                                                          \
  }                                                                          \
  MAKE_FUNCTION(SYMBOL,                                                      \
                hostrpc::detail::SYMBOL##_Trait::this_shimfunction<TargetF>, \
                decltype(&SYMBOL), ARITY)

extern "C" float func(int x, char y, double z);
extern "C" int func2(int x, char y, double z);
extern "C" float func3(char y, double z);

HOSTRPC_SYNTAX(func, 3);
HOSTRPC_SYNTAX(func2, 3);
HOSTRPC_SYNTAX(func3, 2);

template <size_t N, size_t... Is>
constexpr hostrpc::cxx::array<
    bool (*)(unsigned, unsigned char *),
    N> static getFunctionPointerArray(hostrpc::cxx::index_sequence<Is...>)
{
  return {{hostrpc::detail::type::function_pointer<Is>::call()...}};
}

// Call a function based on runtime value of ID, i.e. what was read out of
// the packet. A switch over the functions marked out with SYNTAX above
extern "C" void call_by_integer(unsigned reqID, unsigned char *b)
{
  enum
  {
    max_id = HOSTRPC_CURRENT_IDENTIFIER()
  };
  static constexpr hostrpc::cxx::array<bool (*)(unsigned, unsigned char *),
                                       max_id>
      function_pointers = getFunctionPointerArray<max_id>(
          hostrpc::cxx::make_index_sequence<max_id>());

  // codegens as a jump table
  // function_pointers[reqID](reqID, b);

  // same thing, less inclined to codegen as jump table
#pragma clang loop unroll_count(function_pointers.size())
  for (size_t i = 0; i < function_pointers.size(); i++)
    {
      if (function_pointers[i](reqID, b))
        {
          break;
        }
    }
}

#if 1
__attribute__((noinline)) float codegen2(int x, double y) { return (float)y; }
float codegen(int x, double y)
{
  auto t = hostrpc::cxx::make_tuple(x, y);
  decltype(t) cp;
  unsigned char bytes[t.count_bytes()];
  t.into_bytes(bytes);
  cp.from_bytes(bytes);
  return codegen2(cp.get<0>(), cp.get<1>());
}
#endif

using equal_type_ex = hostrpc::cxx::tuple<int, float, double, int64_t>;
extern "C" bool equal_example(equal_type_ex x, equal_type_ex y)
{
  return x == y;
}

extern "C" bool round_example(equal_type_ex x) { return round_trip(x); }

extern "C" void into_bytes(equal_type_ex *x, unsigned char *b)
{
  x->into_bytes(b);
}

extern "C" void from_bytes(equal_type_ex *x, unsigned char *b)
{
  x->from_bytes(b);
}

#endif

#if 1
MAIN_MODULE()
{
  TEST("")
  {
    hostrpc::cxx::tuple<int, char, int64_t> t = {1, 'c', 4};

    printf("%d %c %ld\n", t.get<0>(), t.get<1>(), t.get<2>());

    decltype(t) cp;

    unsigned char bytes[t.count_bytes()];
    t.into_bytes(bytes);
    cp.from_bytes(bytes);

    printf("%d %c %ld\n", cp.get<0>(), cp.get<1>(), cp.get<2>());

    CHECK(round_trip(t));
  }
}
#endif
