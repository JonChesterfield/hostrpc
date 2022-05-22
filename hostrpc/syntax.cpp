#include "EvilUnit.h"
#include "thirdparty/dlwrap.h"
#include "thirdparty/make_function.h"
#include <assert.h>

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

__attribute__((noinline)) extern "C" void target(unsigned ID,
                                                 unsigned char *bytes)
{
  asm volatile("" : "+r"(bytes), "+r"(ID)::"memory");
}


// https://stackoverflow.com/questions/687490/how-do-i-expand-a-tuple-into-variadic-template-functions-arguments,
// modified to work with local libc++ functions might want get to be a free
// function instead of a member of tuple (matches std, seems to be because
// writing .template everywhere is annoying)

namespace detail
{

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
}  // namespace detail

template <typename F, typename T>
inline auto apply(F &&f, T &&t)
{
  return detail::Apply<hostrpc::cxx::remove_reference<T>::type::size()>::apply(
      hostrpc::cxx::forward<F>(f), hostrpc::cxx::forward<T>(t));
}

namespace dlwrap
{
namespace type
{
template <size_t N>
struct function_pointer;
}
}  // namespace dlwrap

// Each instantiation of the macro allocates a new integer and associates
// it with the current symbol
#define DLWRAP_SYNTAX(SYMBOL, ARITY)                              \
  DLWRAP_INC();                                                   \
  namespace dlwrap                                                \
  {                                                               \
    struct SYMBOL##_Trait                                         \
  {                                                               \
    enum                                                          \
    {                                                             \
      ID = DLWRAP_ID() - 1,                                       \
    };                                                            \
    using T = llvm::make_function::trait<decltype(&SYMBOL)>;      \
    static bool call_by_integer(unsigned reqID, unsigned char *b) \
    {                                                             \
      if (reqID != ID) return false;                              \
      detail::FunctionToTuple<decltype(&SYMBOL)>::Type tmp;       \
      tmp.from_bytes(b);                                          \
      T::ReturnType res = apply(&SYMBOL, tmp.rest);               \
      tmp.value = res;                                            \
      tmp.into_bytes(b);                                          \
      return true;                                                \
    }                                                             \
    /*todo: specify argument types directly? */                   \
    template <typename... Ts>                                     \
    static T::ReturnType this_shimfunction(Ts... t)               \
    {                                                             \
      using R = T::ReturnType;                                    \
      detail::FunctionToTuple<decltype(&SYMBOL)>::Type tup(R{}, t...); \
      unsigned char bytes[tup.count_bytes()];                     \
      tup.into_bytes(bytes);                                      \
      target(ID, bytes);                                          \
      tup.from_bytes(bytes);                                      \
      return tup.template get<0>();                               \
    }                                                             \
    static constexpr T::FunctionType get()                        \
    {                                                             \
      verboseAssert<ARITY, T::nargs>(); \
      return &this_shimfunction;                                  \
    }                                                             \
  };                                                              \
  template <>                                                     \
  struct dlwrap::type::function_pointer<DLWRAP_ID() - 1>          \
  {                                                               \
    static constexpr bool (*call())(unsigned, unsigned char *)    \
    {                                                             \
      return SYMBOL##_Trait::call_by_integer;                     \
    }                                                             \
  };                                                              \
  } MAKE_FUNCTION(SYMBOL, dlwrap::SYMBOL##_Trait::this_shimfunction, decltype(&SYMBOL), ARITY)

extern "C" float func(int x, char y, double z);
extern "C" int func2(int x, char y, double z);
extern "C" float func3(char y, double z);

DLWRAP_SYNTAX(func, 3);
DLWRAP_SYNTAX(func2, 3);
DLWRAP_SYNTAX(func3, 2);

template <size_t N, size_t... Is>
constexpr std::array<
    bool (*)(unsigned, unsigned char *),
    N> static getFunctionPointerArray(std::index_sequence<Is...>)
{
  return {{dlwrap::type::function_pointer<Is>::call()...}};
}

// Call a function based on runtime value of ID, i.e. what was read out of
// the packet. A switch over the functions marked out with SYNTAX above
extern "C" void call_by_integer(unsigned reqID, unsigned char *b)
{
  static constexpr std::array<bool (*)(unsigned, unsigned char *), DLWRAP_ID()>
      function_pointers = getFunctionPointerArray<DLWRAP_ID()>(
          std::make_index_sequence<DLWRAP_ID()>());

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

#if 0
float codegen2(int x, double y);
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
