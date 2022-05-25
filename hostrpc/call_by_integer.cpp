#include "detail/cxx.hpp"
#include "detail/tuple.hpp"
#include "thirdparty/make_function.h"

#include <stdint.h>

namespace hostrpc
{
namespace detail
{

template <typename F>
struct FunctionToTuple;
template <typename R, typename... Ts>
struct FunctionToTuple<R (*)(Ts...)>
{
  using Type = ::hostrpc::cxx::tuple<R, Ts...>;
};

template <size_t N>
struct Apply
{
  template <typename F, typename T, typename... A>
  static inline auto apply(F &&f, T &&t, A &&...a)
  {
    return Apply<N - 1>::apply(
        ::hostrpc::cxx::forward<F>(f), ::hostrpc::cxx::forward<T>(t),
        ::hostrpc::cxx::forward<T>(t).template get<N - 1>(),
        ::hostrpc::cxx::forward<A>(a)...);
  }
};

template <>
struct Apply<0>
{
  template <typename F, typename T, typename... A>
  static inline auto apply(F &&f, T &&, A &&...a)
  {
    return ::hostrpc::cxx::forward<F>(f)(::hostrpc::cxx::forward<A>(a)...);
  }
};

template <typename F, typename T>
inline auto apply(F &&f, T &&t)
{
  return Apply<::hostrpc::cxx::remove_reference<T>::type::size()>::apply(
      ::hostrpc::cxx::forward<F>(f), ::hostrpc::cxx::forward<T>(t));
}

}  // namespace detail

}  // namespace hostrpc

float func(int x, char y, double z);
int func2(int x, char y, double z);
float func3(char y, double z);
// float func3(char y, int over); // not implemented yet

template <size_t S>
struct count
{
  static constexpr size_t N = count<S - 1>::N;
  using Type = typename count<S - 1>::Type;
};

template <>
struct count<0>
{
  static constexpr size_t N = 0;
  using Type = void;
};

template <size_t Idx>
static void call_via_bytes_indirection(unsigned char *b)
{
  (void)b;
}

template <typename FTy, FTy f, size_t N_>
struct call_by_integer_base
{
  static constexpr size_t N() { return N_; }
  static constexpr FTy F() { return f; }

  static bool call_from_bytes(size_t idx, unsigned char *b)
  {
    // branch here folds out to a switch
    if (idx != N())
      {
        return false;
      }
    call_from_bytes_void(b);
    return true;
  }

  static void call_from_bytes_void(unsigned char *b)
  {
    using T = llvm::make_function::trait<FTy>;
    typename ::hostrpc::detail::FunctionToTuple<FTy>::Type tmp;
    tmp.from_bytes(b);
    // todo: indirection needed here as well?
    typename T::ReturnType res = ::hostrpc::detail::apply(f, tmp.rest);
    tmp.value = res;
    tmp.into_bytes(b);
  }

  template <typename R, typename... Args>
  static R call_via_bytes(Args... args)
  {
    // asserts here, though tuple constructor probably has the same
    typename ::hostrpc::detail::FunctionToTuple<FTy>::Type tup(
        R{}, ::hostrpc::cxx::forward<Args>(args)...);
    unsigned char bytes[tup.count_bytes()];
    tup.into_bytes(bytes);
    call_via_bytes_indirection<N>(bytes);
    tup.from_bytes(bytes);
    return tup.template get<0>();
  }
};

template <size_t I>
struct IndexToFunction;

template <typename F, F f>
struct FunctionToIndex;

template <size_t I>
struct IndexToCallFromBytes;

template <typename F, F f, size_t N>
constexpr bool consistent()
{
  auto constexpr Index = FunctionToIndex<F, f>::N();
  auto constexpr Function = IndexToFunction<N>::F();
  return Index == N && Function == f;
}

#define RECORD(Function) RECORD_IMPL(__LINE__, Function)

#define RECORD_IMPL(LINE, Function)                                            \
  template <>                                                                  \
  struct count<LINE>                                                           \
  {                                                                            \
    static constexpr size_t N = 1 + count<LINE - 1>::N;                        \
    using Type = call_by_integer_base<decltype(&Function), Function, N>;       \
  };                                                                           \
  template <>                                                                  \
  struct IndexToFunction<count<LINE>::N>                                       \
  {                                                                            \
    static constexpr auto F() { return count<LINE>::Type::F(); }               \
  };                                                                           \
  template <>                                                                  \
  struct FunctionToIndex<decltype(&Function), Function>                        \
  {                                                                            \
    static constexpr auto N() { return count<LINE>::Type::N(); }               \
  };                                                                           \
  template <>                                                                  \
  struct IndexToCallFromBytes<count<LINE>::N>                                  \
  {                                                                            \
    static constexpr auto get() { return count<LINE>::Type::call_from_bytes; } \
  };                                                                           \
  static_assert(consistent<decltype(&Function), Function, count<LINE>::N>(),   \
                "");

RECORD(func);
RECORD(func2);
RECORD(func3);


template <size_t N, size_t offset, size_t... Is>
constexpr ::hostrpc::cxx::array<
    bool (*)(size_t idx, unsigned char *),
    N> static getFunctionPointerArray(::hostrpc::cxx::index_sequence<Is...>)
{
  return {{IndexToCallFromBytes<Is + offset>::get()...}};
}

template <typename L, L l, typename U, U u>
void call_by_runtime_integer(size_t idx, unsigned char *bytes)
{
  auto constexpr lower = FunctionToIndex<L, l>::N();
  auto constexpr upper = FunctionToIndex<U, u>::N();
  static_assert(lower <= upper, "");
  constexpr size_t width = upper - lower + 1;
  auto constexpr array = getFunctionPointerArray<width, lower>(
      ::hostrpc::cxx::make_index_sequence<width>());

  // codegens as jump table
  // array[idx](bytes);

  // codegens as switch
#pragma clang loop unroll_count(array.size())
  for (size_t i = 0; i < array.size(); i++)
    {
      if (array[i](idx, bytes))
        {
          break;
        }
    }
}

void call_by_runtime_integer_codegen(size_t idx, unsigned char *bytes)
{
  call_by_runtime_integer<decltype(&func), func, decltype(&func3), func3>(
      idx, bytes);
}

int main()
{
  auto a = IndexToFunction<1>::F();
  auto b = FunctionToIndex<decltype(&func3), func3>();
}
