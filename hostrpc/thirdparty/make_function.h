//------ make_function.h - Helper for creating function wrappers -- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_MAKE_FUNCTION_H_INCLUDED
#define _OMPTARGET_MAKE_FUNCTION_H_INCLUDED

#ifdef MAKE_FUNCTION
#error "Unexpected macro MAKE_FUNCTION already defined"
#endif

// Define a function symbol as using a call to another function symbol
// where argument and return types are inferred from WITH_TYPE to allow
// disambiguation with overloaded functions or passing decltype(&SYMBOL)
#define MAKE_FUNCTION(DEFINE_SYMBOL, USING_SYMBOL, WITH_TYPE, ARITY)           \
  MAKE_FUNCTION_IMPL(DEFINE_SYMBOL, USING_SYMBOL, WITH_TYPE, ARITY)

// Example, implement acos as a call to acos_gpu
// double acos(double);
// MAKE_FUNCTION(acos, asos_gpu, decltype(&acos), 1)
// // expands to
// double acos(double x0)
// {
//   return acos_gpu(x0);
// }

// Essentially equivalent to instantiating named copies of:
// template <typename R, typename... Ts>
// R DEFINE_SYMBOL(Ts... ts) {
//   return USING_SYMBOL(ts...);
// }

#if !__has_builtin(__type_pack_element)
// Assumes either this builtin or an implementation of std::tuple_element
#include <tuple>
#endif

namespace llvm {
namespace make_function {

template <unsigned long Requested, unsigned long Required>
constexpr bool verboseAssert() {
  static_assert(Requested == Required, "Arity Error");
  return Requested == Required;
}

template <typename F> struct trait;
template <typename R, typename... Ts> struct trait<R (*)(Ts...)> {
  constexpr static const auto nargs = sizeof...(Ts);
  typedef R ReturnType;
  template </*size_t*/ decltype(sizeof(int)) I> struct arg {
    static_assert(I < nargs, "Argument index out of range");
#if __has_builtin(__type_pack_element)
    using type = __type_pack_element<I, Ts...>;
#else
    typedef typename std::tuple_element<I, std::tuple<Ts...>>::type type;
#endif
  };
  typedef R (*FunctionType)(Ts...);
};

} // namespace make_function
} // namespace llvm

// Convert type to trait and dispatch to matching arity
#define MAKE_FUNCTION_IMPL(SYM_DEF, SYM_USE, WITH_TYPE, ARITY)                 \
  MAKE_FUNCTION_##ARITY(SYM_DEF, SYM_USE,                                      \
                        llvm::make_function::trait<WITH_TYPE>, ARITY)

#define MAKE_FUNCTION_0(SYM_DEF, SYM_USE, T, ARITY)                            \
  T::ReturnType SYM_DEF() {                                                    \
    llvm::make_function::verboseAssert<ARITY, T::nargs>();                     \
    return SYM_USE();                                                          \
  }

#define MAKE_FUNCTION_1(SYM_DEF, SYM_USE, T, ARITY)                            \
  T::ReturnType SYM_DEF(typename T::template arg<0>::type x0) {                \
    llvm::make_function::verboseAssert<ARITY, T::nargs>();                     \
    return SYM_USE(x0);                                                        \
  }
#define MAKE_FUNCTION_2(SYM_DEF, SYM_USE, T, ARITY)                            \
  T::ReturnType SYM_DEF(typename T::template arg<0>::type x0,                  \
                        typename T::template arg<1>::type x1) {                \
    llvm::make_function::verboseAssert<ARITY, T::nargs>();                     \
    return SYM_USE(x0, x1);                                                    \
  }
#define MAKE_FUNCTION_3(SYM_DEF, SYM_USE, T, ARITY)                            \
  T::ReturnType SYM_DEF(typename T::template arg<0>::type x0,                  \
                        typename T::template arg<1>::type x1,                  \
                        typename T::template arg<2>::type x2) {                \
    llvm::make_function::verboseAssert<ARITY, T::nargs>();                     \
    return SYM_USE(x0, x1, x2);                                                \
  }
#define MAKE_FUNCTION_4(SYM_DEF, SYM_USE, T, ARITY)                            \
  T::ReturnType SYM_DEF(typename T::template arg<0>::type x0,                  \
                        typename T::template arg<1>::type x1,                  \
                        typename T::template arg<2>::type x2,                  \
                        typename T::template arg<3>::type x3) {                \
    llvm::make_function::verboseAssert<ARITY, T::nargs>();                     \
    return SYM_USE(x0, x1, x2, x3);                                            \
  }
#define MAKE_FUNCTION_5(SYM_DEF, SYM_USE, T, ARITY)                            \
  T::ReturnType SYM_DEF(typename T::template arg<0>::type x0,                  \
                        typename T::template arg<1>::type x1,                  \
                        typename T::template arg<2>::type x2,                  \
                        typename T::template arg<3>::type x3,                  \
                        typename T::template arg<4>::type x4) {                \
    llvm::make_function::verboseAssert<ARITY, T::nargs>();                     \
    return SYM_USE(x0, x1, x2, x3, x4);                                        \
  }
#define MAKE_FUNCTION_6(SYM_DEF, SYM_USE, T, ARITY)                            \
  T::ReturnType SYM_DEF(typename T::template arg<0>::type x0,                  \
                        typename T::template arg<1>::type x1,                  \
                        typename T::template arg<2>::type x2,                  \
                        typename T::template arg<3>::type x3,                  \
                        typename T::template arg<4>::type x4,                  \
                        typename T::template arg<5>::type x5) {                \
    llvm::make_function::verboseAssert<ARITY, T::nargs>();                     \
    return SYM_USE(x0, x1, x2, x3, x4, x5);                                    \
  }

#define MAKE_FUNCTION_7(SYM_DEF, SYM_USE, T, ARITY)                            \
  T::ReturnType SYM_DEF(typename T::template arg<0>::type x0,                  \
                        typename T::template arg<1>::type x1,                  \
                        typename T::template arg<2>::type x2,                  \
                        typename T::template arg<3>::type x3,                  \
                        typename T::template arg<4>::type x4,                  \
                        typename T::template arg<5>::type x5,                  \
                        typename T::template arg<6>::type x6) {                \
    llvm::make_function::verboseAssert<ARITY, T::nargs>();                     \
    return SYM_USE(x0, x1, x2, x3, x4, x5, x6);                                \
  }

#define MAKE_FUNCTION_8(SYM_DEF, SYM_USE, T, ARITY)                            \
  T::ReturnType SYM_DEF(typename T::template arg<0>::type x0,                  \
                        typename T::template arg<1>::type x1,                  \
                        typename T::template arg<2>::type x2,                  \
                        typename T::template arg<3>::type x3,                  \
                        typename T::template arg<4>::type x4,                  \
                        typename T::template arg<5>::type x5,                  \
                        typename T::template arg<6>::type x6,                  \
                        typename T::template arg<7>::type x7) {                \
    llvm::make_function::verboseAssert<ARITY, T::nargs>();                     \
    return SYM_USE(x0, x1, x2, x3, x4, x5, x6, x7);                            \
  }
#define MAKE_FUNCTION_9(SYM_DEF, SYM_USE, T, ARITY)                            \
  T::ReturnType SYM_DEF(typename T::template arg<0>::type x0,                  \
                        typename T::template arg<1>::type x1,                  \
                        typename T::template arg<2>::type x2,                  \
                        typename T::template arg<3>::type x3,                  \
                        typename T::template arg<4>::type x4,                  \
                        typename T::template arg<5>::type x5,                  \
                        typename T::template arg<6>::type x6,                  \
                        typename T::template arg<7>::type x7,                  \
                        typename T::template arg<8>::type x8) {                \
    llvm::make_function::verboseAssert<ARITY, T::nargs>();                     \
    return SYM_USE(x0, x1, x2, x3, x4, x5, x6, x7, x8);                        \
  }
#define MAKE_FUNCTION_10(SYM_DEF, SYM_USE, T, ARITY)                           \
  T::ReturnType SYM_DEF(typename T::template arg<0>::type x0,                  \
                        typename T::template arg<1>::type x1,                  \
                        typename T::template arg<2>::type x2,                  \
                        typename T::template arg<3>::type x3,                  \
                        typename T::template arg<4>::type x4,                  \
                        typename T::template arg<5>::type x5,                  \
                        typename T::template arg<6>::type x6,                  \
                        typename T::template arg<7>::type x7,                  \
                        typename T::template arg<8>::type x8,                  \
                        typename T::template arg<9>::type x9) {                \
    llvm::make_function::verboseAssert<ARITY, T::nargs>();                     \
    return SYM_USE(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9);                    \
  }
#define MAKE_FUNCTION_11(SYM_DEF, SYM_USE, T, ARITY)                           \
  T::ReturnType SYM_DEF(typename T::template arg<0>::type x0,                  \
                        typename T::template arg<1>::type x1,                  \
                        typename T::template arg<2>::type x2,                  \
                        typename T::template arg<3>::type x3,                  \
                        typename T::template arg<4>::type x4,                  \
                        typename T::template arg<5>::type x5,                  \
                        typename T::template arg<6>::type x6,                  \
                        typename T::template arg<7>::type x7,                  \
                        typename T::template arg<8>::type x8,                  \
                        typename T::template arg<9>::type x9,                  \
                        typename T::template arg<10>::type x10) {              \
    llvm::make_function::verboseAssert<ARITY, T::nargs>();                     \
    return SYM_USE(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10);               \
  }

#endif
