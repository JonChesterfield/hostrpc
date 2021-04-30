#ifndef HOSTRPC_PRINTF_H_INCLUDED
#define HOSTRPC_PRINTF_H_INCLUDED

#include "detail/platform_detect.hpp"
#include <stdint.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if (HOSTRPC_HOST)
#include "hsa.h"
#ifdef __cplusplus
extern "C"
{
#endif
  int hostrpc_print_enable_on_hsa_agent(hsa_executable_t ex,
                                        hsa_agent_t kernel_agent);
#ifdef __cplusplus
}
#endif
#endif

// printf implementation macros
#define __PRINTF_API_EXTERNAL_ HOSTRPC_ANNOTATE __attribute__((noinline))
#define __PRINTF_API_INTERNAL_ \
  HOSTRPC_ANNOTATE static inline __attribute__((unused))

#ifdef __cplusplus
#define __PRINTF_API_EXTERNAL __PRINTF_API_EXTERNAL_ extern "C"
#define __PRINTF_API_INTERNAL __PRINTF_API_INTERNAL_
#else
#define __PRINTF_API_EXTERNAL __PRINTF_API_EXTERNAL_

#ifdef __attribute__
#warning "__attribute__ is a macro, missing freestanding?"
#endif

#define __PRINTF_API_INTERNAL \
  __PRINTF_API_INTERNAL_ __attribute__((overloadable))
#endif

#define __PRINTF_PASTE_(X, Y) X##Y
#define __PRINTF_PASTE(X, Y) __PRINTF_PASTE_(X, Y)

#define __PRINTF_WRAP(FMT, POS, X)                                            \
  __spec_loc = __printf_next_specifier_location(__fmt, __strlen, __spec_loc); \
  piecewise_print_element(__port,                                             \
                          __printf_specifier_classify(FMT, __spec_loc), X);
#define __PRINTF_WRAP1(FMT, U)
#define __PRINTF_WRAP2(FMT, U, X) __PRINTF_WRAP(FMT, 0, X)
#define __PRINTF_WRAP3(FMT, U, X, Y) \
  __PRINTF_WRAP2(FMT, U, X) __PRINTF_WRAP(FMT, 1, Y)
#define __PRINTF_WRAP4(FMT, U, X, Y, Z) \
  __PRINTF_WRAP3(FMT, U, X, Y) __PRINTF_WRAP(FMT, 2, Z)
#define __PRINTF_WRAP5(FMT, U, X0, X1, X2, X3) \
  __PRINTF_WRAP4(FMT, U, X0, X1, X2) __PRINTF_WRAP(FMT, 3, X3)
#define __PRINTF_WRAP6(FMT, U, X0, X1, X2, X3, X4) \
  __PRINTF_WRAP5(FMT, U, X0, X1, X2, X3) __PRINTF_WRAP(FMT, 4, X4)
#define __PRINTF_WRAP7(FMT, U, X0, X1, X2, X3, X4, X5) \
  __PRINTF_WRAP6(FMT, U, X0, X1, X2, X3, X4) __PRINTF_WRAP(FMT, 5, X5)
#define __PRINTF_WRAP8(FMT, U, X0, X1, X2, X3, X4, X5, X6) \
  __PRINTF_WRAP7(FMT, U, X0, X1, X2, X3, X4, X5) __PRINTF_WRAP(FMT, 6, X6)
#define __PRINTF_WRAP9(FMT, U, X0, X1, X2, X3, X4, X5, X6, X7) \
  __PRINTF_WRAP8(FMT, U, X0, X1, X2, X3, X4, X5, X6) __PRINTF_WRAP(FMT, 7, X7)

#if 0
/*
 * I believe I first saw this trick on stack overflow. Possibly at the
 * following:
 * http://stackoverflow.com/questions/11317474/macro-to-count-number-of-arguments
 * This is considered to be a standard preprocessor technique in common
 * knowledge.
 * One place attributes the original implementationo to Laurent Deniau, January 2006
 * Laurent Deniau, "__VA_NARG__," 17 January 2006, <comp.std.c> (29 November 2007).
 * https://groups.google.com/forum/?fromgroups=#!topic/comp.std.c/d-6Mj5Lko_s
 */
#endif

#define __PRINTF_PP_ARG_N(                                                     \
    _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16,     \
    _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, \
    _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, \
    _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61, \
    _62, _63, N, ...)                                                          \
  N

#define __PRINTF_PP_RSEQ_N()                                                  \
  63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, \
      44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, \
      26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9,  \
      8, 7, 6, 5, 4, 3, 2, 1, 0

#define __PRINTF_PP_NARG_(...) __PRINTF_PP_ARG_N(__VA_ARGS__)

#define __PRINTF_PP_NARG(...) \
  __PRINTF_PP_NARG_(__VA_ARGS__, __PRINTF_PP_RSEQ_N())

#define __PRINTF_DISPATCH_ARGS(FMT, ...) \
  __PRINTF_PASTE(__PRINTF_WRAP, __PRINTF_PP_NARG(__VA_ARGS__))(FMT, __VA_ARGS__)

// printf expands as a macro that counts the arguments and passes each one to
// piecewise_print_element

// this is moderately annoying when included into host code, maybe
// rename it when off device
#define printf(FMT, ...)                                  \
  {                                                       \
    size_t __offset = 0;                                  \
    (void)__offset;                                       \
    const char *__fmt = FMT;                              \
    const size_t __strlen = __builtin_constant_p(__fmt)   \
                                ? __builtin_strlen(__fmt) \
                                : __printf_strlen(__fmt); \
    (void)__strlen;                                       \
    uint32_t __port = piecewise_print_start(__fmt);       \
    size_t __spec_loc = 0;                                \
    (void)__spec_loc;                                     \
    __PRINTF_DISPATCH_ARGS(__fmt, UNUSED, ##__VA_ARGS__)  \
    piecewise_print_end(__port);                          \
  }

// Functions implemented out of header. printf resolves to multiple calls to
// these. Some implemented on gcn. All should probably be implemented on
// gcn/ptx/x64
__PRINTF_API_EXTERNAL uint32_t piecewise_print_start(const char *fmt);
__PRINTF_API_EXTERNAL int piecewise_print_end(uint32_t port);

// simple types
__PRINTF_API_EXTERNAL void piecewise_pass_element_int32(uint32_t port,
                                                        int32_t x);
__PRINTF_API_EXTERNAL void piecewise_pass_element_uint32(uint32_t port,
                                                         uint32_t x);
__PRINTF_API_EXTERNAL void piecewise_pass_element_int64(uint32_t port,
                                                        int64_t x);
__PRINTF_API_EXTERNAL void piecewise_pass_element_uint64(uint32_t port,
                                                         uint64_t x);
__PRINTF_API_EXTERNAL void piecewise_pass_element_double(uint32_t port,
                                                         double x);

// copy null terminated string starting at x, print the string
__PRINTF_API_EXTERNAL void piecewise_pass_element_cstr(uint32_t port,
                                                       const char *x);
// print the address of the argument on the gpu
__PRINTF_API_EXTERNAL void piecewise_pass_element_void(uint32_t port,
                                                       const void *x);

// implement %n specifier
__PRINTF_API_EXTERNAL void piecewise_pass_element_write_int32(uint32_t port,
                                                              int32_t *x);
__PRINTF_API_EXTERNAL void piecewise_pass_element_write_int64(uint32_t port,
                                                              int64_t *x);

// approximate compile time calculations in C. TODO: Can force it when in C++
enum __printf_spec_t
{
  spec_normal,
  spec_string,
  spec_output,
  spec_none,
};

// Dispatch based on element type

// Straightforward mapping from integer/double onto the lower calls
__PRINTF_API_INTERNAL void piecewise_print_element(uint32_t port,
                                                   enum __printf_spec_t spec,
                                                   int x)
{
  // (printf)("hit L%u [%s]\n", __LINE__, __PRETTY_FUNCTION__);
  (void)spec;
  _Static_assert(sizeof(int) == sizeof(int32_t), "");
  piecewise_pass_element_int32(port, x);
}

__PRINTF_API_INTERNAL void piecewise_print_element(uint32_t port,
                                                   enum __printf_spec_t spec,
                                                   unsigned x)
{
  // (printf)("hit L%u [%s]\n", __LINE__, __PRETTY_FUNCTION__);
  (void)spec;
  _Static_assert(sizeof(unsigned) == sizeof(uint32_t), "");
  piecewise_pass_element_uint32(port, x);
}

__PRINTF_API_INTERNAL void piecewise_print_element(uint32_t port,
                                                   enum __printf_spec_t spec,
                                                   long x)
{
  // (printf)("hit L%u [%s]\n", __LINE__, __PRETTY_FUNCTION__);
  (void)spec;
  _Static_assert(
      (sizeof(long) == sizeof(int32_t)) || (sizeof(long) == sizeof(int64_t)),
      "");
  if (sizeof(long) == sizeof(int32_t))
    {
      piecewise_pass_element_int32(port, (int32_t)x);
    }
  if (sizeof(long) == sizeof(int64_t))
    {
      piecewise_pass_element_int64(port, (int64_t)x);
    }
}

__PRINTF_API_INTERNAL void piecewise_print_element(uint32_t port,
                                                   enum __printf_spec_t spec,
                                                   unsigned long x)
{
  // (printf)("hit L%u [%s]\n", __LINE__, __PRETTY_FUNCTION__);
  (void)spec;
  _Static_assert((sizeof(unsigned long) == sizeof(uint32_t)) ||
                     (sizeof(unsigned long) == sizeof(uint64_t)),
                 "");
  if (sizeof(unsigned long) == sizeof(uint32_t))
    {
      piecewise_pass_element_uint32(port, (uint32_t)x);
    }
  if (sizeof(unsigned long) == sizeof(uint64_t))
    {
      piecewise_pass_element_uint64(port, (uint64_t)x);
    }
}

__PRINTF_API_INTERNAL void piecewise_print_element(uint32_t port,
                                                   enum __printf_spec_t spec,
                                                   long long x)
{
  // (printf)("hit L%u [%s]\n", __LINE__, __PRETTY_FUNCTION__);
  (void)spec;
  _Static_assert(sizeof(long long) == sizeof(int64_t), "");
  piecewise_pass_element_int64(port, x);
}

__PRINTF_API_INTERNAL void piecewise_print_element(uint32_t port,
                                                   enum __printf_spec_t spec,
                                                   unsigned long long x)
{
  // (printf)("hit L%u [%s]\n", __LINE__, __PRETTY_FUNCTION__);
  (void)spec;
  _Static_assert(sizeof(unsigned long long) == sizeof(uint64_t), "");
  piecewise_pass_element_uint64(port, x);
}

__PRINTF_API_INTERNAL void piecewise_print_element(uint32_t port,
                                                   enum __printf_spec_t spec,
                                                   double x)
{
  // (printf)("hit L%u [%s]\n", __LINE__, __PRETTY_FUNCTION__);
  (void)spec;
  piecewise_pass_element_double(port, x);
}

// char* and void* check the format string to distinguish copy string vs pointer
// signed char* can also used with %n
__PRINTF_API_INTERNAL void piecewise_print_element(uint32_t port,
                                                   enum __printf_spec_t spec,
                                                   const char *x)
{
  // (printf)("hit L%u [%s]\n", __LINE__, __PRETTY_FUNCTION__);
  switch (spec)
    {
      case spec_string:
        return piecewise_pass_element_cstr(port, x);
      case spec_normal:
        return piecewise_pass_element_void(port, (const void *)x);
      case spec_output:
        {
          // This is somewhat dubious.
          // In C, a string literal has type char[]
          // Overloading works on const char* or char*, but clang
          // considers a call with a "literal" ambiguous
          // Bug https://bugs.llvm.org/show_bug.cgi?id=49978
          // Preference would be to instantiate on char* and
          // const char*, where the latter ignores %n
          // For present, passing a const char* too write to via
          // printf is UB anyway, so assume the argument is actually
          // a mutable char that has been cast
          int32_t tmp;
          piecewise_pass_element_write_int32(port, &tmp);
          *(char *)x = (char)tmp;
          break;
        }
      case spec_none:
        break;
    }
}

__PRINTF_API_INTERNAL void piecewise_print_element(uint32_t port,
                                                   enum __printf_spec_t spec,
                                                   signed short *x)
{
  (void)spec;
  // (printf)("hit L%u [%s]", __LINE__, __PRETTY_FUNCTION__);
  int32_t tmp;
  piecewise_pass_element_write_int32(port, &tmp);
  *x = (signed short)tmp;
}

__PRINTF_API_INTERNAL void piecewise_print_element(uint32_t port,
                                                   enum __printf_spec_t spec,
                                                   int *x)
{
  (void)spec;
  // (printf)("hit L%u [%s]", __LINE__, __PRETTY_FUNCTION__);
  _Static_assert(sizeof(int) == sizeof(int32_t), "");
  piecewise_pass_element_write_int32(port, x);
}

__PRINTF_API_INTERNAL void piecewise_print_element(uint32_t port,
                                                   enum __printf_spec_t spec,
                                                   long *x)
{
  (void)spec;
  // (printf)("hit L%u [%s]", __LINE__, __PRETTY_FUNCTION__);

  if (sizeof(long) == sizeof(int32_t))
    {
      piecewise_pass_element_write_int32(port, (int32_t *)x);
    }
  if (sizeof(long) == sizeof(int64_t))
    {
      piecewise_pass_element_write_int64(port, (int64_t *)x);
    }
}

__PRINTF_API_INTERNAL void piecewise_print_element(uint32_t port,
                                                   enum __printf_spec_t spec,
                                                   size_t *x)
{
  (void)spec;
  // (printf)("hit L%u [%s]", __LINE__, __PRETTY_FUNCTION__);

  if (sizeof(size_t) == sizeof(int32_t))
    {
      piecewise_pass_element_write_int32(port, (int32_t *)x);
    }
  if (sizeof(size_t) == sizeof(int64_t))
    {
      piecewise_pass_element_write_int64(port, (int64_t *)x);
    }
}

__PRINTF_API_INTERNAL void piecewise_print_element(uint32_t port,
                                                   enum __printf_spec_t spec,
                                                   long long *x)
{
  (void)spec;
  // (printf)("hit L%u [%s]", __LINE__, __PRETTY_FUNCTION__);
  _Static_assert(sizeof(long long) == sizeof(int64_t), "");
  piecewise_pass_element_write_int64(port, (int64_t *)x);
}

#define __PRINTF_DISPATCH_INDIRECT(TYPE, VIA)           \
  __PRINTF_API_INTERNAL void piecewise_print_element(   \
      uint32_t port, enum __printf_spec_t spec, TYPE x) \
  {                                                     \
    piecewise_print_element(port, spec, (VIA)x);        \
  }

// Implement types that undergo argument promotion as promotions
__PRINTF_DISPATCH_INDIRECT(bool, int)
__PRINTF_DISPATCH_INDIRECT(signed char, int)
__PRINTF_DISPATCH_INDIRECT(unsigned char, unsigned int)
__PRINTF_DISPATCH_INDIRECT(short, int)
__PRINTF_DISPATCH_INDIRECT(unsigned short, unsigned int)
__PRINTF_DISPATCH_INDIRECT(float, double)

// Pointers take behaviour from the format string (%s/%p/%n), so redirect
// signed/unsigned/void via the const char* which contains the switch
__PRINTF_DISPATCH_INDIRECT(const void *, const char *)
__PRINTF_DISPATCH_INDIRECT(const signed char *, const char *)
__PRINTF_DISPATCH_INDIRECT(const unsigned char *, const char *)

#undef __PRINTF_DISPATCH_INDIRECT

// because __builtin_strlen resolves to strlen, which amdgcn does not lower
// todo: check if memcpy on unknown length has the same problem
__PRINTF_API_INTERNAL size_t __printf_strlen(const char *str)
{
  for (size_t i = 0;; i++)
    {
      if (str[i] == '\0')
        {
          return i;
        }
    }
}

__PRINTF_API_INTERNAL
enum __printf_spec_t __printf_specifier_classify(const char *format, size_t loc)
{
  switch (format[loc])
    {
      case '\0':
        {
          // %'\0 ill formed
          return spec_none;
        }
      case 's':
        {
          return spec_string;
        }
      case 'n':
        {
          return spec_output;
        }
      default:
        {
          return spec_normal;
        }
    }
}

__PRINTF_API_INTERNAL bool __printf_haszero(uint32_t v)
{
  return ((v - UINT32_C(0x01010101)) & ~v & UINT32_C(0x80808080));
}

__PRINTF_API_INTERNAL bool __printf_length_modifier_p(char c)
{
  // test if 'c' is one of hljztL, as if so, need to keep scanning to
  // find the s/n/p
  unsigned char uc;
  __builtin_memcpy(&uc, &c, 1);

  // 32 bit codegen is noticably better for amdgcn and ~ same as 64 bit for x64
  // Six characters to check, so checks 'h' repeatedly to avoid a zero.
  uint32_t broadcast = UINT32_C(0x01010101) * uc;
  uint32_t haystackA = ('h' << 0) | ('l' << 8) | ('j' << 16) | ('z' << 24);
  uint32_t haystackB = ('t' << 0) | ('L' << 8) | ('h' << 16) | ('h' << 24);

  // ~ (A & B)
  //  ~A | ~B
  // todo: looks redundant, but manually simplifying doesn't help codegen
  // uint32_t vA = (broadcast ^ haystackA);
  // uint32_t vB = (broadcast ^ haystackB);
  // return
  //      ((vA -UINT32_C(0x01010101)) & ~vA & UINT32_C(0x80808080)) |
  //      ((vB -UINT32_C(0x01010101)) & ~vB & UINT32_C(0x80808080));

  return __printf_haszero(broadcast ^ haystackA) |
         __printf_haszero(broadcast ^ haystackB);
}

__PRINTF_API_INTERNAL
size_t __printf_next_start(const char *format, size_t len, size_t input_offset)
{
  if (len < 2)
    {
      return len;
    }
  for (size_t o = input_offset; o < len - 1; o++)
    {
      if (format[o] == '%')
        {
          if (format[o + 1] == '%')
            {
              o++;
              continue;
            }
          else
            {
              return o;
            }
        }
    }
  return len;
}

__PRINTF_API_INTERNAL
size_t __printf_next_end(const char *format, size_t len, size_t input_offset)
{
  // assert(format[input_offset] == '%');
  size_t o = input_offset;
  o++; /* step over the % */
  for (; o < len; o++)
    {
      { /* todo: this loop should be < 3 iters */
        if (!__printf_length_modifier_p(format[o]))
          {
            return o;
          }
      }
    }
  return len;
}

__PRINTF_API_INTERNAL
__attribute__((always_inline)) size_t __printf_next_specifier_location(
    const char *format, size_t len, size_t input_offset)
{
  // more heuristics than one would like, but probably viable
  // c++ version can be totally robust

  // flatten recursive function is not treated as wholly unrolled loop

  // if unroll is too high, hits an internal compute limit and gives up
  // if unroll is too low, doesn't cover the whole input string
  // changing to scan multiple bytes at a time may help with that, problem
  // is unrolling creates a lot of IR, and instcombine gives up before
  // reducing it fully
#if 0
  size_t length_to_scan = len - input_offset;
  if (__builtin_constant_p(length_to_scan))
    {
#pragma unroll 32
      IMPL();
    }
  else
    {
      IMPL();
    }
#endif
  
  return __printf_next_end(format, len,
                           __printf_next_start(format, len, input_offset));


}

// redundant parts of API / convenience hacks
__PRINTF_API_INTERNAL void print_string(const char *str)
{
  uint32_t port = piecewise_print_start("%s");
  if (port == UINT32_MAX)
    {
      return;
    }

  piecewise_pass_element_cstr(port, str);
  piecewise_print_end(port);
}

#endif
