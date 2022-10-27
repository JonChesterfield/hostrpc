#ifndef HOSTRPC_PRINTF_H_INCLUDED
#define HOSTRPC_PRINTF_H_INCLUDED

#include "platform/detect.hpp"
#include "hostrpc_printf_api_macro.h"
#include "base_types.hpp" // unfortunate but probably doesn't matter

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// printf expands as a macro that counts the arguments and passes each one to
// __printf_print_element

// Expects to be used by #define printf(...) __hostrpc_printf(__VA_ARGS__)
// Should be a function for C++, needs to be a macro in C
// Rest of this header provides implementation details that are usefully
// inlined into the application

// Basically can't use this from C unless the typed_port abstraction is breached
// On balance, more likely to spin printf in the compiler than a library anyway.

#define __hostrpc_printf(FMT, ...)                       \
  {                                                      \
    size_t __offset = 0;                                 \
    (void)__offset;                                      \
    const char *__fmt = FMT;                             \
    const size_t __strlen = __printf_strlen(__fmt);      \
    (void)__strlen;                                      \
    uint32_t __port = __printf_print_start(__fmt);       \
    size_t __spec_loc = 0;      \
    (void)__spec_loc;                                    \
    __PRINTF_DISPATCH_ARGS(__fmt, UNUSED, ##__VA_ARGS__) \
    __printf_print_end(__port);                          \
    }

// Functions implemented out of C header. printf resolves to multiple calls to
// these. Some implemented on gcn. All should probably be implemented on
// gcn/ptx/x64
__PRINTF_API_EXTERNAL uint32_t __printf_print_start(const char *fmt);
__PRINTF_API_EXTERNAL int __printf_print_end(uint32_t port);

// simple types
__PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_int32(uint32_t port,
                                                       int32_t x);
__PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_uint32(uint32_t port,
                                                        uint32_t x);
__PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_int64(uint32_t port,
                                                       int64_t x);
__PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_uint64(uint32_t port,
                                                        uint64_t x);
__PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_double(uint32_t port,
                                                        double x);

// print the address of the argument on the gpu
__PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_void(uint32_t port,
                                                      const void *x);

// copy null terminated string starting at x, print the string
__PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_cstr(uint32_t port,
                                                      const char *x);

// implement %n specifier, may need one per sizeof target
__PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_write_int64(uint32_t port,
                                                             int64_t *x);


#define __PRINTF_PASTE_(X, Y) X##Y
#define __PRINTF_PASTE(X, Y) __PRINTF_PASTE_(X, Y)

#define __PRINTF_WRAP(FMT, POS, X)                                             \
  __spec_loc = __printf_next_specifier_location(__fmt, __strlen, __spec_loc);  \
  __printf_print_element(__port, __printf_specifier_classify(FMT, __spec_loc), \
                         X);
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
#define __PRINTF_WRAP10(FMT, U, X0, X1, X2, X3, X4, X5, X6, X7, X8) \
  __PRINTF_WRAP9(FMT, U, X0, X1, X2, X3, X4, X5, X6, X7)            \
  __PRINTF_WRAP(FMT, 8, X8)

#if 0
/*
 * I believe I first saw this trick on stack overflow. Possibly at the
 * following:
 * http://stackoverflow.com/questions/11317474/macro-to-count-number-of-arguments
 * This is considered to be a standard preprocessor technique in common
 * knowledge.
/* 
 * Counting arguments with the preprocessor. Fairly standard technique, one place
 * attributed the original implementation to:
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
__PRINTF_API_INTERNAL hostrpc::port_t __printf_print_element(uint32_t port,
                                                  enum __printf_spec_t spec,
                                                  int x)
{
  // (printf)("hit L%u [%s]\n", __LINE__, __PRETTY_FUNCTION__);
  (void)spec;
  _Static_assert(sizeof(int) == sizeof(int32_t), "");
  return __printf_pass_element_int32(port, x);
}

__PRINTF_API_INTERNAL hostrpc::port_t __printf_print_element(uint32_t port,
                                                  enum __printf_spec_t spec,
                                                  unsigned x)
{
  // (printf)("hit L%u [%s]\n", __LINE__, __PRETTY_FUNCTION__);
  (void)spec;
  _Static_assert(sizeof(unsigned) == sizeof(uint32_t), "");
  return __printf_pass_element_uint32(port, x);
}

__PRINTF_API_INTERNAL hostrpc::port_t __printf_print_element(uint32_t port,
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
      return __printf_pass_element_int32(port, (int32_t)x);
    }
  else if (sizeof(long) == sizeof(int64_t))
    {
      return __printf_pass_element_int64(port, (int64_t)x);
    }
}

__PRINTF_API_INTERNAL hostrpc::port_t __printf_print_element(uint32_t port,
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
      return __printf_pass_element_uint32(port, (uint32_t)x);
    }
  else if (sizeof(unsigned long) == sizeof(uint64_t))
    {
      return __printf_pass_element_uint64(port, (uint64_t)x);
    }
}

__PRINTF_API_INTERNAL hostrpc::port_t __printf_print_element(uint32_t port,
                                                  enum __printf_spec_t spec,
                                                  long long x)
{
  // (printf)("hit L%u [%s]\n", __LINE__, __PRETTY_FUNCTION__);
  (void)spec;
  _Static_assert(sizeof(long long) == sizeof(int64_t), "");
  return __printf_pass_element_int64(port, x);
}

__PRINTF_API_INTERNAL hostrpc::port_t __printf_print_element(uint32_t port,
                                                  enum __printf_spec_t spec,
                                                  unsigned long long x)
{
  // (printf)("hit L%u [%s]\n", __LINE__, __PRETTY_FUNCTION__);
  (void)spec;
  _Static_assert(sizeof(unsigned long long) == sizeof(uint64_t), "");
  return __printf_pass_element_uint64(port, x);
}

__PRINTF_API_INTERNAL hostrpc::port_t __printf_print_element(uint32_t port,
                                                  enum __printf_spec_t spec,
                                                  double x)
{
  // (printf)("hit L%u [%s]\n", __LINE__, __PRETTY_FUNCTION__);
  (void)spec;
  return __printf_pass_element_double(port, x);
}

// char* and void* check the format string to distinguish copy string vs pointer
// signed char* can also used with %n
__PRINTF_API_INTERNAL hostrpc::port_t __printf_print_element(uint32_t port,
                                                  enum __printf_spec_t spec,
                                                  const void *x)
{
  // (printf)("hit L%u [%s]\n", __LINE__, __PRETTY_FUNCTION__);
  switch (spec)
    {
      case spec_string:
        return __printf_pass_element_cstr(port, (const char *)x);
      case spec_normal:
        return __printf_pass_element_void(port, (const void *)x);
      case spec_output:
        {
          // This is somewhat dubious.
          // In C, a string literal has type char[]
          // Overloading works on const char* or char*, but clang
          // considers a call with a "literal" ambiguous
          // Bug https://bugs.llvm.org/show_bug.cgi?id=49978
          // Preference would be to instantiate on char* and
          // const char*, where the latter ignores %n
          // For present, passing a const char* to write to via
          // printf is UB anyway, so assume the argument is actually
          // a mutable char * to an int64_t that has been cast
          int64_t tmp;
          auto r =__printf_pass_element_write_int64(port, &tmp);
          *(int64_t *)x =
              tmp;  // todo: can't assume this, doesn't work for %n & some-char
          return r; // todo: deal with move
        }
      case spec_none:
        return static_cast<hostrpc::port_t>(port);
    }
}

// todo: can these be patched directly to write int64
#if 1
__PRINTF_API_INTERNAL hostrpc::port_t __printf_print_element(uint32_t port,
                                                  enum __printf_spec_t spec,
                                                  signed short *x)
{
  (void)spec;
  // (printf)("hit L%u [%s]", __LINE__, __PRETTY_FUNCTION__);
  int64_t tmp = 0;
  auto r = __printf_print_element(port, spec, (const char *)&tmp);
  *x = (signed short)tmp;
  return r;
}

__PRINTF_API_INTERNAL hostrpc::port_t __printf_print_element(uint32_t port,
                                                  enum __printf_spec_t spec,
                                                  int *x)
{
  (void)spec;
  // (printf)("hit L%u [%s]", __LINE__, __PRETTY_FUNCTION__);
  int64_t tmp = 0;
  auto r =__printf_print_element(port, spec, (const char *)&tmp);
  *x = (int)tmp;
  return r;
}

__PRINTF_API_INTERNAL hostrpc::port_t __printf_print_element(uint32_t port,
                                                  enum __printf_spec_t spec,
                                                  long *x)
{
  (void)spec;
  // (printf)("hit L%u [%s]", __LINE__, __PRETTY_FUNCTION__);
  int64_t tmp = 0;
  auto r = __printf_print_element(port, spec, (const char *)&tmp);
  *x = (long)tmp;
  return r;
}

__PRINTF_API_INTERNAL hostrpc::port_t __printf_print_element(uint32_t port,
                                                  enum __printf_spec_t spec,
                                                  size_t *x)
{
  (void)spec;
  // (printf)("hit L%u [%s]", __LINE__, __PRETTY_FUNCTION__);
  int64_t tmp = 0;
  auto r = __printf_print_element(port, spec, (const char *)&tmp);
  *x = (size_t)tmp;
  return r;
}

__PRINTF_API_INTERNAL hostrpc::port_t __printf_print_element(uint32_t port,
                                                  enum __printf_spec_t spec,
                                                  long long *x)
{
  (void)spec;
  // (printf)("hit L%u [%s]", __LINE__, __PRETTY_FUNCTION__);
  int64_t tmp = 0;
  auto r = __printf_print_element(port, spec, (const char *)&tmp);
  *x = (long long)tmp;
  return r;
}
#endif

#define __PRINTF_DISPATCH_INDIRECT(TYPE, VIA)           \
  __PRINTF_API_INTERNAL hostrpc::port_t __printf_print_element(    \
      uint32_t port, enum __printf_spec_t spec, TYPE x) \
  {                                                     \
    return __printf_print_element(port, spec, (VIA)x);         \
  }

// Implement types that undergo argument promotion as promotions

__PRINTF_DISPATCH_INDIRECT(bool, int)
__PRINTF_DISPATCH_INDIRECT(signed char, int)
__PRINTF_DISPATCH_INDIRECT(unsigned char, unsigned int)
__PRINTF_DISPATCH_INDIRECT(short, int)
__PRINTF_DISPATCH_INDIRECT(unsigned short, unsigned int)
__PRINTF_DISPATCH_INDIRECT(float, double)

// Pointers take behaviour from the format string (%s/%p/%n), so redirect
// signed/unsigned/void via the const void* which contains the switch
__PRINTF_DISPATCH_INDIRECT(const char *, const void *)
__PRINTF_DISPATCH_INDIRECT(const signed char *, const void *)
__PRINTF_DISPATCH_INDIRECT(const unsigned char *, const void *)

#undef __PRINTF_DISPATCH_INDIRECT

// because __builtin_strlen resolves to strlen, which amdgcn does not lower
// todo: check if memcpy on unknown length has the same problem
__PRINTF_API_INTERNAL size_t __printf_strlen(const char *str)
{
  // unreliable at O0
  if (__builtin_constant_p(str))
    {
      return __builtin_strlen(str);
    }
  else
    {
      for (size_t i = 0;; i++)
        {
          if (str[i] == '\0')
            {
              return i;
            }
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

static uint32_t __printf_haystack(char a, char b, char c, char d)
{
  uint32_t res = (a << 0u) | (b << 8u) | (c << 16u) | (d << 24u);
  return res;
}

__PRINTF_API_INTERNAL bool __printf_conversion_specifier_p(char c)
{
  // excluding % which is handled elsewhere
  unsigned char uc;
  __builtin_memcpy(&uc, &c, 1);

  // 32 bit codegen is noticably better for amdgcn and ~ same as 64 bit for x64
  // Six characters to check, so checks 'h' repeatedly to avoid a zero.
  uint32_t broadcast = UINT32_C(0x01010101) * uc;
  uint32_t A = __printf_haystack('c', 's', 'd', 'i');
  uint32_t B = __printf_haystack('o', 'x', 'X', 'u');
  uint32_t C = __printf_haystack('f', 'F', 'e', 'E');
  uint32_t D = __printf_haystack('a', 'A', 'g', 'G');
  uint32_t E = __printf_haystack('n', 'p', 'n', 'p');

  // Works, but can probably be optimised further
  return (int/*suppress spurious warning*/)__printf_haszero(broadcast ^ A) | __printf_haszero(broadcast ^ B) |
         __printf_haszero(broadcast ^ C) | __printf_haszero(broadcast ^ D) |
         __printf_haszero(broadcast ^ E);
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
      if (__printf_conversion_specifier_p(format[o]))
        {
          return o;
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

#endif
