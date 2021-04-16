#ifndef PRINTF_H_INCLUDED
#define PRINTF_H_INCLUDED

#ifdef PRECOMPILE
#define TMP #include "../../EvilUnit/EvilUnit.h"
TMP
#undef TMP
#else
#include "../../EvilUnit/EvilUnit.h"
#endif

#ifdef PRECOMPILE
#define TMP #include <stdbool.h>
    TMP
#undef TMP
#define TMP #include <stddef.h>
        TMP
#undef TMP
#define TMP #include <stdint.h>
            TMP
#undef TMP
#define TMP #include <stdio.h>
                TMP
#undef TMP
#define TMP #include <assert.h>
                    TMP
#undef TMP
#else
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#endif

// printf call rewritten on the gpu to calls to these

#define API __attribute__((noinline))

                        API uint32_t
                        piecewise_print_start(const char *fmt);
API int piecewise_print_end(uint32_t port);

// simple types
API void piecewise_pass_element_int32(uint32_t port, int32_t x);
API void piecewise_pass_element_uint32(uint32_t port, uint32_t x);
API void piecewise_pass_element_int64(uint32_t port, int64_t x);
API void piecewise_pass_element_uint64(uint32_t port, uint64_t x);
API void piecewise_pass_element_double(uint32_t port, double x);

// copy null terminated string starting at x, print the string
API void piecewise_pass_element_cstr(uint32_t port, const char *x);
// print the address of the argument on the gpu
API void piecewise_pass_element_void(uint32_t port, const void *x);

// implement %n specifier
API void piecewise_pass_element_write_int32(uint32_t port, int32_t *x);
API void piecewise_pass_element_write_int64(uint32_t port, int64_t *x);

enum spec_t
{
  spec_normal,
  spec_string,
  spec_output,
  spec_none,
};

static __attribute__((unused)) const char *spec_str(enum spec_t s)
{
  switch (s)
    {
      case spec_normal:
        return "spec_normal";
      case spec_string:
        return "spec_string";
      case spec_output:
        return "spec_output";
      case spec_none:
        return "spec_none";
    }
}

static bool length_modifier_p(char c)
{
  switch (c)
    {
      case 'h':
      case 'l':
      case 'j':
      case 'z':
      case 't':
      case 'L':
        return true;
      default:
        return false;
    }
}

bool pred(int i)
{
  uint8_t hash = (uint8_t)(i * ((i ^ 218) + 1));
  return hash == 0;
}

void scan(void)
{
  for (int i = 0; i < 256; i++)
    {
      if (pred(i))
        {
          printf("Hit at i = %u\n", i);
        }
    }
}

static size_t find_perc(const char *format, size_t len, size_t from)
{
  printf("Got %zu %zu\n", len, from);
  for (size_t o = from; o < len; o++)
    {
      if (format[o] == '%')
        {
          return o;
        }
    }
  return SIZE_MAX;
}

__attribute__((always_inline)) enum spec_t next_specifier(const char *format,
                                                          size_t len,
                                                          size_t *input_offset)
{
  const bool verbose = false;

  (printf)("got length %zu\n", len);

  size_t perc = find_perc(format, len, *input_offset);
  if (perc == SIZE_MAX)
    {
      return spec_none;
    }

  size_t offset = perc;

  offset++;

  (printf)("got offset %zu\n", offset);

  {
    if (format[offset] == '%')
      {
        // not a specifier after all
        offset++;
        *input_offset = offset;
        return next_specifier(format, len, input_offset);
      }

    while (length_modifier_p(format[offset]))
      {
        offset++;
      }

    switch (format[offset])
      {
        case '\0':
          {
            // %'\0 ill formed
            *input_offset = offset;
            return spec_none;
          }
        case 's':
          {
            offset++;
            *input_offset = offset;
            return spec_string;
          }
        case 'n':
          {
            offset++;
            *input_offset = offset;
            return spec_output;
          }
      }

    if (verbose)
      {
        printf("str %s offset %zu is %%\n", format, offset);
      }
    *input_offset = offset;
    return spec_normal;
  }
}

__attribute__((always_inline)) enum spec_t specifier_classify(
    const char *format, size_t loc)
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

bool is_nonperc_conversion_specifier(char c)
{
  switch (c)
    {
      case 'c':
      case 's':
      case 'd':
      case 'i':
      case 'x':
      case 'X':
      case 'u':
      case 'f':
      case 'F':
      case 'e':
      case 'E':
      case 'a':
      case 'A':
      case 'g':
      case 'G':
      case 'n':
      case 'p':
        return true;
      default:
        return false;
    }
}

__attribute__((always_inline)) size_t next_specifier_location(
    const char *format, size_t len, size_t input_offset)
{
  // more heuristics than one would like, but probably viable
  // c++ version can be totally robust
  if (__builtin_constant_p(len) && (len < 64))
    {
#pragma unroll 16
      for (size_t o = input_offset; o < len; o++)
        {
          if (format[o] == '%')
            {
              o++;
              if (format[o] == '%')
                {
                  continue;
                }

              for (; o < len; o++)
                {
                  if (is_nonperc_conversion_specifier(format[o]))
                    {
                      return o;
                    }
                }
            }
        }
    }
  else
    {
      for (size_t o = input_offset; o < len; o++)
        {
          if (format[o] == '%')
            {
              o++;
              if (format[o] == '%')
                {
                  continue;
                }

              for (; o < len; o++)
                {
                  if (is_nonperc_conversion_specifier(format[o]))
                    {
                      return o;
                    }
                }
            }
        }
    }

  return len;
}

__attribute__((always_inline)) size_t nth_specifier_location(const char *format,
                                                             size_t N)
{
  size_t len = __builtin_strlen(format);

  size_t loc = next_specifier_location(format, len, 0);

#pragma unroll 2
  while (N > 0)
    {
      loc = next_specifier_location(format, len, loc);
      N--;
    }

  return loc;
}

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

#define PP_ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, \
                 _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26,  \
                 _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38,  \
                 _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50,  \
                 _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61, _62,  \
                 _63, N, ...)                                                 \
  N

#define PP_RSEQ_N()                                                           \
  63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, \
      44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, \
      26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9,  \
      8, 7, 6, 5, 4, 3, 2, 1, 0

#define PP_NARG_(...) PP_ARG_N(__VA_ARGS__)

#define PP_NARG(...) PP_NARG_(__VA_ARGS__, PP_RSEQ_N())

#define __PRINTF_DISPATCH \
  __attribute__((overloadable)) __attribute__((unused)) static inline

// Straightforward mapping from integer/double onto the lower calls
__PRINTF_DISPATCH void piecewise_print_element(uint32_t port, enum spec_t spec,
                                               int x)
{
  // (printf)("hit L%u [%s]\n", __LINE__, __PRETTY_FUNCTION__);
  (void)spec;
  _Static_assert(sizeof(int) == sizeof(int32_t), "");
  piecewise_pass_element_int32(port, x);
}

__PRINTF_DISPATCH void piecewise_print_element(uint32_t port, enum spec_t spec,
                                               unsigned x)
{
  // (printf)("hit L%u [%s]\n", __LINE__, __PRETTY_FUNCTION__);
  (void)spec;
  _Static_assert(sizeof(unsigned) == sizeof(uint32_t), "");
  piecewise_pass_element_uint32(port, x);
}

__PRINTF_DISPATCH void piecewise_print_element(uint32_t port, enum spec_t spec,
                                               long x)
{
  // (printf)("hit L%u [%s]\n", __LINE__, __PRETTY_FUNCTION__);
  (void)spec;
  _Static_assert(
      (sizeof(long) == sizeof(int32_t)) || (sizeof(long) == sizeof(int64_t)),
      "");
  if (sizeof(long) == sizeof(int32_t))
    {
      piecewise_pass_element_int32(port, x);
    }
  if (sizeof(long) == sizeof(int64_t))
    {
      piecewise_pass_element_int64(port, x);
    }
}

__PRINTF_DISPATCH void piecewise_print_element(uint32_t port, enum spec_t spec,
                                               unsigned long x)
{
  // (printf)("hit L%u [%s]\n", __LINE__, __PRETTY_FUNCTION__);
  (void)spec;
  _Static_assert((sizeof(unsigned long) == sizeof(uint32_t)) ||
                     (sizeof(unsigned long) == sizeof(uint64_t)),
                 "");
  if (sizeof(unsigned long) == sizeof(uint32_t))
    {
      piecewise_pass_element_uint32(port, x);
    }
  if (sizeof(unsigned long) == sizeof(uint64_t))
    {
      piecewise_pass_element_uint64(port, x);
    }
}

__PRINTF_DISPATCH void piecewise_print_element(uint32_t port, enum spec_t spec,
                                               long long x)
{
  // (printf)("hit L%u [%s]\n", __LINE__, __PRETTY_FUNCTION__);
  (void)spec;
  _Static_assert(sizeof(long long) == sizeof(int64_t), "");
  piecewise_pass_element_int64(port, x);
}

__PRINTF_DISPATCH void piecewise_print_element(uint32_t port, enum spec_t spec,
                                               unsigned long long x)
{
  // (printf)("hit L%u [%s]\n", __LINE__, __PRETTY_FUNCTION__);
  (void)spec;
  _Static_assert(sizeof(unsigned long long) == sizeof(uint64_t), "");
  piecewise_pass_element_uint64(port, x);
}

__PRINTF_DISPATCH void piecewise_print_element(uint32_t port, enum spec_t spec,
                                               double x)
{
  // (printf)("hit L%u [%s]\n", __LINE__, __PRETTY_FUNCTION__);
  (void)spec;
  piecewise_pass_element_double(port, x);
}

// char* and void* check the format string to distinguish copy string vs pointer
// signed char* can also used with %n
__PRINTF_DISPATCH void piecewise_print_element(uint32_t port, enum spec_t spec,
                                               const char *x)
{
  (printf)("hit L%u [%s]\n", __LINE__, __PRETTY_FUNCTION__);
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

__PRINTF_DISPATCH void piecewise_print_element(uint32_t port, enum spec_t spec,
                                               signed short *x)
{
  (printf)("hit L%u [%s]", __LINE__, __PRETTY_FUNCTION__);
  int32_t tmp;
  piecewise_pass_element_write_int32(port, &tmp);
  *x = (signed short)tmp;
}

__PRINTF_DISPATCH void piecewise_print_element(uint32_t port, enum spec_t spec,
                                               int *x)
{
  (printf)("hit L%u [%s]", __LINE__, __PRETTY_FUNCTION__);
  _Static_assert(sizeof(int) == sizeof(int32_t), "");
  piecewise_pass_element_write_int32(port, x);
}

__PRINTF_DISPATCH void piecewise_print_element(uint32_t port, enum spec_t spec,
                                               long *x)
{
  (printf)("hit L%u [%s]", __LINE__, __PRETTY_FUNCTION__);

  if (sizeof(long) == sizeof(int32_t))
    {
      piecewise_pass_element_write_int32(port, (int32_t *)x);
    }
  if (sizeof(long) == sizeof(int64_t))
    {
      piecewise_pass_element_write_int64(port, (int64_t *)x);
    }
}

__PRINTF_DISPATCH void piecewise_print_element(uint32_t port, enum spec_t spec,
                                               size_t *x)
{
  (printf)("hit L%u [%s]", __LINE__, __PRETTY_FUNCTION__);

  if (sizeof(size_t) == sizeof(int32_t))
    {
      piecewise_pass_element_write_int32(port, (int32_t *)x);
    }
  if (sizeof(size_t) == sizeof(int64_t))
    {
      piecewise_pass_element_write_int64(port, (int64_t *)x);
    }
}

__PRINTF_DISPATCH void piecewise_print_element(uint32_t port, enum spec_t spec,
                                               long long *x)
{
  (printf)("hit L%u [%s]", __LINE__, __PRETTY_FUNCTION__);
  _Static_assert(sizeof(long long) == sizeof(int64_t), "");
  piecewise_pass_element_write_int64(port, (int64_t *)x);
}

#define __PRINTF_DISPATCH_INDIRECT(TYPE, VIA)                              \
  __PRINTF_DISPATCH void piecewise_print_element(uint32_t port,            \
                                                 enum spec_t spec, TYPE x) \
  {                                                                        \
    piecewise_print_element(port, spec, (VIA)x);                           \
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
#undef __PRINTF_DISPATCH

#define PASTE_(X, Y) X##Y
#define PASTE(X, Y) PASTE_(X, Y)

#define printf(FMT, ...)                                                       \
  {                                                                            \
    size_t offset = 0;                                                         \
    (void)offset;                                                              \
    uint32_t __port = piecewise_print_start(FMT);                              \
    __lib_printf_args(FMT, UNUSED, ##__VA_ARGS__) piecewise_print_end(__port); \
  }

#define WRAP(FMT, X)       \
  piecewise_print_element( \
      __port, next_specifier(FMT, __builtin_strlen(FMT), &offset), X);
#define WRAP1(FMT, U)
#define WRAP2(FMT, U, X) WRAP(FMT, X)
#define WRAP3(FMT, U, X, Y) WRAP2(FMT, U, X) WRAP(FMT, Y)
#define WRAP4(FMT, U, X, Y, Z) WRAP3(FMT, U, X, Y) WRAP(FMT, Z)

#define __lib_printf_args(FMT, ...) \
  PASTE(WRAP, PP_NARG(__VA_ARGS__))(FMT, __VA_ARGS__)

static enum spec_t next_spec(const char *format)
{
  size_t offset = 0;
  return next_specifier(format, __builtin_strlen(format), &offset);
}

MODULE(format)
{
  TEST("count")
  {
    CHECK(next_spec("%") == spec_none);
    CHECK(next_spec("%a") == spec_normal);
    CHECK(next_spec("a%a") == spec_normal);
    CHECK(next_spec("a%a") == spec_normal);
  }

  TEST("literal %")
  {
    CHECK(next_spec("%%") == spec_none);
    CHECK(next_spec(" %%") == spec_none);
  }

  TEST("string literal")
  {
    CHECK(next_spec("%s") == spec_string);
    CHECK(next_spec("%ls") == spec_string);
    CHECK(next_spec(" %s") == spec_string);
    CHECK(next_spec(" %ls") == spec_string);
    CHECK(next_spec("s%s") == spec_string);
    CHECK(next_spec("s%ls") == spec_string);
    CHECK(next_spec("s%%s") == spec_none);
    CHECK(next_spec("s%%ls") == spec_none);
  }

  TEST("output")
  {
    CHECK(next_spec("%hhn") == spec_output);
    CHECK(next_spec("%hn") == spec_output);
    CHECK(next_spec("%n") == spec_output);
    CHECK(next_spec("%ln") == spec_output);
    CHECK(next_spec("%lln") == spec_output);
    CHECK(next_spec("%jn") == spec_output);
    CHECK(next_spec("%zn") == spec_output);
    CHECK(next_spec("%tn") == spec_output);
  }

  TEST("check stays within bounds")
  {
    size_t offset = 0;

    // empty
    offset = 0;
    CHECK(next_specifier("", 0, &offset) == spec_none);
    CHECK(offset == 0);

    // single %
    offset = 0;
    CHECK(next_specifier("%", __builtin_strlen("%"), &offset) == spec_none);
    CHECK(offset == 1);
    CHECK(next_specifier("%", __builtin_strlen("%"), &offset) == spec_none);
    CHECK(offset == 1);

    // literal
    offset = 0;
    CHECK(next_specifier("%%", __builtin_strlen("%%"), &offset) == spec_none);
    CHECK(offset == 2);
    CHECK(next_specifier("%%", __builtin_strlen("%%"), &offset) == spec_none);
    CHECK(offset == 2);

    // string
    offset = 0;
    CHECK(next_specifier("%s", __builtin_strlen("%s"), &offset) == spec_string);
    CHECK(offset == 2);
    CHECK(next_specifier("%s", __builtin_strlen("%s"), &offset) == spec_none);
    CHECK(offset == 2);
  }
}

void codegenA(uint32_t __port)
{
  size_t offset = 0;

  const char *fmt = "flt %g %d";

  size_t first = next_specifier_location(fmt, __builtin_strlen(fmt), 0);
  size_t second = next_specifier_location(fmt, __builtin_strlen(fmt), first);

  (printf)("size %zu, first %zu, second %zu\n", __builtin_strlen(fmt), first,
           second);
}

void codegen(uint32_t __port)
{
  const char *fmt = "flst %g %d %s longer!!!!";

  (printf)("size %zu, first %zu, second %zu, third %zu\n",
           __builtin_strlen(fmt), nth_specifier_location(fmt, 0),
           nth_specifier_location(fmt, 1), nth_specifier_location(fmt, 2)

  );
}

EVILUNIT_MAIN_MODULE()
{
  DEPENDS(format);

  TEST("ex")
  {
#define LIFE 43
    // printf(); // error: to few arguments too function call, single argument
    // 'fmt' not specified
    printf("str");
    printf("uint %u", 42);
    printf("int %d", LIFE);
    printf("flt %g %d", 3.14, 81);
    printf("str %s", "strings");

    void *p = (void *)0;
    printf("void1 %p", p);

    printf("void2 %p", (void *)4);

    char mutable[8] = "mutable";

    printf("fmt ptr %p take ptr", p);
    printf("fmt ptr %p take str", "careful");
    printf("fmt ptr %p take mut", mutable);
    printf("fmt str %s take ptr", (void *)"suspect");

    printf("fmt str %s take str", "good");
    printf("fmt str %s take cstr", (const char *)"const");
    printf("fmt str %s take mut", mutable);

    printf("fmt str %n mutable", mutable);

    printf("bool %u", false);
    printf("raw char %c", (char)11);
    printf("signed char %c", (signed char)12);
    printf("unsigned char %c", (unsigned char)13);

    printf("signed char int %hhd", (signed char)14);
    printf("signed char int %hhi", (signed char)15);
    printf("signed short int %hd", (signed short int)16);
    printf("signed short int %hi", (signed short int)17);
    printf("signed int %d", (signed int)18);
    printf("signed int %i", (signed int)19);
    printf("signed long %ld", (signed long)20);
    printf("signed long %li", (signed long)21);
    printf("signed long long %lld", (signed long long)22);
    printf("signed long long %lli", (signed long long)23);

    printf("unsigned char int %hhu", (unsigned char)24);
    printf("unsigned short int %hu", (unsigned short int)25);
    printf("unsigned int %u", (unsigned int)26);
    printf("unsigned long %lu", (unsigned long)27);
    printf("unsigned long long %llu", (unsigned long long)28);

    printf("intmax %jd", (intmax_t)29);
    printf("intmax %ji", (intmax_t)30);
    printf("size_t %zd", (size_t)31);
    printf("size_t %zi", (size_t)32);
    printf("ptrdiff %td", (ptrdiff_t)33);
    printf("ptrdiff %ti", (ptrdiff_t)34);

    printf("uintmax %jd", (uintmax_t)35);
    printf("size_t %zd", (size_t)36);
    printf("ptrdiff %td", (ptrdiff_t)37);

    printf("float %f", (float)3.1f);
    printf("double %lf", (double)4.1f);
  }

  TEST("long double")
  {
    // Postponing until checking what long double turns into on
    // amdgcn. May need to accept passing a 16 byte value.
    _Static_assert(sizeof(long double) == 2 * sizeof(double), "");
    // printf("long double %Lf", (long double)4.1f);
  }

  TEST("output")
  {
    {
      signed char x = 0;
      printf("%n sc", &x);
      CHECK(x == 0);
    }
    {
      short x = 0;
      printf("%n sh", &x);
      CHECK(x == 0);
    }
    {
      int x = 0;
      printf("%n it", &x);
      CHECK(x == 0);
    }
    {
      long x = 0;
      printf("%n lg", &x);
      CHECK(x == 0);
    }
    {
      long long x = 0;
      printf("%n ll ", &x);
      CHECK(x == 0);
    }
    {
      intmax_t x = 0;
      printf("%n im", &x);
      CHECK(x == 0);
    }
    {
      size_t x = 0;
      printf("%n st", &x);
      CHECK(x == 0);
    }
    {
      ptrdiff_t x = 0;
      printf("%n pd", &x);
      CHECK(x == 0);
    }
  }
}

#include "printf_stub.h"

#endif
