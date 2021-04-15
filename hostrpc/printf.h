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
#define TMP #include <stdint.h>
    TMP
#undef TMP
#define TMP #include <stdio.h>
        TMP
#undef TMP
#define TMP #include <stdbool.h>
            TMP
#undef TMP
#else
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#endif

                size_t
                next_specifier(const char *input_format, size_t input_offset)
{
  const bool verbose = true;

  // offset assumed within nul terminated format
  if (input_offset == SIZE_MAX)
    {
      if (verbose)
        {
          printf("at limit, ret max\n");
          return SIZE_MAX;
        }
    }

  const char *format = &input_format[input_offset];
  size_t offset = input_offset;

  for (;;)
    {
      switch (*format)
        {
          case '\0':
            {
              printf("str %s offset %zu is '0\n", format, offset);
              return SIZE_MAX;
            }
          case '%':
            {
              if (format[1] == '%')
                {
                  format += 2;
                  offset += 2;
                  break;
                }
              else
                {
                  printf("str %s offset %zu is %%\n", format, offset);
                  return offset;
                }
            }

          default:
            {
              format++;
              offset++;
              break;
            }
        }
    }
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

__attribute__((overloadable)) void piecewise_print_element(int x)
{
  printf("%d", x);
}

__attribute__((overloadable)) void piecewise_print_element(unsigned x)
{
  printf("%u", x);
}

__attribute__((overloadable)) void piecewise_print_element(long x)
{
  printf("%ld", x);
}

__attribute__((overloadable)) void piecewise_print_element(long long x)
{
  printf("%lld", x);
}

__attribute__((overloadable)) void piecewise_print_element(unsigned long x)
{
  printf("%lu", x);
}

__attribute__((overloadable)) void piecewise_print_element(unsigned long long x)
{
  printf("%llu", x);
}

__attribute__((overloadable)) void piecewise_print_element(double x)
{
  printf("%f", x);
}

__attribute__((overloadable)) void piecewise_print_element(const void *x)
{
  printf("%%P: %p", x);
}

// Redundant types via argument promotion
__attribute__((overloadable)) void piecewise_print_element(bool x)
{
  piecewise_print_element((int)x);
}
__attribute__((overloadable)) void piecewise_print_element(signed char x)
{
  piecewise_print_element((int)x);
}
__attribute__((overloadable)) void piecewise_print_element(unsigned char x)
{
  piecewise_print_element((unsigned int)x);
}
__attribute__((overloadable)) void piecewise_print_element(short x)
{
  piecewise_print_element((int)x);
}
__attribute__((overloadable)) void piecewise_print_element(unsigned short x)
{
  piecewise_print_element((unsigned int)x);
}
__attribute__((overloadable)) void piecewise_print_element(float x)
{
  piecewise_print_element((double)x);
}

uint32_t piecewise_print_start(const char *fmt)
{
  printf("Start: %s ", fmt);
  return 101;
}

int piecewise_print_end(uint32_t port)
{
  (void)port;
  printf("\n");
  return 0;
}

void piecewise_print_pass_uint64(uint32_t port, uint64_t v);
void piecewise_print_pass_cstr(uint32_t port, const char *str);

#define PASTE_(X, Y) X##Y
#define PASTE(X, Y) PASTE_(X, Y)
#define EXPAND(X) X

#define printf(FMT, ...)                                                  \
  {                                                                       \
    uint32_t __port = piecewise_print_start(FMT);                         \
    __lib_printf_args(UNUSED, ##__VA_ARGS__) piecewise_print_end(__port); \
  }

#define WRAP(X) piecewise_print_element(X);
#define WRAP1(U)
#define WRAP2(U, X) WRAP(X)
#define WRAP3(U, X, Y) WRAP2(U, X) WRAP(Y)
#define WRAP4(U, X, Y, Z) WRAP3(U, X, Y) WRAP(Z)

#define __lib_printf_args(...) PASTE(WRAP, PP_NARG(__VA_ARGS__))(__VA_ARGS__)

MODULE(format)
{
  TEST("count")
  {
    CHECK(next_specifier("%", 0) == 0);
    CHECK(next_specifier("%a", 0) == 0);
    CHECK(next_specifier("a%a", 0) == 1);
    CHECK(next_specifier("a%a", 1) == 1);
    CHECK(next_specifier("a%a", 2) == SIZE_MAX);
  }

  TEST("literal %")
  {
    CHECK(next_specifier("%%", 0) == SIZE_MAX);
    CHECK(next_specifier(" %%", 0) == SIZE_MAX);

    // Strictly a parsing failure, but fine assuming iteration from start
    CHECK(next_specifier("%%", 1) == 1);
  }
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

    printf("fmt ptr %p take ptr", p);
    printf("fmt ptr %p take str", "careful");
    printf("fmt str %s take ptr", (void *)"suspect");
    printf("fmt str %s take str", "good");

    printf("raw char %c", (char)11);
    printf("signed char %c", (signed char)12);
    printf("unsigned char %c", (unsigned char)13);
  }
}

#endif
