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

    enum spec_t {
      spec_normal,
      spec_string,
      spec_output,
      spec_none,
    };

const char *spec_str(enum spec_t s)
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

enum spec_t next_specifier(const char *input_format, size_t *input_offset)
{
  const bool verbose = false;

  size_t offset = *input_offset;
  const char *format = &input_format[offset];

  for (;;)
    {
      switch (*format)
        {
          case '\0':
            {
              if (verbose)
                {
                  printf("str %s offset %zu is '0\n", format, offset);
                }
              *input_offset = offset;
              return spec_none;
            }
          case '%':
            {
              offset++;
              format++;
              if (*format == '%')
                {
                  // not a specifier after all
                  format++;
                  offset++;
                  break;
                }

              while (length_modifier_p(*format))
                {
                  format++;
                  offset++;
                }

              switch (*format)
                {
                  case '\0':
                    {
                      // %'\0 ill formed
                      *input_offset = offset;
                      return spec_none;
                    }
                  case 's':
                    {
                      format++;
                      offset++;
                      *input_offset = offset;
                      return spec_string;
                    }
                  case 'n':
                    {
                      format++;
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

__attribute__((overloadable)) void piecewise_print_element(enum spec_t spec,
                                                           int x)
{
  printf("%d", x);
}

__attribute__((overloadable)) void piecewise_print_element(enum spec_t spec,
                                                           unsigned x)
{
  printf("%u", x);
}

__attribute__((overloadable)) void piecewise_print_element(enum spec_t spec,
                                                           long x)
{
  printf("%ld", x);
}

__attribute__((overloadable)) void piecewise_print_element(enum spec_t spec,
                                                           long long x)
{
  printf("%lld", x);
}

__attribute__((overloadable)) void piecewise_print_element(enum spec_t spec,
                                                           unsigned long x)
{
  printf("%lu", x);
}

__attribute__((overloadable)) void piecewise_print_element(enum spec_t spec,
                                                           unsigned long long x)
{
  printf("%llu", x);
}

__attribute__((overloadable)) void piecewise_print_element(enum spec_t spec,
                                                           double x)
{
  printf("%f", x);
}

// Need the complete overload set in scope or implicit casts to void cause self
// recursion
__attribute__((overloadable)) void piecewise_print_element(enum spec_t spec,
                                                           const char *x);
__attribute__((overloadable)) void piecewise_print_element(
    enum spec_t spec, const signed char *x);
__attribute__((overloadable)) void piecewise_print_element(
    enum spec_t spec, const unsigned char *x);
__attribute__((overloadable)) void piecewise_print_element(enum spec_t spec,
                                                           const void *x);

__attribute__((overloadable)) void piecewise_print_element(enum spec_t spec,
                                                           const char *x)
{
  // (printf)("In func %s/L%u\n", __func__,__LINE__);
  if (spec != spec_string)
    {
      piecewise_print_element(spec, (const void *)x);
    }
  else
    {
      printf("%%_S: %s", x);
    }
}

__attribute__((overloadable)) void piecewise_print_element(enum spec_t spec,
                                                           const signed char *x)
{
  // (printf)("In func %s/L%u\n", __func__,__LINE__);
  printf("%%sS: %s", x);
}

__attribute__((overloadable)) void piecewise_print_element(
    enum spec_t spec, const unsigned char *x)
{
  // (printf)("In func %s/L%u\n", __func__,__LINE__);
  printf("%%uS: %s", x);
}

__attribute__((overloadable)) void piecewise_print_element(enum spec_t spec,
                                                           const void *x)
{
  // (printf)("In func %s/L%u\n", __func__,__LINE__);
  if (spec == spec_string)
    {
      piecewise_print_element(spec, (const char *)x);
    }
  else
    {
      printf("%%P: %p", x);
    }
}

// Redundant types via argument promotion
__attribute__((overloadable)) void piecewise_print_element(enum spec_t spec,
                                                           bool x)
{
  piecewise_print_element(spec, (int)x);
}
__attribute__((overloadable)) void piecewise_print_element(enum spec_t spec,
                                                           signed char x)
{
  piecewise_print_element(spec, (int)x);
}
__attribute__((overloadable)) void piecewise_print_element(enum spec_t spec,
                                                           unsigned char x)
{
  piecewise_print_element(spec, (unsigned int)x);
}
__attribute__((overloadable)) void piecewise_print_element(enum spec_t spec,
                                                           short x)
{
  piecewise_print_element(spec, (int)x);
}
__attribute__((overloadable)) void piecewise_print_element(enum spec_t spec,
                                                           unsigned short x)
{
  piecewise_print_element(spec, (unsigned int)x);
}
__attribute__((overloadable)) void piecewise_print_element(enum spec_t spec,
                                                           float x)
{
  piecewise_print_element(spec, (double)x);
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

#define printf(FMT, ...)                                                       \
  {                                                                            \
    size_t offset = 0;                                                         \
    (void)offset;                                                              \
    uint32_t __port = piecewise_print_start(FMT);                              \
    __lib_printf_args(FMT, UNUSED, ##__VA_ARGS__) piecewise_print_end(__port); \
  }

#define WRAP(FMT, X) piecewise_print_element(next_specifier(FMT, &offset), X);
#define WRAP1(FMT, U)
#define WRAP2(FMT, U, X) WRAP(FMT, X)
#define WRAP3(FMT, U, X, Y) WRAP2(FMT, U, X) WRAP(FMT, Y)
#define WRAP4(FMT, U, X, Y, Z) WRAP3(FMT, U, X, Y) WRAP(FMT, Z)

#define __lib_printf_args(FMT, ...) \
  PASTE(WRAP, PP_NARG(__VA_ARGS__))(FMT, __VA_ARGS__)

static enum spec_t next_spec(const char *format)
{
  size_t offset = 0;
  return next_specifier(format, &offset);
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
    CHECK(next_specifier("", &offset) == spec_none);
    CHECK(offset == 0);

    // single %
    offset = 0;
    CHECK(next_specifier("%", &offset) == spec_none);
    CHECK(offset == 1);
    CHECK(next_specifier("%", &offset) == spec_none);
    CHECK(offset == 1);

    // literal
    offset = 0;
    CHECK(next_specifier("%%", &offset) == spec_none);
    CHECK(offset == 2);
    CHECK(next_specifier("%%", &offset) == spec_none);
    CHECK(offset == 2);

    // string
    offset = 0;
    CHECK(next_specifier("%s", &offset) == spec_string);
    CHECK(offset == 2);
    CHECK(next_specifier("%s", &offset) == spec_none);
    CHECK(offset == 2);
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
    {
      size_t offset = 0;
      enum spec_t s = next_specifier("str %s", &offset);
      (printf)("calling print_element on %s, moved offset to %zu\n",
               spec_str(s), offset);
    }
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
