#include "printf.h"

#ifdef PRECOMPILE
#define TMP #include "../../EvilUnit/EvilUnit.h"
TMP
#undef TMP
#else
#include "../../EvilUnit/EvilUnit.h"
#endif

// Functions to be implemented via hostrpc, or possibly as test stubs. Presently
// calls through the glibc printf, which works other than for %n
#include "printf_stub.h"

static __attribute__((unused)) const char *
spec_str(enum __printf_spec_t s)
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

static bool length_modifier_p_slow(char c)
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

MODULE(length_modifier)
{
  TEST("compare")
  {
    for (unsigned i = 0; i < 256; i++)
      {
        unsigned char uc = (unsigned char)i;
        char tmp;
        __builtin_memcpy(&tmp, &uc, 1);
        CHECK(__printf_length_modifier_p(tmp) == length_modifier_p_slow(tmp));
      }
  }
}

static enum __printf_spec_t next_specifier(const char *format, size_t len,
                                  size_t *input_offset)
{
  size_t loc = __printf_next_specifier_location(format, len, *input_offset);
  *input_offset = loc;
  return __printf_specifier_classify(format, loc);
}

static enum __printf_spec_t next_spec(const char *format)
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
    CHECK(offset == 1);  // return offset of s
    CHECK(next_specifier("%s", __builtin_strlen("%s"), &offset) == spec_none);
    CHECK(offset == 2);
  }
}

void codegenA(uint32_t __port)
{
  const char *fmt = "flt %g %d";

  size_t first = __printf_next_specifier_location(fmt, __builtin_strlen(fmt), 0);
  size_t second = __printf_next_specifier_location(fmt, __builtin_strlen(fmt), first);

  (printf)("size %zu, first %zu, second %zu\n", __builtin_strlen(fmt), first,
           second);
}

#define EVILUNIT_ANSI_COLOUR_RED "\x1b[31m"
#define EVILUNIT_ANSI_COLOUR_GREEN "\x1b[32m"
#define EVILUNIT_ANSI_COLOUR_RESET "\x1b[0m"

void codegen_evilunit_pass(uint32_t failures, uint32_t checks,
                           const char *modulename)
{
  const char *fmt =
      "[ " EVILUNIT_ANSI_COLOUR_GREEN "Pass" EVILUNIT_ANSI_COLOUR_RESET
      " ]"
      "%u/%u %s\n";
  printf(fmt, failures, checks, modulename);
}

void codegen_evilunit_fail(uint32_t failures, uint32_t checks,
                           const char *modulename, const char *filename,
                           uint32_t line, const char *testname,
                           const char *check)
{
  printf("[ " EVILUNIT_ANSI_COLOUR_RED "Fail" EVILUNIT_ANSI_COLOUR_RESET
         " ] %u/%u %s %s(%u): \"%s\" %s\n",
         failures, checks, modulename, filename, line, testname, check);
}

#undef EVILUNIT_ANSI_COLOUR_RED
#undef EVILUNIT_ANSI_COLOUR_GREEN
#undef EVILUNIT_ANSI_COLOUR_RESET

EVILUNIT_MAIN_MODULE()
{
  DEPENDS(format);
  DEPENDS(length_modifier);

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
