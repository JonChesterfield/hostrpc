#include "hostrpc_printf.h"

#include "detail/platform_detect.hpp"

#ifdef PRECOMPILE
#define TMP #define EVILUNIT_USE_STDIO 0
TMP
#undef TMP
#define TMP #include "EvilUnit.h"
    TMP
#undef TMP
#else
#define EVILUNIT_USE_STDIO 0
#include "EvilUnit.h"
#endif

// Functions to be implemented via hostrpc, or possibly as test stubs. Presently
// calls through the glibc printf, which works other than for %n
#if 0
#include "printf_stub.h"
#endif

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

#if 0
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
#endif

static inline bool is_master_lane(void);
#if HOSTRPC_HOST
static inline bool is_master_lane(void) { return true; }
#endif
#if HOSTRPC_AMDGCN
static inline uint32_t get_lane_id(void)
{
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}
static inline bool is_master_lane(void)
{
  // TODO: 32 wide wavefront, consider not using raw intrinsics here
  uint64_t activemask = __builtin_amdgcn_read_exec();

  // TODO: check codegen for trunc lowest_active vs expanding lane_id
  // TODO: ffs is lifted from openmp runtime, looks like it should be ctz
  uint32_t lowest_active = __builtin_ffsl(activemask) - 1;
  uint32_t lane_id = get_lane_id();

  // TODO: readfirstlane(lane_id) == lowest_active?
  return lane_id == lowest_active;
}
#endif

#include "printf_specifier.data"

EVILUNIT_MAIN_MODULE()
{
  DEPENDS(specifier);

  TEST("ex")
  {
    if (is_master_lane())
      {
#define LIFE 43
        // printf(); // error: to few arguments too function call, single
        // argument 'fmt' not specified
        printf("str\n");
        printf("uint %u\n", 42);
        printf("int %d\n", LIFE);
        printf("flt %g %d\n", 3.14, 81);
        printf("str %s", "strings\n");

        void *p = (void *)0;
        printf("void1 %p\n", p);

        printf("void2 %p\n", (void *)4);

        char mutable_buffer[8] = "mutable";

        printf("fmt ptr %p take ptr\n", p);
        printf("fmt ptr %p take str\n", "careful");
        printf("fmt ptr %p take mut\n", mutable_buffer);
        printf("fmt str %s take ptr\n", (void *)"suspect");

        printf("fmt str %s take str\n", "good");
        printf("fmt str %s take cstr\n", (const char *)"const");
        printf("fmt str %s take mut\n", mutable_buffer);

        printf("fmt str %n mutable\n", mutable_buffer);

        printf("bool %u\n", false);

        printf("raw char %c\n", (char)'a');
        printf("signed char %c\n", (signed char)'b');
        printf("unsigned char %c\n", (unsigned char)'c');

        printf("signed char int %hhd\n", (signed char)14);
        printf("signed char int %hhi\n", (signed char)15);
        printf("signed short int %hd\n", (signed short int)16);
        printf("signed short int %hi\n", (signed short int)17);
        printf("signed int %d\n", (signed int)18);
        printf("signed int %i\n", (signed int)19);
        printf("signed long %ld\n", (signed long)20);
        printf("signed long %li\n", (signed long)21);
        printf("signed long long %lld\n", (signed long long)22);
        printf("signed long long %lli\n", (signed long long)23);

        printf("unsigned char int %hhu\n", (unsigned char)24);
        printf("unsigned short int %hu\n", (unsigned short int)25);
        printf("unsigned int %u\n", (unsigned int)26);
        printf("unsigned long %lu\n", (unsigned long)27);
        printf("unsigned long long %llu\n", (unsigned long long)28);

        printf("intmax %jd\n", (intmax_t)29);
        printf("intmax %ji\n", (intmax_t)30);
        printf("size_t %zd\n", (size_t)31);
        printf("size_t %zi\n", (size_t)32);
        printf("ptrdiff %td\n", (ptrdiff_t)33);
        printf("ptrdiff %ti\n", (ptrdiff_t)34);

        printf("uintmax %jd\n", (uintmax_t)35);
        printf("size_t %zd\n", (size_t)36);
        printf("ptrdiff %td\n", (ptrdiff_t)37);

        printf("float %f\n", (float)3.1f);
        printf("double %lf\n", (double)4.1f);
      }
  }

  TEST("long double")
  {
    // x64 thinks long double is 16 bytes
    // gcn thinks long double i 8 bytes
    // so, that seems broken
    //    _Static_assert(sizeof(long double) == 1 * sizeof(double), "");
    // _Static_assert(sizeof(long double) == 2 * sizeof(double), "");
    // printf("long double %Lf\n", (long double)4.1f);
  }

  TEST("output")
  {
    // if (is_master_lane())
    {
      {
        signed char x = 0;
        printf("%n sc\n", &x);
        // both these checks fail
        CHECK((int)x == 0); // getting paranoid
        CHECK(x == 0);       
        printf("sc should be 0, got %d\n", (int)(x-1));
      }

      {
        signed char x = 0;
        printf(" %n sc\n", &x);
        CHECK(x == 1);
      }
      
      {
        short x = 0;
        printf("%n sh0\n", &x);
        CHECK(x == 0);
      }
      
      {
        short x = 0;
        printf(" %n sh\n", &x);
        CHECK(x == 1);
        // printf("x should be 1, got %d\n", (int)x);
      }
      {
        int x = 0;
        printf("  %n it\n", &x);
        CHECK(x == 2);
      }
      {
        long x = 0;
        printf("   %n lg\n", &x);
        CHECK(x == 3);
      }
      {
        long long x = 0;
        printf("    %n ll \n", &x);
        CHECK(x == 4);
      }
      {
        intmax_t x = 0;
        printf("     %n im\n", &x);
        CHECK(x == 5);
      }
      {
        size_t x = 0;
        printf("      %n st\n", &x);
        CHECK(x == 6);
      }
      {
        ptrdiff_t x = 0;
        printf("       %n pd\n", &x);
        CHECK(x == 7);
      }
    }
  }
}
