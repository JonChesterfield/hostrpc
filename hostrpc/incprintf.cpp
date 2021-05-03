#include "EvilUnit.h"

#include <stdint.h>

#include <cassert>
#include <cstring>
#include <tuple>
#include <vector>

#include "hostrpc_printf.h"

#include "incprintf.hpp"

namespace
{
void append_bytes(const char *from, size_t N, incr &glob)
{
  assert(*glob.output.rbegin() == '\0');
  glob.output.pop_back();
  for (size_t i = 0; i < N; i++)
    {
      if (i != (N - 1))
        {
          if (from[i] == '%' && from[i + 1] == '%')
            {
              glob.output.push_back('%');
              i++;
              continue;
            }
        }

      glob.output.push_back(from[i]);
    }
  glob.output.push_back('\0');
}

void space_for_n_bytes(incr &glob, size_t bytes)
{
  for (size_t i = 0; i < bytes; i++)
    {
      glob.output.push_back('\0');
    }
}

void append_until_next_loc(incr &glob)
{
  char *fmt = glob.format.data();
  size_t len = glob.format.size();
  size_t next_start = __printf_next_start(fmt, len, glob.loc);

  if (next_start > glob.loc)
    {
      append_bytes(&fmt[glob.loc], next_start - glob.loc, glob);
      glob.loc = next_start;
    }
}

}  // namespace


namespace
{
struct inject_nul_in_format
{
  // write a \0 to the format string at position at,
  // reversing the write in the destructor
  char *loc;
  char previous;
  inject_nul_in_format(char *fmt, size_t at)
  {
    loc = &fmt[at];
    previous = *loc;
    *loc = '\0';
  }

  ~inject_nul_in_format() { *loc = previous; }
};

template <typename T>
size_t bytes_for_arg(bool verbose, char *fmt, size_t next_start, size_t next_end, T v)
{
  if (verbose) {
    (printf)("bytes_for_arg [%zu %zu] format:\n%s\n", next_start, next_end, fmt);
    if (next_start > 0) {
      (printf)("%*c", (int)(next_start -1), ' ');
    }
    (printf)("^");
    (printf)("\n");
  
  }
  
  

  size_t one_past = next_end + 1;
  inject_nul_in_format nul(fmt, one_past);

  if (verbose) {
    (printf)("Invoking sprintf on format %s, value %lu\n", &fmt[next_start], (uint64_t)v);
  }
  return snprintf(NULL, 0, &fmt[next_start], v);
}

}  // namespace




template <typename T>
void incr::piecewise_pass_element_T(T value)
{
  char *fmt = format.data();
  size_t len = format.size();
  size_t next_start = __printf_next_start(fmt, len, loc);
  size_t next_end = __printf_next_end(fmt, len, next_start);
  const bool verbose = false;
  if (verbose)
    (printf)("glob loc %zu, Format %s/%zu, split [%zu %zu]\n", loc, fmt,
             len, next_start, next_end);

  append_until_next_loc( *this);

  if (next_start == len)
    {
      return;
    }

  size_t bytes = bytes_for_arg(verbose, fmt, next_start, next_end, value);

  if (verbose)
    (printf)("output size %zu, increasing by %zu\n", output.size(), bytes);

  size_t output_end = output.size() - 1 /*trim off trailing nul*/;

  space_for_n_bytes(*this, bytes);

  {
    if (verbose)
      (printf)(
          "writing fmt %s into loc %zu, giving %zu bytes. Next end at %zu\n",
          &fmt[next_start], output_end, bytes, next_end);
    size_t one_past = next_end + 1;
    inject_nul_in_format nul(fmt, one_past);
    snprintf(output.data() + output_end,
             bytes + 1,  // overwrites the trailing nul that is already there
             &fmt[next_start], value);
  }

  loc = next_end + 1;
  if (verbose) (printf)("Setting loc to %zu\n", loc);
}

void incr::set_format(const char *fmt)
{
    loc = 0;
    format = {fmt, fmt + strlen(fmt)};
    output.push_back('\0');
}

std::vector<char> incr::finalize()
{
  append_until_next_loc( *this);
  return output;
}

static void piecewise_pass_element_uint64(uint64_t value,
                                          incr &glob)
{
  return glob.piecewise_pass_element_T( value);
}

// TODO: Audit list
template void incr::piecewise_pass_element_T(const char *);
template void incr::piecewise_pass_element_T(const void *);
template void incr::piecewise_pass_element_T(char);
template void incr::piecewise_pass_element_T(signed char);
template void incr::piecewise_pass_element_T(unsigned char);

template void incr::piecewise_pass_element_T(short);
template void incr::piecewise_pass_element_T(int);
template void incr::piecewise_pass_element_T(long);
template void incr::piecewise_pass_element_T(long long);

template void incr::piecewise_pass_element_T(unsigned short);
template void incr::piecewise_pass_element_T(unsigned int);
template void incr::piecewise_pass_element_T(unsigned long);
template void incr::piecewise_pass_element_T(unsigned long long);


#include "printf_specifier.data"

static MODULE(writeint)
{
  TEST("null string")
  {
    int data = 10;
    snprintf(NULL, 0, "chars %n\n", &data);
    CHECK(data == 6);
  }
}

static size_t nsl(const char *format, size_t len, size_t input_offset)
{
  return __printf_next_specifier_location(format, len, input_offset);
}

static MODULE(next_spec_loc)
{
  TEST("no format") { CHECK(0 == nsl("", 0, 0)); }

  TEST("skip second perc") { CHECK(4 == nsl("s%%s", 4, 0)); }
}

void dump(std::vector<char> v)
{
  (printf)("%zu: ", v.size());
  for (size_t i = 0; i < v.size(); i++)
    {
      if (v[i] == '\0')
        {
          if (i == v.size() - 1)
            {
              continue;
            }
        }
      (printf)("%c", v[i]);
    }
  (printf)("\n");
}

std::vector<char> no_format(const char *str)
{
  incr tmp;
  tmp.set_format(str);
  return tmp.finalize();
}

static MODULE(incr)
{
  TEST("no format")
  {
    static const char *cases[] = {
        "", "a", "bc", "\t", " -- ",
    };
    constexpr size_t N = sizeof(cases) / sizeof(cases[0]);

    for (size_t i = 0; i < N; i++)
      {
        auto r = no_format(cases[i]);
        CHECK(r.size() == strlen(cases[i]) + 1);
        CHECK(strcmp(r.data(), cases[i]) == 0);
      }
  }

  TEST("single int")
  {
    std::tuple<const char *, uint64_t, const char *> cases[] = {
        {"%lu", 10, "10"},       {" %lu", 20, " 20"},  {"%lu ", 30, "30 "},
        {" %lu ", 40, " 40 "},   {"%%lu", 100, "%lu"}, {"%%%lu", 100, "%100"},
        {"%%%%lu", 100, "%%lu"},
    };
    constexpr size_t N = sizeof(cases) / sizeof(cases[0]);

    for (size_t i = 0; i < N; i++)
      {
        incr tmp;
        tmp.set_format(std::get<0>(cases[i]));
        piecewise_pass_element_uint64( std::get<1>(cases[i]), tmp);
        auto r = tmp.finalize();
        CHECK(r.size() == strlen(std::get<2>(cases[i])) + 1);
        CHECK(strcmp(r.data(), std::get<2>(cases[i])) == 0);
      }

    TEST("two int")
    {
      std::tuple<const char *, uint64_t, uint64_t, const char *> cases[] = {
          {"%lu%lu", 1, 2, "12"},       {"%lu %lu", 1, 2, "1 2"},
          {" %lu %lu", 1, 2, " 1 2"},   {"%lu %lu ", 1, 2, "1 2 "},
          {" %lu %lu ", 1, 2, " 1 2 "}, {" %lu%%%lu ", 1, 2, " 1%2 "},
      };
      constexpr size_t N = sizeof(cases) / sizeof(cases[0]);

      for (size_t i = 0; i < N; i++)
        {
          incr tmp;
        tmp.set_format(std::get<0>(cases[i]));
          
          piecewise_pass_element_uint64( std::get<1>(cases[i]), tmp);
          piecewise_pass_element_uint64( std::get<2>(cases[i]), tmp);
        auto r = tmp.finalize();
          CHECK(r.size() == strlen(std::get<3>(cases[i])) + 1);
          CHECK(strcmp(r.data(), std::get<3>(cases[i])) == 0);
        }
    }
  }
}

MODULE(list)
{
  DEPENDS(writeint);
  DEPENDS(specifier);
  DEPENDS(next_spec_loc);
  DEPENDS(incr);
}

#if 0
MAIN_MODULE()
{
  DEPENDS(list);
}
#endif
