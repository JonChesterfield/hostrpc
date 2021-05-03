#include "EvilUnit.h"

#include <stdint.h>

#include <cassert>
#include <cstring>
#include <tuple>
#include <vector>

#include "hostrpc_printf.h"

struct incr
{
  std::vector<char> format;
  size_t loc;

  std::vector<char> output;

  incr() : incr("") {}
  incr(const char *fmt)
  {
    loc = 0;
    format = {fmt, fmt + strlen(fmt)};
    output.push_back('\0');
  }
};

incr glob("");

const char *as_cstr(incr *glob) { return glob->output.data(); }

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

void append_until_next_loc(uint32_t port, incr &glob)
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

uint32_t piecewise_print_start(const char *fmt, incr &glob)
{
  uint32_t port = 0;
  incr tmp(fmt);
  glob = tmp;
  append_until_next_loc(port, glob);
  return port;
}

int piecewise_print_end(uint32_t port, incr &glob)
{
  (void)port;
  (void)glob;
  append_until_next_loc(port, glob);
  return 0;
}

int _exit(int);

size_t bytes_for_arg(char *fmt, size_t next_start, size_t next_end, uint64_t v)
{
  size_t one_past = next_end + 1;
  char term = fmt[one_past];
  fmt[one_past] = '\0';
  size_t r = snprintf(NULL, 0, &fmt[next_start], v);
  fmt[one_past] = term;
  return r;
}

void piecewise_pass_element_uint64(uint32_t port, uint64_t v, incr &glob)
{
  char *fmt = glob.format.data();
  size_t len = glob.format.size();
  size_t next_start = __printf_next_start(fmt, len, glob.loc);
  size_t next_end = __printf_next_end(fmt, len, next_start);
  const bool verbose = false;
  if (verbose)
    (printf)("glob loc %zu, Format %s/%zu, split [%zu %zu]\n", glob.loc, fmt,
             len, next_start, next_end);

  append_until_next_loc(port, glob);

  if (next_start == len)
    {
      return;
    }

  size_t bytes = bytes_for_arg(fmt, next_start, next_end, v);

  if (verbose)
    (printf)("output size %zu, increasing by %zu\n", glob.output.size(), bytes);

  size_t output_end = glob.output.size() - 1 /*trim off trailing nul*/;
  for (size_t i = 0; i < bytes; i++)
    {
      glob.output.push_back('\0');
    }

  {
    if (verbose)
      (printf)(
          "writing fmt %s into loc %zu, giving %zu bytes. Next end at %zu\n",
          &fmt[next_start], output_end, bytes, next_end);
    size_t one_past = next_end + 1;
    char term = fmt[one_past];
    fmt[one_past] = '\0';
    snprintf(glob.output.data() + output_end,
             bytes + 1,  // overwrites the trailing nul that is already there
             &fmt[next_start], v);
    fmt[one_past] = term;
  }

  glob.loc = next_end + 1;
  if (verbose) (printf)("Setting glob.loc to %zu\n", glob.loc);
}

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
  uint32_t port = piecewise_print_start(str, tmp);
  piecewise_print_end(port, tmp);
  return tmp.output;
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
        uint32_t port = piecewise_print_start(std::get<0>(cases[i]), tmp);
        piecewise_pass_element_uint64(port, std::get<1>(cases[i]), tmp);
        piecewise_print_end(port, tmp);
        auto &r = tmp.output;
        CHECK(r.size() == strlen(std::get<2>(cases[i])) + 1);
        CHECK(strcmp(r.data(), std::get<2>(cases[i])) == 0);
      }
  }
}

MAIN_MODULE()
{
  DEPENDS(writeint);
  DEPENDS(specifier);
  DEPENDS(next_spec_loc);
  DEPENDS(incr);
}
