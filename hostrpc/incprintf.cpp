#include "EvilUnit.h"

#include <stdint.h>

#include <cstring>
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

const char *as_cstr(incr &glob) { return glob.output.data(); }

uint32_t piecewise_print_start(const char *fmt, incr &glob)
{
  incr tmp(fmt);
  glob = tmp;
  return 0;
}

void append_bytes(const char *from, size_t N, incr &glob)
{
  for (size_t i = 0; i < N; i++)
    {
      glob.output.push_back(from[i]);
    }
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

  if (0)
    (printf)("glob loc %zu, Format %s/%zu, split [%zu %zu]\n", glob.loc, fmt,
             len, next_start, next_end);

  if (next_start > glob.loc)
    {
      if (0) (printf)("Appending %lu bytes\n", next_start - glob.loc);
      append_bytes(fmt, next_start - glob.loc, glob);
    }

  if (next_start == len)
    {
      return;
    }

  size_t bytes = bytes_for_arg(fmt, next_start, next_end, v);

  if (0)
    (printf)("output size %zu, increasing by %zu\n", glob.output.size(), bytes);

  size_t output_end = glob.output.size() - 1 /*trim off trailing nul*/;
  for (size_t i = 0; i < bytes; i++)
    {
      glob.output.push_back('\0');
    }

  {
    if (0)
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

static MODULE(incr)
{
  TEST("single int")
  {
    incr tmp;
    uint32_t port = piecewise_print_start("%lu", tmp);
    piecewise_pass_element_uint64(port, 42127, tmp);
    CHECK(strcmp(as_cstr(tmp), "42127") == 0);
  }
}

MAIN_MODULE()
{
  DEPENDS(writeint);
  DEPENDS(specifier);
  DEPENDS(next_spec_loc);
  DEPENDS(incr);
}
