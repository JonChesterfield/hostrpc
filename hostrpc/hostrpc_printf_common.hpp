#ifndef HOSTRPC_PRINTF_COMMON_HPP_INCLUDED
#define HOSTRPC_PRINTF_COMMON_HPP_INCLUDED

// Types, macros used by client and server of printf

#include <stdint.h>

#include "detail/platform_detect.hpp"

enum func_type
{
  hostrpc_printf_print_nop = 0,

  // malloc = 1
  // free = 2,

  hostrpc_printf_print_start = 5,
  hostrpc_printf_print_end = 6,
  hostrpc_printf_pass_element_cstr = 7,

  hostrpc_printf_pass_element_scalar = 8,

  hostrpc_printf_pass_element_int32 = 9,
  hostrpc_printf_pass_element_uint32 = 10,
  hostrpc_printf_pass_element_int64 = 11,
  hostrpc_printf_pass_element_uint64 = 12,
  hostrpc_printf_pass_element_double = 13,
  hostrpc_printf_pass_element_void = 14,
  hostrpc_printf_pass_element_write_int64 = 16,

};

namespace
{
struct __printf_print_start_t
{
  uint64_t ID = hostrpc_printf_print_start;
  char unused[56];
  HOSTRPC_ANNOTATE __printf_print_start_t() {}
};

struct __printf_print_end_t
{
  uint64_t ID = hostrpc_printf_print_end;
  char unused[56];
  HOSTRPC_ANNOTATE __printf_print_end_t() {}
};

struct __printf_pass_element_cstr_t
{
  uint64_t ID = hostrpc_printf_pass_element_cstr;
  enum
  {
    width = 56
  };
  char payload[width];
  HOSTRPC_ANNOTATE __printf_pass_element_cstr_t(const char *s, size_t N)
  {
    __builtin_memset(payload, 0, width);
    __builtin_memcpy(payload, s, N);
  }
};

struct __printf_pass_element_scalar_t
{
  uint64_t ID = hostrpc_printf_pass_element_scalar;
  uint64_t Type;
  uint64_t payload;
  char unused[40];
  HOSTRPC_ANNOTATE __printf_pass_element_scalar_t(enum func_type type,
                                                  uint64_t x)
      : Type(type), payload(x)
  {
  }
};

struct __printf_pass_element_write_t
{
  uint64_t ID = hostrpc_printf_pass_element_write_int64;
  int64_t payload = 0;
  enum
  {
    width = 48
  };
  char unused[width];
  HOSTRPC_ANNOTATE __printf_pass_element_write_t() {}
};

}  // namespace

#endif
