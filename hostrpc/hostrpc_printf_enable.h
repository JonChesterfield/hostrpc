#ifndef HOSTRPC_PRINTF_ENABLE_H_INCLUDED
#define HOSTRPC_PRINTF_ENABLE_H_INCLUDED

#include "detail/platform_detect.hpp"

#include <stdint.h>
#include <stddef.h>

enum func_type : uint64_t
{
  func___printf_print_nop = 0,

  func___printf_print_start = 5,
  func___printf_print_end = 6,
  func___printf_pass_element_cstr = 7,

  func___printf_pass_element_scalar = 8,

  func___printf_pass_element_int32 = 9,
  func___printf_pass_element_uint32 = 10,
  func___printf_pass_element_int64 = 11,
  func___printf_pass_element_uint64 = 12,
  func___printf_pass_element_double = 13,
  func___printf_pass_element_void = 14,
  func___printf_pass_element_write_int64 = 16,

};

namespace
{
struct __printf_print_start_t
{
  uint64_t ID = func___printf_print_start;
  char unused[56];
  HOSTRPC_ANNOTATE __printf_print_start_t() {}
};

struct __printf_print_end_t
{
  uint64_t ID = func___printf_print_end;
  char unused[56];
  HOSTRPC_ANNOTATE __printf_print_end_t() {}
};

struct __printf_pass_element_cstr_t
{
  uint64_t ID = func___printf_pass_element_cstr;
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
  uint64_t ID = func___printf_pass_element_scalar;
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
  uint64_t ID = func___printf_pass_element_write_int64;
  int64_t payload = 0;
  enum
  {
    width = 48
  };
  char unused[width];
  HOSTRPC_ANNOTATE __printf_pass_element_write_t() {}
};

// using SZ = hostrpc::size_runtime;

}  // namespace


#if (HOSTRPC_HOST)
#include "hsa.h"
#ifdef __cplusplus
extern "C"
{
#endif
  int hostrpc_print_enable_on_hsa_agent(hsa_executable_t ex,
                                        hsa_agent_t kernel_agent);
#ifdef __cplusplus
}
#endif
#endif

#endif
