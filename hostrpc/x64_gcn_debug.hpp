#ifndef HOSTRPC_X64_GCN_DEBUG_HPP_INCLUDE
#define HOSTRPC_X64_GCN_DEBUG_HPP_INCLUDE

#include "detail/platform_detect.hpp"
#include <stdint.h>

namespace hostrpc
{
// Can be used to print a string, provided it doesn't contain any
// format characters (as it will be passed to printf with no args)
void print_base(const char *, uint64_t, uint64_t, uint64_t);

inline void print(const char *str, uint64_t x0, uint64_t x1, uint64_t x2)
{
  print_base(str, x0, x1, x2);
}
inline void print(const char *str, uint64_t x0, uint64_t x1)
{
  print(str, x0, x1, 0);
}
inline void print(const char *str, uint64_t x0) { print(str, x0, 0); }
inline void print(const char *str) { return print(str, 0); }

}  // namespace hostrpc

uint32_t piecewise_print_start(const char *fmt);
void piecewise_print_pass_uint64(uint32_t port, uint64_t v);
void piecewise_print_pass_cstr(uint32_t port, const char *str);
int piecewise_print_end(uint32_t port);

#if (HOSTRPC_HOST)
#include "hsa.h"

namespace hostrpc
{
int print_enable(hsa_executable_t ex, hsa_agent_t kernel_agent);
}
#endif

#endif
