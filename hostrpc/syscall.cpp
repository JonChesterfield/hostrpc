#include "syscall.hpp"

namespace hostrpc
{
#if HOSTRPC_HOST
uint64_t syscall6(uint64_t n, uint64_t a0, uint64_t a1, uint64_t a2,
                  uint64_t a3, uint64_t a4, uint64_t a5)
{
  const bool verbose = true;
  uint64_t ret;
  // not in a target region, but clang errors on the unknown register anyway
  register uint64_t r10 __asm__("r10") = a3;
  register uint64_t r8 __asm__("r8") = a4;
  register uint64_t r9 __asm__("r9") = a5;

  ret = 0;
  __asm__ volatile("syscall"
                   : "=a"(ret)
                   : "a"(n), "D"(a0), "S"(a1), "d"(a2), "r"(r10), "r"(r8),
                     "r"(r9)
                   : "rcx", "r11", "memory");

  if (verbose)
    {
      fprintf(stderr, "%lu <- syscall %lu %lu %lu %lu %lu %lu %lu\n", ret, n,
              a0, a1, a2, a3, a4, a5);
    }
  return ret;
}
#endif

}  // namespace hostrpc
