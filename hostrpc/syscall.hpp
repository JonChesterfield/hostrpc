#ifndef HOSTRPC_SYSCALL_H_INCLUDED
#define HOSTRPC_SYSCALL_H_INCLUDED

#include <x86_64-linux-gnu/asm/unistd_64.h>

#include "allocator.hpp"  // uses the openmp_impl for shared
#include "base_types.hpp"
#include "detail/platform_detect.h"

#if HOSTRPC_HOST
#include <stdio.h>
#endif

namespace hostrpc
{
static const uint64_t no_op = 0;

static const uint64_t syscall_op = 42;
static const uint64_t allocate_op = 21;
static const uint64_t free_op = 22;

#if HOSTRPC_HOST
// call syscall n on arguments ai, does not set errno
uint64_t syscall6(uint64_t n, uint64_t a0, uint64_t a1, uint64_t a2,
                  uint64_t a3, uint64_t a4, uint64_t a5);

inline void syscall_on_cache_line(unsigned index, hostrpc::cacheline_t *line)
{
  const bool verbose = false;
  if (verbose)
    {
      printf("%u: (%lu, %lu, %lu, %lu, %lu, %lu, %lu, %lu)\n", index,
             line->element[0], line->element[1], line->element[2],
             line->element[3], line->element[4], line->element[5],
             line->element[6], line->element[7]);
    }

  if (line->element[0] == no_op)
    {
      return;
    }

  if (line->element[0] == allocate_op)
    {
      uint64_t size = line->element[1];
      fprintf(stderr, "Call allocate_shared\n");
      void *res = hostrpc::allocator::hsa_impl::allocate_fine_grain(size);
      fprintf(stderr, "Called allocate_shared -> %lu\n", (uint64_t)res);
      line->element[0] = (uint64_t)res;
      return;
    }

  if (line->element[0] == free_op)
    {
      void *ptr = (void *)line->element[1];
      line->element[0] = hostrpc::allocator::hsa_impl::deallocate(ptr);

      return;
    }

  if (line->element[0] == syscall_op)
    {
      line->element[0] =
          syscall6(line->element[1], line->element[2], line->element[3],
                   line->element[4], line->element[5], line->element[6],
                   line->element[7]);
      return;
    }
}

#endif

}  // namespace hostrpc

#endif
