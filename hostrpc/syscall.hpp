#ifndef HOSTRPC_SYSCALL_H_INCLUDED
#define HOSTRPC_SYSCALL_H_INCLUDED

#include <x86_64-linux-gnu/asm/unistd_64.h>

#include "allocator.hpp"  // uses the openmp_impl for shared
#include "base_types.hpp"
#include "detail/platform_detect.hpp"

#if HOSTRPC_HOST
#include <stdio.h>
#endif

// TODO: Factor out the cuda/nvptx part
#if !(DEMO_AMDGCN) && !(DEMO_NVPTX)
#error "Missing macro"
#endif

namespace hostrpc
{
static const uint64_t no_op = 0;

static const uint64_t syscall_op = 42;

#if DEMO_NVPTX
static const uint64_t allocate_op_cuda = 22;
static const uint64_t free_op_cuda = 32;
// as hsa is known to be a no-op
static const uint64_t device_to_host_pointer_cuda = 35;
#endif

#if DEMO_AMDGCN
static const uint64_t allocate_op_hsa = 21;
static const uint64_t free_op_hsa = 31;
#endif

#if HOSTRPC_HOST
// call syscall n on arguments ai, does not set errno
uint64_t syscall6(uint64_t n, uint64_t a0, uint64_t a1, uint64_t a2,
                  uint64_t a3, uint64_t a4, uint64_t a5);

inline void syscall_on_cache_line(unsigned index, hostrpc::cacheline_t *line)
{
  const bool verbose = false;

  if (verbose)
    {
      printf("SYSCALL %u: (%lu, %lu, %lu, %lu, %lu, %lu, %lu, %lu)\n", index,
             line->element[0], line->element[1], line->element[2],
             line->element[3], line->element[4], line->element[5],
             line->element[6], line->element[7]);
    }

  if (line->element[0] == no_op)
    {
      return;
    }

#if DEMO_NVPTX

  if (line->element[0] == allocate_op_cuda)
    {
      fprintf(stderr, "Call allocate_shared_cuda\n");
      uint64_t size = line->element[1];
      void *host = hostrpc::allocator::cuda_impl::allocate_shared(size);
      void *res = hostrpc::allocator::cuda_impl::device_ptr_from_host_ptr(host);
      fprintf(stderr, "Called allocate_shared -> %lu/%lu\n", (uint64_t)host,
              (uint64_t)res);
      line->element[0] = (uint64_t)res;
      return;
    }
  if (line->element[0] == free_op_cuda)
    {
      fprintf(stderr, "Call free_cuda\n");
      void *ptr = (void *)line->element[1];
      uint64_t size = line->element[2];
      (void)size;
      void *host = hostrpc::allocator::cuda_impl::host_ptr_from_device_ptr(ptr);
      line->element[0] = hostrpc::allocator::cuda_impl::deallocate_shared(host);
      return;
    }

  if (line->element[0] == device_to_host_pointer_cuda)
    {
      void *ptr = (void *)line->element[1];
      void *host = hostrpc::allocator::cuda_impl::host_ptr_from_device_ptr(ptr);
      line->element[0] = (uint64_t)host;
    }

#endif

#if DEMO_AMDGCN
  if (line->element[0] == free_op_hsa)
    {
      fprintf(stderr, "Call free_hsa\n");
      void *ptr = (void *)line->element[1];
      uint64_t size = line->element[2];
      (void)size;
      line->element[0] = hostrpc::allocator::hsa_impl::deallocate(ptr);

      return;
    }
  if (line->element[0] == allocate_op_hsa)
    {
      uint64_t size = line->element[1];
      fprintf(stderr, "Call allocate_shared_hsa\n");
      void *res = hostrpc::allocator::hsa_impl::allocate_fine_grain(size);

      fprintf(stderr, "Called allocate_shared -> %lu\n", (uint64_t)res);
      line->element[0] = (uint64_t)res;
      return;
    }
#endif

  if (line->element[0] == syscall_op)
    {
      fprintf(stderr, "Call syscall<%lu>\n", line->element[1]);

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
