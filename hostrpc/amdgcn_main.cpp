#include "detail/platform.hpp"
#undef printf
#include "hostcall.hpp"
#include <stddef.h>

#include "EvilUnit.h"

__attribute__((unused)) static const uint64_t no_op =
    UINT64_MAX;  // Warning: Update hostcall.cpp if changing this

static const uint64_t syscall_op = 42;

static const uint64_t allocate_op = 21;
static const uint64_t free_op = 22;

// Included on amdgcn.
#include <x86_64-linux-gnu/asm/unistd_64.h>

#if HOSTRPC_HOST
#include "hsa.h"
#include "hsa.hpp"

static uint64_t syscall6(uint64_t n, uint64_t a0, uint64_t a1, uint64_t a2,
                         uint64_t a3, uint64_t a4, uint64_t a5)
{
  const bool verbose = false;
  uint64_t ret;
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

ssize_t write(unsigned int fd, const char *buf, size_t count)
{
  uint64_t s = 0;
  uint64_t sr =
      syscall6((uint64_t)fd, (uint64_t)buf, (uint64_t)count, s, s, s, s);
  ssize_t r;
  __builtin_memcpy(&r, &sr, 8);
  return r;
}

#endif

namespace hostcall_ops
{
#if HOSTRPC_HOST

void operate(unsigned lane, hostrpc::cacheline_t *line)
{
  if (line->element[0] == no_op)
    {
      return;
    }

  if (line->element[0] == allocate_op)
    {
      void *res = nullptr;
      uint64_t size = line->element[1];
      // todo: line up properly, check for errors
      hsa_agent_t kernel_agent = hsa::find_a_gpu_or_exit();
      hsa_region_t fine_grained_region = hsa::region_fine_grained(kernel_agent);

      hsa_memory_allocate(fine_grained_region, size, &res);

      line->element[0] = (uint64_t)res;
      fprintf(stderr, "Alloc %lu bytes => %p\n", size, res);
      return;
    }

  if (line->element[0] == free_op)
    {
      void *ptr = (void *)line->element[1];
      hsa_memory_free(ptr);
      line->element[0] = 0;

      fprintf(stderr, "Free %p\n", ptr);
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

  if (1)
    fprintf(stderr, "[%u] %lu %lu %lu %lu %lu %lu %lu %lu\n", lane,
            line->element[0], line->element[1], line->element[2],
            line->element[3], line->element[4], line->element[5],
            line->element[6], line->element[7]);
}

void operate(hostrpc::page_t *page)
{
  for (unsigned c = 0; c < 64; c++)
    {
      hostrpc::cacheline_t *line = &page->cacheline[c];
      operate(c, line);
    }
}

void clear(hostrpc::page_t *page)
{
  for (unsigned c = 0; c < 64; c++)
    {
      hostrpc::cacheline_t &line = page->cacheline[c];
      for (unsigned i = 0; i < 8; i++)
        {
          line.element[i] = no_op;
        }
    }
}

#endif

#if defined __AMDGCN__

void pass_arguments(hostrpc::page_t *page, uint64_t d[8])
{
  hostrpc::cacheline_t *line = &page->cacheline[platform::get_lane_id()];
  for (unsigned i = 0; i < 8; i++)
    {
      line->element[i] = d[i];
    }
}
void use_result(hostrpc::page_t *page, uint64_t d[8])
{
  hostrpc::cacheline_t *line = &page->cacheline[platform::get_lane_id()];
  for (unsigned i = 0; i < 8; i++)
    {
      d[i] = line->element[i];
    }
}
#endif

}  // namespace hostcall_ops

#if defined __AMDGCN__

static uint64_t syscall6(uint64_t n, uint64_t a0, uint64_t a1, uint64_t a2,
                         uint64_t a3, uint64_t a4, uint64_t a5)
{
  uint64_t scratch[8];
  scratch[0] = syscall_op;
  scratch[1] = n;
  scratch[2] = a0;
  scratch[3] = a1;
  scratch[4] = a2;
  scratch[5] = a3;
  scratch[6] = a4;
  scratch[7] = a5;
  hostcall_client(scratch);
  return scratch[0];
}

static void *allocate(uint64_t size)
{
  uint64_t scratch[8];
  scratch[0] = allocate_op;
  scratch[1] = size;
  hostcall_client(scratch);
  return (void *)scratch[0];
}

static void deallocate(void *ptr)
{
  uint64_t scratch[8];
  scratch[0] = free_op;
  scratch[1] = (uint64_t)ptr;
  hostcall_client(scratch);
}

extern "C" __attribute__((visibility("default"))) int main(int argc,
                                                           char **argv);

MAIN_MODULE()
{
  TEST("first print")
  {
    if (platform::get_lane_id() % 2 == 1)
      {
        printf("EvUn with %s syntax\n", "printf");
        CHECK(0);
      }
  }

  TEST("syscall")
  {
    if (platform::get_lane_id() == 0)
      {
        char *buf = (char *)allocate(16);

        buf[0] = 'h';
        buf[1] = 'i';
        buf[2] = '\n';
        buf[3] = '\0';

        syscall6(__NR_write, 2, (uint64_t)buf, 3, 0, 0, 0);

        syscall6(__NR_fsync, 2, 0, 0, 0, 0, 0);

        deallocate(buf);
      }
  }

  TEST("second print\n")
  {
    if (platform::get_lane_id() % 2 == 0)
      {
        CHECK(0);
        printf("second\n");
      }
  }
}

#endif
