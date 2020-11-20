#pragma omp declare target

#include "allocator.hpp"
#include "base_types.hpp"
#include "client_server_pair.hpp"
#include "detail/client_impl.hpp"
#include "detail/server_impl.hpp"
#include "host_client.hpp"

namespace hostrpc
{
template <typename SZ, int device_num>
using x64_device_type_base =
    client_server_pair_t<SZ, copy_functor_given_alias, uint64_t,
                         allocator::openmp_shared<alignof(page_t)>,
                         allocator::openmp_shared<64>, allocator::host_libc<64>,
                         allocator::openmp_device<64, device_num>>;

template <typename SZ, int device_num>
struct x64_device_type : public x64_device_type_base<SZ, device_num>
{
  using base = x64_device_type_base<SZ, device_num>;
  HOSTRPC_ANNOTATE x64_device_type(SZ sz)
      : base(sz, typename base::AllocBuffer(),
             typename base::AllocInboxOutbox(), typename base::AllocLocal(),
             typename base::AllocRemote())
  {
  }
};
}  // namespace hostrpc

template <typename C, bool have_continuation>
static bool invoke(C *client, uint64_t x[8])
{
  auto fill = [&](hostrpc::page_t *page) -> void {
    hostrpc::cacheline_t *line = &page->cacheline[platform::get_lane_id()];
    line->element[0] = x[0];
    line->element[1] = x[1];
    line->element[2] = x[2];
    line->element[3] = x[3];
    line->element[4] = x[4];
    line->element[5] = x[5];
    line->element[6] = x[6];
    line->element[7] = x[7];
  };

  auto use = [&](hostrpc::page_t *page) -> void {
    hostrpc::cacheline_t *line = &page->cacheline[platform::get_lane_id()];
    x[0] = line->element[0];
    x[1] = line->element[1];
    x[2] = line->element[2];
    x[3] = line->element[3];
    x[4] = line->element[4];
    x[5] = line->element[5];
    x[6] = line->element[6];
    x[7] = line->element[7];
  };

  return client
      ->template rpc_invoke<decltype(fill), decltype(use), have_continuation>(
          fill, use);
}

#pragma omp end declare target

#include <omp.h>

#include "hostrpc_thread.hpp"
#include "openmp_plugins.hpp"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <x86_64-linux-gnu/asm/unistd_64.h>

__attribute__((unused)) static const uint64_t no_op = 0;

static const uint64_t syscall_op = 42;
static const uint64_t allocate_op = 21;
static const uint64_t free_op = 22;

static uint64_t syscall6(uint64_t n, uint64_t a0, uint64_t a1, uint64_t a2,
                         uint64_t a3, uint64_t a4, uint64_t a5)
{
  const bool verbose = false;
  uint64_t ret;
#if HOSTRPC_HOST
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
#endif
  return ret;
}

using SZ = hostrpc::size_compiletime<1920>;
constexpr static int device_num = 0;

using base_type = hostrpc::x64_device_type<SZ, device_num>;

base_type::client_type client_instance;

struct operate_test
{
  void operator()(hostrpc::page_t *page)
  {
    fprintf(stderr, "Invoked operate\n");
    for (unsigned i = 0; i < 64; i++)
      {
        operator()(i, &page->cacheline[i]);
      }
  }

  void operator()(unsigned index, hostrpc::cacheline_t *line)
  {
    printf("%u: (%lu, %lu, %lu, %lu, %lu, %lu, %lu, %lu)\n", index,
           line->element[0], line->element[1], line->element[2],
           line->element[3], line->element[4], line->element[5],
           line->element[6], line->element[7]);

    if (line->element[0] == no_op)
      {
        return;
      }

    if (line->element[0] == allocate_op)
      {
        uint64_t size = line->element[1];
        fprintf(stderr, "Call allocate_shared\n");
        void *res = hostrpc::allocator::openmp_impl::allocate_shared(size);
        fprintf(stderr, "Called allocate_shared\n");
        line->element[0] = (uint64_t)res;
        return;
      }

    if (line->element[0] == free_op)
      {
        void *ptr = (void *)line->element[1];
        line->element[0] =
            hostrpc::allocator::openmp_impl::deallocate_shared(ptr);

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
};

struct clear_test
{
  void operator()(hostrpc::page_t *page)
  {
    for (unsigned c = 0; c < 64; c++)
      {
        hostrpc::cacheline_t &line = page->cacheline[c];
        line.element[0] = no_op;
      }
  }
};

int main()
{
#pragma omp target
  asm("// less lazy");

  hostrpc::plugins got = hostrpc::find_plugins();

  fprintf(stderr, "amd: %u, ptx: %u. Found %u devices\n", got.amdgcn, got.nvptx,
          omp_get_num_devices());

  {
    SZ sz;

    base_type p(sz);

    HOSTRPC_ATOMIC(uint32_t) server_control;
    platform::atomic_store<uint32_t, __ATOMIC_RELEASE,
                           __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
        &server_control, 1);

    auto serv_func = [&]() {
      uint32_t location = 0;

      auto serv_func_busy = [&]() {
        bool r = true;
        while (r)
          {
            r = p.server.rpc_handle<operate_test, clear_test>(
                operate_test{}, clear_test{}, &location);
          }
      };

      for (;;)
        {
          serv_func_busy();

          // ran out of work, has client set control to cease?
          uint32_t ctrl =
              platform::atomic_load<uint32_t, __ATOMIC_ACQUIRE,
                                    __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
                  &server_control);

          if (ctrl == 0)
            {
              // client called to cease, empty any clear jobs in the pipeline
              serv_func_busy();
              break;
            }

          // nothing to do, but not told to stop. spin.
          for (unsigned j = 0; j < 1000; j++)
            {
              platform::sleep();
            }
        }
    };

    auto serv = hostrpc::make_thread(&serv_func);

    client_instance = p.client;

#pragma omp target map(tofrom : client_instance) device(0)
    {
      uint64_t tmp[8];
      tmp[0] = allocate_op;
      tmp[1] = 16;
      invoke<decltype(client_instance), true>(&client_instance, tmp);

      char *buf = (char *)tmp[0];

      buf[0] = 'h';
      buf[1] = 'i';
      buf[2] = '\n';
      buf[3] = '\0';

      tmp[0] = syscall_op;
      tmp[1] = __NR_write;
      tmp[2] = 2;
      tmp[3] = (uint64_t)buf;
      tmp[4] = 3;

      invoke<decltype(client_instance), true>(&client_instance, tmp);

      tmp[0] = syscall_op;
      tmp[1] = __NR_fsync;
      tmp[2] = 2;

      invoke<decltype(client_instance), true>(&client_instance, tmp);

      tmp[0] = free_op;
      tmp[1] = (uint64_t)buf;
      invoke<decltype(client_instance), true>(&client_instance, tmp);
    }

    fprintf(stderr, "Post target region\n");

    platform::atomic_store<uint32_t, __ATOMIC_RELEASE,
                           __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
        &server_control, 0);

    serv.join();
    fprintf(stderr, "Joined\n");
  }
}
