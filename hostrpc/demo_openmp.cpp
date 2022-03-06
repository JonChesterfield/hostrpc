#define HOSTRPC_HAVE_STDIO 1
#include <stdio.h>

#if !(DEMO_AMDGCN) && !(DEMO_NVPTX)
#error "Missing macro"
#endif

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
    client_server_pair_t<SZ, arch::x64, arch::openmp_target<device_num> >;

template <typename SZ, int device_num>
struct x64_device_type : public x64_device_type_base<SZ, device_num>
{
  using base = x64_device_type_base<SZ, device_num>;
  HOSTRPC_ANNOTATE x64_device_type(SZ sz) : base(sz, {}, {}) {}
};
}  // namespace hostrpc

template <typename C>
static bool invoke(C *client, uint64_t x[8])
{
  auto fill = [&](hostrpc::port_t, hostrpc::page_t *page) -> void {
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

  auto use = [&](hostrpc::port_t, hostrpc::page_t *page) -> void {
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

  return client->template rpc_invoke(fill, use);
}

#pragma omp end declare target

#include <omp.h>

#include "hostrpc_thread.hpp"
#include "openmp_plugins.hpp"
#include "syscall.hpp"

#include "server_thread_state.hpp"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <x86_64-linux-gnu/asm/unistd_64.h>

using SZ = hostrpc::size_compiletime<1920>;
constexpr static int device_num = 0;

using base_type = hostrpc::x64_device_type<SZ, device_num>;

struct operate_test
{
  void operator()(hostrpc::port_t port, hostrpc::page_t *page)
  {
    fprintf(stderr, "Invoked operate\n");
    for (unsigned i = 0; i < 64; i++)
      {
        operator()(port, i, &page->cacheline[i]);
      }
  }

  void operator()(hostrpc::port_t, unsigned index, hostrpc::cacheline_t *line)
  {
#if HOSTRPC_HOST
    hostrpc::syscall_on_cache_line(index, line);
#endif
  }
};

struct clear_test
{
  void operator()(hostrpc::port_t, hostrpc::page_t *page)
  {
    for (unsigned c = 0; c < 64; c++)
      {
        hostrpc::cacheline_t &line = page->cacheline[c];
        line.element[0] = hostrpc::no_op;
      }
  }
};

#if DEMO_NVPTX
#include <cuda_runtime.h>
#endif

int main()
{
  // this is probably the default, setting it doesn't seem to help
#if DEMO_NVPTX
  cudaError_t rc = cudaSetDeviceFlags(cudaDeviceMapHost);
  if (rc != cudaSuccess)
    {
      fprintf(stderr, "Failed to set device flags\n");
      exit(1);
    }
  hostrpc::disable_amdgcn();
#else
  hostrpc::disable_nvptx();
#endif

#pragma omp target
  asm("// less lazy");

  hostrpc::plugins got = hostrpc::find_plugins();
  fprintf(stderr, "amd: %u, ptx: %u\n", got.amdgcn, got.nvptx);

  {
    SZ sz;

    base_type p(sz);

    if (!p.valid())
      {
        fprintf(stderr, "%s: Failed to allocate\n", __func__);
        return 1;
      }

    HOSTRPC_ATOMIC(uint32_t) server_control;
    platform::atomic_store<uint32_t, __ATOMIC_RELEASE,
                           __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
        &server_control, 1);

    auto s = hostrpc::make_server_thread_state(&p.server, &server_control,
                                               operate_test{}, clear_test{});
    auto serv = hostrpc::make_thread(&s);

#if DEMO_NVPTX
    hostrpc::allocator::openmp_device<64, device_num> alloc;
    auto dev_client_raw = alloc.allocate(sizeof(p.client));
    void *dev_client = dev_client_raw.remote_ptr();

    cudaError_t rc = cudaMemcpy(dev_client, &p.client, sizeof(p.client),
                                cudaMemcpyHostToDevice);
    if (rc != cudaSuccess)
      {
        fprintf(stderr, "Failed to copy client to gpu memory\n");
        return 1;
      }
#endif
#if DEMO_AMDGCN
    auto client = p.client;
#endif
    // A target region that maps data in/out resolves to:
    // data_submit_async
    // run_target_region_async
    // data_retrieve_async
    // The retrieve_async is launched before the kernel has finished running
    // which means the memcpy_async it invokes can deadlock with a call within
    // the target region
    // This avoids map, thus avoids the memcpy_async deadlock, but a less
    // fragile solution is required

#if DEMO_NVPTX
#pragma omp target device(0) is_device_ptr(dev_client)
    {
      base_type::client_type *client = (base_type::client_type *)dev_client;
      const uint64_t alloc_op = hostrpc::allocate_op_cuda;
      const uint64_t free_op = hostrpc::free_op_cuda;
      auto inv = [&](uint64_t x[8]) -> bool {
        return invoke<base_type::client_type>(client, x);
      };
#endif
#if DEMO_AMDGCN
#pragma omp target device(0) map(to : client)
      {
        const uint64_t alloc_op = hostrpc::allocate_op_hsa;
        const uint64_t free_op = hostrpc::free_op_hsa;
        auto inv = [&](uint64_t x[8]) -> bool {
          return invoke<base_type::client_type>(&client, x);
        };
#endif

        printf("target region start\n");
        constexpr const uint64_t buffer_size = 16;

        uint64_t tmp[8];
        tmp[0] = alloc_op;
        tmp[1] = buffer_size;
        inv(tmp);

        char *buf = (char *)tmp[0];

        buf[0] = 'h';
        buf[1] = 'i';
        buf[2] = '\n';
        buf[3] = '\0';

#if DEMO_AMDGCN
        uint64_t host_buf = tmp[0];
#else
    tmp[0] = hostrpc::device_to_host_pointer_cuda;
    tmp[1] = (uint64_t)buf;
    inv(tmp);
    uint64_t host_buf = tmp[0];
#endif

        tmp[0] = hostrpc::syscall_op;
        tmp[1] = __NR_write;
        tmp[2] = 2;
        tmp[3] = host_buf;  // needs to be a host pointer here
        tmp[4] = 3;

        inv(tmp);

        tmp[0] = hostrpc::syscall_op;
        tmp[1] = __NR_fsync;
        tmp[2] = 2;

        inv(tmp);

        tmp[0] = free_op;
        tmp[1] = (uint64_t)buf;  // assuming a device pointer here
        tmp[2] = buffer_size;
        inv(tmp);
        printf("target region done\n");
      }

      fprintf(stderr, "Post target region\n");

      // isn't waiting for the previous kernel to finish before launching the
      // async exit

      platform::atomic_store<uint32_t, __ATOMIC_RELEASE,
                             __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
          &server_control, 0);

      serv.join();
      fprintf(stderr, "Joined\n");
    }
  }
