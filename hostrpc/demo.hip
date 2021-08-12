#define DEMO_AMDGCN 1

#include "hip/hip_runtime.h"

#include "detail/client_impl.hpp"
#include "detail/server_impl.hpp"

#include "allocator.hpp"
#include "host_client.hpp"

#include "x64_gcn_type.hpp"

#include "hsa.hpp"

#include "syscall.hpp"

#include "hostrpc_thread.hpp"
#include "server_thread_state.hpp"

using SZ = hostrpc::size_runtime;

__device__ extern "C" int printf(const char *format, ...);

template <typename C>
__device__ static bool invoke(C *client, uint64_t x[8])
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

  return client->template rpc_invoke<decltype(fill), decltype(use)>(fill, use);
}

#include "demo_kernel.hip"

struct operate_test
{
  void operator()(uint32_t slot, hostrpc::page_t *page)
  {
    fprintf(stderr, "Invoked operate\n");
    for (unsigned i = 0; i < 64; i++)
      {
        operator()(slot, i, &page->cacheline[i]);
      }
  }

  void operator()(uint32_t slot, unsigned index, hostrpc::cacheline_t *line)
  {
#if HOSTRPC_HOST
    hostrpc::syscall_on_cache_line(index, line);
#endif
  }
};

struct clear_test
{
  void operator()(uint32_t, hostrpc::page_t *page)
  {
    for (unsigned c = 0; c < 64; c++)
      {
        hostrpc::cacheline_t &line = page->cacheline[c];
        line.element[0] = hostrpc::no_op;
      }
  }
};

int main()
{
  hsa::init with_hsa;

  hsa_agent_t kernel_agent = hsa::find_a_gpu_or_exit();

  hsa_region_t fine_grained_region = hsa::region_fine_grained(kernel_agent);
  hsa_region_t coarse_grained_region = hsa::region_coarse_grained(kernel_agent);
  {
    uint64_t fail = reinterpret_cast<uint64_t>(nullptr);
    if (fine_grained_region.handle == fail ||
        coarse_grained_region.handle == fail)
      {
        fprintf(stderr, "Failed to find allocation region on kernel agent\n");
        exit(1);
      }
  }

  SZ sz(1920);
  hostrpc::x64_gcn_type<SZ> p(sz, fine_grained_region.handle,
                              coarse_grained_region.handle);

  hostrpc::x64_gcn_type<SZ>::client_type *ptr;
  hipMalloc((void **)&ptr, sizeof(hostrpc::x64_gcn_type<SZ>::client_type));
  hipMemcpyHtoD(ptr, &p.client, sizeof(hostrpc::x64_gcn_type<SZ>::client_type));

  HOSTRPC_ATOMIC(uint32_t) server_control;
  platform::atomic_store<uint32_t, __ATOMIC_RELEASE,
                         __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(&server_control,
                                                                1);

  auto s = hostrpc::make_server_thread_state(&p.server, &server_control,
                                             operate_test{}, clear_test{});
  auto serv = hostrpc::make_thread(&s);

  fprintf(stderr, "passing addr %p\n", ptr);
  hipDeviceSynchronize();

  on_gpu<<<1, 1>>>(ptr, 0, 0, 0);

  fprintf(stderr, "got back from on_gpu launch\n");

  hipDeviceSynchronize();

  hipDeviceReset();

  return 0;
}
