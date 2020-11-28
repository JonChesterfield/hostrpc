#include "x64_gcn_debug.hpp"
#include "detail/platform_detect.h"
#include "x64_gcn_type.hpp"

using SZ = hostrpc::size_runtime;

#if (HOSTRPC_AMDGCN)

__attribute__((visibility("default")))
hostrpc::x64_gcn_type<SZ>::client_type hostrpc_x64_gcn_debug_client[1];

#endif

#if (HOSTRPC_HOST)

int main() {}
#include "hostrpc_thread.hpp"
#include "hsa.hpp"
#include "server_thread_state.hpp"
#include <stdio.h>
#include <utility>

namespace
{
struct operate
{
  void operator()(hostrpc::page_t *page)
  {
    (void)page;
    fprintf(stderr, "Invoked operate\n");
  }
};

struct clear
{
  void operator()(hostrpc::page_t *page)
  {
    (void)page;
    fprintf(stderr, "Invoked clear\n");
  }
};

using sts_ty =
    hostrpc::server_thread_state<hostrpc::x64_gcn_type<SZ>::server_type,
                                 operate, clear>;

struct global
{
  hsa::init hsa_instance;

  std::unique_ptr<hostrpc::x64_gcn_type<SZ>> p;
  HOSTRPC_ATOMIC(uint32_t) server_control;

  sts_ty server_state;
  std::unique_ptr<hostrpc::thread<sts_ty>> thrd;

  global() : hsa_instance{}
  {
    fprintf(stderr, "ctor\n");
    hsa_agent_t kernel_agent = hsa::find_a_gpu_or_exit();
    hsa_region_t fine_grained_region = hsa::region_fine_grained(kernel_agent);
    hsa_region_t coarse_grained_region =
        hsa::region_coarse_grained(kernel_agent);

    uint64_t fail = reinterpret_cast<uint64_t>(nullptr);
    if (fine_grained_region.handle == fail ||
        coarse_grained_region.handle == fail)
      {
        fprintf(stderr, "Failed to find allocation region on kernel agent\n");
        exit(1);
      }

    SZ N{1920};

    // having trouble getting clang to call the move constructor, work around
    // with heap
    p = std::make_unique<hostrpc::x64_gcn_type<SZ>>(
        N, fine_grained_region.handle, coarse_grained_region.handle);
    platform::atomic_store<uint32_t, __ATOMIC_RELEASE,
                           __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
        &server_control, 1);

    server_state = sts_ty(&p->server, &server_control, operate{}, clear{});

    thrd =
        std::make_unique<hostrpc::thread<sts_ty>>(make_thread(&server_state));

    if (!thrd->valid())
      {
        fprintf(stderr, "Failed to spawn thread\n");
        exit(1);
      }
  }

  ~global()
  {
    fprintf(stderr, "dtor\n");

    platform::atomic_store<uint32_t, __ATOMIC_RELEASE,
                           __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
        &server_control, 0);

    thrd->join();
  }
} global_instance;

}  // namespace
#endif
