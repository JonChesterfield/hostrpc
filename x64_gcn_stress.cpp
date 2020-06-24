#define VISIBLE __attribute__((visibility("default")))

// Kernel entry
#if defined __OPENCL__
void __device_start_cast(__global void *asargs);
kernel void __device_start(__global void *args) { __device_start_cast(args); }
#endif

#if !defined __OPENCL__

#include "base_types.hpp"
#include "detail/platform.hpp"
#include "interface.hpp"

#if defined(__AMDGCN__)
#include <stdint.h>
#else
#include <cstdint>
#endif

struct kernel_args
{
  uint32_t id;
  uint32_t reps;
  uint32_t result;
};

#if defined(__AMDGCN__)

// Wire opencl kernel up to c++ implementation

static uint32_t gpu_call(uint32_t id, uint32_t reps);
extern "C" void __device_start_main(kernel_args *args)
{
  uint32_t r = gpu_call(args->id, args->reps);
  args->result = r;
}
extern "C" void __device_start_cast(
    __attribute__((address_space(1))) void *asargs)
{
  kernel_args *args = (kernel_args *)asargs;
  __device_start_main(args);
}

// memcmp from musl, with type casts for c++
static int memcmp(const void *vl, const void *vr, size_t n)
{
  const unsigned char *l = static_cast<const unsigned char *>(vl);
  const unsigned char *r = static_cast<const unsigned char *>(vr);
  for (; n && *l == *r; n--, l++, r++)
    ;
  return n ? *l - *r : 0;
}

static void init_page(hostrpc::page_t *page, uint64_t v)
{
  platform::init_inactive_lanes(page, v);
  hostrpc::cacheline_t *line = &page->cacheline[platform::get_lane_id()];
  for (unsigned e = 0; e < 8; e++)
    {
      line->element[e] = v;
    }
}

VISIBLE
hostrpc::x64_gcn_t::client_t hostrpc_pair_client;
uint32_t gpu_call(uint32_t id, uint32_t reps)
{
  // unsigned reps = 1000;
  // unsigned id = 42;
  hostrpc::page_t scratch;
  hostrpc::page_t expect;
  unsigned failures = 0;
  for (unsigned r = 0; r < reps; r++)
    {
      init_page(&scratch, id);
      init_page(&expect, id + 1);
      if (hostrpc_pair_client.invoke(
              [&](hostrpc::page_t *page) {
                __builtin_memcpy(page, &scratch, sizeof(hostrpc::page_t));
              },
              [&](hostrpc::page_t *page) {
                __builtin_memcpy(&scratch, page, sizeof(hostrpc::page_t));
              }))
        {
          if (memcmp(&scratch, &expect, sizeof(hostrpc::page_t)) != 0)
            {
              failures++;
            }
        }
    }

  return failures;
}

#else

#include "catch.hpp"
#include "hsa.hpp"
#include <cstring>
#include <thread>

template <typename T>
struct launch_t
{
  launch_t(hsa_agent_t, hsa_queue_t *, const char *, T)
  {
    ready = false;
    state = nullptr;
  }

  T operator()()
  {
    assert(state);
    wait();
    return *state;
  }
  ~launch_t()
  {
    if (state)
      {
        wait();
        state->~T();
        // free
      }
  }

 private:
  void wait()
  {
    while (!ready)
      {
      }
  }
  bool ready;
  T *state;
};
template <typename T>
launch_t<T> launch(hsa_agent_t kernel_agent, hsa_queue_t *queue,
                   const char *kernel, T arg)
{
  return {kernel_agent, queue, kernel, arg};
}

// executable is named x64_gcn_stress.gcn.so

#include "incbin.h"

INCBIN(x64_gcn_stress_so, "x64_gcn_stress.gcn.so");

TEST_CASE("x64_gcn_stress")
{
  hsa::init hsa;
  {
    using namespace hostrpc;

    hsa_agent_t kernel_agent = hsa::find_a_gpu_or_exit();
    auto ex = hsa::executable(kernel_agent, x64_gcn_stress_so_data,
                              x64_gcn_stress_so_size);
    CHECK(ex.valid());

    uint64_t entry = ex.get_symbol_address_by_name("__device_start.kd");
    uint64_t client = ex.get_symbol_address_by_name("hostrpc_pair_client");

    printf("gcn stress: kernel at %lu, client at %lu\n", entry, client);

    int life = 42;
    auto v = launch(kernel_agent, 0, 0, life);
    (void)v;

    hsa_region_t kernarg_region = hsa::region_kernarg(kernel_agent);
    hsa_region_t fine_grained_region = hsa::region_fine_grained(kernel_agent);
    hsa_region_t coarse_grained_region =
        hsa::region_coarse_grained(kernel_agent);
    {
      uint64_t fail = reinterpret_cast<uint64_t>(nullptr);
      if (kernarg_region.handle == fail || fine_grained_region.handle == fail ||
          coarse_grained_region.handle == fail)
        {
          fprintf(stderr, "Failed to find allocation region on kernel agent\n");
          exit(1);
        }
    }

    _Atomic bool server_live(true);
    size_t N = 1920;
    hostrpc::x64_gcn_t p(N, fine_grained_region.handle,
                         coarse_grained_region.handle);

    auto op_func = [](hostrpc::page_t *page) {
      for (unsigned c = 0; c < 64; c++)
        {
          hostrpc::cacheline_t &line = page->cacheline[c];
          std::swap(line.element[0], line.element[7]);
          std::swap(line.element[1], line.element[6]);
          std::swap(line.element[2], line.element[5]);
          std::swap(line.element[3], line.element[4]);
          for (unsigned i = 0; i < 8; i++)
            {
              line.element[i]++;
            }
        }
    };

    auto server_worker = [&](unsigned id) {
      unsigned count = 0;

      uint64_t server_location = 0;
      for (;;)
        {
          if (!server_live)
            {
              printf("server %u did %u tasks\n", id, count);
              break;
            }
          bool did_work = p.server().handle(op_func, &server_location);
          if (did_work)
            {
              count++;
            }
        }
    };

    unsigned nservers = 8;

    printf("x64-gcn spawning %u x64 servers\n", nservers);

    std::vector<std::thread> server_store;
    for (unsigned i = 0; i < nservers; i++)
      {
        server_store.emplace_back(std::thread(server_worker, i));
      }

    server_live = false;
    for (auto &i : server_store)
      {
        i.join();
      }
  }
}

#endif
#endif
