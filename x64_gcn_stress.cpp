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
#include "timer.hpp"

#define MAXCLIENT (4 * 1024)  // lazy memory management, could malloc it

#if defined(__AMDGCN__)
#include <stdint.h>
#else
#include "launch.hpp"
#include <cstdint>
#endif

struct kernel_args
{
  uint32_t id;
  uint32_t reps;
  uint64_t result;
};

#if defined(__AMDGCN__)

// Wire opencl kernel up to c++ implementation

VISIBLE
hostrpc::x64_gcn_t::client_t hostrpc_pair_client[1];

static uint64_t gpu_call(hostrpc::x64_gcn_t::client_t *client, uint32_t id,
                         uint32_t reps);
extern "C" void __device_start_main(kernel_args *args)
{
  uint64_t r = gpu_call(hostrpc_pair_client, args->id, args->reps);
  args->result = r;
}
extern "C" void __device_start_cast(
    __attribute__((address_space(1))) void *asargs)
{
  kernel_args *args = (kernel_args *)asargs;
  __device_start_main(args);
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

static bool equal_page(hostrpc::page_t *lhs, hostrpc::page_t *rhs)
{
  bool eq = true;
  for (unsigned i = 0; i < 64; i++)
    {
      for (unsigned e = 0; e < 8; e++)
        {
          eq &= (lhs->cacheline[i].element[e] == rhs->cacheline[i].element[e]);
        }
    }
  return eq;
}

hostrpc::page_t scratch_store[MAXCLIENT];
hostrpc::page_t expect_store[MAXCLIENT];

uint64_t gpu_call(hostrpc::x64_gcn_t::client_t *client, uint32_t id,
                  uint32_t reps)
{
  const bool check_result = true;  // false => memory access fault?

#if 1
  hostrpc::page_t *scratch = &scratch_store[id];
  hostrpc::page_t *expect = &expect_store[id];
#else
  // This is associated with memory corruption. 8k per thread may be too much.
  hostrpc::page_t stack_scratch;
  hostrpc::page_t stack_expect;
  hostrpc::page_t *scratch = &stack_scratch;
  hostrpc::page_t *expect = &stack_expect;
#endif

  uint64_t failures = 0;
  for (unsigned r = 0; r < reps; r++)
    {
      if (check_result)
        {
          init_page(scratch, id + r);
          init_page(expect, id + r + 1);
        }

      client->invoke(scratch);

      if (0 && check_result)
        {
          if (!equal_page(scratch, expect))
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

    const char *kernel_entry = "__device_start.kd";
    uint64_t kernel_address = ex.get_symbol_address_by_name(kernel_entry);
    uint64_t client_address =
        ex.get_symbol_address_by_name("hostrpc_pair_client");

    uint32_t kernel_private_segment_fixed_size = 0;
    uint32_t kernel_group_segment_fixed_size = 0;

    {
      auto m = ex.get_kernel_info();
      auto it = m.find(std::string(kernel_entry));
      if (it != m.end())
        {
          kernel_private_segment_fixed_size =
              it->second.private_segment_fixed_size;
          kernel_group_segment_fixed_size = it->second.group_segment_fixed_size;
        }
      else
        {
          printf("fatal: get_kernel_info failed\n");
          exit(1);
        }
    }

    hsa_queue_t *queue;
    {
      hsa_status_t rc = hsa_queue_create(
          kernel_agent /* make the queue on this agent */,
          131072 /* todo: size it, this hardcodes max size for vega20 */,
          HSA_QUEUE_TYPE_SINGLE /* baseline */,
          NULL /* called on every async event? */,
          NULL /* data passed to previous */,
          // If sizes exceed these values, things are supposed to work slowly
          UINT32_MAX /* private_segment_size, 32_MAX is unknown */,
          UINT32_MAX /* group segment size, as above */, &queue);
      if (rc != HSA_STATUS_SUCCESS)
        {
          fprintf(stderr, "Failed to create queue\n");
          exit(1);
        }
    }

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

    hostrpc::x64_gcn_t::client_t *client =
        reinterpret_cast<hostrpc::x64_gcn_t::client_t *>(client_address);
    client[0] = p.client();
    printf("Initialized gpu client state\n");

    auto op_func = [](hostrpc::page_t *page) {
#if 0
      printf("gcn stress hit server function\n");
      printf("first values %lu/%lu/%lu\n",
             page->cacheline[0].element[0],
             page->cacheline[0].element[1],
             page->cacheline[1].element[0]);
#endif
      for (unsigned c = 0; c < 64; c++)
        {
          hostrpc::cacheline_t &line = page->cacheline[c];
#if 0
          std::swap(line.element[0], line.element[7]);
          std::swap(line.element[1], line.element[6]);
          std::swap(line.element[2], line.element[5]);
          std::swap(line.element[3], line.element[4]);
#endif
          for (unsigned i = 0; i < 8; i++)
            {
              line.element[i]++;
            }
        }
    };

    auto server_worker = [&](unsigned id) {
      unsigned count = 0;

      uint64_t server_location = 0;
      hostrpc::x64_gcn_t::server_t s = p.server();
      for (;;)
        {
          if (!server_live)
            {
              printf("server %u did %u tasks\n", id, count);
              break;
            }
          bool did_work = s.handle(op_func, &server_location);
          if (did_work)
            {
              count++;
            }
          else
            {
              platform::sleep_briefly();
            }
        }
    };

    unsigned nservers = 4;
    unsigned nclients = 1;
    unsigned per_client = 4096;

    unsigned derive = 5;
    for (unsigned i = 0; i < derive; i++)
      {
        nclients *= 2;
        per_client /= 2;
      }

    assert(nclients <= MAXCLIENT);

    // Looks like contention.
    // derive clients per-cl clock   run2   run3
    //  0         1    4096   4111   1420    588
    //  1         2    2048   4148   1416    750
    //  2         4    1024   4189   1571   1075
    //  3         8     512   4211   1899   1690
    //  4        16     256   4293   2549   1731
    //  5        32     128   4462   3835   2978
    //  6        64      64   4810   6427   5523
    //  7       128      32   5492  11592  10660
    //  8       256      16   6889  21974  21000
    //  9       512       8   9702  42725  41682
    // 10      1024       4  15174
    // 11      2048       2  26427
    // 12      4098       1  47647

    printf("x64-gcn spawning %u x64 servers, %u gcn clients\n", nservers,
           nclients);

    std::vector<launch_t<kernel_args> > client_store;
    {
      timer t("Launching clients");
      for (unsigned i = 0; i < nclients; i++)
        {
          kernel_args example = {
              .id = i, .reps = per_client, .result = UINT64_MAX};
          launch_t<kernel_args> tmp(kernel_agent, queue, kernel_address,
                                    kernel_private_segment_fixed_size,
                                    kernel_group_segment_fixed_size, example);
          client_store.emplace_back(std::move(tmp));
        }
    }
    printf("Spawn server workers\n");

    std::vector<std::thread> server_store;
    {
      timer t("Launching servers");
      for (unsigned i = 0; i < nservers; i++)
        {
          server_store.emplace_back(std::thread(server_worker, i));
        }
    }
    printf("Servers running\n");

    // make sure there's a server running before we wait for the result
    {
      timer t("Collect results");
      for (unsigned i = 0; i < nclients; i++)
        {
          kernel_args res = client_store[i]();
          CHECK(res.result == 0);
        }
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
