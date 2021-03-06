#define VISIBLE __attribute__((visibility("default")))

#include "cxa_atexit.hpp"

// Kernel entry
#if defined __OPENCL__
void __device_start_cast(__global void *asargs);
kernel void __device_start(__global void *args) { __device_start_cast(args); }
#endif

#if !defined __OPENCL__

#include "base_types.hpp"
#include "detail/client_impl.hpp"
#include "detail/platform.hpp"
#include "detail/platform_detect.hpp"
#include "detail/server_impl.hpp"
#include "timer.hpp"
#include "x64_gcn_type.hpp"

#if (HOSTRPC_HOST)
#include "hsa.h"

#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

HOSTRPC_ANNOTATE static inline uint64_t get_thread_id()
{
  pid_t x = syscall(__NR_gettid);
  return x;
}

#endif

#include "allocator.hpp"
#include "host_client.hpp"

#if defined(__AMDGCN__)
static void copy_page(hostrpc::page_t *dst, hostrpc::page_t *src)
{
  unsigned id = platform::get_lane_id();
  hostrpc::cacheline_t *dline = &dst->cacheline[id];
  hostrpc::cacheline_t *sline = &src->cacheline[id];
  for (unsigned e = 0; e < 8; e++)
    {
      dline->element[e] = sline->element[e];
    }
}
#endif

struct fill
{
  fill(hostrpc::page_t *d) : d(d) {}
  hostrpc::page_t *d;

  void operator()(hostrpc::page_t *page)
  {
#if defined(__AMDGCN__)
    copy_page(page, d);
#else
    (void)page;
#endif
  };
};

struct use
{
  use(hostrpc::page_t *d) : d(d) {}
  hostrpc::page_t *d;

  void operator()(hostrpc::page_t *page)
  {
#if defined(__AMDGCN__)
    copy_page(d, page);
#else
    (void)page;
#endif
  };
};

#define MAXCLIENT (8 * 1024)  // lazy memory management, could malloc it

#if defined(__AMDGCN__)
#include <stdint.h>
#else
#include "launch.hpp"
#include <cstdint>
#endif

#define MAX_WAVES (1)
struct kernel_args
{
  uint32_t id;
  uint32_t reps;
  uint64_t result[MAX_WAVES];
  _Atomic(uint64_t) * control;
};

using SZ = hostrpc::size_runtime;

#if defined(__AMDGCN__)

// Wire opencl kernel up to c++ implementation
VISIBLE
hostrpc::x64_gcn_type<SZ>::client_type hostrpc_pair_client[1];

static uint64_t gpu_call(hostrpc::x64_gcn_type<SZ>::client_type *client,
                         uint32_t id, uint32_t reps);
extern "C" void __device_start_main(kernel_args *args)
{
  while (platform::atomic_load<uint64_t, __ATOMIC_ACQUIRE,
                               __OPENCL_MEMORY_SCOPE_DEVICE>(args->control) ==
         0)
    {
    }
  uint64_t wg = __builtin_amdgcn_workgroup_id_x();
  uint64_t r = gpu_call(hostrpc_pair_client, args->id + wg, args->reps);

  if (platform::is_master_lane())
    {
      args->result[wg] = r;
    }
}
extern "C" void __device_start_cast(
    __attribute__((address_space(1))) void *asargs)
{
  kernel_args *args = (kernel_args *)asargs;
  __device_start_main(args);
}

static void init_page(hostrpc::page_t *page, uint64_t v)
{
  // Only need to initialize lines corresponding to live lanes
  hostrpc::cacheline_t *line = &page->cacheline[platform::get_lane_id()];
  for (unsigned e = 0; e < 8; e++)
    {
      line->element[e] = v;
    }
}

static bool equal_page(hostrpc::page_t *lhs, hostrpc::page_t *rhs)
{
  // Would like to only check live lanes
  // TODO: Work out how reduction ops work on amdgcn
  unsigned id = platform::get_lane_id();
  hostrpc::cacheline_t *line_lhs = &lhs->cacheline[id];
  hostrpc::cacheline_t *line_rhs = &rhs->cacheline[id];

  uint32_t diff = 0;
  for (unsigned e = 0; e < 8; e++)
    {
      diff |= line_lhs->element[e] != line_rhs->element[e];
    }

  // TODO: Don't think this sort of reduction is defined on nvptx
  // amdgcn treats inactive lanes as containing zero
  diff = platform::reduction_sum(diff);
  diff = platform::broadcast_master(diff);

  return diff == 0;
}

// error: 'alignas' attribute cannot be applied to types ?

VISIBLE
hostrpc::page_t scratch_store[MAXCLIENT];
VISIBLE
hostrpc::page_t expect_store[MAXCLIENT];

uint64_t gpu_call(hostrpc::x64_gcn_type<SZ>::client_type *client, uint32_t id,
                  uint32_t reps)
{
  const bool check_result = true;
  id = platform::broadcast_master(id);
#if 1
  hostrpc::page_t *scratch = &scratch_store[id];
  hostrpc::page_t *expect = &expect_store[id];
  // asm volatile ("// " :"+r"(scratch), "+r"(expect) ::);

#else
  // This is associated with memory corruption. 8k per thread may be too much.
  hostrpc::page_t stack_scratch;
  hostrpc::page_t stack_expect;
  hostrpc::page_t *scratch = &stack_scratch;
  hostrpc::page_t *expect = &stack_expect;
#endif

  fill f(scratch);
  use u(scratch);

  uint32_t failures = 0;
  for (unsigned r = 0; r < reps; r++)
    {
      if (check_result)
        {
          if (0)
            {
              init_page(scratch, id + r);
              init_page(expect, id + r + 1);
            }

          init_page(scratch, 42);
          init_page(expect, 43);
        }

      bool rb = false;
      do
        {
          rb = client->rpc_invoke<fill, use>(f, u);
        }
      while (rb == false);

      if (check_result)
        {
          if (!equal_page(scratch, expect))
            {
              failures++;
            }
        }
    }

  failures = platform::reduction_sum(failures);
  failures = platform::broadcast_master(failures);

  return failures;
}

#else

#include "catch.hpp"
#include "hsa.hpp"
#include <cstring>
#include <thread>

#include "incbin.h"
namespace hostrpc
{
void init();
}

INCBIN(x64_gcn_stress_so, "x64_gcn_stress.gcn.so");

TEST_CASE("x64_gcn_stress")
{
  hsa::init hsa;
  {
    {
      hostrpc::init();
    }
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

    hsa_queue_t *queue = hsa::create_queue(kernel_agent);
    if (!queue)
    {
      fprintf(stderr, "Failed to create queue\n");
      exit(1);
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

    HOSTRPC_ATOMIC(bool) server_live(true);
    SZ N{1920};
    hostrpc::x64_gcn_type<SZ> p(N, fine_grained_region.handle,
                                coarse_grained_region.handle);

    // Great error from valgrind on gfx1010:
    // Address 0x8e08000 is in a --- mapped file /dev/dri/renderD128 segment
    // client[0] = p.client();

    {
      // put a default constructed instance in fine grain memory then overwrite
      // it with the p instance. May instead want to construct p into fine grain
      auto c = hsa::allocate(fine_grained_region,
                             sizeof(hostrpc::x64_gcn_type<SZ>::client_type));
      void *vc = c.get();
      memcpy(vc, &p.client, sizeof(p.client));

      int rc = hsa::copy_host_to_gpu(
          kernel_agent, reinterpret_cast<void *>(client_address),
          reinterpret_cast<const void *>(vc),
          sizeof(hostrpc::x64_gcn_type<SZ>::client_type));
      if (rc != 0)
        {
          fprintf(stderr, "Failed to copy client state to gpu\n");
          exit(1);
        }
    }

    printf("Initialized gpu client state\n");

    // This is presently being called for 'operate' and for 'clear', which is OK
    // for this test but not right in general.
    // Reasonable chance we also want to initialize the data before the first
    // call

    auto page_to_index = [&](hostrpc::page_t *page) -> int64_t {
      hostrpc::page_t *base = p.client.shared_buffer;

      intptr_t d = page - base;
      // fprintf(stderr,"base %lx, page %lx, diff %ld\n", (uint64_t)base,
      // (uint64_t)page, d);
      return d;
    };

    auto str = [](bool hit) -> const char * { return hit ? "FAIL" : "pass"; };
    auto op_func = [&](hostrpc::page_t *page) {
#if 1
      // printf("gcn stress hit server function\n");
      uint64_t f[3] = {
          page->cacheline[0].element[0],
          page->cacheline[0].element[1],
          page->cacheline[1].element[0],
      };
      uint64_t e = 42;
      bool hit = false;
      for (int i = 0; i < 3; i++)
        {
          if (f[i] != e) hit = true;
        }
      if (hit)
        {
          fprintf(stderr, "Operate (%s)(%ld): first values %lu/%lu/%lu\n",
                  str(hit), page_to_index(page), f[0], f[1], f[2]);
        }

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

    auto cl_func = [&](hostrpc::page_t *page) {
#if 1
      //   printf("gcn stress hit clear function\n");
      uint64_t f[3] = {
          page->cacheline[0].element[0],
          page->cacheline[0].element[1],
          page->cacheline[1].element[0],
      };
      uint64_t e = 43;
      bool hit = false;
      for (int i = 0; i < 3; i++)
        {
          if (f[i] != e) hit = true;
        }
      if (hit)
        {
          fprintf(stderr, "Clear (%s)(%ld): first values %lu/%lu/%lu\n",
                  str(hit), page_to_index(page), f[0], f[1], f[2]);
        }

#endif
      for (unsigned c = 0; c < 64; c++)
        {
          hostrpc::cacheline_t &line = page->cacheline[c];
          for (unsigned i = 0; i < 8; i++)
            {
              line.element[i] = 0;
            }
        }
    };

    auto server_worker = [&](unsigned id) {
      uint64_t thread_id = get_thread_id();
      fprintf(stderr, "Server worker id %u => %lx\n", id, thread_id);
      unsigned count = 0;

      uint32_t server_location = 0;

      for (;;)
        {
          if (!server_live)
            {
              printf("server %u did %u tasks\n", id, count);
              break;
            }
          bool did_work =
              p.server.rpc_handle<decltype(op_func), decltype(cl_func)>(
                  op_func, cl_func, &server_location);
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

    // number tasks = MAX_WAVES * nclients * per_client
    unsigned nservers = 2;
    unsigned nclients = 1;
    unsigned per_client = 64;  // 4096 * 2;

#ifndef DERIVE_VAL
#error "Req derive_val"
#endif

    unsigned derive = DERIVE_VAL;
    for (unsigned i = 0; i < derive; i++)
      {
        nclients *= 2;
        per_client /= 2;
      }

    assert(nclients <= MAXCLIENT);

    // doing 32k calls in about 4 seconds. 8k/s.
    // best effort 64k calls in 1 seconds, 64k/s, using 8 servers procs.
    // pcie latency is about 1us, best possible round trip order of 1M/s.
    // multiple round trips involved at present.

    // Looks like contention.
    // run5 is for 32k tasks, clients kernels, 8 waves, 4 servers
    // run6 is for 64k tasks
    // run7 is for 64k tasks, 8 servers, 8 waves
    // derive clients per-cl clock   run2   run3    run4   run5   run6   run7
    //  0         1    4096   4111   1420    588     646    644   1191    708
    //  1         2    2048   4148   1416    750     375    853   1391    913
    //  2         4    1024   4189   1571   1075     239   1267   1804   1330
    //  3         8     512   4211   1899   1690     170   2102   2647   2167
    //  4        16     256   4293   2549   1731     378   3743   3270   2454
    //  5        32     128   4462   3835   2978     487   3972   7070   4794
    //  6        64      64   4810   6427   5523     490   4682   8731   5160
    //  7       128      32   5492  11592  10660     828   9715   9832   9761
    //  8       256      16   6889  21974  21000    2027  19890
    //  9       512       8   9702  42725  41682    4554
    // 10      1024       4  15174                  9728
    // 11      2048       2  26427
    // 12      4098       1  47647

    fprintf(stderr,
            "x64-gcn spawning %u x64 servers, %u gcn clients, each doing %u "
            "reps in batches of %u\n",
            nservers, nclients, per_client, MAX_WAVES);

    auto ctrl_holder = hsa::allocate(fine_grained_region, 8);
    _Atomic(uint64_t) *control =
        (_Atomic(uint64_t) *)ctrl_holder.get();  // placement new

    platform::atomic_store<uint64_t, __ATOMIC_RELEASE,
                           __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(control, 0);

    // derive 0 x64-gcn spawning 4 x64 servers, 1 gcn clients, each doing 4096
    // reps in batches of 8 derive 1 x64-gcn spawning 4 x64 servers, 2 gcn
    // clients, each doing 2048 reps in batches of 8
    std::vector<launch_t<kernel_args> > client_store;
    {
      uint64_t max_id = (nclients - 1) * MAX_WAVES;
      if (max_id >= MAXCLIENT)
        {
          fprintf(stderr, "Max id %lu exceeds MAXCLIENT %lu\n", max_id,
                  (uint64_t)MAXCLIENT);
          exit(1);
        }

      uint64_t total = (uint64_t)nclients * MAX_WAVES * per_client;
      fprintf(
          stderr,
          "Launching %u kernels each doing %u waves with %u reps, total %lu\n",
          nclients, MAX_WAVES, per_client, total);

      timer t("Launching clients");
      for (unsigned i = 0; i < nclients; i++)
        {
          fprintf(stderr, "client %u, id %u\n", (4 + i), MAX_WAVES * (4 + i));
          kernel_args example = {.id = MAX_WAVES * (4 + i),
                                 .reps = per_client,
                                 .result = {0},
                                 .control = control};

          for (size_t i = 0; i < MAX_WAVES; i++)
            {
              example.result[i] = UINT64_MAX;
            }
          hsa_signal_t sig;
          auto rc = hsa_signal_create(1, 0, NULL, &sig);
          if (rc != HSA_STATUS_SUCCESS)
            {
              exit(1);
            }

          launch_t<kernel_args> tmp(
              kernel_agent, queue, std::move(sig), kernel_address,
              kernel_private_segment_fixed_size,
              kernel_group_segment_fixed_size, MAX_WAVES, example);
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

    // Release
    platform::atomic_store<uint64_t, __ATOMIC_RELEASE,
                           __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(control, 1);

    // make sure there's a server running before we wait for the result
    {
      timer t("Collect results");
      for (size_t i = 0; i < client_store.size(); i++)
        {
          kernel_args res = client_store[i]();
          for (size_t i = 0; i < MAX_WAVES; i++)
            {
              CHECK(res.result[i] == 0);
            }
        }
    }
    server_live = false;

    printf("Servers halting\n");
    for (auto &i : server_store)
      {
        i.join();
      }

    // Counters are incremented on the instance on the gpu. Retrieve before
    // dump.
    {
      auto c = hsa::allocate(fine_grained_region,
                             sizeof(hostrpc::x64_gcn_type<SZ>::client_type));
      void *vc = c.get();

      int rc = hsa::copy_host_to_gpu(
          kernel_agent, vc, reinterpret_cast<const void *>(client_address),
          sizeof(hostrpc::x64_gcn_type<SZ>::client_type));
      if (rc != 0)
        {
          fprintf(stderr, "Failed to copy client state back from gpu\n");
          exit(1);
        }

      memcpy(&p.client, vc, sizeof(p.client));
    }

    p.server_counters().dump();
    p.client_counters().dump();
    client_store.clear();  // tear down clients before closing hsa
    hsa_queue_destroy(queue);
  }
}

#endif
#endif
