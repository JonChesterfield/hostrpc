#pragma omp declare target
#include "detail/client_impl.hpp"
#include "detail/server_impl.hpp"

#include "allocator.hpp"
#include "host_client.hpp"

#include "x64_gcn_type.hpp"
#include "x64_target_type.hpp"

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

struct fill
{
  fill(hostrpc::page_t *d) : d(d) {}
  hostrpc::page_t *d;

  void operator()(hostrpc::page_t *page) { copy_page(page, d); };
};

struct use
{
  use(hostrpc::page_t *d) : d(d) {}
  hostrpc::page_t *d;

  void operator()(hostrpc::page_t *page) { copy_page(d, page); };
};

#pragma omp end declare target

#include <stdio.h>
#include <thread>
#include <unistd.h>

#include "hsa.hpp"

hostrpc::x64_gcn_type::client_type client_instance;
hostrpc::page_t scratch;

struct operate_test
{
  void operator()(hostrpc::page_t *) { fprintf(stderr, "Invoked operate\n"); }
};
struct clear_test
{
  void operator()(hostrpc::page_t *) { fprintf(stderr, "Invoked clear\n"); }
};

int main()
{
  hsa::init hsa;
  {
    printf("in openmp host\n");
    hsa_agent_t kernel_agent = hsa::find_a_gpu_or_exit();
    hsa_region_t fine_grained_region = hsa::region_fine_grained(kernel_agent);
    hsa_region_t coarse_grained_region =
        hsa::region_coarse_grained(kernel_agent);
    size_t N = 1920;
    hostrpc::x64_target_type<0> p(N);

    std::thread serv([&]() {
      uint32_t location = 0;

      for (unsigned i = 0; i < 16; i++)
        {
          bool r = p.server.rpc_handle<operate_test, clear_test>(
              operate_test{}, clear_test{}, &location);
          fprintf(stderr, "server ret %u\n", r);
          for (unsigned j = 0; j < 1000; j++)
            {
              usleep(100);
            }
        }
    });

    client_instance = p.client;

#pragma omp target map(tofrom : client_instance, scratch)
    {
      fill f(&scratch);
      use u(&scratch);
      client_instance.rpc_invoke<fill, use, true>(f, u);
    }

    serv.join();
  }
}
