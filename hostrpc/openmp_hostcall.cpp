

#include "detail/common.hpp"
#include "detail/platform.hpp"
#include "detail/platform_detect.hpp"
#include "hostcall.hpp"
#include "hostcall_hsa.hpp"
#include "x64_gcn_type.hpp"

#include "detail/client_impl.hpp"
#include "detail/server_impl.hpp"

enum opcodes
{
  opcodes_nop = 0,
  opcodes_malloc = 1,
  opcodes_free = 2,
};

#if HOSTRPC_AMDGCN

// overrides weak functions in target_impl.hip
extern "C"
{
  void *__kmpc_impl_malloc(size_t);
  void __kmpc_impl_free(void *);
}

using client_type = hostrpc::x64_gcn_type<hostrpc::size_runtime>::client_type;

client_type *get_client()
{
  // Less obvious is that some asm is needed on the device to trigger the
  // handling,
  __asm__("; hostcall_invoke: record need for hostcall support\n\t"
          ".type needs_hostcall_buffer,@object\n\t"
          ".global needs_hostcall_buffer\n\t"
          ".comm needs_hostcall_buffer,4":::);

  // and that the result of hostrpc_assign_buffer, if zero is failure, else
  // it's written into a point in the implicit arguments where the GPU can
  // retrieve it from
  // size_t* argptr = (size_t *)__builtin_amdgcn_implicitarg_ptr();
  // result is found in argptr[3]

  size_t *argptr = (size_t *)__builtin_amdgcn_implicitarg_ptr();
  return (client_type *)argptr[3];
}

struct fill
{
  uint64_t *d;
  fill(uint64_t *d) : d(d) {}
  void operator()(hostrpc::page_t *page)
  {
    hostrpc::cacheline_t *line = &page->cacheline[platform::get_lane_id()];
    for (unsigned i = 0; i < 8; i++)
      {
        line->element[i] = d[i];
      }
  }
};

struct use
{
  uint64_t *d;
  use(uint64_t *d) : d(d) {}
  void operator()(hostrpc::page_t *page)
  {
    hostrpc::cacheline_t *line = &page->cacheline[platform::get_lane_id()];
    for (unsigned i = 0; i < 8; i++)
      {
        d[i] = line->element[i];
      }
  }
};

extern "C"
{
  void *__kmpc_impl_malloc(size_t x)
  {
    uint64_t data[8] = {0};
    data[0] = opcodes_malloc;
    data[1] = x;
    fill f(&data[0]);
    use u(&data[0]);
    client_type *c = get_client();
    bool success = false;
    while (!success)
      {
        success = c->rpc_invoke(f, u);
      }

    void *res;
    __builtin_memcpy(&res, &data[0], 8);
    return res;
  }

  void __kmpc_impl_free(void *x)
  {
    uint64_t data[8] = {0};
    data[0] = opcodes_free;
    __builtin_memcpy(&data[1], &x, 8);
    fill f(&data[0]);
    client_type *c = get_client();
    bool success = false;
    while (!success)
      {
        success = c->rpc_invoke(f);
      }
  }
}

#endif

#if HOSTRPC_HOST

#include "hsa.hpp"

// overrides weak functions in rtl.cpp
extern "C"
{
  // gets called repeatedly
  // uses agent to size queue
  unsigned long hostrpc_assign_buffer(hsa_agent_t agent, hsa_queue_t *this_Q,
                                      uint32_t device_id);
  hsa_status_t hostrpc_init();
  hsa_status_t hostrpc_terminate();
}

struct operate
{
  static void op( hostrpc::cacheline_t *line)
  {
    uint64_t op = line->element[0];
    switch (op)
      {
        case opcodes_nop:
          {
            break;
          }
        case opcodes_malloc:
          {
            uint64_t size;
            __builtin_memcpy(&size, &line->element[1], 8);
            // needs a memory region derived from a kernel_agent
            (void)size;  // todo
            break;
          }
        case opcodes_free:
          {
            void *ptr;
            __builtin_memcpy(&ptr, &line->element[1], 8);
            hsa_memory_free(ptr);
            break;
          }
      }
    return;
  }

  void operator()(hostrpc::page_t *page)
  {
    for (unsigned c = 0; c < 64; c++)
      {
        hostrpc::cacheline_t *line = &page->cacheline[c];
        op(line);
      }
  }
};

struct clear
{
  void operator()(hostrpc::page_t *page)
  {
    for (unsigned c = 0; c < 64; c++)
      {
        page->cacheline[c].element[0] = opcodes_nop;
      }
  }
};

struct device_state
{
  using T = hostrpc::x64_gcn_type<hostrpc::size_runtime>;
  T storage; // move-only type
  unsigned long device_client_pointer = 0;
  T::server_type * host_server_pointer = 0;
};

std::vector<device_state> hostrpc_storage;

unsigned long hostrpc_assign_buffer(hsa_agent_t agent, hsa_queue_t *this_Q,
                                    uint32_t device_id)
{
  (void)this_Q;
  while (hostrpc_storage.size() < device_id)
    {
      hostrpc_storage.emplace_back();
    }

  if (hostrpc_storage[device_id].device_client_pointer)
    {
      return hostrpc_storage[device_id].device_client_pointer;
    }


  // gets called a lot, so if the result is already available, need to return it
  // probably a vector<unsigned long>
  // is called under a lock

  uint32_t numCu;
  hsa_status_t err = hsa_agent_get_info(
      agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, &numCu);

  if (err != HSA_STATUS_SUCCESS)
    {
      return 0;
    }
  uint32_t waverPerCu;
  err = hsa_agent_get_info(
      agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU,
      &waverPerCu);
  if (err != HSA_STATUS_SUCCESS)
    {
      return 0;
    }

  auto size = hostrpc::size_runtime(numCu * waverPerCu);

  hsa_region_t fine_grain = hsa::region_fine_grained(agent);
  hsa_region_t coarse_grain = hsa::region_coarse_grained(agent);


  auto state = device_state::T (size, fine_grain.handle, coarse_grain.handle);
  if (!state.valid()) { return 0; }


  // todo: client, server pointers (vector resizes), spin up a thread

  return 0;
}

hsa_status_t hostrpc_init() { return HSA_STATUS_SUCCESS; }
hsa_status_t hostrpc_terminate() { hostrpc_storage.clear(); return HSA_STATUS_SUCCESS; }

#endif

