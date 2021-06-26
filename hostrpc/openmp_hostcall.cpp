#include "../../../hostrpc/src/hostrpc.h"
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
  /*
  __asm__("; hostcall_invoke: record need for hostcall support\n\t"
          ".type needs_hostcall_buffer,@object\n\t"
          ".global needs_hostcall_buffer\n\t"
          ".comm needs_hostcall_buffer,4":::);

   */
  // and that the result of hostrpc_assign_buffer, if zero is failure, else
  // it's written into a point in the implicit arguments where the GPU can
  // retrieve it from
  // size_t* argptr = (size_t *)__builtin_amdgcn_implicitarg_ptr();
  // result is found in argptr[3]

  __asm__(
      "; hostcall_invoke: record need for hostcall support\n\t"
      ".type needs_hostcall_buffer,@object\n\t"
      ".global needs_hostcall_buffer\n\t"
      ".comm needs_hostcall_buffer,4" ::
          :);
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
  static void op(unsigned lane, hostrpc::cacheline_t *line)
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
        op(c, line);
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

unsigned long hostrpc_assign_buffer(hsa_agent_t agent, hsa_queue_t *this_Q,
                                    uint32_t device_id)
{
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

  size_t size = hostrpc::runtime_size(numCu * waverPerCu);

  //todo: make the client object and also spin up a thread
}

hsa_status_t hostrpc_init() { return HSA_STATUS_SUCCESS; }
hsa_status_t hostrpc_terminate() { return HSA_STATUS_SUCCESS; }

#endif

namespace hostcall_ops
{
#if HOSTRPC_HOST
void operate(hostrpc::page_t *page)
{
  for (unsigned c = 0; c < 64; c++)
    {
      hostrpc::cacheline_t &line = page->cacheline[c];

      assert(line.element[0] <= UINT32_MAX);
      uint32_t service_id = (uint32_t)line.element[0];
      uint64_t *payload = &line.element[1];

      // A bit dubious in that the existing code expects payload to have
      // length 8 and we're passing one of length 7, but nothing yet
      // implemented goes beyond [3]
      uint32_t device_id = 0;  // todo
      hostrpc_execute_service(static_cast<uint32_t>(service_id), device_id,
                              payload);
    }
}
void clear(hostrpc::page_t *page)
{
  for (unsigned c = 0; c < 64; c++)
    {
      hostrpc::cacheline_t &line = page->cacheline[c];
      for (unsigned i = 0; i < 8; i++)
        {
          line.element[i] = HOSTRPC_SERVICE_NO_OPERATION;
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

extern "C" hostrpc_result_t __attribute__((used))
hostrpc_invoke(uint32_t service_id, uint64_t arg0, uint64_t arg1, uint64_t arg2,
               uint64_t arg3, uint64_t arg4, uint64_t arg5, uint64_t arg6,
               uint64_t arg7)
{
  // changes the control flow in hsa/impl
  __asm__(
      "; hostcall_invoke: record need for hostcall support\n\t"
      ".type needs_hostcall_buffer,@object\n\t"
      ".global needs_hostcall_buffer\n\t"
      ".comm needs_hostcall_buffer,4" ::
          :);

  uint64_t buf[8] = {service_id, arg0, arg1, arg2, arg3, arg4, arg5, arg6};

  hostcall_client(buf);

  return {buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7], UINT64_MAX};
}

#endif

}  // namespace hostcall_ops
