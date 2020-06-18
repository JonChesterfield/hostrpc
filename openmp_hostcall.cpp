#include "../../../src/hostrpc_service_id.h"
#include "detail/common.hpp"
#include "detail/platform.hpp"
#include "hostcall.hpp"

#if defined(__x86_64__)

extern "C" void handlePayload(uint32_t service, uint64_t *payload);

#endif

namespace hostcall_ops
{
#if defined(__x86_64__)
void operate(hostrpc::page_t *page)
{
  for (unsigned c = 0; c < 64; c++)
    {
      hostrpc::cacheline_t &line = page->cacheline[c];
      uint64_t service_id = line.element[0];
      uint64_t *payload = &line.element[1];

      service_id = (service_id << 16u) >> 16u;
      assert(service_id <= UINT32_MAX);

      // A bit dubious in that the existing code expects payload to have
      // length 8 and we're passing one of length 7, but nothing yet
      // implemented goes beyond [3]
      handlePayload(static_cast<uint32_t>(service_id), payload);
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

typedef struct
{
  uint64_t arg0;
  uint64_t arg1;
  uint64_t arg2;
  uint64_t arg3;
  uint64_t arg4;
  uint64_t arg5;
  uint64_t arg6;
  uint64_t arg7;
} __ockl_hostcall_result_t;

extern "C" __ockl_hostcall_result_t hostcall_invoke(
    uint32_t service_id, uint64_t arg0, uint64_t arg1, uint64_t arg2,
    uint64_t arg3, uint64_t arg4, uint64_t arg5, uint64_t arg6, uint64_t arg7)
{
  uint64_t buf[8] = {service_id, arg0, arg1, arg2, arg3, arg4, arg5, arg6};
  uint64_t activemask = __builtin_amdgcn_read_exec();
  if (platform::is_master_lane())
    {
      // TODO: manipulate exec mask directly instead of looping
      for (uint64_t i = 0; i < 64; i++)
        {
          if (!hostrpc::detail::nthbitset64(activemask, i))
            {
              buf[0] = HOSTCALL_SERVICE_NO_OPERATION;
            }
        }
    }

  hostcall_client(buf);

  return {buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7], UINT64_MAX};
}

#endif

}  // namespace hostcall_ops
