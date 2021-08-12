#include "../detail/platform_detect.hpp"
#include "../detail/server_impl.hpp"

using SZ = hostrpc::size_compiletime<128>;

HOSTRPC_ANNOTATE void op_target(hostrpc::page_t *);
HOSTRPC_ANNOTATE void cl_target(hostrpc::page_t *);

namespace hostrpc
{
struct operate_direct
{
  HOSTRPC_ANNOTATE void operator()(uint32_t, hostrpc::page_t *page)
  {
    op_target(page);
  }

  operate_direct(const operate_direct&) = delete;
  operate_direct(operate_direct&&) = delete;

};

struct clear_direct
{
  HOSTRPC_ANNOTATE void operator()(uint32_t, hostrpc::page_t *page)
  {
    cl_target(page);
  }
  clear_direct(const clear_direct&) = delete;
  clear_direct(clear_direct&&) = delete;
};
}  // namespace hostrpc

using server_type_direct = hostrpc::server_impl<uint64_t, SZ>;

extern "C" HOSTRPC_ANNOTATE void server_instance_direct(
    server_type_direct::inbox_t inbox, server_type_direct::outbox_t outbox,
    server_type_direct::lock_t active, server_type_direct::staging_t staging,
    hostrpc::page_t *shared_buffer, hostrpc::operate_direct op,
    hostrpc::clear_direct cl)
{
  SZ sz;
  server_type_direct s = {sz, active, inbox, outbox, staging, shared_buffer};

  for (;;)
    {
      s.rpc_handle<hostrpc::operate_direct, hostrpc::clear_direct>(op, cl);
    }
}
