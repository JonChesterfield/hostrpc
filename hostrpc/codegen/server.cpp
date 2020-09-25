#include "../detail/server_impl.hpp"

using SZ = hostrpc::size_compiletime<128>;

void op_target(hostrpc::page_t *, void *);
void cl_target(hostrpc::page_t *, void *);

struct pack
{
  void (*func)(hostrpc::page_t *, void *);
  void *state;
};

namespace hostrpc
{
struct operate_indirect
{
  static void call(hostrpc::page_t *page, void *pv)
  {
    pack *p = static_cast<pack *>(pv);
    p->func(page, p->state);
  }
};

struct operate_direct
{
  static void call(hostrpc::page_t *page, void *pv) { op_target(page, pv); }
};

struct clear_indirect
{
  static void call(hostrpc::page_t *page, void *pv)
  {
    pack *p = static_cast<pack *>(pv);
    p->func(page, p->state);
  }
};

struct clear_direct
{
  static void call(hostrpc::page_t *page, void *pv) { cl_target(page, pv); }
};
}  // namespace hostrpc

using server_type_direct =
    hostrpc::server_impl<uint32_t, SZ, hostrpc::copy_functor_memcpy_pull,
                         hostrpc::operate_direct, hostrpc::clear_direct,
                         hostrpc::nop_stepper>;

extern "C" void server_instance_direct(server_type_direct::inbox_t inbox,
                                       server_type_direct::outbox_t outbox,
                                       server_type_direct::lock_t active,
                                       server_type_direct::staging_t staging,
                                       hostrpc::page_t *remote_buffer,
                                       hostrpc::page_t *local_buffer,
                                       void *state_arg)
{
  SZ sz;
  server_type_direct s = {sz,      active,        inbox,       outbox,
                          staging, remote_buffer, local_buffer};

  for (;;)
    {
      s.rpc_handle(state_arg);
    }
}

using server_type_indirect =
    hostrpc::server_impl<uint32_t, SZ, hostrpc::copy_functor_memcpy_pull,
                         hostrpc::operate_indirect, hostrpc::clear_indirect,
                         hostrpc::nop_stepper>;

extern "C" void server_instance_indirect(
    server_type_indirect::inbox_t inbox, server_type_indirect::outbox_t outbox,
    server_type_indirect::lock_t active,
    server_type_indirect::staging_t staging, hostrpc::page_t *remote_buffer,
    hostrpc::page_t *local_buffer, void *state_arg)
{
  SZ sz;
  server_type_indirect s = {sz,      active,        inbox,       outbox,
                            staging, remote_buffer, local_buffer};

  pack arg = {.func = op_target, .state = state_arg};

  for (;;)
    {
      s.rpc_handle(static_cast<void *>(&arg));
    }
}
