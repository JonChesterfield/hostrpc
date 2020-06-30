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

extern "C" void server_instance_direct(hostrpc::slot_bitmap_all_svm inbox,
                                       hostrpc::slot_bitmap_all_svm outbox,
                                       hostrpc::slot_bitmap_device active,
                                       hostrpc::page_t *remote_buffer,
                                       hostrpc::page_t *local_buffer,
                                       void *state_arg)
{
  using server_type =
      hostrpc::server_impl<SZ, hostrpc::copy_functor_memcpy_pull,
                           hostrpc::operate_direct, hostrpc::clear_direct,
                           hostrpc::nop_stepper>;

  SZ sz;
  server_type s = {sz, inbox, outbox, active, remote_buffer, local_buffer};

  for (;;)
    {
      s.rpc_handle(state_arg);
    }
}

extern "C" void server_instance_indirect(hostrpc::slot_bitmap_all_svm inbox,
                                         hostrpc::slot_bitmap_all_svm outbox,
                                         hostrpc::slot_bitmap_device active,
                                         hostrpc::page_t *remote_buffer,
                                         hostrpc::page_t *local_buffer,
                                         void *state_arg)
{
  using server_type =
      hostrpc::server_impl<SZ, hostrpc::copy_functor_memcpy_pull,
                           hostrpc::operate_indirect, hostrpc::clear_indirect,
                           hostrpc::nop_stepper>;

  SZ sz;
  server_type s = {sz, inbox, outbox, active, remote_buffer, local_buffer};

  pack arg = {.func = op_target, .state = state_arg};

  for (;;)
    {
      s.rpc_handle(static_cast<void *>(&arg));
    }
}
