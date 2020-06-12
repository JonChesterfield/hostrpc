#include "memory.hpp"
#include "server_impl.hpp"

using SZ = hostrpc::size_compiletime<128>;

void server_instance(hostrpc::slot_bitmap_all_svm<SZ> inbox,
                     hostrpc::slot_bitmap_all_svm<SZ> outbox,
                     hostrpc::slot_bitmap_device<SZ> active,
                     hostrpc::page_t* remote_buffer,
                     hostrpc::page_t* local_buffer)
{
  struct copy_functor_nop
      : public hostrpc::copy_functor_interface<copy_functor_nop>
  {
  };

  using server_type =
      hostrpc::server_impl<SZ, copy_functor_nop, hostrpc::operate_nop,
                           hostrpc::nop_stepper>;

  server_type s = {inbox, outbox, active, remote_buffer, local_buffer};

  for (;;)
    {
      s.rpc_handle(nullptr);
    }
}
