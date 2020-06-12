#include "memory.hpp"
#include "server_impl.hpp"

#define N 128
void server_instance(hostrpc::slot_bitmap_all_svm<N> inbox,
                     hostrpc::slot_bitmap_all_svm<N> outbox,
                     hostrpc::slot_bitmap_device<N> active,
                     hostrpc::page_t* remote_buffer,
                     hostrpc::page_t* local_buffer)
{
  struct copy_functor_nop
      : public hostrpc::copy_functor_interface<copy_functor_nop>
  {
  };

  using server_type =
      hostrpc::server_impl<N, copy_functor_nop, hostrpc::operate_nop,
                           hostrpc::nop_stepper>;

  server_type s = {inbox, outbox, active, remote_buffer, local_buffer};

  for (;;)
    {
      s.rpc_handle(nullptr);
    }
}
