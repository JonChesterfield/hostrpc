#ifndef HOSTRPC_SERVER_HPP_INCLUDED
#define HOSTRPC_SERVER_HPP_INCLUDED

#include "common.hpp"

namespace hostrpc
{
template <size_t N>
struct server
{
  server(const mailbox_t<N>* inbox, mailbox_t<N>* outbox, page_t* buffer)
      : inbox(inbox), outbox(outbox), buffer(buffer)
  {
  }

  void rpc_handle() {}

  const mailbox_t<N>* inbox;
  mailbox_t<N>* outbox;
  page_t* buffer;

  slot_bitmap<N, __OPENCL_MEMORY_SCOPE_DEVICE> active;
};

}  // namespace hostrpc

#endif
