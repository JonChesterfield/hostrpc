#ifndef HOSTRPC_CLIENT_HPP_INCLUDED
#define HOSTRPC_CLIENT_HPP_INCLUDED

#include "common.hpp"

// Intend to have call and service working across gcn and x86
// The normal terminology is:
// Client makes a call to the server, which does ome work and sends back a reply

namespace hostrpc
{
template <size_t N>
struct client
{
  client(const mailbox_t<N>* inbox, mailbox_t<N>* outbox, page_t* buffer)
      : inbox(inbox), outbox(outbox), buffer(buffer)
  {
  }

  size_t spin_until_claimed_slot()
  {
    for (;;)
      {
        size_t slot = active.try_claim_any_empty_slot();
        if (slot != SIZE_MAX) { return slot; }
      }
  }

  void fill_buffer(page_t*) {}
  void use_buffer(page_t*) {}

  void rpc_invoke()
  {
    // wave_acquire_slot
    size_t slot = spin_until_claimed_slot();

    // wave_populate
    fill_buffer(&buffer[slot]);

    // wave_publish work
    outbox->claim_slot(slot);

    // wait for H1
    while ((*inbox)[slot] != 1)
      ;

    // recieve
    use_buffer(&buffer[slot]);

    // wave publish done
    outbox->release_slot(slot);

    // wait for H0
    while ((*inbox)[slot] != 0)
      ;

    // wave release slot
    active.release_slot(slot);
  }

  const mailbox_t<N>* inbox;
  mailbox_t<N>* outbox;
  page_t* buffer;

  slot_bitmap<N, __OPENCL_MEMORY_SCOPE_DEVICE> active;
};
}  // namespace hostrpc

#endif
