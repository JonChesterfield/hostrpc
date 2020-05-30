#ifndef HOSTRPC_CLIENT_HPP_INCLUDED
#define HOSTRPC_CLIENT_HPP_INCLUDED

#include "common.hpp"
#include <functional>

// Intend to have call and service working across gcn and x86
// The normal terminology is:
// Client makes a call to the server, which does ome work and sends back a reply

namespace hostrpc
{
void fill_nop(page_t*) {}
void use_nop(page_t*) {}

template <size_t N>
struct client
{
  client(const mailbox_t<N>* inbox, mailbox_t<N>* outbox, page_t* buffer,
         std::function<void(page_t*)> fill = fill_nop,
         std::function<void(page_t*)> use = use_nop)

      : inbox(inbox), outbox(outbox), buffer(buffer), fill(fill), use(use)
  {
  }

  size_t spin_until_claimed_slot()
  {
    for (;;)
      {
        size_t slot = active.try_claim_any_empty_slot();
        if (slot != SIZE_MAX)
          {
            return slot;
          }
      }
  }

  void rpc_invoke()
  {
    // wave_acquire_slot
    size_t slot = spin_until_claimed_slot();

    // wave_populate
    fill(&buffer[slot]);

    // wave_publish work
    outbox->claim_slot(slot);

    // wait for H1
    while ((*inbox)[slot] != 1)
      ;

    // recieve
    use(&buffer[slot]);

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

  std::function<void(page_t*)> fill;
  std::function<void(page_t*)> use;
};
}  // namespace hostrpc

#endif
