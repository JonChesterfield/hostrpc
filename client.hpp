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

template <size_t N, typename S>
struct client
{
  client(const mailbox_t<N>* inbox, mailbox_t<N>* outbox, page_t* buffer,
         S step, std::function<void(page_t*)> fill = fill_nop,
         std::function<void(page_t*)> use = use_nop)

      : inbox(inbox),
        outbox(outbox),
        buffer(buffer),
        step(step),
        fill(fill),
        use(use)
  {
  }

  size_t spin_until_claimed_slot()
  {
    for (;;)
      {
        step(__LINE__);
        size_t slot = active.try_claim_any_empty_slot();
        if (slot != SIZE_MAX)
          {
            step(__LINE__);
            return slot;
          }
      }
  }

  void rpc_invoke()
  {
    step(__LINE__);

    // wave_acquire_slot
    size_t slot = spin_until_claimed_slot();
    step(__LINE__);

    // wave_populate
    fill(&buffer[slot]);
    step(__LINE__);

    // wave_publish work
    outbox->claim_slot(slot);
    step(__LINE__);

    // wait for H1
    while ((*inbox)[slot] != 1)
      {
        step(__LINE__);
      }

    step(__LINE__);
    // recieve
    use(&buffer[slot]);

    step(__LINE__);
    // wave publish done
    outbox->release_slot(slot);
    step(__LINE__);

    // wait for H0
    while ((*inbox)[slot] != 0)
      {
        step(__LINE__);
      }

    // wave release slot
    step(__LINE__);
    active.release_slot(slot);
  }

  const mailbox_t<N>* inbox;
  mailbox_t<N>* outbox;
  page_t* buffer;
  S step;
  std::function<void(page_t*)> fill;
  std::function<void(page_t*)> use;
  slot_bitmap<N, __OPENCL_MEMORY_SCOPE_DEVICE> active;
};
}  // namespace hostrpc

#endif
