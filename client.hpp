#ifndef HOSTRPC_CLIENT_HPP_INCLUDED
#define HOSTRPC_CLIENT_HPP_INCLUDED

#include "common.hpp"
#include <functional>
#include <unistd.h>

// Intend to have call and service working across gcn and x86
// The normal terminology is:
// Client makes a call to the server, which does ome work and sends back a reply

namespace hostrpc
{
void fill_nop(page_t*) {}
void use_nop(page_t*) {}

enum class client_state : uint8_t
{
  // inbox outbox active
  idle_client = 0b000,
  active_thread = 0b001,
  work_available = 0b011,
  unknownA = 0b010,  // Invalid? Would mean work posted, nothing returned yet,
                     // nothing waiting
  unknownB = 0b100,  // waiting for server to garbage collect,, no local thread
  garbage_with_thread = 0b101,  // transient state, 0b100 with local thread
  unknownC = 0b110,             // async call, server
  result_available = 0b111,     // thread waiting
};

// if inbox is set and outbox not, we are waiting for the server to collect
// garbage that is, can't claim the slot for a new thread is that a sufficient
// criteria for the slot to be awaiting gc?

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
        size_t slot = active.try_claim_any_empty_slot();
        if (slot != SIZE_MAX)
          {
            step(__LINE__);
            return slot;
          }
      }
  }

  size_t words()
  {
    // todo: constexpr, static assert matches outbox and active
    return inbox->words();
  }

  size_t find_candidate_client_slot()
  {
    for (uint64_t w = 0; w < words(); w++)
      {
        size_t f = find_candidate_client_slot(w);
        if (f != SIZE_MAX)
          {
            return f;
          }
      }
    return SIZE_MAX;
  }

  size_t find_candidate_client_slot(uint64_t w)
  {
    // find a slot which is currently available

    // active must be clear (no other thread using it)
    // outbox must be clear (no data in use)
    // server must also be clear (otherwise waiting on GC)
    // previous sketch featured inbox and outbox clear if active is clear,
    // as a thread waits on the gpu. Going to require all clear here:
    // Checking inbox means we can miss garbage collection at the end of a
    // synchronous task
    // Checking outbox opens the door to async launch

    uint64_t i = inbox->load_word(w);
    uint64_t o = outbox->load_word(w);
    uint64_t a = active.load_word(w);

    uint64_t some_use = i | o | a;

    uint64_t available = ~some_use;
    if (available != 0)
      {
        return 64 * w + detail::ctz64(available);
      }

    return SIZE_MAX;
  }

  void rpc_invoke()
  {
    step(__LINE__);

    // wave_acquire_slot
    // can currently acquire any slot, considering only acquiring a slot
    // where inbox is clear
    size_t slot = SIZE_MAX;
    while (slot == SIZE_MAX)
      {
        // spin until a 000 slot is found
        slot = find_candidate_client_slot();
        if (slot != SIZE_MAX)
          {
            if (active.try_claim_empty_slot(slot))
              {
                break;
              }
          }
      }

    // 0b001
    step(__LINE__);

    // wave_populate
    fill(&buffer[slot]);
    step(__LINE__);

    // wave_publish work
    outbox->claim_slot(slot);
    // 0b011

    step(__LINE__);

    // wait for H1, result available
    // is this necessary for async?
    while ((*inbox)[slot] != 1)
      {
        usleep(100);
      }
    // 0b111

    step(__LINE__);
    // recieve, nop if async
    use(&buffer[slot]);

    step(__LINE__);

    // current strategy is drop interest in the slot, then wait for the
    // server to confirm, then drop local thread

    // wave publish done
    outbox->release_slot(slot);
    // 0b101
    step(__LINE__);

    // wait for H0, result has been garbage collected by the host
    // todo: want to get rid of this busy spin in favour of deferred collection
    // I think that will need an extra client side bitmap

    // if we don't wait, would transition to 0b100
    // that is, no thread running, client not interested in the slot
    //
    while ((*inbox)[slot] != 0)
      {
        usleep(100);
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
