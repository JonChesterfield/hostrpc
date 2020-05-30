#ifndef HOSTRPC_SERVER_HPP_INCLUDED
#define HOSTRPC_SERVER_HPP_INCLUDED

#include "common.hpp"
#include <functional>

namespace hostrpc
{
void operate_nop(page_t*) {}

template <size_t N, typename S>
struct server
{
  server(const mailbox_t<N>* inbox, mailbox_t<N>* outbox, page_t* buffer,
         S stepper,
         std::function<void(page_t*)> operate = operate_nop)
    : inbox(inbox), outbox(outbox), buffer(buffer), stepper(stepper), operate(operate)
  {
  }

  uint64_t work_todo(uint64_t word)
  {
    uint64_t i = inbox->load_word(word);
    uint64_t o = outbox->load_word(word);
    return i & ~o;
  }

  size_t find_and_claim_slot()  // or SIZE_MAX
  {
    // static_assert(decltype(*inbox)::words() == decltype(active)::words(),
    // "");
    for (uint64_t w = 0; w < inbox->words(); w++)
      {
        uint64_t work_available = work_todo(w) & ~active.load_word(w);

        while (work_available != 0)
          {
            uint64_t idx = detail::ctz64(work_available);
            assert(detail::nthbitset64(work_available, idx));
            uint64_t slot = 64 * w + idx;
            // attempt to get that slot
            bool r = active.try_claim_empty_slot(slot);
            if (r)
              {
                // got the slot, check the work is still available
                if (detail::nthbitset64(work_todo(w), idx))
                  {
                    // got lock on a slot with work to do
                    // said work is no longer available to another thread

                    assert(!detail::nthbitset64(
                        work_todo(w) & ~active.load_word(w), idx));
                    return slot;
                  }
              }

            // cas failed, or lost race, assume something else claimed it
            work_available = detail::clearnthbit64(work_available, idx);

            // some things which were availabe in the inbox won't be anymore
            // only clear those that are no longer present, don't insert ones
            // that have just arrived, in order to preserve termination
            // this is a potential optimisation - reduces trips through the loop
            // work_available &= inbox->load_word(w);
          }
      }
    return SIZE_MAX;
  }

  void rpc_handle()
  {
    size_t slot = find_and_claim_slot();
    if (slot == SIZE_MAX)
      {
        return;
      }

    operate(&buffer[slot]);

    // publish result
    outbox->claim_slot(slot);

    // wait for G0
    // this will change when supporting async transitions
    while ((*inbox)[slot] != 0)
      ;

    outbox->release_slot(slot);
    active.release_slot(slot);
  }

  const mailbox_t<N>* inbox;
  mailbox_t<N>* outbox;
  page_t* buffer;
  S stepper;
  std::function<void(page_t*)> operate;
  slot_bitmap<N, __OPENCL_MEMORY_SCOPE_DEVICE> active;

};

}  // namespace hostrpc

#endif
