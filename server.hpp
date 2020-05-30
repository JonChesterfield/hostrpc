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

  uint64_t work_todo(uint64_t word)
  {
    uint64_t i = inbox->load_word(word);
    uint64_t o = outbox->load_word(word);
    return i & ~o;
  }

  size_t find_and_claim_slot()  // or SIZE_MAX
  {
    static_assert(inbox->words() == active.words(), "");
    for (uint64_t w = 0; w < inbox->words(); w++)
      {
        uint64_t a = active.load_word(w);
        uint64_t work_available = work_todo() & ~a;

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
                if (detail::nthbitset64(work_todo(w)))
                  {
                    // got lock on a slot with work to do
                    // said work is no longer available to another thread

                    assert(!detail::nthbitset64(
                        work_todo() & ~active.load_word(w), idx));
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

        return SIZE_MAX;
      }
  }

  void rpc_handle() {}

  const mailbox_t<N>* inbox;
  mailbox_t<N>* outbox;
  page_t* buffer;

  slot_bitmap<N, __OPENCL_MEMORY_SCOPE_DEVICE> active;
};

}  // namespace hostrpc

#endif
