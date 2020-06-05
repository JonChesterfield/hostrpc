#ifndef HOSTRPC_SERVER_HPP_INCLUDED
#define HOSTRPC_SERVER_HPP_INCLUDED

#include "common.hpp"

namespace hostrpc
{
inline void operate_nop(page_t*, void*) {}

enum class server_state : uint8_t
{
  // inbox outbox active
  idle_server = 0b000,
  idle_thread = 0b001,
  garbage_available = 0b010,
  garbage_with_thread = 0b011,
  work_available = 0b100,
  work_with_thread = 0b101,
  result_available = 0b110,
  result_with_thread = 0b111,
};

template <size_t N, typename C, typename Op, typename S>
struct server
{
  server(C copy, const mailbox_t<N>* inbox, mailbox_t<N>* outbox,
         slot_bitmap<N, __OPENCL_MEMORY_SCOPE_DEVICE>* active,
         page_t* remote_buffer, page_t* local_buffer, S step,
         Op operate = operate_nop)
      : copy(copy),
        inbox(inbox),
        outbox(outbox),
        active(active),
        remote_buffer(remote_buffer),
        local_buffer(local_buffer),
        step(step),
        operate(operate)
  {
  }

  void dump_word(uint64_t word)
  {
    uint64_t i = inbox->load_word(word);
    uint64_t o = outbox->load_word(word);
    uint64_t a = active->load_word(word);
    printf("%lu %lu %lu\n", i, o, a);
  }

  uint64_t work_todo(uint64_t word)
  {
    uint64_t i = inbox->load_word(word);
    uint64_t o = outbox->load_word(word);
    return i & ~o;
  }

  size_t find_and_claim_slot(uint64_t w)  // or SIZE_MAX
  {
    uint64_t work_visible = work_todo(w);
    uint64_t work_available = work_visible & ~active->load_word(w);
    // tries each bit in the work available at he call
    // doesn't load new information for work_available to preserve termination

    while (work_available != 0)
      {
        uint64_t idx = detail::ctz64(work_available);
        assert(detail::nthbitset64(work_available, idx));
        uint64_t slot = 64 * w + idx;
        // attempt to get that slot
        uint64_t active_word;
        bool r = active->try_claim_empty_slot(slot, &active_word);

        if (r)
          {
            // got the slot, check the work is still available
            uint64_t td = work_todo(w);
            if (detail::nthbitset64(td, idx))
              {
                // got lock on a slot with work to do
                // said work is no longer available to another thread

                assert(!detail::nthbitset64(td & ~active->load_word(w), idx));
                step(__LINE__);

                return slot;
              }
          }

        // cas failed, or lost race, assume something else claimed it
        assert(detail::nthbitset64(work_available, idx));
        work_available = detail::clearnthbit64(work_available, idx);

        // some things which were availabe in the inbox won't be anymore
        // only clear those that are no longer present, don't insert ones
        // that have just arrived, in order to preserve termination
        // this is a potential optimisation - reduces trips through the loop
        // work_available &= inbox->load_word(w);
      }
    return SIZE_MAX;
  }

  // return true if no garbage (briefly) during call
  void try_garbage_collect_word_server(uint64_t w)
  {
    auto c = [](uint64_t i, uint64_t o) -> uint64_t { return ~i & o; };
    try_garbage_collect_word<N, decltype(c)>(c, inbox, outbox, active, w);
  }

  size_t words()
  {
    // todo: constexpr, static assert matches outbox and active
    return inbox->words();
  }

  void rpc_handle_given_slot(void* application_state, size_t slot)
  {
    assert(slot != SIZE_MAX);

    const uint64_t element = index_to_element(slot);
    const uint64_t subindex = index_to_subindex(slot);

    auto lock_held = [&]() -> bool {
      return detail::nthbitset64(active->load_word(element), subindex);
    };
    (void)lock_held;

    cache<N> c;
    c.init(slot);

    uint64_t i = inbox->load_word(element);
    uint64_t o = outbox->load_word(element);
    uint64_t a = active->load_word(element);
    __c11_atomic_thread_fence(__ATOMIC_ACQUIRE);
    c.i = i;
    c.o = o;
    c.a = a;

    // Called with a lock. The corresponding slot can be:
    //  inbox outbox    state  action
    //      0      0     idle    none
    //      1      0     work    work
    //      0      1  garbage collect
    //      1      1  waiting    none

    uint64_t this_slot = detail::setnthbit64(0, subindex);
    uint64_t work_todo = (i & ~o) & this_slot;
    uint64_t garbage_todo = (~i & o) & this_slot;

    assert((work_todo & garbage_todo) == 0);  // disjoint
    assert(lock_held());

    if (garbage_todo)
      {
        assert((o & this_slot) != 0);
        __c11_atomic_thread_fence(__ATOMIC_RELEASE);
        uint64_t updated_out =
            outbox->release_slot_returning_updated_word(slot);
        assert((updated_out & this_slot) == 0);

        return;
      }

    if (work_todo)
      {
        tracker.claim(slot);

        assert(c.is(0b101));

        step(__LINE__);

        copy.pull_to_server_from_client((void*)&local_buffer[slot],
                                        (void*)&remote_buffer[slot],
                                        sizeof(page_t));
        step(__LINE__);

        operate(&local_buffer[slot], application_state);
        step(__LINE__);

        copy.push_from_server_to_client((void*)&remote_buffer[slot],
                                        (void*)&local_buffer[slot],
                                        sizeof(page_t));
        step(__LINE__);

        assert(c.is(0b101));

        tracker.release(slot);

        // publish result
        {
          __c11_atomic_thread_fence(__ATOMIC_RELEASE);
          uint64_t o = outbox->claim_slot_returning_updated_word(slot);
          c.o = o;
        }
        assert(c.is(0b111));
        // leaves outbox live
        return;
      }

    step(__LINE__);

    assert(lock_held());
    return;
  }

  // Returns true if it handled one task. Does not attempt multiple tasks
  bool rpc_handle(void* application_state)
  {
    // printf("Server rpc_handle\n");

    step(__LINE__);

    // garbage collection should be fairly cheap when there is none,
    // and the presence of any occupied slots can starve the client
    for (uint64_t w = 0; w < inbox->words(); w++)
      {
        try_garbage_collect_word_server(w);
      }

    step(__LINE__);

    size_t slot = SIZE_MAX;
    {
      // TODO: probably better to give up if there's no work to do instead of
      // keep waiting for some. That means this call always completes in
      // bounded time, after handling zero or one call

      // always trying words in order is a potential problem in that later words
      // may never be collected. probably need the api to take a indicator of
      // where to start scanning from

      for (uint64_t w = 0; w < inbox->words(); w++)
        {
          // if there is no inbound work, it can be because the slots are
          // all filled with garbage on the server side
          try_garbage_collect_word_server(w);
          slot = find_and_claim_slot(w);
          if (slot != SIZE_MAX)
            {
              break;
            }
        }
    }

    if (slot == SIZE_MAX)
      {
        return false;
      }

    rpc_handle_given_slot(application_state, slot);

    uint64_t a = active->release_slot_returning_updated_word(slot);
    assert(!detail::nthbitset64(a, index_to_subindex(slot)));
    (void)a;

    return true;
  }

  C copy;
  const mailbox_t<N>* inbox;
  mailbox_t<N>* outbox;
  slot_bitmap<N, __OPENCL_MEMORY_SCOPE_DEVICE>* active;
  page_t* remote_buffer;
  page_t* local_buffer;
  S step;
  Op operate;
};

template <size_t N, typename C, typename Op, typename S>
server<N, C, Op, S> make_server(
    C copy, const mailbox_t<N>* inbox, mailbox_t<N>* outbox,
    slot_bitmap<N, __OPENCL_MEMORY_SCOPE_DEVICE>* active, page_t* remote_buffer,
    page_t* local_buffer, S step, Op operate = operate_nop)
{
  return {copy,          inbox,        outbox, active,
          remote_buffer, local_buffer, step,   operate};
}

}  // namespace hostrpc
#endif
