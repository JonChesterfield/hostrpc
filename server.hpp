#ifndef HOSTRPC_SERVER_HPP_INCLUDED
#define HOSTRPC_SERVER_HPP_INCLUDED

#include "common.hpp"
#include <functional>
#include <unistd.h>

namespace hostrpc
{
void operate_nop(page_t*) {}

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

inline server_state operator|(server_state lhs, server_state rhs)
{
  using T = std::underlying_type<server_state>::type;
  return static_cast<server_state>(static_cast<T>(lhs) | static_cast<T>(rhs));
}

inline server_state operator&(server_state lhs, server_state rhs)
{
  using T = std::underlying_type<server_state>::type;
  return static_cast<server_state>(static_cast<T>(lhs) & static_cast<T>(rhs));
}

inline server_state& operator|=(server_state& lhs, server_state rhs)
{
  lhs = lhs | rhs;
  return lhs;
}
inline server_state& operator&=(server_state& lhs, server_state rhs)
{
  lhs = lhs & rhs;
  return lhs;
}

template <size_t N, typename S>
struct server
{
  server(const mailbox_t<N>* inbox, mailbox_t<N>* outbox, page_t* buffer,
         S step, std::function<void(page_t*)> operate = operate_nop)
      : inbox(inbox),
        outbox(outbox),
        buffer(buffer),
        step(step),
        operate(operate)
  {
    for (size_t i = 0; i < N; i++)
      {
        assert(state[i] == server_state::idle_server);
      }
  }

  server_state state[N] = {};

  void transition(size_t slot, server_state to) { state[slot] = to; }

  void dump_word(uint64_t word)
  {
    uint64_t i = inbox->load_word(word);
    uint64_t o = outbox->load_word(word);
    uint64_t a = active.load_word(word);
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

                assert(!detail::nthbitset64(work_todo(w) & ~active.load_word(w),
                                            idx));
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
  bool try_garbage_collect_word(uint64_t w)
  {
    uint64_t i = inbox->load_word(w);
    uint64_t o = outbox->load_word(w);
    uint64_t a = active.load_word(w);

    uint64_t garbage_available = ~i & o & ~a;

    if (garbage_available == 0)
      {
        return true;
      }

    // Try to claim the locks on each garbage slot, if there is any
    {
      // propsed set of locks is the current set and the ones we're claiming
      assert((garbage_available & a) == 0);  // disjoint
      uint64_t proposed = garbage_available | a;
      uint64_t result;
      bool got = active.cas(w, a, proposed, &result);
      if (!got)
        {
          // lost the cas
          return false;
        }

      uint64_t locks_held = garbage_available;
      // Some of the slots may have already been garbage collected
      // in which case some of the input may be work available again
      i = inbox->load_word(w);
      o = outbox->load_word(w);

      uint64_t garbage_and_locked = ~i & o & locks_held;

      // clear locked bits in outbox
      uint64_t before = outbox->fetch_and(w, ~garbage_and_locked);
      // assert(before == (~i & o & ~locks_held));  // may be the wrong value

      // drop locks
      active.fetch_and(w, ~locks_held);

      return true;
    }
  }

  size_t words()
  {
    // todo: constexpr, static assert matches outbox and active
    return inbox->words();
  }

  void rpc_handle()
  {
    printf("Server rpc_handle\n");

    step(__LINE__);

    // garbage collection should be fairly cheap when there is none,
    // and the presence of any occupied slots can starve the client
    for (uint64_t w = 0; w < inbox->words(); w++)
      {
        try_garbage_collect_word(w);
      }

    size_t slot = SIZE_MAX;
    while (slot == SIZE_MAX)
      {
        // TODO: probably better to give up if there's no work to do instead of
        // keep waiting for some. That means this call always completes in
        // bounded time, after handling zero or one call
        for (uint64_t w = 0; w < inbox->words(); w++)
          {
            // if there is no inbound work, it can be because the slots are
            // all filled with garbage on the server side
            try_garbage_collect_word(w);
            slot = find_and_claim_slot(w);
            if (slot != SIZE_MAX)
              {
                break;
              }
          }
      }
    step(__LINE__);

    printf("got slot %lu\n", slot);
    assert((*inbox)[slot] == 1);
    step(__LINE__);

    operate(&buffer[slot]);
    step(__LINE__);

    // publish result
    assert((*inbox)[slot] == 1);
    outbox->claim_slot(slot);

    step(__LINE__);

    // can wait for G0 and then drop outbox, active slots
    // but that suspending a server thread until the client
    // drops the data. instead we can drop the lock and garbage collect later
    active.release_slot(slot);

    step(__LINE__);
    // leaves outbox live
  }

  const mailbox_t<N>* inbox;
  mailbox_t<N>* outbox;
  page_t* buffer;
  S step;
  std::function<void(page_t*)> operate;
  slot_bitmap<N, __OPENCL_MEMORY_SCOPE_DEVICE> active;
};  // namespace hostrpc

}  // namespace hostrpc
#endif