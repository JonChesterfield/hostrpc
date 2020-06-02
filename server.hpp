#ifndef HOSTRPC_SERVER_HPP_INCLUDED
#define HOSTRPC_SERVER_HPP_INCLUDED

#include "common.hpp"

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
  // no type_traits, thus no std::underlying_type
  using T = uint8_t;
  return static_cast<server_state>(static_cast<T>(lhs) | static_cast<T>(rhs));
}

inline server_state operator&(server_state lhs, server_state rhs)
{
  using T = uint8_t;
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

template <size_t N, typename Op, typename S>
struct server
{
  server(const mailbox_t<N>* inbox, mailbox_t<N>* outbox,
         slot_bitmap<N, __OPENCL_MEMORY_SCOPE_DEVICE>* active,
         page_t* remote_buffer, page_t* local_buffer, S step,
         Op operate = operate_nop)
      : inbox(inbox),
        outbox(outbox),
        active(active),
        remote_buffer(remote_buffer),
        local_buffer(local_buffer),

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
    uint64_t a = active->load_word(word);
    printf("%lu %lu %lu\n", i, o, a);
  }

  uint64_t work_todo(uint64_t word, uint64_t* inbox_word, uint64_t* outbox_word)
  {
    uint64_t i = inbox->load_word(word);
    uint64_t o = outbox->load_word(word);
    *inbox_word = i;
    *outbox_word = o;
    return i & ~o;
  }

  size_t find_and_claim_slot(uint64_t w, uint64_t* inbox_word,
                             uint64_t* outbox_word,
                             uint64_t* active_word)  // or SIZE_MAX
  {
    uint64_t work_available =
        work_todo(w, inbox_word, outbox_word) & ~active->load_word(w);
    // tries each bit in the work available at he call
    // doesn't load new information for work_available to preserve termination

    while (work_available != 0)
      {
        uint64_t idx = detail::ctz64(work_available);
        assert(detail::nthbitset64(work_available, idx));
        uint64_t slot = 64 * w + idx;
        // attempt to get that slot
        bool r = active->try_claim_empty_slot(slot, active_word);

        if (r)
          {
            // got the slot, check the work is still available
            uint64_t td = work_todo(w, inbox_word, outbox_word);
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

  // Returns true if it handled one task. Does not attempt multiple tasks
  bool rpc_handle()
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

    cache<N> c;

    size_t slot = SIZE_MAX;
    {
      // TODO: probably better to give up if there's no work to do instead of
      // keep waiting for some. That means this call always completes in
      // bounded time, after handling zero or one call
      for (uint64_t w = 0; w < inbox->words(); w++)
        {
          uint64_t inbox_word, outbox_word, active_word;
          // if there is no inbound work, it can be because the slots are
          // all filled with garbage on the server side
          try_garbage_collect_word_server(w);
          slot =
              find_and_claim_slot(w, &inbox_word, &outbox_word, &active_word);
          if (slot != SIZE_MAX)
            {
              c.i = inbox_word;
              c.o = outbox_word;
              c.a = active_word;
              break;
            }
        }
    }

    if (slot == SIZE_MAX)
      {
        return false;
      }

    c.init(slot);
    assert(c.is(0b101));

    step(__LINE__);

    __builtin_memcpy((void*)&local_buffer[slot], (void*)&remote_buffer[slot], sizeof(page_t));
   
    operate(&local_buffer[slot]);
    step(__LINE__);

    assert(c.is(0b101));

    // publish result
    {
      uint64_t o = outbox->claim_slot_returning_updated_word(slot);
      c.o = o;
    }
    assert(c.is(0b111));

    step(__LINE__);

    // can wait for G0 and then drop outbox, active slots
    // but that suspending a server thread until the client
    // drops the data. instead we can drop the lock and garbage collect later
    {
      uint64_t a = active->release_slot_returning_updated_word(slot);
      c.a = a;
    }

    assert(c.is(0b110));

    step(__LINE__);
    // leaves outbox live

    return true;
  }

  const mailbox_t<N>* inbox;
  mailbox_t<N>* outbox;
  slot_bitmap<N, __OPENCL_MEMORY_SCOPE_DEVICE>* active;
  page_t* remote_buffer;
  page_t* local_buffer;
  S step;
  Op operate;
};

template <size_t N, typename Op, typename S>
server<N,Op,S> make_server(const mailbox_t<N>* inbox, mailbox_t<N>* outbox,
         slot_bitmap<N, __OPENCL_MEMORY_SCOPE_DEVICE>* active,
         page_t* remote_buffer, page_t* local_buffer, S step,
         Op operate = operate_nop)
{
  return {inbox,outbox,active,remote_buffer,local_buffer,step,operate};
}

}  // namespace hostrpc
#endif
