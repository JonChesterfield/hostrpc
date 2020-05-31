#ifndef HOSTRPC_SERVER_HPP_INCLUDED
#define HOSTRPC_SERVER_HPP_INCLUDED

#include "common.hpp"
#include <functional>

namespace hostrpc
{
void operate_nop(page_t*) {}

enum class server_state : uint8_t
{
  // bits inbox outbox active
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

            if (state[slot] == server_state::idle_server)
              {
                // Just discovered there is work available
                transition(slot, server_state::work_available);
              }

            // assert(state[slot] == server_state::work_available);
            bool r = active.try_claim_empty_slot(slot);
            if (r)
              {
                state[slot] |= static_cast<server_state>(0b001);
                assert(state[slot] == server_state::work_with_thread ||
                       state[slot] == server_state::idle_thread);

                // got the slot, check the work is still available
                if (detail::nthbitset64(work_todo(w), idx))
                  {
                    // got lock on a slot with work to do
                    // said work is no longer available to another thread

                    assert(!detail::nthbitset64(
                        work_todo(w) & ~active.load_word(w), idx));
                    step(__LINE__);
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
    size_t slot = SIZE_MAX;
    while (slot == SIZE_MAX)
      {
        slot = find_and_claim_slot();
      }
    step(__LINE__);

    operate(&buffer[slot]);
    step(__LINE__);

    // publish result
    outbox->claim_slot(slot);

    step(__LINE__);
    // wait for G0
    // this will change when supporting async transitions
    while ((*inbox)[slot] != 0)
      {
      }

    step(__LINE__);
    outbox->release_slot(slot);
    step(__LINE__);
    active.release_slot(slot);
    step(__LINE__);
  }

  const mailbox_t<N>* inbox;
  mailbox_t<N>* outbox;
  page_t* buffer;
  S step;
  std::function<void(page_t*)> operate;
  slot_bitmap<N, __OPENCL_MEMORY_SCOPE_DEVICE> active;
};

}  // namespace hostrpc

#endif
