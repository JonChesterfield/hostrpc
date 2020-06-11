#ifndef HOSTRPC_SERVER_HPP_INCLUDED
#define HOSTRPC_SERVER_HPP_INCLUDED

#include "common.hpp"

namespace hostrpc
{
struct operate_nop
{
  static void call(page_t*, void*) {}
};

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

template <size_t N, template <size_t> class bitmap_types, typename Copy,
          typename Op, typename Step>
struct server
{
  using bt = bitmap_types<N>;

  server(typename bt::inbox_t inbox, typename bt::outbox_t outbox,
         typename bt::locks_t active, page_t* remote_buffer,
         page_t* local_buffer)
      : inbox(inbox),
        outbox(outbox),
        active(active),
        remote_buffer(remote_buffer),
        local_buffer(local_buffer)
  {
  }

  server()
      : inbox{},
        outbox{},
        active{},
        remote_buffer(nullptr),
        local_buffer(nullptr)
  {
  }

  static constexpr size_t serialize_size() { return 5; }
  void serialize(uint64_t* to)
  {
    inbox.serialize(&to[0]);
    outbox.serialize(&to[1]);
    active.serialize(&to[2]);
    __builtin_memcpy(&to[3], &remote_buffer, 8);
    __builtin_memcpy(&to[4], &local_buffer, 8);
  }

  void deserialize(uint64_t* from)
  {
    inbox.deserialize(&from[0]);
    outbox.deserialize(&from[1]);
    active.deserialize(&from[2]);
    __builtin_memcpy(&remote_buffer, &from[3], 8);
    __builtin_memcpy(&local_buffer, &from[4], 8);
  }

  void step(int x, void* y) { Step::call(x, y); }

  void dump_word(uint64_t word)
  {
    uint64_t i = inbox.load_word(word);
    uint64_t o = outbox.load_word(word);
    uint64_t a = active.load_word(word);
    printf("%lu %lu %lu\n", i, o, a);
  }

  uint64_t work_todo(uint64_t word)
  {
    uint64_t i = inbox.load_word(word);
    uint64_t o = outbox.load_word(word);
    return i & ~o;
  }

  size_t find_and_claim_slot(uint64_t w,
                             void* application_state)  // or SIZE_MAX
  {
    uint64_t work_visible = work_todo(w);
    uint64_t work_available = work_visible & ~active.load_word(w);
    __c11_atomic_thread_fence(__ATOMIC_ACQUIRE);

    // tries each bit in the work available at the time of the call
    // doesn't load new information for work_available to preserve termination

    while (work_available != 0)
      {
        // this tries each slot in the
        uint64_t idx = detail::ctz64(work_available);
        assert(detail::nthbitset64(work_available, idx));
        uint64_t slot = 64 * w + idx;
        // attempt to get that slot
        uint64_t active_word;
        bool r = active.try_claim_empty_slot(slot, &active_word);
        if (r)
          {
            step(__LINE__, application_state);
            return slot;
          }

        // cas failed, or lost race, assume something else claimed it
        assert(detail::nthbitset64(work_available, idx));
        work_available = detail::clearnthbit64(work_available, idx);

        // some things which were available in the inbox won't be anymore
        // only clear those that are no longer present, don't insert ones
        // that have just arrived, in order to preserve termination
        // this is a potential optimisation - reduces trips through the loop
        // work_available &= inbox.load_word(w);
      }
    return SIZE_MAX;
  }

  // return true if no garbage (briefly) during call
  void try_garbage_collect_word_server(uint64_t w)
  {
    auto c = [](uint64_t i, uint64_t o) -> uint64_t { return ~i & o; };
    try_garbage_collect_word<N, bitmap_types, decltype(c)>(c, inbox, outbox,
                                                           active, w);
  }

  size_t words()
  {
    // todo: constexpr, static assert matches outbox and active
    return inbox.words();
  }

  void rpc_handle_given_slot(void* application_state, size_t slot)
  {
    assert(slot != SIZE_MAX);

    const uint64_t element = index_to_element(slot);
    const uint64_t subindex = index_to_subindex(slot);

    auto lock_held = [&]() -> bool {
      return detail::nthbitset64(active.load_word(element), subindex);
    };
    (void)lock_held;

    cache<N> c;
    c.init(slot);

    uint64_t i = inbox.load_word(element);
    uint64_t o = outbox.load_word(element);
    uint64_t a = active.load_word(element);
    __c11_atomic_thread_fence(__ATOMIC_ACQUIRE);
    c.i = i;
    c.o = o;
    c.a = a;

    // Called with a lock. The corresponding slot can be:
    //  inbox outbox    state  action
    //      0      0     idle    none
    //      0      1  garbage collect
    //      1      0     work    work
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
        uint64_t updated_out = outbox.release_slot_returning_updated_word(slot);
        assert((updated_out & this_slot) == 0);
        (void)updated_out;
        return;
      }

    if (work_todo)
      {
        tracker.claim(slot);

        assert(c.is(0b101));

        step(__LINE__, application_state);

        Copy::pull_to_server_from_client((void*)&local_buffer[slot],
                                         (void*)&remote_buffer[slot],
                                         sizeof(page_t));
        step(__LINE__, application_state);

        Op::call(&local_buffer[slot], application_state);
        step(__LINE__, application_state);

        Copy::push_from_server_to_client((void*)&remote_buffer[slot],
                                         (void*)&local_buffer[slot],
                                         sizeof(page_t));
        step(__LINE__, application_state);

        assert(c.is(0b101));

        tracker.release(slot);

        // publish result
        {
          __c11_atomic_thread_fence(__ATOMIC_RELEASE);
          uint64_t o = outbox.claim_slot_returning_updated_word(slot);
          c.o = o;
        }
        assert(c.is(0b111));
        // leaves outbox live
        return;
      }

    step(__LINE__, application_state);

    assert(lock_held());
    return;
  }

  // Returns true if it handled one task. Does not attempt multiple tasks
  bool rpc_handle(void* application_state) noexcept
  {
    step(__LINE__, application_state);

    // garbage collection should be fairly cheap when there is none,
    // and the presence of any occupied slots can starve the client
    for (uint64_t w = 0; w < inbox.words(); w++)
      {
        // try_garbage_collect_word_server(w);
      }

    step(__LINE__, application_state);

    size_t slot = SIZE_MAX;
    {
      // Always trying words in order is a potential problem in that later words
      // may never be collected. Probably need the api to take a indicator of
      // where to start scanning from

      for (uint64_t w = 0; w < inbox.words(); w++)
        {
          // if there is no inbound work, it can be because the slots are
          // all filled with garbage on the server side
          try_garbage_collect_word_server(w);
          slot = find_and_claim_slot(w, application_state);
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

    uint64_t a = active.release_slot_returning_updated_word(slot);
    assert(!detail::nthbitset64(a, index_to_subindex(slot)));
    (void)a;

    return true;
  }

  typename bt::inbox_t inbox;
  typename bt::outbox_t outbox;
  typename bt::locks_t active;
  page_t* remote_buffer;
  page_t* local_buffer;
};

}  // namespace hostrpc
#endif
