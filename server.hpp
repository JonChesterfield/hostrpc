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

  size_t find_candidate_server_available_bitmap(uint64_t w, uint64_t mask)
  {
    uint64_t i = inbox.load_word(w);
    uint64_t o = outbox.load_word(w);
    uint64_t a = active.load_word(w);
    __c11_atomic_thread_fence(__ATOMIC_ACQUIRE);

    uint64_t work = i & ~o;
    uint64_t garbage = ~i & o;
    uint64_t todo = work | garbage;
    uint64_t available = todo & ~a & mask;
    return available;
  }

  size_t find_candidate_server_slot(uint64_t w, uint64_t mask)
  {
    uint64_t available = find_candidate_server_available_bitmap(w, mask);
    if (available != 0)
      {
        return 64 * w + detail::ctz64(available);
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

  // may want to rename this, number-slots?
  size_t size() { return inbox.size(); }

  bool rpc_handle_given_slot(void* application_state, size_t slot)
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

        if (platform::is_master_lane())
          {
            uint64_t updated_out =
                outbox.release_slot_returning_updated_word(slot);
            assert((updated_out & this_slot) == 0);
            (void)updated_out;
          }
        assert(lock_held());
        return false;
      }

    if (!work_todo)
      {
        step(__LINE__, application_state);
        assert(lock_held());
        return false;
      }

    assert(c.is(0b101));
    step(__LINE__, application_state);
    tracker.claim(slot);

    // make the calls
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
      uint64_t o = platform::critical<uint64_t>(
          [&]() { return outbox.claim_slot_returning_updated_word(slot); });
      c.o = o;
    }
    assert(c.is(0b111));
    // leaves outbox live
    assert(lock_held());
    return true;
  }

  // Returns true if it handled one task. Does not attempt multiple tasks

  bool rpc_handle(void* application_state) noexcept
  {
    uint64_t location = 0;
    return rpc_handle(application_state, &location);
  }

  // location != NULL, used to round robin across slots
  bool rpc_handle(void* application_state, uint64_t* location_arg) noexcept
  {
    step(__LINE__, application_state);

    // garbage collection should be fairly cheap when there is none,
    // and the presence of any occupied slots can starve the client
    for (uint64_t w = 0; w < inbox.words(); w++)
      {
        // try_garbage_collect_word_server(w);
      }

    step(__LINE__, application_state);

    const uint64_t location = *location_arg % size();
    const uint64_t element = index_to_element(location);

    // skip bits in the first word <= subindex
    uint64_t mask = detail::setbitsrange64(index_to_subindex(location), 63);

    // Tries a few bits in element, then all bits in all the other words, then
    // all bits in element. This overshoots somewhat but ensures that all slots
    // are checked. Could truncate the last word to check each slot exactly once
    for (uint64_t wc = 0; wc < words() + 1; wc++)
      {
        uint64_t w = (element + wc) % words();
        uint64_t available = find_candidate_server_available_bitmap(w, mask);
        while (available != 0)
          {
            uint64_t idx = detail::ctz64(available);
            assert(detail::nthbitset64(available, idx));
            uint64_t slot = 64 * w + idx;
            uint64_t active_word;
            if (active.try_claim_empty_slot(slot, &active_word))
              {
                // Success, got the lock. Aim location_arg at next slot
                assert(active_word != 0);
                *location_arg = slot + 1;

                bool r = rpc_handle_given_slot(application_state, slot);

                step(__LINE__, application_state);

                if (platform::is_master_lane())
                  {
                    uint64_t a =
                        active.release_slot_returning_updated_word(slot);
                    assert(!detail::nthbitset64(a, index_to_subindex(slot)));
                    (void)a;
                  }

                return r;
              }

            // don't try the same slot repeatedly
            available = detail::clearnthbit64(available, idx);
          }

        mask = UINT64_MAX;
      }

    // Nothing hit, may as well go from the same location on the next call
    step(__LINE__, application_state);
    return false;
  }

  typename bt::inbox_t inbox;
  typename bt::outbox_t outbox;
  typename bt::locks_t active;
  page_t* remote_buffer;
  page_t* local_buffer;
};

}  // namespace hostrpc
#endif
