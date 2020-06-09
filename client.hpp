#ifndef HOSTRPC_CLIENT_HPP_INCLUDED
#define HOSTRPC_CLIENT_HPP_INCLUDED

#include "common.hpp"
#include "memory.hpp"
// Intend to have call and service working across gcn and x86
// The normal terminology is:
// Client makes a call to the server, which does ome work and sends back a reply

namespace hostrpc
{
struct fill_nop
{
  static void call(page_t*, void*) {}
};

struct use_nop
{
  static void call(page_t*, void*) {}
};

enum class client_state : uint8_t
{
  // inbox outbox active
  idle_client = 0b000,
  active_thread = 0b001,
  work_available = 0b011,
  async_work_available = 0b010,
  done_pending_server_gc =
      0b100,  // waiting for server to garbage collect, no local thread
  garbage_with_thread = 0b101,  // transient state, 0b100 with local thread
  done_pending_client_gc =
      0b110,                 // created work, result available, no continuation
  result_available = 0b111,  // thread waiting
};

// if inbox is set and outbox not, we are waiting for the server to collect
// garbage that is, can't claim the slot for a new thread is that a sufficient
// criteria for the slot to be awaiting gc?

template <size_t N, template <size_t> class bitmap_types, typename Copy,
          typename Fill, typename Use, typename Step>
struct client
{
  using bt = bitmap_types<N>;

  client(typename bt::inbox_t inbox, typename bt::outbox_t outbox,
         typename bt::locks_t active, page_t* remote_buffer,
         page_t* local_buffer)

      : inbox(inbox),
        outbox(outbox),
        active(active),
        remote_buffer(remote_buffer),
        local_buffer(local_buffer)
  {
    static_assert(sizeof(client) == 40, "");
  }


  
  void step(int x, void* y) { Step::call(x, y); }

  size_t words()
  {
    // todo: constexpr, static assert matches outbox and active
    return inbox.words();
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

    // choosing to ignore inbox here - if inbox is set there's garbage to
    // collect
    uint64_t o = outbox.load_word(w);
    uint64_t a = active.load_word(w);
    __c11_atomic_thread_fence(__ATOMIC_ACQUIRE);

    uint64_t some_use = o | a;

    uint64_t available = ~some_use;
    if (available != 0)
      {
        return 64 * w + detail::ctz64(available);
      }

    return SIZE_MAX;
  }

  // return true if no garbage (briefly) during call
  void try_garbage_collect_word_client(uint64_t w)
  {
    auto c = [](uint64_t i, uint64_t) -> uint64_t { return i; };
    try_garbage_collect_word<N, bitmap_types, decltype(c)>(c, inbox, outbox,
                                                           active, w);
  }

  void dump_word(uint64_t word)
  {
    uint64_t i = inbox.load_word(word);
    uint64_t o = outbox.load_word(word);
    uint64_t a = active.load_word(word);
    printf("%lu %lu %lu\n", i, o, a);
  }

  // true if did work
  template <bool have_continuation>
  __attribute__((noinline))
  bool rpc_invoke_given_slot(void* application_state, size_t slot)
  {
    assert(slot != SIZE_MAX);
    const uint64_t element = index_to_element(slot);
    const uint64_t subindex = index_to_subindex(slot);

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
    //      0      0     work    work
    //      0      1     done    none
    //      1      0  garbage    none (waiting on server)
    //      1      1  garbage   clean
    // Inbox true means the result has come back
    // That this lock has been taken means no other thread is
    // waiting for that result
    uint64_t this_slot = detail::setnthbit64(0, subindex);

    uint64_t garbage = i & o & this_slot;
    uint64_t available = ~i & ~o & this_slot;

    assert((garbage & available) == 0);  // disjoint
    if (garbage)
      {
        __c11_atomic_thread_fence(__ATOMIC_RELEASE);
        outbox.release_slot_returning_updated_word(slot);
        return false;
      }

    if (!available)
      {
        return false;
      }

    assert(c.is(0b001));
    step(__LINE__, application_state);
    tracker.claim(slot);

    // wave_populate
    Fill::call(&local_buffer[slot], application_state);
    step(__LINE__, application_state);
    Copy::push_from_client_to_server((void*)&remote_buffer[slot],
                                     (void*)&local_buffer[slot],
                                     sizeof(page_t));
    step(__LINE__, application_state);

    tracker.release(slot);

    // wave_publish work
    {
      __c11_atomic_thread_fence(__ATOMIC_RELEASE);
      uint64_t o = outbox.claim_slot_returning_updated_word(slot);
      c.o = o;
    }

    assert(c.is(0b011));

    step(__LINE__, application_state);

    // current strategy is drop interest in the slot, then wait for the
    // server to confirm, then drop local thread

    // with a continuation, outbox is cleared before this thread returns
    // otherwise, garbage collection eneds to clear that outbox
    if (have_continuation)
      {
        // wait for H1, result available
        uint64_t loaded;
        unsigned rep = 0;
        unsigned max_rep = 10000;
        while (inbox(slot, &loaded) != 1)
          {
            c.i = loaded;
            assert(c.is(0b011));
            platform::sleep();
            rep++;
            if (rep == max_rep)
              {
                rep = 0;
                if (tracker.slots[slot] != UINT32_MAX)
                  {
                    printf("probably stalled here: waiting on slot %zu\n",
                           slot);
                    printf("slot %lu owned by %u\n", slot, tracker.slots[slot]);
                    // e.g. inbox 0, outbox 1, active 1,
                    inbox.dump();
                    outbox.dump();
                    active.dump();
                  }
              }
          }

        __c11_atomic_thread_fence(__ATOMIC_ACQUIRE);
        c.i = loaded;
        assert(c.is(0b111));

        tracker.claim(slot);

        step(__LINE__, application_state);
        Copy::pull_to_client_from_server((void*)&local_buffer[slot],
                                         (void*)&remote_buffer[slot],
                                         sizeof(page_t));
        step(__LINE__, application_state);
        // call the continuation
        Use::call(&local_buffer[slot], application_state);

        step(__LINE__, application_state);

        tracker.release(slot);

        // mark the work as no longer in use
        // todo: is it better to leave this for the GC?
        {
          __c11_atomic_thread_fence(__ATOMIC_RELEASE);
          uint64_t o = outbox.release_slot_returning_updated_word(slot);
          c.o = o;
        }

        assert(c.is(0b101));
        step(__LINE__, application_state);
      }

    // if we don't have a continuation, would return on 0b010
    // this wouldn't be considered garbage by client as inbox is clear
    // the server gets 0b100, does the work, sets the result to 0b110
    // that is then picked up by the client as 0b110

    // wait for H0, result has been garbage collected by the host
    // todo: want to get rid of this busy spin in favour of deferred collection
    // I think that will need an extra client side bitmap

    // We could wait for inbox[slot] != 0 which indicates the result
    // has been garbage collected, but that stalls the wave waiting for the hose
    // Instead, drop the warp and let the allocator skip occupied inbox slots
    return true;
  }

  // Returns true if it successfully launched the task
  template <bool have_continuation>
  __attribute__((noinline))
  bool rpc_invoke(void* application_state)
  {
    step(__LINE__, application_state);

    // 0b111 is posted request, waited for it, got it
    // 0b110 is posted request, nothing waited, got one
    // 0b101 is got a result, don't need it, only spun up a thread for cleanup
    // 0b100 is got a result, don't need it
    for (uint64_t w = 0; w < inbox.words(); w++)
      {
        // try_garbage_collect_word_client(w);
      }

    step(__LINE__, application_state);

    // wave_acquire_slot
    // can only acquire a slot which is 000
    size_t slot = SIZE_MAX;

    for (uint64_t w = 0; w < words(); w++)
      {
        uint64_t active_word;
        // may need to gc for there to be a slot
        // try_garbage_collect_word_client(w);
        slot = find_candidate_client_slot(w);
        if (slot != SIZE_MAX)
          {
            if (active.try_claim_empty_slot(slot, &active_word))
              {
                assert(active_word != 0);
                // printf("try_claim succeeded\n");
                // found a slot and locked it
                break;
              }
            else
              {
                slot = SIZE_MAX;
              }
          }
      }

    if (slot == SIZE_MAX)
      {
        // couldn't get a slot, won't launch
        step(__LINE__, application_state);
        return false;
      }

    bool r = rpc_invoke_given_slot<have_continuation>(application_state, slot);

    // wave release slot
    step(__LINE__, application_state);
    {
      uint64_t a = active.release_slot_returning_updated_word(slot);
      (void)a;
    }

    return r;
  }

  typename bt::inbox_t inbox;
  typename bt::outbox_t outbox;
  typename bt::locks_t active;
  page_t* remote_buffer;
  page_t* local_buffer;
};

}  // namespace hostrpc

#endif
