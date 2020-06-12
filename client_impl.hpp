#ifndef HOSTRPC_CLIENT_HPP_INCLUDED
#define HOSTRPC_CLIENT_HPP_INCLUDED

#include "common.hpp"
#include "memory.hpp"
// Intend to have call and service working across gcn and x86
// The normal terminology is:
// Client makes a call to the server, which does some work and sends back a
// reply

// Layering falling apart a bit. Trying to work out if a signal is the missing
// piece for memory visibility
#include "x64_host_amdgcn_client_api.hpp"

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

template <size_t N, typename Copy, typename Fill, typename Use, typename Step>
struct client_impl
{
  using inbox_t = slot_bitmap_all_svm<N>;
  using outbox_t = slot_bitmap_all_svm<N>;
  using locks_t = slot_bitmap_device<N>;

  client_impl(inbox_t inbox, outbox_t outbox, locks_t active,
              page_t* remote_buffer, page_t* local_buffer)

      : inbox(inbox),
        outbox(outbox),
        active(active),
        remote_buffer(remote_buffer),
        local_buffer(local_buffer)
  {
    static_assert(sizeof(client_impl) == 40, "");
    static_assert(alignof(client_impl) == 8, "");
  }

  client_impl()
      : inbox{},
        outbox{},
        active{},
        remote_buffer(nullptr),
        local_buffer(nullptr)
  {
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
    try_garbage_collect_word<N, decltype(c)>(c, inbox, outbox, active, w);
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
  __attribute__((noinline)) bool rpc_invoke_given_slot(void* application_state,
                                                       size_t slot) noexcept
  {
    assert(slot != SIZE_MAX);
    const uint64_t element = index_to_element(slot);
    const uint64_t subindex = index_to_subindex(slot);

    cache c;
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
    uint64_t garbage_todo = i & o & this_slot;
    uint64_t available = ~i & ~o & this_slot;

    assert((garbage_todo & available) == 0);  // disjoint

    if (garbage_todo)
      {
        __c11_atomic_thread_fence(__ATOMIC_RELEASE);
        if (platform::is_master_lane())
          {
            outbox.release_slot_returning_updated_word(slot);
          }
        return false;
      }

    if (!available)
      {
        step(__LINE__, application_state);
        return false;
      }

    assert(c.is(0b001));
    step(__LINE__, application_state);
    tracker().claim(slot);

    // wave_populate
    Fill::call(&local_buffer[slot], application_state);
    step(__LINE__, application_state);
    Copy::push_from_client_to_server((void*)&remote_buffer[slot],
                                     (void*)&local_buffer[slot],
                                     sizeof(page_t));
    step(__LINE__, application_state);

    tracker().release(slot);

    // wave_publish work
    {
      __c11_atomic_thread_fence(__ATOMIC_RELEASE);
      uint64_t o = platform::critical<uint64_t>(
          [&]() { return outbox.claim_slot_returning_updated_word(slot); });
      c.o = o;
      assert(detail::nthbitset64(o, subindex));
      assert(c.is(0b011));
    }

    step(__LINE__, application_state);

    // current strategy is drop interest in the slot, then wait for the
    // server to confirm, then drop local thread

    // with a continuation, outbox is cleared before this thread returns
    // otherwise, garbage collection eneds to clear that outbox

    if (have_continuation)
      {
        // wait for H1, result available
        uint64_t loaded = 0;

        while (true)
          {
            uint32_t got = platform::critical<uint32_t>([&]() {
              // I think this should be relaxed, existing hostcall uses
              // acquire
              return inbox(slot, &loaded);
            });

            loaded = platform::broadcast_master(loaded);

            c.i = loaded;

            assert(got == 1 ? c.is(0b111) : c.is(0b011));

            if (got == 1)
              {
                break;
              }

            // make this spin slightly cheaper
            // todo: can the client do useful work while it waits? e.g. gc?
            platform::sleep();
          }

        __c11_atomic_thread_fence(__ATOMIC_ACQUIRE);

        assert(c.is(0b111));
        tracker().claim(slot);

        step(__LINE__, application_state);
        Copy::pull_to_client_from_server((void*)&local_buffer[slot],
                                         (void*)&remote_buffer[slot],
                                         sizeof(page_t));
        step(__LINE__, application_state);
        // call the continuation
        Use::call(&local_buffer[slot], application_state);

        step(__LINE__, application_state);

        tracker().release(slot);

        // mark the work as no longer in use
        // todo: is it better to leave this for the GC?

        __c11_atomic_thread_fence(__ATOMIC_RELEASE);
        uint64_t o = platform::critical<uint64_t>(
            [&]() { return outbox.release_slot_returning_updated_word(slot); });

        c.o = o;
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
  __attribute__((noinline)) bool rpc_invoke(void* application_state) noexcept
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

    size_t slot = SIZE_MAX;
    // tries each word in sequnce. A cas failing suggests contention, in which
    // case try the next word instead of the next slot
    // may be worth supporting non-zero starting word for cache locality effects
    for (uint64_t w = 0; w < words(); w++)
      {
        uint64_t active_word;
        slot = find_candidate_client_slot(w);
        if (slot != SIZE_MAX)
          {
            if (active.try_claim_empty_slot(slot, &active_word))
              {
                // Success, got the lock.
                assert(active_word != 0);

                bool r = rpc_invoke_given_slot<have_continuation>(
                    application_state, slot);

                // wave release slot
                step(__LINE__, application_state);
                if (platform::is_master_lane())
                  {
                    active.release_slot_returning_updated_word(slot);
                  }
                return r;
              }
          }
      }

    // couldn't get a slot, won't launch
    step(__LINE__, application_state);
    return false;
  }

  inbox_t inbox;
  outbox_t outbox;
  locks_t active;
  page_t* remote_buffer;
  page_t* local_buffer;
};

}  // namespace hostrpc

#endif
