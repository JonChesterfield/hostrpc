#ifndef HOSTRPC_CLIENT_HPP_INCLUDED
#define HOSTRPC_CLIENT_HPP_INCLUDED

#include "common.hpp"
// Intend to have call and service working across gcn and x86
// The normal terminology is:
// Client makes a call to the server, which does some work and sends back a
// reply

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

template <typename SZ, typename Copy, typename Fill, typename Use,
          typename Step>
struct client_impl : public SZ
{
  using inbox_t = slot_bitmap_all_svm;
  using outbox_t = slot_bitmap_all_svm;
  using locks_t = slot_bitmap_device;
  using outbox_staging_t = slot_bitmap_coarse;

  client_impl(SZ sz, inbox_t inbox, outbox_t outbox, locks_t active,
              outbox_staging_t outbox_staging, page_t* remote_buffer,
              page_t* local_buffer)

      : SZ{sz},
        remote_buffer(remote_buffer),
        local_buffer(local_buffer),
        inbox(inbox),
        outbox(outbox),
        active(active),
        outbox_staging(outbox_staging)
  {
    // SZ is expected to be zero bytes or a uint64_t
    struct local : public SZ
    {
      float x;
    };
    constexpr bool sz_empty = sizeof(local) == sizeof(float);
    static_assert(sizeof(client_impl) == (sz_empty ? 48 : 56), "");
    static_assert(alignof(client_impl) == 8, "");
  }

  client_impl()
      : SZ{0},
        remote_buffer(nullptr),
        local_buffer(nullptr),
        inbox{},
        outbox{},
        active{},
        outbox_staging{}
  {
  }

  static void* operator new(size_t, client_impl* p) { return p; }

  void step(int x, void* y, void* z)
  {
    Step::call(x, y);
    Step::call(x, z);
  }

  size_t size() { return SZ::N(); }
  size_t words() { return size() / 64; }

  size_t find_candidate_client_slot(uint64_t w)
  {
    uint64_t i = inbox.load_word(size(), w);
    uint64_t o = outbox_staging.load_word(size(), w);
    uint64_t a = active.load_word(size(), w);
    __c11_atomic_thread_fence(__ATOMIC_ACQUIRE);

    uint64_t available = ~o & ~a;
    uint64_t garbage = i & o & ~a;

    uint64_t candidate = available | garbage;
    if (candidate != 0)
      {
        return 64 * w + detail::ctz64(candidate);
      }

    return SIZE_MAX;
  }

  void dump_word(size_t size, uint64_t word)
  {
    uint64_t i = inbox.load_word(size, word);
    uint64_t o = outbox_staging.load_word(size, word);
    uint64_t a = active.load_word(size, word);
    (void)(i + o + a);
    printf("%lu %lu %lu\n", i, o, a);
  }

  // true if it successfully made a call, false if no work to do or only gc
  // If there's no continuation, shouldn't require a use_application_state
  template <bool have_continuation>
  bool rpc_invoke_given_slot(void* fill_application_state,
                             void* use_application_state, size_t slot) noexcept
  {
    assert(slot != SIZE_MAX);
    const uint64_t element = index_to_element(slot);
    const uint64_t subindex = index_to_subindex(slot);

    cache c;
    c.init(slot);
    const size_t size = this->size();
    uint64_t i = inbox.load_word(size, element);
    uint64_t o = outbox_staging.load_word(size, element);
    uint64_t a = active.load_word(size, element);
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
            staged_release_slot_returning_updated_word(
                size, slot, &outbox_staging, &outbox);

            // outbox.release_slot_returning_updated_word(size, slot);
          }
        return false;
      }

    if (!available)
      {
        step(__LINE__, fill_application_state, use_application_state);
        return false;
      }

    assert(c.is(0b001));
    step(__LINE__, fill_application_state, use_application_state);
    tracker().claim(slot);

    // wave_populate

    // Fill may have no precondition, in which case this doesn't need to run
    Copy::pull_to_client_from_server((void*)&local_buffer[slot],
                                     (void*)&remote_buffer[slot],
                                     sizeof(page_t));
    step(__LINE__, fill_application_state, use_application_state);
    Fill::call(&local_buffer[slot], fill_application_state);
    step(__LINE__, fill_application_state, use_application_state);
    Copy::push_from_client_to_server((void*)&remote_buffer[slot],
                                     (void*)&local_buffer[slot],
                                     sizeof(page_t));
    step(__LINE__, fill_application_state, use_application_state);

    tracker().release(slot);

    // wave_publish work
    {
      __c11_atomic_thread_fence(__ATOMIC_RELEASE);
      uint64_t o = platform::critical<uint64_t>([&]() {
        return staged_claim_slot_returning_updated_word(
            size, slot, &outbox_staging, &outbox);
        // return outbox.claim_slot_returning_updated_word(size, slot);
      });
      c.o = o;
      assert(detail::nthbitset64(o, subindex));
      assert(c.is(0b011));
    }

    step(__LINE__, fill_application_state, use_application_state);

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
            uint32_t got = platform::critical<uint32_t>(
                [&]() { return inbox(size, slot, &loaded); });

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

        step(__LINE__, fill_application_state, use_application_state);
        Copy::pull_to_client_from_server((void*)&local_buffer[slot],
                                         (void*)&remote_buffer[slot],
                                         sizeof(page_t));
        step(__LINE__, fill_application_state, use_application_state);
        // call the continuation
        Use::call(&local_buffer[slot], use_application_state);

        step(__LINE__, fill_application_state, use_application_state);

        // Copying the state back to the server is a nop for aliased case,
        // and is only necessary if the server has a non-nop garbage clear
        // callback
        Copy::push_from_client_to_server((void*)&remote_buffer[slot],
                                         (void*)&local_buffer[slot],
                                         sizeof(page_t));

        step(__LINE__, fill_application_state, use_application_state);

        tracker().release(slot);

        // mark the work as no longer in use
        // todo: is it better to leave this for the GC?

        __c11_atomic_thread_fence(__ATOMIC_RELEASE);
        uint64_t o = platform::critical<uint64_t>([&]() {
          return staged_release_slot_returning_updated_word(
              size, slot, &outbox_staging, &outbox);

          // return outbox.release_slot_returning_updated_word(size, slot);
        });

        c.o = o;
        assert(c.is(0b101));

        step(__LINE__, fill_application_state, use_application_state);
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
  // dropping noinline here makes tests at least very slow and
  // possibly non-terminating.
  __attribute__((noinline)) bool rpc_invoke(
      void* fill_application_state, void* use_application_state) noexcept
  {
    step(__LINE__, fill_application_state, use_application_state);

    const size_t size = this->size();
    const size_t words = size / 64;
    // 0b111 is posted request, waited for it, got it
    // 0b110 is posted request, nothing waited, got one
    // 0b101 is got a result, don't need it, only spun up a thread for cleanup
    // 0b100 is got a result, don't need it

    step(__LINE__, fill_application_state, use_application_state);

    size_t slot = SIZE_MAX;
    // tries each word in sequnce. A cas failing suggests contention, in which
    // case try the next word instead of the next slot
    // may be worth supporting non-zero starting word for cache locality effects

    // the array is somewhat contended - attempt to spread out the load by
    // starting clients off at different points in the array. Doesn't make an
    // observable difference in the current benchmark.

    // if the invoke call performed garbage collection, the word is not
    // known to be contended so it may be worth trying a different slot
    // before trying a different word
#define CLIENT_OFFSET 0

#if CLIENT_OFFSET
    const uint32_t wstart = platform::client_start_slot();
    const uint32_t wend = wstart + words;
    for (uint64_t wi = wstart; wi != wend; wi++)
      {
        uint64_t w =
            wi % words;  // modulo may hurt here, and probably want 32 bit iv
#else
    for (uint64_t w = 0; w < words; w++)
      {
#endif
        uint64_t active_word;
        slot = find_candidate_client_slot(w);
        if (slot != SIZE_MAX)
          {
            if (active.try_claim_empty_slot(size, slot, &active_word))
              {
                // Success, got the lock.
                assert(active_word != 0);

                bool r = rpc_invoke_given_slot<have_continuation>(
                    fill_application_state, use_application_state, slot);

                // wave release slot
                step(__LINE__, fill_application_state, use_application_state);
                if (platform::is_master_lane())
                  {
                    active.release_slot_returning_updated_word(size, slot);
                  }
                  // returning if the invoke garbage collected is inefficient
                  // as the caller will need to try again, better to keep the
                  // position in the loop. This raises a memory access error
                  // however HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION: The
                  // agent attempted to access memory beyond the largest legal
                  // address.
#if 0
                if (r)
                  {
                    return true;
                  }
#else
                return r;
#endif
              }
          }
      }

    // couldn't get a slot, won't launch
    step(__LINE__, fill_application_state, use_application_state);
    return false;
  }

  page_t* remote_buffer;
  page_t* local_buffer;
  inbox_t inbox;
  outbox_t outbox;
  locks_t active;
  outbox_staging_t outbox_staging;
};

namespace indirect
{
struct fill
{
  static void call(hostrpc::page_t* page, void* pv)
  {
    hostrpc::closure_pair* p = static_cast<hostrpc::closure_pair*>(pv);
    p->func(page, p->state);
  };
};

struct use
{
  static void call(hostrpc::page_t* page, void* pv)
  {
    hostrpc::closure_pair* p = static_cast<hostrpc::closure_pair*>(pv);
    p->func(page, p->state);
  };
};

}  // namespace indirect

template <typename SZ, typename Copy, typename Step>
using client_indirect_impl =
    client_impl<SZ, Copy, indirect::fill, indirect::use, Step>;

}  // namespace hostrpc

#endif
