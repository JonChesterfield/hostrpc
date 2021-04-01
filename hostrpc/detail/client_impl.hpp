#ifndef HOSTRPC_CLIENT_HPP_INCLUDED
#define HOSTRPC_CLIENT_HPP_INCLUDED

#include "common.hpp"
#include "counters.hpp"
#include "platform_detect.hpp"

// Intend to have call and service working across gcn and x86
// The normal terminology is:
// Client makes a call to the server, which does some work and sends back a
// reply

namespace hostrpc
{
struct fill_nop
{
  HOSTRPC_ANNOTATE void operator()(page_t*) {}
};

struct use_nop
{
  HOSTRPC_ANNOTATE void operator()(page_t*) {}
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

// enabling counters breaks codegen for amdgcn,
template <typename WordT, typename SZT, typename Copy,
          typename Counter = counters::client>
struct client_impl : public SZT, public Counter
{
  using Word = WordT;
  using SZ = SZT;
  using lock_t = lock_bitmap<Word>;
  using inbox_t = message_bitmap<Word>;
  using outbox_t = message_bitmap<Word>;
  using staging_t = slot_bitmap_device_local<Word>;
  HOSTRPC_ANNOTATE constexpr size_t wordBits() const
  {
    return 8 * sizeof(Word);
  }
  HOSTRPC_ANNOTATE uint32_t size() const { return SZ::N(); }
  HOSTRPC_ANNOTATE uint32_t words() const { return size() / wordBits(); }

  page_t* remote_buffer;
  page_t* local_buffer;
  lock_t active;

  inbox_t inbox;
  outbox_t outbox;
  staging_t staging;

  HOSTRPC_ANNOTATE client_impl()
      : SZ{0},
        Counter{},
        remote_buffer(nullptr),
        local_buffer(nullptr),
        active{},
        inbox{},
        outbox{},
        staging{}
  {
  }
  HOSTRPC_ANNOTATE ~client_impl() {}
  HOSTRPC_ANNOTATE client_impl(SZ sz, lock_t active, inbox_t inbox,
                               outbox_t outbox, staging_t staging,
                               page_t* remote_buffer, page_t* local_buffer)

      : SZ{sz},
        Counter{},
        remote_buffer(remote_buffer),
        local_buffer(local_buffer),
        active(active),
        inbox(inbox),
        outbox(outbox),
        staging(staging)
  {
    constexpr size_t client_size = 48;

    // SZ is expected to be zero bytes or a uint64_t
    struct SZ_local : public SZ
    {
      float x;
    };
    // Counter is zero bytes for nop or potentially many
    struct Counter_local : public Counter
    {
      float x;
    };
    constexpr bool SZ_empty = sizeof(SZ_local) == sizeof(float);
    constexpr bool Counter_empty = sizeof(Counter_local) == sizeof(float);

    constexpr size_t SZ_size = hostrpc::round8(SZ_empty ? 0 : sizeof(SZ));
    constexpr size_t Counter_size = Counter_empty ? 0 : sizeof(Counter);

    constexpr size_t total_size = client_size + SZ_size + Counter_size;

    static_assert(sizeof(client_impl) == total_size, "");
    static_assert(alignof(client_impl) == 8, "");
  }

  HOSTRPC_ANNOTATE void dump()
  {
#if HOSTRPC_HAVE_STDIO
    fprintf(stderr, "remote_buffer %p\n", remote_buffer);
    fprintf(stderr, "local_buffer  %p\n", local_buffer);
    fprintf(stderr, "inbox         %p\n", inbox.a);
    fprintf(stderr, "outbox        %p\n", outbox.a);
    fprintf(stderr, "active        %p\n", active.a);
    fprintf(stderr, "outbox stg    %p\n", staging.a);
#endif
  }

  HOSTRPC_ANNOTATE static void* operator new(size_t, client_impl* p)
  {
    return p;
  }

  HOSTRPC_ANNOTATE client_counters get_counters() { return Counter::get(); }

  // rpc_invoke returns true if it successfully launched the task
  // returns false if no slot was available

  // Return after calling fill(), i.e. does not wait for server
  template <typename Fill>
  HOSTRPC_ANNOTATE bool rpc_invoke(Fill fill) noexcept
  {
    // maybe name this async
    struct Use
    {
      HOSTRPC_ANNOTATE void operator()(hostrpc::page_t*){};
    };

    Use u;
    return rpc_invoke_impl<Fill, Use, false>(fill, u);
  }

  // Return after calling use(), i.e. waits for server
  template <typename Fill, typename Use>
  HOSTRPC_ANNOTATE bool rpc_invoke(Fill fill, Use use) noexcept
  {
    return rpc_invoke_impl<Fill, Use, true>(fill, use);
  }

 private:
  template <typename Fill, typename Use, bool have_continuation>
  HOSTRPC_ANNOTATE bool rpc_invoke_impl(Fill fill, Use use) noexcept
  {
    const uint32_t size = this->size();
    const uint32_t words = this->words();
    // 0b111 is posted request, waited for it, got it
    // 0b110 is posted request, nothing waited, got one
    // 0b101 is got a result, don't need it, only spun up a thread for cleanup
    // 0b100 is got a result, don't need it

    uint32_t slot = UINT32_MAX;
    // tries each word in sequnce. A cas failing suggests contention, in which
    // case try the next word instead of the next slot
    // may be worth supporting non-zero starting word for cache locality effects

    // the array is somewhat contended - attempt to spread out the load by
    // starting clients off at different points in the array. Doesn't make an
    // observable difference in the current benchmark.

    // if the invoke call performed garbage collection, the word is not
    // known to be contended so it may be worth trying a different slot
    // before trying a different word
    for (uint32_t w = 0; w < words; w++)
      {
        slot = find_candidate_client_slot(w);
        if (slot == UINT32_MAX)
          {
            // no slot
            Counter::no_candidate_slot();
          }
        else
          {
            uint64_t cas_fail_count = 0;
            if (active.try_claim_empty_slot(size, slot, &cas_fail_count))
              {
                // Success, got the lock.
                Counter::cas_lock_fail(cas_fail_count);

                // Test if it is available, e.g. isn't garbage
                if (rpc_invoke_verify_slot_available(slot))
                  {
                    rpc_invoke_fill_given_slot<Fill>(fill, slot);

                    if (have_continuation)
                      {
                        // if only garbage collected, don't call use
                        rpc_invoke_use_given_slot(use, slot);
                      }

                    // wave release slot
                    if (platform::is_master_lane())
                      {
                        active.release_slot(size, slot);
                      }

                    // invoke() garbage collecting is transparent
                    // to caller, so only returns true on non-garbage
                    // work done
                    return true;
                  }
              }
            else
              {
                Counter::missed_lock_on_candidate_slot();
              }
          }
      }

    // couldn't get a slot in any word, won't launch
    return false;
  }

  HOSTRPC_ANNOTATE uint32_t find_candidate_client_slot(uint32_t w)
  {
    Word i = inbox.load_word(size(), w);
    Word o = staging.load_word(size(), w);
    Word a = active.load_word(size(), w);
    platform::fence_acquire();

    // inbox == outbox == 0 => available for use
    Word available = ~i & ~o & ~a;

    // 1 0 => garbage waiting on server
    // 1 1 => garbage that client can act on
    // Take those that client can act on and are not locked
    Word garbage_todo = i & o & ~a;

    Word candidate = available | garbage_todo;
    if (candidate != 0)
      {
        return wordBits() * w + bits::ctz(candidate);
      }

    return UINT32_MAX;
  }

  HOSTRPC_ANNOTATE void release_slot(uint32_t slot)
  {
    const uint32_t size = this->size();
    platform::fence_release();
    uint64_t cas_fail_count = 0;
    uint64_t cas_help_count = 0;
    // opencl has incomplete support for lambdas, can't pass address of
    // captured variable.
    if (platform::is_master_lane())
      {
        staged_release_slot(size, slot, &staging, &outbox, &cas_fail_count,
                            &cas_help_count);
      }
    cas_fail_count = platform::broadcast_master(cas_fail_count);
    cas_help_count = platform::broadcast_master(cas_help_count);
    Counter::garbage_cas_fail(cas_fail_count);
    Counter::garbage_cas_help(cas_help_count);
  }

  HOSTRPC_ANNOTATE void dump_word(uint32_t size, Word word)
  {
    Word i = inbox.load_word(size, word);
    Word o = staging.load_word(size, word);
    Word a = active.load_word(size, word);
    (void)(i + o + a);
    printf("%lu %lu %lu\n", i, o, a);
  }

  // true if it successfully made a call, false if no work to do or only gc
  // If there's no continuation, shouldn't require a use_application_state

  // Series of functions called with lock[slot] held. Garbage collect if
  // necessary, use slot if possible, wait & call continuationo if necessary

  HOSTRPC_ANNOTATE bool rpc_invoke_verify_slot_available(uint32_t slot) noexcept
  {
    assert(slot != UINT32_MAX);
    const uint32_t element = index_to_element<Word>(slot);
    const uint32_t subindex = index_to_subindex<Word>(slot);

    const uint32_t size = this->size();
    Word i = inbox.load_word(size, element);
    Word o = staging.load_word(size, element);
    platform::fence_acquire();

    // Called with a lock. The corresponding slot can be:
    //  inbox outbox    state  action outbox'
    //      0      0    avail    work       1
    //      0      1     done    none       -
    //      1      0  garbage    none       -
    //      1      1  garbage   clean       0
    // Inbox true means the result has come back
    // That this lock has been taken means no other thread is
    // waiting for that result

    Word this_slot = bits::setnthbit((Word)0, subindex);
    Word garbage_todo = i & o & this_slot;
    Word available = ~i & ~o & this_slot;

    assert((garbage_todo & available) == 0);  // disjoint

    if (garbage_todo)
      {
        release_slot(slot);
        return false;
      }

    if (!available)
      {
        Counter::got_lock_after_work_done();
        return false;
      }

    // Slot is available for use.
    return true;
  }

  template <typename Fill>
  HOSTRPC_ANNOTATE void rpc_invoke_fill_given_slot(Fill fill,
                                                   uint32_t slot) noexcept
  {
    assert(slot != UINT32_MAX);
    const uint32_t size = this->size();

    // wave_populate
    // Fill may have no precondition, in which case this doesn't need to run
    Copy::pull_to_client_from_server(&local_buffer[slot], &remote_buffer[slot]);
    fill(&local_buffer[slot]);
    Copy::push_from_client_to_server(&remote_buffer[slot], &local_buffer[slot]);

    // wave_publish work
    {
      platform::fence_release();
      uint64_t cas_fail_count = 0;
      uint64_t cas_help_count = 0;
      if (platform::is_master_lane())
        {
          staged_claim_slot(size, slot, &staging, &outbox, &cas_fail_count,
                            &cas_help_count);
        }
      cas_fail_count = platform::broadcast_master(cas_fail_count);
      cas_help_count = platform::broadcast_master(cas_help_count);
      Counter::publish_cas_fail(cas_fail_count);
      Counter::publish_cas_help(cas_help_count);
    }

    // current strategy is drop interest in the slot, then wait for the
    // server to confirm, then drop local thread

    // with a continuation, outbox is cleared before this thread returns
    // otherwise, garbage collection needed to clear that outbox

    // if we don't have a continuation, would return on 0b010
    // this wouldn't be considered garbage by client as inbox is clear
    // the server gets 0b100, does the work, sets the result to 0b110
    // that is then picked up by the client as 0b110

    // wait for H0, result has been garbage collected by the host
    // todo: want to get rid of this busy spin in favour of deferred collection
    // I think that will need an extra client side bitmap

    // We could wait for inbox[slot] != 0 which indicates the result
    // has been garbage collected, but that stalls the wave waiting for the host
    // Instead, drop the warp and let the allocator skip occupied inbox slots
  }

  template <typename Use>
  HOSTRPC_ANNOTATE void rpc_invoke_use_given_slot(Use use,
                                                  uint32_t slot) noexcept
  {
    const uint32_t size = this->size();
    {
      // wait for H1, result available

      while (true)
        {
          Word loaded = 0;
          uint32_t got = 0;
          if (platform::is_master_lane())
            {
              got = inbox(size, slot, &loaded);
            }
          got = platform::broadcast_master(got);

          if (got == 1)
            {
              break;
            }

          Counter::waiting_for_result();

          // make this spin slightly cheaper
          // todo: can the client do useful work while it waits? e.g. gc?
          // need to avoid taking too many locks at a time given forward
          // progress which makes gc tricky
          // could attempt to propagate the current word from staging to
          // outbox - that's safe because a lock is held, maintaining linear
          // time - but may conflict with other clients trying to do the same
          platform::sleep();
        }

      platform::fence_acquire();

      Copy::pull_to_client_from_server(&local_buffer[slot],
                                       &remote_buffer[slot]);
      // call the continuation
      use(&local_buffer[slot]);

      // Copying the state back to the server is a nop for aliased case,
      // and is only necessary if the server has a non-nop garbage clear
      // callback
      Copy::push_from_client_to_server(&remote_buffer[slot],
                                       &local_buffer[slot]);

      // mark the work as no longer in use
      // todo: is it better to leave this for the GC?
      // can free slots more lazily by updating the staging outbox and
      // leaving the visible one. In that case the update may be transfered
      // for free, or it may never become visible in which case the server
      // won't realise the slot is no longer in use
      release_slot(slot);
    }
  }
};

}  // namespace hostrpc

#endif
