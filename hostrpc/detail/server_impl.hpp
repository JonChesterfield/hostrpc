#ifndef HOSTRPC_SERVER_HPP_INCLUDED
#define HOSTRPC_SERVER_HPP_INCLUDED

#include "common.hpp"
#include "counters.hpp"
#include "platform_detect.hpp"

namespace hostrpc
{
struct operate_nop
{
  HOSTRPC_ANNOTATE static void call(page_t*, void*) {}
};
struct clear_nop
{
  HOSTRPC_ANNOTATE static void call(page_t*, void*) {}
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

template <typename WordT, typename SZT, typename Copy,
          typename Counter = counters::server>
struct server_impl : public SZT, public Counter
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
  // may want to rename this, number-slots?
  HOSTRPC_ANNOTATE uint32_t size() const { return SZ::N(); }
  HOSTRPC_ANNOTATE uint32_t words() const { return size() / wordBits(); }

  page_t* remote_buffer;
  page_t* local_buffer;
  lock_t active;

  inbox_t inbox;
  outbox_t outbox;
  staging_t staging;

  HOSTRPC_ANNOTATE server_impl()
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
  HOSTRPC_ANNOTATE ~server_impl() {}
  HOSTRPC_ANNOTATE server_impl(SZ sz, lock_t active, inbox_t inbox,
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

  HOSTRPC_ANNOTATE static void* operator new(size_t, server_impl* p)
  {
    return p;
  }

  HOSTRPC_ANNOTATE server_counters get_counters() { return Counter::get(); }

  HOSTRPC_ANNOTATE void dump_word(uint32_t size, Word word)
  {
    Word i = inbox.load_word(size, word);
    Word o = staging.load_word(size, word);
    Word a = active.load_word(size, word);
    (void)(i + o + a);
    printf("%lu %lu %lu\n", i, o, a);
  }

  // rpc_handle return true if it handled one task, does not attempt multiple.

  // Requires Operate argument.
  // If passed Clear, will call it on garbage collection
  // If passed location, will use it to round robin across slots

  template <typename Operate, typename Clear>
  __attribute__((always_inline)) HOSTRPC_ANNOTATE bool rpc_handle(
      Operate op, Clear cl, uint32_t* location) noexcept
  {
    return rpc_handle_impl<Operate, Clear, true>(op, cl, location);
  }

  template <typename Operate>
  __attribute__((always_inline)) HOSTRPC_ANNOTATE bool rpc_handle(
      Operate op, uint32_t* location) noexcept
  {
    struct Clear
    {
      HOSTRPC_ANNOTATE void operator()(hostrpc::page_t*){};
    };
    Clear cl;
    return rpc_handle_impl<Operate, Clear, false>(op, cl, location);
  }

  // Default location to always start from zero
  template <typename Operate, typename Clear>
  __attribute__((always_inline)) HOSTRPC_ANNOTATE bool rpc_handle(
      Operate op, Clear cl) noexcept
  {
    uint32_t location = 0;
    return rpc_handle<Operate, Clear>(op, cl, &location);
  }

  template <typename Operate>
  __attribute__((always_inline)) HOSTRPC_ANNOTATE bool rpc_handle(
      Operate op) noexcept
  {
    uint32_t location = 0;
    return rpc_handle<Operate>(op, &location);
  }

 private:
  HOSTRPC_ANNOTATE Word find_candidate_server_available_bitmap(uint32_t w,
                                                               Word mask)
  {
    const uint32_t size = this->size();
    Word i = inbox.load_word(size, w);
    Word o = staging.load_word(size, w);
    Word a = active.load_word(size, w);
    platform::fence_acquire();

    Word work = i & ~o;
    Word garbage = ~i & o;
    Word todo = work | garbage;
    Word available = todo & ~a & mask;
    return available;
  }

  template <typename Operate, typename Clear, bool have_precondition>
  __attribute__((always_inline)) HOSTRPC_ANNOTATE bool rpc_handle_impl(
      Operate op, Clear cl, uint32_t* location_arg) noexcept
  {
    const uint32_t size = this->size();
    const uint32_t words = this->words();

    const uint32_t location = *location_arg % size;
    const uint32_t element = index_to_element<Word>(location);

    // skip bits in the first word <= subindex
    static_assert((sizeof(Word) == 8) || (sizeof(Word) == 4), "");
    Word mask = (sizeof(Word) == 8)
                    ? detail::setbitsrange64(index_to_subindex<Word>(location),
                                             wordBits() - 1)
                    : detail::setbitsrange32(index_to_subindex<Word>(location),
                                             wordBits() - 1);

    // Tries a few bits in element, then all bits in all the other words, then
    // all bits in element. This overshoots somewhat but ensures that all slots
    // are checked. Could truncate the last word to check each slot exactly once
    for (uint32_t wc = 0; wc < words + 1; wc++)
      {
        uint32_t w = (element + wc) % words;
        Word available = find_candidate_server_available_bitmap(w, mask);
        if (available == 0)
          {
            Counter::no_candidate_bitmap();
          }
        while (available != 0)
          {
            uint32_t idx = bits::ctz(available);
            assert(bits::nthbitset(available, idx));
            uint32_t slot = wordBits() * w + idx;
            uint64_t cas_fail_count = 0;
            if (active.try_claim_empty_slot(size, slot, &cas_fail_count))
              {
                Counter::cas_lock_fail(cas_fail_count);
                // Success, got the lock. Aim location_arg at next slot
                *location_arg = slot + 1;

                bool r =
                    rpc_handle_given_slot<Operate, Clear, have_precondition>(
                        op, cl, slot);

                platform::critical<uint32_t>([&]() {
                  active.release_slot(size, slot);
                  return 0;
                });

                return r;
              }
            else
              {
                Counter::missed_lock_on_candidate_bitmap();
              }

            // don't try the same slot repeatedly
            available = bits::clearnthbit(available, idx);
          }

        mask = ~((Word)0);
        Counter::missed_lock_on_word();
      }

    // Nothing hit, may as well go from the same location on the next call
    return false;
  }

  template <typename Operate, typename Clear, bool have_precondition>
  __attribute__((always_inline)) HOSTRPC_ANNOTATE bool rpc_handle_given_slot(
      Operate op, Clear cl, uint32_t slot)
  {
    assert(slot != SIZE_MAX);

    const uint32_t element = index_to_element<Word>(slot);
    const uint32_t subindex = index_to_subindex<Word>(slot);

    auto lock_held = [&]() -> bool {
      return bits::nthbitset(active.load_word(size(), element), subindex);
    };
    (void)lock_held;

    const uint32_t size = this->size();

    Word i = inbox.load_word(size, element);
    Word o = staging.load_word(size, element);
    platform::fence_acquire();

    // Called with a lock. The corresponding slot can be:
    //  inbox outbox    state  action outbox'
    //      0      0     idle    none       -
    //      0      1  garbage collect       0
    //      1      0     work    work       1
    //      1      1  waiting    none       -

    Word this_slot = bits::setnthbit((Word)0, subindex);
    Word work_todo = (i & ~o) & this_slot;
    Word garbage_todo = (~i & o) & this_slot;

    assert((work_todo & garbage_todo) == 0);  // disjoint
    assert(lock_held());

    if (garbage_todo)
      {
        assert((o & this_slot) != 0);
        if (have_precondition)
          {
            // Move data and clear. TODO: Elide the copy for nop clear
            Copy::pull_to_server_from_client(&local_buffer[slot],
                                             &remote_buffer[slot]);
            cl(&local_buffer[slot]);
            Copy::push_from_server_to_client(&remote_buffer[slot],
                                             &local_buffer[slot]);
          }

        platform::fence_release();
        uint64_t cas_fail_count = 0;
        uint64_t cas_help_count = 0;
        if (platform::is_master_lane())
          {
            staged_release_slot(size, slot, &staging, &outbox, &cas_fail_count,
                                &cas_help_count);
          }
        cas_fail_count = platform::broadcast_master(cas_fail_count);
        cas_help_count = platform::broadcast_master(cas_help_count);
        Counter::garbage_cas_fail(cas_fail_count);
        Counter::garbage_cas_help(cas_help_count);

        assert(lock_held());
        return false;
      }

    if (!work_todo)
      {
        Counter::got_lock_after_work_done();
        assert(lock_held());
        return false;
      }

    // make the calls
    Copy::pull_to_server_from_client(&local_buffer[slot], &remote_buffer[slot]);
    op(&local_buffer[slot]);
    Copy::push_from_server_to_client(&remote_buffer[slot], &local_buffer[slot]);

    // publish result
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

    // leaves outbox live
    assert(lock_held());
    return true;
  }
};

}  // namespace hostrpc
#endif
