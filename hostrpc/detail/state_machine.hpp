#ifndef HOSTRPC_STATE_MACHINE_HPP_INCLUDED
#define HOSTRPC_STATE_MACHINE_HPP_INCLUDED

#include "../platform/detect.hpp"
#include "common.hpp"
#include "counters.hpp"
#include "cxx.hpp"

namespace hostrpc
{
// inbox == outbox == 0:
//   ready to call on_lo
//
// inbox == outbox == 1:
//   ready to call on_hi
//
// inbox != outbox:
//   waiting on other agent

template <typename WordT, typename SZT, typename Counter, typename inbox_t_arg,
          typename outbox_t_arg>
struct state_machine_impl;

template <unsigned I, unsigned O>
class typed_port_t
{
 public:
  // can move one and convert it to a uint32, but can't make one
  HOSTRPC_ANNOTATE typed_port_t(typed_port_t&&) = default;
  HOSTRPC_ANNOTATE typed_port_t& operator=(typed_port_t&&) = default;
  HOSTRPC_ANNOTATE operator uint32_t() const { return value; }

  HOSTRPC_ANNOTATE typed_port_t()
      : value(static_cast<uint32_t>(port_t::unavailable))
  {
  }

 private:
  template <typename WordT, typename SZT, typename Counter,
            typename inbox_t_arg, typename outbox_t_arg>
  friend struct state_machine_impl;

  HOSTRPC_ANNOTATE typed_port_t(uint32_t v) : value(v)
  {
    static_assert((I <= 1) && (O <= 1), "");
  }
  typed_port_t(const typed_port_t&) = delete;
  typed_port_t& operator=(const typed_port_t&) = delete;
  uint32_t value;
};

template <typename WordT, typename SZT, typename Counter, typename inbox_t_arg,
          typename outbox_t_arg>
struct state_machine_impl : public SZT, public Counter
{
  using Word = WordT;
  using SZ = SZT;
  using lock_t = lock_bitmap<Word>;
  using inbox_t = inbox_t_arg;
  using outbox_t = outbox_t_arg;
  using staging_t = slot_bitmap_device_local<Word>;

  // TODO: Better assert is same_type
  static_assert(sizeof(Word) == sizeof(typename inbox_t::Word), "");
  static_assert(sizeof(Word) == sizeof(typename outbox_t::Word), "");

  HOSTRPC_ANNOTATE constexpr size_t wordBits() const
  {
    return 8 * sizeof(Word);
  }
  // may want to rename this, number-slots?
  HOSTRPC_ANNOTATE uint32_t size() const { return SZ::value(); }
  HOSTRPC_ANNOTATE uint32_t words() const { return size() / wordBits(); }

  page_t* shared_buffer;
  lock_t active;

  inbox_t inbox;
  outbox_t outbox;
  staging_t staging;

  static_assert(cxx::is_trivially_copyable<page_t*>::value, "");
  static_assert(cxx::is_trivially_copyable<lock_t>::value, "");
  static_assert(cxx::is_trivially_copyable<inbox_t>::value, "");
  static_assert(cxx::is_trivially_copyable<outbox_t>::value, "");
  static_assert(cxx::is_trivially_copyable<staging_t>::value, "");

  HOSTRPC_ANNOTATE state_machine_impl()
      : SZ{}, Counter{}, active{}, inbox{}, outbox{}, staging{}
  {
  }
  HOSTRPC_ANNOTATE ~state_machine_impl() = default;
  HOSTRPC_ANNOTATE state_machine_impl(SZ sz, lock_t active, inbox_t inbox,
                                      outbox_t outbox, staging_t staging,
                                      page_t* shared_buffer)
      : SZ{sz},
        Counter{},
        shared_buffer(shared_buffer),
        active(active),
        inbox(inbox),
        outbox(outbox),
        staging(staging)
  {
  }

  HOSTRPC_ANNOTATE void dump()
  {
#if HOSTRPC_HAVE_STDIO
    fprintf(stderr, "shared_buffer %p\n", shared_buffer);
    fprintf(stderr, "inbox         %p\n", inbox.a);
    inbox.dump(size());
    fprintf(stderr, "outbox        %p\n", outbox.a);
    outbox.dump(size());
    fprintf(stderr, "active        %p\n", active.a);
    active.dump(size());
    fprintf(stderr, "outbox stg    %p\n", staging.a);
    staging.dump(size());
#endif
  }

  HOSTRPC_ANNOTATE static void* operator new(size_t, state_machine_impl* p)
  {
    return p;
  }

  HOSTRPC_ANNOTATE void dump_word(uint32_t size, Word word)
  {
    Word i = inbox.load_word(size, word);
    Word o = staging.load_word(size, word);
    Word a = active.load_word(size, word);
    (void)(i + o + a);
#if HOSTRPC_HAVE_STDIO
    // printf("%lu %lu %lu\n", i, o, a);
#endif
  }

  // -> true if successfully opened port, false if failed
  enum class port_state : uint8_t
  {
    unavailable = 0,
    low_values = 1,
    high_values = 2,
    either_low_or_high = 3,
  };

  // port_t::unavailable on failure
  // scan_from can be any uint32_t, it gets % size of the buffer
  // a reasonable guess is a previously used port_t cast to uint32_t and +1

  // open a port with inbox == outbox == 0
  template <typename T>
  HOSTRPC_ANNOTATE port_t rpc_open_port_lo(T active_threads,
                                           uint32_t scan_from = 0)
  {
    return rpc_open_port<T, port_state::low_values>(active_threads, scan_from,
                                                    nullptr);
  }

  // open a port with inbox == outbox == 1
  template <typename T>
  HOSTRPC_ANNOTATE port_t rpc_open_port_hi(T active_threads,
                                           uint32_t scan_from = 0)
  {
    return rpc_open_port<T, port_state::high_values>(active_threads, scan_from,
                                                     nullptr);
  }

  // open a port with inbox == outbox, writes which type was opened iff passed a
  // port_state
  template <typename T>
  HOSTRPC_ANNOTATE port_t rpc_open_port(T active_threads,
                                        uint32_t scan_from = 0,
                                        port_state* which = nullptr)
  {
    return rpc_open_port<T, port_state::either_low_or_high>(active_threads,
                                                            scan_from, which);
  }

  template <typename T>
  HOSTRPC_ANNOTATE typed_port_t<0, 0> rpc_open_typed_port_lo(
      T active_threads, uint32_t scan_from = 0)
  {
    return rpc_open_typed_port_impl<0, 0, T>(active_threads, scan_from);
  }

  template <typename T>
  HOSTRPC_ANNOTATE void rpc_close_port(
      T active_threads,
      port_t port);  // Require != port_t::unavailable, not already closed

  // Can only wait on the inbox to change state as this thread will not change
  // the outbox during the busy wait
  template <typename T, port_state Req>
  HOSTRPC_ANNOTATE void rpc_port_wait_until_state(T active_threads, port_t port,
                                                  port_state* which = nullptr);

  template <typename T, typename Op>
  HOSTRPC_ANNOTATE void rpc_port_apply_lo(T active_threads, port_t port,
                                          Op&& op)
  {
    assert(loadPortState(port) == port_state::low_values);
    rpc_port_apply_given_state<T, Op, port_state::low_values>(
        active_threads, port, cxx::forward<Op>(op));
  }

  template <typename T, typename Op>
  HOSTRPC_ANNOTATE void rpc_port_apply_hi(T active_threads, port_t port,
                                          Op&& op)
  {
    assert(loadPortState(port) == port_state::high_values);
    rpc_port_apply_given_state<T, Op, port_state::high_values>(
        active_threads, port, cxx::forward<Op>(op));
  }

  template <typename T, typename Op>
  HOSTRPC_ANNOTATE void rpc_port_apply(T active_threads, port_t port, Op&& op)
  {
#ifdef NDEBUG
    port_state s = loadPortState(port);
    assert((s == port_state::low_values) || (s == port_state::high_values));
#endif
    rpc_port_apply_given_state<T, Op, port_state::either_low_or_high_values>(
        active_threads, port, cxx::forward<Op>(op));
  }

  // Apply will leave input unchanged and toggle output
  // passed <0, 0> returns <0, 1>, i.e. output changed
  // passed <1, 1> returns <1, 0>, i.e. output changed
  template <unsigned IandO, typename T, typename Op>
  HOSTRPC_ANNOTATE typed_port_t<IandO, !IandO> rpc_port_apply(
      T active_threads, typed_port_t<IandO, IandO>&& port, Op&& op)
  {
    uint32_t raw = static_cast<uint32_t>(port);
    port_t tmp = static_cast<port_t>(raw);
    rpc_port_apply_hi(active_threads, tmp, cxx::forward<Op>(op));
    return typed_port_t<IandO, !IandO>(raw);
  }

  template <unsigned I, typename T>
  HOSTRPC_ANNOTATE bool rpc_port_inbox_has_changed(T,
                                                   typed_port_t<I, !I> const&)
  {
    // todo, intended as a means for caller to tell if rpc_port_wait will
    // transition without spinning
    return false;
  }

  template <unsigned I, typename T>
  HOSTRPC_ANNOTATE typed_port_t<!I, !I> rpc_port_wait(
      T active_threads, typed_port_t<I, !I>&& port)
  {
    // can only wait on the inbox to change
    constexpr port_state Req =
        I == 1 ? port_state::low_values : port_state::high_values;
    uint32_t raw = static_cast<uint32_t>(port);
    port_t tmp = static_cast<port_t>(raw);
    rpc_port_wait_until_state<T, Req>(active_threads, tmp);
    return typed_port_t<!I, !I>(raw);
  }

  template <unsigned I, unsigned O, typename T>
  HOSTRPC_ANNOTATE void rpc_close_port(T active_threads,
                                       typed_port_t<I, O>&& port)
  {
    //
    const uint32_t size = this->size();
    const port_t slot = static_cast<port_t>(static_cast<uint32_t>(port));
    if (platform::is_master_lane(active_threads))
      {
        active.release_slot(size, slot);
      }
  }

  template <typename T, typename CB00, typename CB11, typename CBNone>
  HOSTRPC_ANNOTATE uint32_t /* port used */ rpc_with_opened_port(
      T active_threads, uint32_t scan_from, CB00&& On00, CB11&& On11,
      CBNone&& None)
  {
    port_state ps;
    port_t p = rpc_open_port<T, port_state::either_low_or_high>(active_threads,
                                                                scan_from, &ps);
    if (ps == port_state::unavailable)
      {
        None(active_threads);
        return scan_from;
      }
    else
      {
        if (ps == port_state::low_values)
          {
            On00(active_threads, typed_port_t<0, 0>(static_cast<uint32_t>(p)));
            return static_cast<uint32_t>(p);
          }
        else
          {
            On11(active_threads, typed_port_t<1, 1>(static_cast<uint32_t>(p)));
            return static_cast<uint32_t>(p);
          }
      }
  }

 private:
  HOSTRPC_ANNOTATE
  port_state loadPortState(port_t port)
  {
    assert(port != port_t::unavailable);
    const uint32_t size = this->size();
    const uint32_t w = index_to_element<Word>(port);
    const uint32_t subindex = index_to_subindex<Word>(port);
    Word i = inbox.load_word(size, w);
    Word o = staging.load_word(size, w);

    bool out = bits::nthbitset(o, subindex);
    bool in = bits::nthbitset(i, subindex);

    if ((in == false) && (out == false))
      {
        return port_state::low_values;
      }

    if ((in == true) && (out == true))
      {
        return port_state::high_values;
      }

    return port_state::unavailable;
  }

  template <port_state Req>
  static constexpr HOSTRPC_ANNOTATE Word available_bitmap(Word i, Word o)
  {
    switch (Req)
      {
        case port_state::low_values:
          {
            return ~i & ~o;
          }
        case port_state::high_values:
          {
            return i & o;
          }
        case port_state::either_low_or_high:
          {
            // both 0 or both 1
            return !(i ^ o);
          }
        case port_state::unavailable:
          {
            // different
            return i ^ o;
          }
      }
  }

  static_assert((available_bitmap<port_state::low_values>(0, 0) & 1) == 1, "");
  static_assert((available_bitmap<port_state::low_values>(1, 0) & 1) == 0, "");
  static_assert((available_bitmap<port_state::low_values>(0, 1) & 1) == 0, "");
  static_assert((available_bitmap<port_state::low_values>(1, 1) & 1) == 0, "");

  static_assert((available_bitmap<port_state::high_values>(0, 0) & 1) == 0, "");
  static_assert((available_bitmap<port_state::high_values>(1, 0) & 1) == 0, "");
  static_assert((available_bitmap<port_state::high_values>(0, 1) & 1) == 0, "");
  static_assert((available_bitmap<port_state::high_values>(1, 1) & 1) == 1, "");

  static_assert((available_bitmap<port_state::either_low_or_high>(0, 0) & 1) ==
                    1,
                "");
  static_assert((available_bitmap<port_state::either_low_or_high>(1, 0) & 1) ==
                    0,
                "");
  static_assert((available_bitmap<port_state::either_low_or_high>(0, 1) & 1) ==
                    0,
                "");
  static_assert((available_bitmap<port_state::either_low_or_high>(1, 1) & 1) ==
                    1,
                "");

  static_assert((available_bitmap<port_state::unavailable>(0, 0) & 1) == 0, "");
  static_assert((available_bitmap<port_state::unavailable>(1, 0) & 1) == 1, "");
  static_assert((available_bitmap<port_state::unavailable>(0, 1) & 1) == 1, "");
  static_assert((available_bitmap<port_state::unavailable>(1, 1) & 1) == 0, "");

  template <port_state Req>
  HOSTRPC_ANNOTATE bool is_slot_still_available(uint32_t w, uint32_t idx,
                                                port_state* which)
  {
    const uint32_t size = this->size();
    Word i = inbox.load_word(size, w);
    Word o = staging.load_word(size, w);
    platform::fence_acquire();
    Word r = available_bitmap<Req>(i, o);
    bool available = bits::nthbitset(r, idx);

    if ((Req == port_state::either_low_or_high) && available)
      {
        assert(bits::nthbitset(i, idx) == bits::nthbitset(o, idx));
        if (which != nullptr)
          {
            *which = bits::nthbitset(i, idx) ? port_state::high_values
                                             : port_state::low_values;
          }
      }
    return available;
  }

  template <port_state Req>
  HOSTRPC_ANNOTATE Word find_candidate_available_bitmap(uint32_t w, Word mask)
  {
    const uint32_t size = this->size();
    Word i = inbox.load_word(size, w);
    Word o = staging.load_word(size, w);
    Word a = active.load_word(size, w);
    platform::fence_acquire();

    Word r = available_bitmap<Req>(i, o);
    return r & ~a & mask;
  }

  template <typename T, port_state Req>
  HOSTRPC_ANNOTATE port_t rpc_open_port(T active_threads, uint32_t scan_from,
                                        port_state* which)
  {
    static_assert(Req != port_state::unavailable, "");
    const uint32_t size = this->size();
    const uint32_t words = this->words();
    const uint32_t location = scan_from % size;
    const uint32_t element = index_to_element<Word>(location);

    // skip bits in the first word <= subindex
    static_assert((sizeof(Word) == 8) || (sizeof(Word) == 4), "");
    Word mask = (sizeof(Word) == 8)
                    ? detail::setbitsrange64(index_to_subindex<Word>(location),
                                             wordBits() - 1)
                    : detail::setbitsrange32(index_to_subindex<Word>(location),
                                             wordBits() - 1);

    // Tries a few bits in element, then all bits in all the other words, then
    // all bits in element. This sometimes overshoots by less than a word.
    // Simpler control flow to check a few values twice when almost none
    // available.
    for (uint32_t wc = 0; wc < words + 1; wc++)
      {
        uint32_t w = (element + wc) % words;
        Word available = find_candidate_available_bitmap<Req>(w, mask);
        if (available == 0)
          {
            // Counter::no_candidate_bitmap(active_threads);
          }
        while (available != 0)
          {
            // tries each bit in incrementing order, clearing them on failure
            const uint32_t idx = bits::ctz(available);
            assert(bits::nthbitset(available, idx));
            const uint32_t slot = wordBits() * w + idx;
            uint64_t cas_fail_count = 0;
            if (active.try_claim_empty_slot(active_threads, size, slot,
                                            &cas_fail_count))
              {
                // Counter::cas_lock_fail(active_threads, cas_fail_count);

                // Got the lock. Is the slot still available?
                if (is_slot_still_available<Req>(w, idx, which))
                  {
                    // Success. Got a port with the right mailbox settings.
                    assert(slot < size);
                    return static_cast<port_t>(slot);
                  }
                else
                  {
                    // Failed, drop the lock before continuing to search
                    if (platform::is_master_lane(active_threads))
                      {
                        active.release_slot(size, static_cast<port_t>(slot));
                      }
                  }
              }
            else
              {
                // Counter::missed_lock_on_candidate_bitmap(active_threads);
              }

            // don't try the same slot repeatedly
            available = bits::clearnthbit(available, idx);
          }

        mask = ~((Word)0);
        // Counter::missed_lock_on_word(active_threads);
      }

    if ((Req == port_state::either_low_or_high) && (which != nullptr))
      {
        *which = port_state::unavailable;
      }
    return port_t::unavailable;
  }

  template <typename T, typename Op, port_state Req>
  HOSTRPC_ANNOTATE void rpc_port_apply_given_state(T active_threads,
                                                   port_t port, Op&& op)
  {
    assert(port != port_t::unavailable);
    static_assert(Req != port_state::unavailable, "");
    const uint32_t size = this->size();

    op(port, &shared_buffer[static_cast<uint32_t>(port)]);
    platform::fence_release();

    switch (Req)
      {
        case port_state::low_values:
          {
            if (platform::is_master_lane(active_threads))
              {
                uint64_t cas_fail_count = 0;
                uint64_t cas_help_count = 0;
                staged_claim_slot(size, port, &staging, &outbox,
                                  &cas_fail_count, &cas_help_count);
              }
            break;
          }
        case port_state::high_values:
          {
            if (platform::is_master_lane(active_threads))
              {
                uint64_t cas_fail_count = 0;
                uint64_t cas_help_count = 0;
                staged_release_slot(size, port, &staging, &outbox,
                                    &cas_fail_count, &cas_help_count);
              }
            break;
          }
        case port_state::either_low_or_high:
          {
            if (platform::is_master_lane(active_threads))
              {
                uint64_t cas_fail_count = 0;
                uint64_t cas_help_count = 0;
                staged_toggle_slot(size, port, &staging, &outbox,
                                   &cas_fail_count, &cas_help_count);
              }
            break;
          }
      }
  }

  template <unsigned I, unsigned O, typename T>
  HOSTRPC_ANNOTATE typed_port_t<I, O> rpc_open_typed_port_impl(
      T active_threads, uint32_t scan_from)
  {
    static_assert(I == O, "");
    constexpr port_state Req =
        I == 0 ? port_state::low_values : port_state::high_values;

    port_t p = rpc_open_port<T, Req>(active_threads, scan_from, nullptr);
    // ugly...
    if (static_cast<uint32_t>(p) !=
        static_cast<uint32_t>(port_state::unavailable))
      {
        return typed_port_t<I, O>(static_cast<uint32_t>(p));
      }

    // todo: do amdgpu and nvptx need to implement this or is IR xform
    // sufficient note: returning a typed port means spinning if none is
    // available
    [[clang::musttail]] return rpc_open_typed_port_impl<I, O>(active_threads,
                                                              scan_from);
  }
};

template <typename WordT, typename SZT, typename Counter, typename inbox_t,
          typename outbox_t>
template <typename T>
HOSTRPC_ANNOTATE void
state_machine_impl<WordT, SZT, Counter, inbox_t, outbox_t>::rpc_close_port(
    T active_threads,
    port_t port)  // Require != port_t::unavailable
{
  const uint32_t size = this->size();
  // warning: something needs to release() the buffer element before
  // dropping this lock

  assert(port != port_t::unavailable);
  if (platform::is_master_lane(active_threads))
    {
      active.release_slot(size, port);
    }
}

template <typename WordT, typename SZT, typename Counter, typename inbox_t,
          typename outbox_t>
template <typename T, typename state_machine_impl<WordT, SZT, Counter, inbox_t,
                                                  outbox_t>::port_state Req>
HOSTRPC_ANNOTATE void
state_machine_impl<WordT, SZT, Counter, inbox_t,
                   outbox_t>::rpc_port_wait_until_state(T active_threads,
                                                        port_t port,
                                                        port_state* which)
{
  (void)active_threads;
  const uint32_t size = this->size();
  const uint32_t w = index_to_element<Word>(port);
  const uint32_t subindex = index_to_subindex<Word>(port);

  Word i = inbox.load_word(size, w);
  Word o = staging.load_word(size, w);

  // current thread assumed to hold lock, thus lock is held
  assert(bits::nthbitset(active.load_word(size, w), subindex));

  platform::fence_acquire();

  bool out = bits::nthbitset(o, subindex);
  bool in = bits::nthbitset(i, subindex);

  static_assert(Req != port_state::unavailable, "");
  switch (Req)
    {
      case port_state::low_values:
        {
        lo:;
          // waiting for outbox to change means deadlock
          assert(out == false);
          while (in)
            {
              Word i = inbox.load_word(size, w);
              in = bits::nthbitset(i, subindex);
            }
          platform::fence_acquire();
          break;
        }
      case port_state::high_values:
        {
        hi:;
          assert(out == true);
          while (in == false)
            {
              Word i = inbox.load_word(size, w);
              in = bits::nthbitset(i, subindex);
            }
          platform::fence_acquire();
          break;
        }
      case port_state::either_low_or_high:
        {
          if (out)
            {
              if (which)
                {
                  *which = port_state::high_values;
                }
              goto hi;
            }
          else
            {
              if (which)
                {
                  *which = port_state::low_values;
                }
              goto lo;
            }
        }
    }
}

}  // namespace hostrpc

#endif
