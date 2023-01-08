#ifndef HOSTRPC_STATE_MACHINE_HPP_INCLUDED
#define HOSTRPC_STATE_MACHINE_HPP_INCLUDED

#include "../platform/detect.hpp"
#include "common.hpp"
#include "cxx.hpp"
#include "maybe.hpp"
#include "tuple.hpp"
#include "typed_port_t.hpp"

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

// BufferElementT is currently always a hostrpc::page_t but should be an
// arbitrary trivially copyable value, where callbacks get passed a mutable
// reference to one and the state machine stores a pointer to N of them
//
// WordT is a property of the architecture - some unsigned integer used as the
// building block of bitmaps. No particular reason why it needs to be the same
// for different bitmaps or for different ends of the machine, though that is
// currently assumed to be the case

// SZT is either a hostrpc::size_runtime or a hostrpc::size_compiletime, which
// is an application choice based on how specialised it wants to be

// InvertedInboxLoad is a way for a client/server pair to configure themselves

template <typename BufferElementT, typename WordT, typename SZT,
          bool InvertedInboxLoadT>
struct state_machine_impl : public SZT
{
  using BufferElement = BufferElementT;
  using Word = WordT;
  using SZ = SZT;
  static constexpr bool InvertedInboxLoad() { return InvertedInboxLoadT; }

  using state_machine_impl_t =
      state_machine_impl<BufferElementT, WordT, SZT, InvertedInboxLoadT>;

  using lock_t = lock_bitmap<state_machine_impl_t>;

  using mailbox_t = mailbox_bitmap<state_machine_impl_t>;

  using inbox_t = inbox_bitmap<state_machine_impl_t>;
  using outbox_t = outbox_bitmap<state_machine_impl_t>;

  template <unsigned I, unsigned O>
  using typed_port_t = typed_port_impl_t<state_machine_impl, I, O>;

  template <unsigned S>
  using partial_port_t = partial_port_impl_t<state_machine_impl, S>;

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

  BufferElement* shared_buffer;
  lock_t active;

  inbox_t inbox;
  outbox_t outbox;

  // Require that instances of this class can be trivially copied
  static_assert(cxx::is_trivially_copyable<BufferElement*>() /*::value*/, "");
  static_assert(cxx::is_trivially_copyable<lock_t>() /*::value*/, "");
  static_assert(cxx::is_trivially_copyable<inbox_t>() /*::value*/, "");
  static_assert(cxx::is_trivially_copyable<outbox_t>() /*::value*/, "");

  // Also need to check somewhere that the contents of the shared buffer is
  // copyable
  static_assert(cxx::is_trivially_copyable<BufferElement>() /*::value*/, "");

  HOSTRPC_ANNOTATE state_machine_impl() : SZ{}, active{}, inbox{}, outbox{} {}
  HOSTRPC_ANNOTATE ~state_machine_impl() = default;
  HOSTRPC_ANNOTATE state_machine_impl(SZ sz, lock_t active, inbox_t inbox,
                                      outbox_t outbox,
                                      BufferElement* shared_buffer)
      : SZ{sz},
        shared_buffer(shared_buffer),
        active(active),
        inbox(inbox),
        outbox(outbox)
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
#endif
  }

  HOSTRPC_ANNOTATE static void* operator new(size_t, state_machine_impl* p)
  {
    return p;
  }

  HOSTRPC_ANNOTATE void dump_word(uint32_t size, Word word)
  {
    Word i = inbox.load_word(size, word);
    Word o = outbox.load_word(size, word);
    Word a = active.load_word(size, word);
    (void)(i + o + a);
#if HOSTRPC_HAVE_STDIO
    // printf("%lu %lu %lu\n", i, o, a);
#endif
  }

  // The state machine type can construct and drop ports.
  // The non-default constructor and drop() methods are private to each.
  // The equivalence of states is defined as traits in typed_ports.
  static_assert(traits::traits_consistent<state_machine_impl>());

  template <unsigned I, unsigned O>
  using typed_to_partial_trait =
      traits::typed_to_partial_trait<state_machine_impl, typed_port_t<I, O>>;

  template <unsigned I, unsigned O>
  HOSTRPC_ANNOTATE HOSTRPC_CREATED_RES
      typename typed_to_partial_trait<I, O>::type
      typed_to_partial(typed_port_t<I, O>&& port HOSTRPC_CONSUMED_ARG)
  {
    return cxx::move(port);
  }

  template <unsigned S, bool OutboxState>
  using partial_to_typed_trait =
      traits::partial_to_typed_trait<state_machine_impl, partial_port_t<S>,
                                     OutboxState>;

  template <bool OutboxState, unsigned S, typename T>
  HOSTRPC_ANNOTATE HOSTRPC_RETURN_UNKNOWN
      maybe<typename partial_to_typed_trait<S, OutboxState>::type>
      partial_to_typed(T active_threads,
                       partial_port_t<S>&& port HOSTRPC_CONSUMED_ARG)
  {
#if 0
    // Directly construct it from within state machine
    uint32_t v = port.value;
    bool state = port.state;

    if (OutboxState == state)
      {
        port.kill();
        typename partial_to_typed_trait<S, OutboxState>::type new_port(v);
        return new_port;
      }
    else
      {
        rpc_close_port(active_threads, cxx::move(port));
        return {};
      }
#else
    // Go via port conversion functions
    either<typename partial_to_typed_trait<S, OutboxState>::type,
           typename partial_to_typed_trait<S, !OutboxState>::type>
        either = port;
    if (either)
      {
        return either.on_true();
      }
    else
      {
        auto m = either.on_false();
        if (m)
          {
            rpc_close_port(active_threads, m.value());
          }
        return {};
      }
#endif
  }

  // These should probably be gated behind an explicit opt in, another
  // template parameter or similar.

  template <unsigned I, unsigned O>
  HOSTRPC_ANNOTATE static uint32_t unsafe_port_escape(
      typed_port_t<I, O>&& port HOSTRPC_CONSUMED_ARG)
  {
    uint32_t v = port;
    port.kill();
    return v;
  }

  template <unsigned I, unsigned O>
  HOSTRPC_ANNOTATE HOSTRPC_CREATED_RES static typed_port_t<I, O>
  unsafe_port_create(uint32_t value)
  {
    return {value};
  }

  template <typename P>
  struct port_trait
  {
    HOSTRPC_ANNOTATE static constexpr bool openable() { return false; }
    HOSTRPC_ANNOTATE static constexpr Word available_bitmap(Word, Word)
    {
      return 0;
    }
  };
  template <>
  struct port_trait<typed_port_t<0, 0>>
  {
    HOSTRPC_ANNOTATE static constexpr bool openable() { return true; }
    HOSTRPC_ANNOTATE static constexpr Word available_bitmap(Word i, Word o)
    {
      return ~i & ~o;
    }
  };
  template <>
  struct port_trait<typed_port_t<1, 1>>
  {
    HOSTRPC_ANNOTATE static constexpr bool openable() { return true; }
    HOSTRPC_ANNOTATE static constexpr Word available_bitmap(Word i, Word o)
    {
      return i & o;
    }
  };
  template <>
  struct port_trait<partial_port_t<1>>
  {
    HOSTRPC_ANNOTATE static constexpr bool openable() { return true; }
    HOSTRPC_ANNOTATE static constexpr Word available_bitmap(Word i, Word o)
    {
      return ~(i ^ o);
    }
  };

  template <typename T>
  HOSTRPC_ANNOTATE static constexpr Word available_bitmap(Word i, Word o)
  {
    return port_trait<T>::available_bitmap(i, o);
  }
  template <typename T>
  HOSTRPC_ANNOTATE static constexpr bool port_openable()
  {
    return port_trait<T>::openable();
  }

  static_assert(available_bitmap<typed_port_t<0, 0>>(0, 0) == ~0);
  static_assert(available_bitmap<typed_port_t<0, 0>>(0, ~0) == 0);
  static_assert(available_bitmap<typed_port_t<0, 0>>(~0, 0) == 0);
  static_assert(available_bitmap<typed_port_t<0, 0>>(~0, ~0) == 0);

  static_assert(available_bitmap<typed_port_t<1, 1>>(0, 0) == 0);
  static_assert(available_bitmap<typed_port_t<1, 1>>(0, ~0) == 0);
  static_assert(available_bitmap<typed_port_t<1, 1>>(~0, 0) == 0);
  static_assert(available_bitmap<typed_port_t<1, 1>>(~0, ~0) == ~0);

  static_assert(available_bitmap<partial_port_t<1>>(0, 0) == ~0);
  static_assert(available_bitmap<partial_port_t<1>>(0, ~0) == 0);
  static_assert(available_bitmap<partial_port_t<1>>(~0, 0) == 0);
  static_assert(available_bitmap<partial_port_t<1>>(~0, ~0) == ~0);

  template <unsigned I, unsigned O, typename T>
  HOSTRPC_ANNOTATE HOSTRPC_RETURN_UNKNOWN typename typed_port_t<I, O>::maybe
  rpc_try_open_typed_port(T active_threads, uint32_t scan_from = 0)
  {
    static_assert(port_openable<typed_port_t<I, O>>(), "");
    static_assert(I == O, "");
    return try_open_typed_port<typed_port_t<I, O>, T>(active_threads,
                                                      scan_from);
  }

  template <unsigned I, unsigned O, typename T>
  HOSTRPC_ANNOTATE typed_port_t<I, O> rpc_open_typed_port(
      T active_threads, uint32_t scan_from = 0)
  {
    static_assert(port_openable<typed_port_t<I, O>>(), "");
    static_assert(I == O, "");
    return open_typed_port<typed_port_t<I, O>, T>(active_threads, scan_from);
  }

  template <unsigned S, typename T>
  HOSTRPC_ANNOTATE void rpc_close_port(T active_threads,
                                       partial_port_t<S>&& port)
  {
    const uint32_t size = this->size();
    const uint32_t slot = static_cast<uint32_t>(port);
    platform::fence_release();
    active.release_slot(active_threads, size, slot);
    port.kill();
  }

  template <typename T>
  HOSTRPC_ANNOTATE HOSTRPC_RETURN_UNKNOWN maybe<partial_port_t<1>>
  rpc_try_open_partial_port(T active_threads, uint32_t scan_from = 0)
  {
    static_assert(port_openable<partial_port_t<1>>(), "");
    return try_open_typed_port<partial_port_t<1>>(active_threads, scan_from);
  }

  template <typename T>
  HOSTRPC_ANNOTATE partial_port_t<1> rpc_open_partial_port(
      T active_threads, uint32_t scan_from = 0)
  {
    static_assert(port_openable<partial_port_t<1>>(), "");
    return open_typed_port<partial_port_t<1>>(active_threads, scan_from);
  }

  template <unsigned I, unsigned O, typename T>
  HOSTRPC_ANNOTATE void rpc_close_port(T active_threads,
                                       typed_port_t<I, O>&& port)
  {
    rpc_close_port(active_threads, typed_to_partial(cxx::move(port)));
  }

  // TODO: Want a function which can be called on stable ports, the same
  // ones as apply, but takes the port by const& and does not change it
  // This can be used to read the buffer without triggering a request
  // for the other machine to act. It could probably take the buffer
  // by const& as well, as a reminder that writes to it aren't going
  // to be seen by the other side, though one could make multiple calls
  // before following with an apply.

  // Apply will leave input unchanged and toggle output
  // passed <0, 0> returns <0, 1>, i.e. output changed
  // passed <1, 1> returns <1, 0>, i.e. output changed
  template <unsigned IandO, typename T, typename Op>
  HOSTRPC_ANNOTATE void rpc_port_on_element(
      T active_threads,
      HOSTRPC_CONST_REF_ARG typed_port_t<IandO, IandO> const& port, Op&& op)
  {
    static_assert(IandO == 0 || IandO == 1, "");

    return read_typed_port<typed_port_t<IandO, IandO>, T>(active_threads, port,
                                                          cxx::forward<Op>(op));
  }

  template <typename T, typename Op>
  HOSTRPC_ANNOTATE void rpc_port_on_element(
      T active_threads, HOSTRPC_CONST_REF_ARG partial_port_t<1> const& port,
      Op&& op)
  {
    return read_typed_port<partial_port_t<1>, T>(active_threads, port,
                                                 cxx::forward<Op>(op));
  }

  template <unsigned IandO, typename T, typename Op>
  HOSTRPC_ANNOTATE typed_port_t<IandO, !IandO> rpc_port_apply(
      T active_threads, typed_port_t<IandO, IandO>&& port, Op&& op)
  {
    static_assert(IandO == 0 || IandO == 1, "");

    read_typed_port<typed_port_t<IandO, IandO>, T, Op>(active_threads, port,
                                                       cxx::forward<Op>(op));

    const uint32_t size = this->size();

    platform::fence_release();

    // Toggle the outbox slot
    // partial port implementation could call outbox.toggle here

    // could pass a typed port representation here and get compile time checking
    // of the comments 'assumes slot taken' and similar

    // I think the is_master_lane handling needs to be under the control of the
    // bitmap, which is going to mean passing active threads down into those
    // operations That means it'll be possible to replace this branch with
    // having every active thread perform the atomic operation - that would make
    // the CFG simpler but I don't know what the effect on memory traffic would
    // be. E.g. one lane fetch_or's in a bit, all the others fetch_or in zero,
    // but aimed at the same word in memory - what does that mean across pcie?
    // Don't know, should find out.

    if constexpr (IandO == 0)
      {
        return outbox.claim_slot(active_threads, size, cxx::move(port));
      }
    else
      {
        return outbox.release_slot(active_threads, size, cxx::move(port));
      }
  }

  template <typename T, typename Op>
  HOSTRPC_ANNOTATE partial_port_t<0> rpc_port_apply(T active_threads,
                                                    partial_port_t<1>&& port,
                                                    Op&& op)
  {
    either<typed_port_t<0, 0>, typed_port_t<1, 1>> either = port;
    if (either)
      {
        typename typed_port_t<0, 0>::maybe maybe = either.on_true();
        if (maybe)
          {
            return rpc_port_apply(active_threads, maybe.value(),
                                  cxx::forward<Op>(op));
          }
        else
          {
            __builtin_unreachable();
          }
      }
    else
      {
        typename typed_port_t<1, 1>::maybe maybe = either.on_false();
        if (maybe)
          {
            return rpc_port_apply(active_threads, maybe.value(),
                                  cxx::forward<Op>(op));
          }
        else
          {
            __builtin_unreachable();
          }
      }
  }

  // Can only wait on the inbox to change state as this thread will not change
  // the outbox during the busy wait (sleep? callback? try/test-wait?)
  template <unsigned I, typename T>
  HOSTRPC_ANNOTATE typed_port_t<!I, !I> rpc_port_wait(
      T active_threads, typed_port_t<I, !I>&& port)
  {
#if 1
    // can only wait on the inbox to change
    (void)active_threads;
    static_assert(I == 0 || I == 1, "");

    uint32_t raw = static_cast<uint32_t>(port);

    const uint32_t size = this->size();
    const uint32_t w = index_to_element<Word>(raw);
    const uint32_t subindex = index_to_subindex<Word>(raw);

    constexpr bool req = I == 1;
    bool in = req;

    while (in == req)
      {
        // until inbox changes
        in = bits::nthbitset(inbox.load_word(size, w), subindex);
      }

    platform::fence_acquire();
    // return port.invert_inbox(); // should be fine here but trips up clang's
    // consumed checking
    return typed_port_t<!I, !I>(raw);
#else
    // wait can be implemented in terms of query
    // calling wait vs the following inlined showed identical codegen,
    // implementing wait as the following seems to miss a loop optimisation
    // but the change is very minor

    for (;;)
      {
        auto an_either = rpc_port_query(active_threads, cxx::move(port));
        if (an_either)
          {
            auto a_maybe = an_either.on_true();
            if (a_maybe)
              {
                auto a = a_maybe.value();
                port = cxx::move(a);
              }
            else
              {
                __builtin_unreachable();
              }
          }
        else
          {
            auto a_maybe = an_either.on_false();
            if (a_maybe)
              {
                auto a = a_maybe.value();
                return cxx::move(a);
              }
            else
              {
                __builtin_unreachable();
              }
          }
      }

#endif
  }

  template <typename T>
  HOSTRPC_ANNOTATE partial_port_t<1> rpc_port_wait(T active_threads,
                                                   partial_port_t<0>&& port)
  {
    // Implementing via typed port produces two distinct loops, one for each
    // type it can branch to. Implementing via query resolves to a single loop
    for (;;)
      {
        auto an_either = rpc_port_query(active_threads, cxx::move(port));
        if (an_either)
          {
            auto a_maybe = an_either.on_true();
            if (a_maybe)
              {
                auto a = a_maybe.value();
                port = cxx::move(a);
              }
            else
              {
                __builtin_unreachable();
              }
          }
        else
          {
            auto a_maybe = an_either.on_false();
            if (a_maybe)
              {
                auto a = a_maybe.value();
                return cxx::move(a);
              }
            else
              {
                __builtin_unreachable();
              }
          }
      }

    // Implementing in terms of typed port in the first instance.
    // Codegens roughly as expected - the two waits turn into distinct loops
  }

  template <unsigned I, typename T>
  HOSTRPC_ANNOTATE either<
      /* might want return values swapped over */
      typed_port_t<I, !I>,  /* no change */
      typed_port_t<!I, !I>> /* inbox changed */

  rpc_port_query(T active_threads, typed_port_t<I, !I>&& port)
  {
    static_assert(I == 0 || I == 1, "");
    const uint32_t size = this->size();
    port.unconsumed();
    return inbox.query(size, active_threads, cxx::move(port));
  }

  template <typename T>
  HOSTRPC_ANNOTATE either<partial_port_t<0>, /* no change */
                          partial_port_t<1>> /* inbox changed */
  rpc_port_query(T active_threads, partial_port_t<0>&& port)
  {
    using EitherTy = typename partial_port_t<0>::partial_to_typed_result_type;
    EitherTy either = hostrpc::partial_to_typed(cxx::move(port));

#if 0
    // Visit avoids the builtin unreachable branches but captures active threads by
    // reference. OpenCL may struggle with that.
    using ResultTy = hostrpc::either<partial_port_t<0>, partial_port_t<1>>;
    return either.template visit<ResultTy>(
        [=](typename EitherTy::TrueTy&& port) -> ResultTy {
          return hostrpc::typed_to_partial(
              rpc_port_query(active_threads, cxx::move(port)));
        },
        [=](typename EitherTy::FalseTy&& port) -> ResultTy {
          return hostrpc::typed_to_partial(
              rpc_port_query(active_threads, cxx::move(port)));
        });
#else
    if (either)
      {
        auto maybe = either.on_true();
        if (maybe)
          {
            return hostrpc::typed_to_partial(
                rpc_port_query(active_threads, maybe.value()));
          }
        else
          {
            __builtin_unreachable();
          }
      }
    else
      {
        auto maybe = either.on_false();
        if (maybe)
          {
            return hostrpc::typed_to_partial(
                rpc_port_query(active_threads, maybe.value()));
          }
        else
          {
            __builtin_unreachable();
          }
      }
#endif
  }

 private:
  template <typename PortType>
  HOSTRPC_ANNOTATE HOSTRPC_RETURN_UNKNOWN typename PortType::maybe
  try_construct_port(bool inbox_high, bool outbox_high, uint32_t slot)
  {
    // The make functions branch on a compile time constant
    // Ports can only be opened in inbox==outbox state
    // TODO: Check codegen.

    if (inbox_high & outbox_high)
      {
        return PortType::template make<true, true>(slot);
      }

    if (!inbox_high & !outbox_high)
      {
        return PortType::template make<false, false>(slot);
      }

    return {};
  }

  template <typename PortType, typename T>
  HOSTRPC_ANNOTATE PortType open_typed_port(T active_threads,
                                            uint32_t scan_from)
  {
    static_assert(port_openable<PortType>(), "");
    for (;;)
      {
        auto r = try_open_typed_port<PortType, T>(active_threads, scan_from);
        if (r)
          {
            return r.value();
          }
      }
  }

  template <typename PortType, typename T>
  HOSTRPC_ANNOTATE HOSTRPC_RETURN_UNKNOWN typename PortType::maybe
  try_open_typed_port(T active_threads, uint32_t scan_from)
  {
    static_assert(port_openable<PortType>(), "");
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
#if 0
        // Equivalent but slower
        // Taken locks have bits set in active, need a clear bit to open a port
        Word available = ~active.load_word(size, w);
        available &= mask;
        platform::fence_acquire();
#else
        Word i = inbox.load_word(size, w);
        Word o = outbox.load_word(size, w);
        Word a = active.load_word(size, w);
        platform::fence_acquire();
        Word r = available_bitmap<PortType>(i, o);
        Word available = r & ~a & mask;
#endif

        while (available != 0)
          {
            // tries each bit in incrementing order, clearing them on failure
            const uint32_t idx = bits::ctz(available);
            assert(bits::nthbitset(available, idx));
            const uint32_t slot = wordBits() * w + idx;
            assert(slot < size);

            if (active.try_claim_empty_slot(active_threads, size, slot))
              {
                // Previous versions had no fence here, relying on the
                // acquire-release on the lock. I'm 95% sure that was a race,
                // the acq/rel on the lock CAS has no relation to these loads.
                // Didn't show up in testing.
                platform::fence_acquire();
                static_assert(port_openable<PortType>(), "");
                Word i = inbox.load_word(size, w);
                Word o = outbox.load_word(size, w);
                platform::fence_acquire();

                typename PortType::maybe maybe = try_construct_port<PortType>(
                    bits::nthbitset(i, idx), bits::nthbitset(o, idx), slot);
                if (maybe)
                  {
                    return maybe.value();
                  }
                else
                  {
                    // Failed, drop the lock before continuing to search
                    active.release_slot(active_threads, size, slot);
                  }

                available &= port_trait<PortType>::available_bitmap(i, o);
              }

            available = bits::clearnthbit(available, idx);
          }

        mask = ~((Word)0);
      }

    return {};
  }

  template <typename PortArg, typename T, typename Op>
  HOSTRPC_ANNOTATE void read_typed_port(
      T active_threads, HOSTRPC_CONST_REF_ARG PortArg const& port, Op&& op)
  {
    (void)active_threads;
    static_assert(port_trait<PortArg>::openable(), "");
    uint32_t raw = static_cast<uint32_t>(port);
    op(raw, &shared_buffer[raw]);
  }
};

}  // namespace hostrpc

#endif
