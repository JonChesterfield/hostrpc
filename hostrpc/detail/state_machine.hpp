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
    // TODO: Delete this member, better options implemented now
    // Go via port conversion functions
    either<typename partial_to_typed_trait<S, OutboxState>::type,
           typename partial_to_typed_trait<S, !OutboxState>::type>
        either = port;
    return either.left(rpc_port_closer(active_threads));
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

  template <>
  struct port_trait<either<typed_port_t<0, 0>, typed_port_t<1, 1>>>
  {
    HOSTRPC_ANNOTATE static constexpr bool openable() { return true; }
    HOSTRPC_ANNOTATE static constexpr Word available_bitmap(Word i, Word o)
    {
      return (~i & ~o) | (i & o);
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

  template <typename T>
  HOSTRPC_ANNOTATE HOSTRPC_RETURN_UNKNOWN
      typename either<typed_port_t<0, 0>, typed_port_t<1, 1>>::maybe
      rpc_try_open_port(T active_threads, uint32_t scan_from = 0)
  {
    return try_open_typed_port<either<typed_port_t<0, 0>, typed_port_t<1, 1>>,
                               T>(active_threads, scan_from);
  }

  template <typename T>
  HOSTRPC_ANNOTATE either<typed_port_t<0, 0>, typed_port_t<1, 1>> rpc_open_port(
      T active_threads, uint32_t scan_from = 0)
  {
    return open_typed_port<either<typed_port_t<0, 0>, typed_port_t<1, 1>>, T>(
        active_threads, scan_from);
  }

  template <unsigned S, typename T>
  HOSTRPC_ANNOTATE void rpc_close_port(T active_threads,
                                       partial_port_t<S>&& port)
  {
    platform::fence_release();
    const uint32_t size = this->size();
    active.close_port(active_threads, size, cxx::move(port));
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
    platform::fence_release();
    const uint32_t size = this->size();
    active.close_port(active_threads, size, cxx::move(port));
  }

  template <typename T>
  struct rpc_port_closer_t
  {
    HOSTRPC_ANNOTATE
    rpc_port_closer_t(T active_threads, state_machine_impl* M)
        : active_threads(active_threads), M(M)
    {
    }

    template <unsigned I, unsigned O>
    HOSTRPC_ANNOTATE void operator()(typed_port_t<I, O>&& port)
    {
      M->rpc_close_port(active_threads, cxx::move(port));
    }

    template <unsigned S>
    HOSTRPC_ANNOTATE void operator()(partial_port_t<S>&& port)
    {
      M->rpc_close_port(active_threads, cxx::move(port));
    }

    T active_threads;
    state_machine_impl* M;
  };

  template <typename T>
  HOSTRPC_ANNOTATE rpc_port_closer_t<T> rpc_port_closer(T active_threads)
  {
    return rpc_port_closer_t<T>(active_threads, this);
  }

  template <unsigned IA, unsigned OA, unsigned IB, unsigned OB, typename T>
  HOSTRPC_ANNOTATE void rpc_close_port(
      T active_threads,
      either<typed_port_t<IA, OA>, typed_port_t<IB, OB>>&& port)
  {
    port.template visit<void>(rpc_port_closer(active_threads),
                              rpc_port_closer(active_threads));
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
  HOSTRPC_ANNOTATE void rpc_port_use(
      T active_threads,
      HOSTRPC_CONST_REF_ARG typed_port_t<IandO, IandO> const& port, Op&& op)
  {
    static_assert(IandO == 0 || IandO == 1, "");

    return read_typed_port<typed_port_t<IandO, IandO>, T>(active_threads, port,
                                                          cxx::forward<Op>(op));
  }

  template <typename T, typename Op>
  HOSTRPC_ANNOTATE void rpc_port_use(
      T active_threads, HOSTRPC_CONST_REF_ARG partial_port_t<1> const& port,
      Op&& op)
  {
    return read_typed_port<partial_port_t<1>, T>(active_threads, port,
                                                 cxx::forward<Op>(op));
  }

  template <unsigned IandO, typename T>
  HOSTRPC_ANNOTATE typed_port_t<IandO, !IandO> rpc_port_send(
      T active_threads, typed_port_t<IandO, IandO>&& port)
  {
    static_assert(IandO == 0 || IandO == 1, "");
    platform::fence_release();

    const uint32_t size = this->size();

    if constexpr (IandO == 0)
      {
        return outbox.claim_slot(active_threads, size, cxx::move(port));
      }
    else
      {
        return outbox.release_slot(active_threads, size, cxx::move(port));
      }
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
    // Currently ICE in clang for opencl:
    // Assertion `!MemberQuals.hasAddressSpace()' failed.
    using ResultTy = hostrpc::either<partial_port_t<0>, partial_port_t<1>>;
    return either.template visit<ResultTy>(
        [=](auto&& port) -> ResultTy {
          return hostrpc::typed_to_partial(
              rpc_port_query(active_threads, cxx::move(port)));
        },
        [=](auto&& port) -> ResultTy {
          return hostrpc::typed_to_partial(
              rpc_port_query(active_threads, cxx::move(port)));
        });
#else
    if (either)
      {
        auto maybe = either.left(rpc_port_closer(active_threads));
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
        auto maybe = either.right(rpc_port_closer(active_threads));
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

  // Can only wait on the inbox to change state as this thread will not change
  // the outbox during the busy wait (sleep? callback? try/test-wait?)
  template <unsigned I, typename T>
  HOSTRPC_ANNOTATE typed_port_t<!I, !I> rpc_port_recv(
      T active_threads, typed_port_t<I, !I>&& port)
  {
    for (;;)
      {
        auto an_either = rpc_port_query(active_threads, cxx::move(port));
        if (an_either)
          {
            auto a_maybe = an_either.left(rpc_port_closer(active_threads));
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
            auto a_maybe = an_either.right(rpc_port_closer(active_threads));
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
  }

  template <unsigned IandO, typename T>
  HOSTRPC_ANNOTATE either<typed_port_t<!IandO, !IandO>,
                          typed_port_t<IandO, IandO>>
  rpc_port_recv(
      T active_threads,
      either<typed_port_t<IandO, !IandO>, typed_port_t<!IandO, IandO>>&& port)
  {
    return port.foreach (
        [&](typed_port_t<IandO, !IandO>&& port) {
          return rpc_port_recv(active_threads, cxx::move(port));
        },
        [&](typed_port_t<!IandO, IandO>&& port) {
          return rpc_port_recv(active_threads, cxx::move(port));
        });
  }

  template <typename T>
  HOSTRPC_ANNOTATE partial_port_t<1> rpc_port_recv(T active_threads,
                                                   partial_port_t<0>&& port)
  {
    // Implementing via typed port produces two distinct loops, one for each
    // type it can branch to. Implementing via query resolves to a single loop
    // todo: fold with typed implementation
    for (;;)
      {
        auto an_either = rpc_port_query(active_threads, cxx::move(port));
        if (an_either)
          {
            auto a_maybe = an_either.left(rpc_port_closer(active_threads));
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
            auto a_maybe = an_either.right(rpc_port_closer(active_threads));
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
  }

  template <unsigned IandO, typename T, typename Op>
  HOSTRPC_ANNOTATE typed_port_t<IandO, !IandO> rpc_port_apply(
      T active_threads, typed_port_t<IandO, IandO>&& port, Op&& op)
  {
    static_assert(IandO == 0 || IandO == 1, "");
    rpc_port_use<IandO, T, Op>(active_threads, port, cxx::forward<Op>(op));
    return rpc_port_send(active_threads, cxx::move(port));
  }

  template <unsigned IandO, typename T, typename Op>
  HOSTRPC_ANNOTATE either<typed_port_t<IandO, !IandO>,
                          typed_port_t<!IandO, IandO>>
  rpc_port_apply(
      T active_threads,
      either<typed_port_t<IandO, IandO>, typed_port_t<!IandO, !IandO>>&& port,
      Op&& op)
  {
    return port.foreach (
        [&](typed_port_t<IandO, IandO>&& port) {
          return rpc_port_apply(active_threads, cxx::move(port),
                                cxx::forward<Op>(op));
        },
        [&](typed_port_t<!IandO, !IandO>&& port) {
          return rpc_port_apply(active_threads, cxx::move(port),
                                cxx::forward<Op>(op));
        });
  }

  template <typename T, typename Op>
  HOSTRPC_ANNOTATE partial_port_t<0> rpc_port_apply(T active_threads,
                                                    partial_port_t<1>&& port,
                                                    Op&& op)
  {
    either<typed_port_t<0, 0>, typed_port_t<1, 1>> either = port;
    if (either)
      {
        typename typed_port_t<0, 0>::maybe maybe =
            either.left(rpc_port_closer(active_threads));
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
        typename typed_port_t<1, 1>::maybe maybe =
            either.right(rpc_port_closer(active_threads));
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

 private:

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

  // Only writes to inbox/outbox pointers if it loaded the corresponding
  template <typename T>
  HOSTRPC_ANNOTATE HOSTRPC_RETURN_UNKNOWN
      typename either<typed_port_t<0, 0>, typed_port_t<1, 1>>::maybe
      try_open_specific_port(T active_threads, uint32_t slot,
                             Word* inbox_to_update, Word* outbox_to_update)
  {
    using EitherStable = either<typed_port_t<0, 0>, typed_port_t<1, 1>>;

    const uint32_t size = this->size();
    auto maybe_port = active.try_open_port(active_threads, size, slot);
    if (!maybe_port)
      {
        return {};
      }

    // Need reads from inbox, outbox to occur after the locked read
    platform::fence_acquire();

    // Might be faster to check outbox first
    either<typed_port_t<0, 2>, typed_port_t<1, 2>> with_inbox =
        inbox.refine(size, active_threads, maybe_port.value(), inbox_to_update);

    // Neither visit nor open coding look great. Also this is calling
    // rpc_port_closer which will introduce spurious fences.


    return with_inbox.template visit<typename EitherStable::maybe>(

        [&](typed_port_t<0, 2>&& port) HOSTRPC_RETURN_UNKNOWN ->
        typename EitherStable::maybe {
          either<typed_port_t<0, 0>, typed_port_t<0, 1>> with_outbox =
              outbox.refine(size, active_threads, cxx::move(port),
                            outbox_to_update);
          if (auto maybe = with_outbox.left(rpc_port_closer(active_threads)))
            {
              return EitherStable::Left(maybe.value());
            }
          else
            {
              return {};
            }
        },

        [&](typed_port_t<1, 2>&& port) HOSTRPC_RETURN_UNKNOWN ->
        typename EitherStable::maybe {
          either<typed_port_t<1, 0>, typed_port_t<1, 1>> with_outbox =
              outbox.refine(size, active_threads, cxx::move(port),
                            outbox_to_update);
          if (auto maybe = with_outbox.right(rpc_port_closer(active_threads)))
            {
              return EitherStable::Right(maybe.value());
            }
          else
            {
              return {};
            }
        });
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

            Word latest_known_inbox = i;
            Word latest_known_outbox = o;

            if (auto specific = try_open_specific_port<T>(active_threads, slot,
                                                          &latest_known_inbox,
                                                          &latest_known_outbox))
              {
                either<typed_port_t<0, 0>, typed_port_t<1, 1>> port =
                    specific.value();

                if constexpr (cxx::is_same<typed_port_t<0, 0>, PortType>())
                  {
                    if (port)
                      {
                        platform::fence_acquire();
                        return port.left(rpc_port_closer(active_threads));
                      }
                  }

                if constexpr (cxx::is_same<typed_port_t<1, 1>, PortType>())
                  {
                    if (!port)
                      {
                        platform::fence_acquire();
                        return port.right(rpc_port_closer(active_threads));
                      }
                  }

                if constexpr (cxx::is_same<either<typed_port_t<0, 0>,
                                                  typed_port_t<1, 1>>,
                                           PortType>())
                  {
                    platform::fence_acquire();
                    return port;
                  }

                if constexpr (cxx::is_same<partial_port_t<1>, PortType>())
                  {
                    platform::fence_acquire();
                    return port.template visit<partial_port_t<1>>(
                        [](auto&& port) -> partial_port_t<1> {
                          return hostrpc::typed_to_partial(cxx::move(port));
                        },
                        [](auto&& port) -> partial_port_t<1> {
                          return hostrpc::typed_to_partial(cxx::move(port));
                        });
                  }

                // Otherwise the port didn't match what was requested, close it
                port.template visit<void>(rpc_port_closer(active_threads),
                                          rpc_port_closer(active_threads));
              }

            available &= port_trait<PortType>::available_bitmap(
                latest_known_inbox, latest_known_outbox);

            // Don't try to lock this slot again
            // TODO: Also mask off the locks which we just learned have
            // been taken by other threads
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
