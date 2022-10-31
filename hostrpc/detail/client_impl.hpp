#ifndef HOSTRPC_CLIENT_HPP_INCLUDED
#define HOSTRPC_CLIENT_HPP_INCLUDED

#include "../platform/detect.hpp"
#include "common.hpp"
#include "counters.hpp"
#include "cxx.hpp"
#include "state_machine.hpp"

// Intend to have call and service working across gcn and x86
// The normal terminology is:
// Client makes a call to the server, which does some work and sends back a
// reply

namespace hostrpc
{
struct fill_nop
{
  HOSTRPC_ANNOTATE void operator()(hostrpc::port_t, page_t *) {}
  fill_nop() = default;
  fill_nop(const fill_nop &) = delete;
  fill_nop(fill_nop &&) = delete;
};

struct use_nop
{
  HOSTRPC_ANNOTATE void operator()(hostrpc::port_t, page_t *) {}
  use_nop() = default;
  use_nop(const use_nop &) = delete;
  use_nop(use_nop &&) = delete;
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

template <typename WordT, typename SZT, typename Counter = counters::client>
struct client_impl : public state_machine_impl<WordT, SZT, Counter,
                                               false>
{
  using base =
    state_machine_impl<WordT, SZT, Counter, false>;
  using typename base::state_machine_impl;

  using Word = typename base::Word;
  using SZ = typename base::SZ;
  using lock_t = typename base::lock_t;
  using mailbox_t = typename base::mailbox_t;
  using inbox_t = typename base::inbox_t;
  using outbox_t = typename base::outbox_t;
  template <unsigned I, unsigned O>
  using typed_port_t = typename base::template typed_port_t<I, O>;

  HOSTRPC_ANNOTATE client_impl() : base() {}
  HOSTRPC_ANNOTATE ~client_impl() = default;
  HOSTRPC_ANNOTATE client_impl(SZ sz, lock_t active, inbox_t inbox,
                               outbox_t outbox,
                               page_t *shared_buffer)

      : base(sz, active, inbox, outbox, shared_buffer)
  {
    constexpr size_t client_size = 32;

    // SZ is expected to be zero bytes or a uint
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

  HOSTRPC_ANNOTATE static void *operator new(size_t, client_impl *p)
  {
    return p;
  }

  HOSTRPC_ANNOTATE client_counters get_counters() { return Counter::get(); }

  template <typename Op, typename T>
  HOSTRPC_ANNOTATE typed_port_t<0, 1> rpc_port_send_given_available(
      T active_threads, typed_port_t<0, 0> &&port, Op &&op)
  {
    // TODO: given_available is implicit in type now
    return base::template rpc_port_apply(active_threads, cxx::move(port),
                                         cxx::forward<Op>(op));
  }

  template <typename Op, typename T, unsigned I>
  HOSTRPC_ANNOTATE typed_port_t<0, 1> rpc_port_send(T active_threads,
                                                    typed_port_t<I, 0> &&port,
                                                    Op &&op)
  {
    // we know outbox is low (so we can send), but don't know whether inbox is
    // yet

    typed_port_t<0, 0> ready;
    if constexpr (I == 1)
      {
        ready = base::template rpc_port_wait(active_threads, cxx::move(port));
      }
    else
      {
        ready = cxx::move(port);
      }

    return rpc_port_send_given_available(active_threads, cxx::move(ready),
                                         cxx::forward<Op>(op));
  }

  template <typename T>
  HOSTRPC_ANNOTATE typed_port_t<1, 1> rpc_port_wait_for_result(
      T active_threads, typed_port_t<0, 1> &&port)
  {
    return base::template rpc_port_wait(active_threads, cxx::move(port));
  }

  template <typename Use, typename T, unsigned I>
  HOSTRPC_ANNOTATE typed_port_t<1, 0> rpc_port_recv(T active_threads,
                                                    typed_port_t<I, 1> &&port,
                                                    Use &&use)
  {
    // we know outbox is high (so we can recv), but don't know whether inbox is
    // yet
    typed_port_t<1, 1> ready;
    if constexpr (I == 0)
      {
        ready = base::template rpc_port_wait(active_threads, cxx::move(port));
      }
    else
      {
        ready = cxx::move(port);
      }

    return base::template rpc_port_apply(active_threads, cxx::move(ready),
                                         cxx::forward<Use>(use));
  }

  template <typename T>
  HOSTRPC_ANNOTATE typed_port_t<0, 0> rpc_port_wait_until_available(
      T active_threads, typed_port_t<1, 0> &&port)
  {
    return base::template rpc_port_wait(active_threads, cxx::move(port));
  }

  template <typename T, unsigned I, unsigned O>
  HOSTRPC_ANNOTATE void rpc_close_port(T active_threads,
                                       typed_port_t<I, O> &&port)
  {
    base::template rpc_close_port(active_threads, cxx::move(port));
  }

  template <unsigned I, unsigned O, typename T>
  HOSTRPC_ANNOTATE HOSTRPC_RETURN_UNKNOWN typename typed_port_t<I, O>::maybe rpc_try_open_typed_port(
      T active_threads, uint32_t scan_from = 0)
  {
    static_assert(I == O, "");
    return base::template rpc_try_open_typed_port<I, O, T>(active_threads, scan_from);
  }

  template <unsigned I, unsigned O, typename T>
  HOSTRPC_ANNOTATE typed_port_t<I, O> rpc_open_typed_port(
      T active_threads, uint32_t scan_from = 0)
  {
    static_assert(I == O, "");
    return base::template rpc_open_typed_port<I, O, T>(active_threads, scan_from);
  }
  
  template <typename T>
  HOSTRPC_ANNOTATE HOSTRPC_RETURN_UNKNOWN typename typed_port_t<0, 0>::maybe rpc_try_open_typed_port_lo(
      T active_threads, uint32_t scan_from = 0)
  {
    return base::template rpc_try_open_typed_port<0, 0, T>(active_threads, scan_from);
  }

#if 1
  // Would like to delete these.
  // printf_client.hpp makes that challenging as it's all written in terms of raw uint32_t
  // It includes interesting use cases like multiple calls to send one after another on the same port
  template <typename T>
  HOSTRPC_ANNOTATE port_t rpc_open_port(T active_threads)
  {
    return base::rpc_open_port_lo(active_threads);
  }

  template <typename T>
  HOSTRPC_ANNOTATE void rpc_port_wait_until_available(T active_threads,
                                                      port_t port)
  {
    typename base::port_state s;
    base::template rpc_port_wait_until_state<
        T, base::port_state::either_low_or_high>(active_threads, port, &s);

    if (s == base::port_state::high_values)
      {
        rpc_port_discard_result(active_threads, port);
        base::template rpc_port_wait_until_state<T,
                                                 base::port_state::low_values>(
            active_threads, port);
      }
  }

  template <typename Op, typename T>
  HOSTRPC_ANNOTATE void rpc_port_send(T active_threads, port_t port, Op &&op)
  {
    // If the port has just been opened, we know it is available to
    // submit work to. In general, send might be called while the
    // state machine is elsewhere, so conservatively progress it
    // until the slot is empty.
    // There is a potential bug here if 'use' is being used to
    // reset the state, instead of the server clean, as 'use'
    // is not being called, but that might be deemed a API misuse
    // as the callee could have used recv() explicitly instead of
    // dropping the result
    rpc_port_wait_until_available(active_threads, port);  // expensive
    rpc_port_send_given_available<Op>(active_threads, port,
                                      cxx::forward<Op>(op));
  }

  template <typename Op, typename T>
  HOSTRPC_ANNOTATE void rpc_port_send_given_available(T active_threads,
                                                      port_t port, Op &&op)
  {
    base::template rpc_port_apply_lo(active_threads, port,
                                     cxx::forward<Op>(op));
  }

  template <typename T>
  HOSTRPC_ANNOTATE void rpc_port_wait_for_result(T active_threads, port_t port)
  {
    // assumes output live
    assert(bits::nthbitset(
        base::outbox.load_word(this->size(), index_to_element<Word>(port)),
        index_to_subindex<Word>(port)));
    base::template rpc_port_wait_until_state<T, base::port_state::high_values>(
        active_threads, port);
  }

  template <typename T>
  HOSTRPC_ANNOTATE void rpc_port_discard_result(T active_threads, port_t port)
  {
    base::template rpc_port_apply_hi(active_threads, port,
                                     [](hostrpc::port_t, page_t *) {});
  }

  template <typename Use, typename T>
  HOSTRPC_ANNOTATE void rpc_port_recv(T active_threads, port_t port, Use &&use)
  {
    rpc_port_wait_for_result(active_threads, port);
    base::template rpc_port_apply_hi(active_threads, port,
                                     cxx::forward<Use>(use));
  }

  template <typename T>
  HOSTRPC_ANNOTATE void rpc_close_port(
      T active_threads,
      port_t port)  // Require != port_t::unavailable, not already closed
  {
    base::rpc_close_port(active_threads, port);
  }

  #endif
};

template <typename WordT, typename SZT, typename Counter = counters::client>
struct client : public client_impl<WordT, SZT, Counter>
{
  using base = client_impl<WordT, SZT, Counter>;
  using base::client_impl;
  template <unsigned I, unsigned O>
  using typed_port_t = typename base::template typed_port_t<I, O>;
  
  static_assert(cxx::is_trivially_copyable<base>::value, "");
  
  template <typename T, typename Fill>
  HOSTRPC_ANNOTATE bool rpc_invoke_async(T active_threads, Fill &&fill) noexcept
  {
    auto ApplyFill = hostrpc::make_apply<Fill>(cxx::forward<Fill>(fill));

    if (auto maybe = base::rpc_try_open_typed_port_lo(active_threads))
      {
        auto send = base::rpc_port_send(active_threads, maybe.value(), cxx::move(ApplyFill));
        base::rpc_close_port(active_threads, cxx::move(send));
        return true;
      }
    else
      {
       return false;
      }
  }

  // rpc_invoke returns true if it successfully launched the task
  // returns false if no slot was available

  // Return after calling use(), i.e. waits for server
  template <typename T, typename Fill, typename Use>
  HOSTRPC_ANNOTATE bool rpc_invoke(T active_threads, Fill &&fill,
                                   Use &&use) noexcept
  {
    auto ApplyFill = hostrpc::make_apply<Fill>(cxx::forward<Fill>(fill));
    auto ApplyUse = hostrpc::make_apply<Use>(cxx::forward<Use>(use));
    
    if (auto maybe = base::rpc_try_open_typed_port_lo(active_threads))
      {
        auto send = base::rpc_port_send(active_threads, maybe.value(), cxx::move(ApplyFill));
        auto recv = base::rpc_port_recv(active_threads, cxx::move(send), cxx::move(ApplyUse));
        base::rpc_close_port(active_threads, cxx::move(recv));
        return true;
      }
    else
      {
        return false;
      }   
  }

  // TODO: Probably want one of these convenience functions for each rpc_invoke,
  // but perhaps not on volta

  // Return after calling fill(), i.e. does not wait for server
  template <typename Fill>
  HOSTRPC_ANNOTATE bool rpc_invoke(Fill &&fill) noexcept
  {
    auto active_threads = platform::active_threads();
    return rpc_invoke_async(active_threads, cxx::forward<Fill>(fill));
  }

  template <typename Fill, typename Use>
  HOSTRPC_ANNOTATE bool rpc_invoke(Fill &&f, Use &&u) noexcept
  {
    auto active_threads = platform::active_threads();
    return rpc_invoke(active_threads, cxx::forward<Fill>(f),
                      cxx::forward<Use>(u));
  }
};

}  // namespace hostrpc

#endif
