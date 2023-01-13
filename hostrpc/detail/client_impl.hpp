#ifndef HOSTRPC_CLIENT_HPP_INCLUDED
#define HOSTRPC_CLIENT_HPP_INCLUDED

#include "../platform/detect.hpp"
#include "common.hpp"
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
  template <typename BufferElement>
  HOSTRPC_ANNOTATE void operator()(uint32_t, BufferElement *)
  {
  }
  fill_nop() = default;
  fill_nop(const fill_nop &) = delete;
  fill_nop(fill_nop &&) = delete;
};

struct use_nop
{
  template <typename BufferElement>
  HOSTRPC_ANNOTATE void operator()(uint32_t, BufferElement *)
  {
  }
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

template <typename BufferElementT, typename WordT, typename SZT>
struct client_impl
    : public state_machine_impl<BufferElementT, WordT, SZT, false>
{
  using base = state_machine_impl<BufferElementT, WordT, SZT, false>;
  using typename base::state_machine_impl;

  using BufferElement = typename base::BufferElement;
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
                               outbox_t outbox, BufferElement *shared_buffer)

      : base(sz, active, inbox, outbox, shared_buffer)
  {
    constexpr size_t client_size = 32;

    // SZ is expected to be zero bytes or a uint
    struct SZ_local : public SZ
    {
      float x;
    };
    constexpr bool SZ_empty = sizeof(SZ_local) == sizeof(float);

    constexpr size_t SZ_size = hostrpc::round8(SZ_empty ? 0 : sizeof(SZ));

    constexpr size_t total_size = client_size + SZ_size;

    static_assert(sizeof(client_impl) == total_size, "");
    static_assert(alignof(client_impl) == 8, "");
  }

  HOSTRPC_ANNOTATE static void *operator new(size_t, client_impl *p)
  {
    return p;
  }

  template <typename T>
  HOSTRPC_ANNOTATE typed_port_t<0, 0> rpc_open_typed_port(
      T active_threads, uint32_t scan_from = 0)
  {
    // Warning: If no other call opens ports in <1, 1> state, it's possible
    // to end up with all ports in <0, 0> and hit exhaustion.
    // This would be safer if it garbage collected <1,1> ports on the fly.
    constexpr unsigned I = 0;
    constexpr unsigned O = 0;
    return base::template rpc_open_typed_port<I, O, T>(active_threads,
                                                       scan_from);
  }

  template <typename T>
  HOSTRPC_ANNOTATE HOSTRPC_RETURN_UNKNOWN typename typed_port_t<0, 0>::maybe
  rpc_try_open_typed_port(T active_threads, uint32_t scan_from = 0)
  {
    constexpr unsigned I = 0;
    constexpr unsigned O = 0;
    return base::template rpc_try_open_typed_port<I, O, T>(active_threads,
                                                           scan_from);
  }

  template <typename T, unsigned I, unsigned O>
  HOSTRPC_ANNOTATE void rpc_close_port(T active_threads,
                                       typed_port_t<I, O> &&port)
  {
    base::template rpc_close_port(active_threads, cxx::move(port));
  }

  template <typename T>
  HOSTRPC_ANNOTATE typed_port_t<0, 0> rpc_port_wait_until_available(
      T active_threads, typed_port_t<1, 0> &&port)
  {
    return base::template rpc_port_recv(active_threads, cxx::move(port));
  }

  template <typename T>
  HOSTRPC_ANNOTATE typed_port_t<1, 1> rpc_port_wait_for_result(
      T active_threads, typed_port_t<0, 1> &&port)
  {
    return base::template rpc_port_recv(active_threads, cxx::move(port));
  }

  template <typename Op, typename T, unsigned I>
  HOSTRPC_ANNOTATE typed_port_t<0, 1> rpc_port_send(T active_threads,
                                                    typed_port_t<I, 0> &&port,
                                                    Op &&op)
  {
    // can only try to send with a low outbox, but can wait for inbox to clear

    typed_port_t<0, 0> ready;
    if constexpr (I == 1)
      {
        ready = rpc_port_wait_until_available(active_threads, cxx::move(port));
      }
    else
      {
        ready = cxx::move(port);
      }

    return base::template rpc_port_apply(active_threads, cxx::move(ready),
                                         cxx::forward<Op>(op));
  }

  template <typename Use, typename T, unsigned I>
  HOSTRPC_ANNOTATE typed_port_t<1, 0> rpc_port_wait(T active_threads,
                                                    typed_port_t<I, 1> &&port,
                                                    Use &&use)
  {
    // can only try to recv with a high outbox, but can wait for the inbox
    typed_port_t<1, 1> ready;
    if constexpr (I == 0)
      {
        ready = rpc_port_wait_for_result(active_threads, cxx::move(port));
      }
    else
      {
        ready = cxx::move(port);
      }

    return base::template rpc_port_apply(active_threads, cxx::move(ready),
                                         cxx::forward<Use>(use));
  }

  template <typename T>
  HOSTRPC_ANNOTATE typed_port_t<1, 0> rpc_port_discard_result(
      T active_threads, typed_port_t<1, 1> &&port)
  {
    return rpc_port_wait(active_threads, hostrpc::cxx::move(port),
                         [](uint32_t, BufferElement *) {});
  }

  template <typename T, unsigned I, unsigned O>
  HOSTRPC_ANNOTATE typed_port_t<0, 0> rpc_port_wait_until_available(
      T active_threads, typed_port_t<I, O> &&port)
  {
    if constexpr (I == 0 && O == 0)
      {
        return cxx::move(port);
      }

    if constexpr (I == 0 && O == 1)
      {
        auto tmp = rpc_port_wait_for_result(active_threads, cxx::move(port));
        return rpc_port_wait_until_available<T, 1, 1>(active_threads,
                                                      cxx::move(tmp));
      }

    if constexpr (I == 1 && O == 1)
      {
        auto tmp = rpc_port_discard_result(active_threads, cxx::move(port));
        return rpc_port_wait_until_available<T, 1, 0>(active_threads,
                                                      cxx::move(tmp));
      }

    if constexpr (I == 1 && O == 0)
      {
        return base::rpc_port_recv(active_threads, cxx::move(port));
      }
  }
};

template <typename BufferElementT, typename WordT, typename SZT>
struct client : public client_impl<BufferElementT, WordT, SZT>
{
  // TODO: Write directly in terms of state machine? Credible chance that any
  // program which wants more control than the invoke/invoke_async api will do
  // better with the raw state_machine.
  using base = client_impl<BufferElementT, WordT, SZT>;
  using base::client_impl;
  template <unsigned I, unsigned O>
  using typed_port_t = typename base::template typed_port_t<I, O>;

  static_assert(cxx::is_trivially_copyable<base>() /*::value*/, "");

  template <typename T, typename Fill>
  HOSTRPC_ANNOTATE bool rpc_invoke_async(T active_threads, Fill &&fill) noexcept
  {
    if (auto maybe = base::rpc_try_open_typed_port(active_threads))
      {
        auto send =
            base::rpc_port_send(active_threads, maybe.value(), cxx::move(fill));
        base::rpc_close_port(active_threads, cxx::move(send));
        return true;
      }
    else
      {
        return false;
      }
  }

  template <typename T, typename Fill, typename Use>
  HOSTRPC_ANNOTATE bool rpc_invoke(T active_threads, Fill &&fill,
                                   Use &&use) noexcept
  {
    if (auto maybe = base::rpc_try_open_typed_port(active_threads))
      {
        auto send =
            base::rpc_port_send(active_threads, maybe.value(), cxx::move(fill));
        auto recv = base::rpc_port_recv(active_threads, cxx::move(send),
                                        cxx::move(use));
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
