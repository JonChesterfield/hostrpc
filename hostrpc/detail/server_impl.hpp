#ifndef HOSTRPC_SERVER_HPP_INCLUDED
#define HOSTRPC_SERVER_HPP_INCLUDED

#include "../platform/detect.hpp"
#include "common.hpp"
#include "counters.hpp"
#include "cxx.hpp"
#include "state_machine.hpp"

namespace hostrpc
{
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

template <typename BufferElementT, typename WordT, typename SZT, typename Counter = counters::server>
struct server_impl : public state_machine_impl<BufferElementT, WordT, SZT, Counter, true>
{
  using base = state_machine_impl<BufferElementT, WordT, SZT, Counter, true>;
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

  HOSTRPC_ANNOTATE constexpr size_t wordBits() const
  {
    return 8 * sizeof(Word);
  }
  // may want to rename this, number-slots?
  HOSTRPC_ANNOTATE uint32_t size() const { return SZ::value(); }
  HOSTRPC_ANNOTATE uint32_t words() const { return size() / wordBits(); }

  HOSTRPC_ANNOTATE server_impl() : base() {}
  HOSTRPC_ANNOTATE ~server_impl() = default;
  HOSTRPC_ANNOTATE server_impl(SZ sz, lock_t active, inbox_t inbox,
                               outbox_t outbox, BufferElement* shared_buffer)
      : base(sz, active, inbox, outbox, shared_buffer)
  {
    constexpr size_t server_size = 32;

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

    constexpr size_t total_size = server_size + SZ_size + Counter_size;

    static_assert(sizeof(server_impl) == total_size, "");
    static_assert(alignof(server_impl) == 8, "");
  }

  HOSTRPC_ANNOTATE static void* operator new(size_t, server_impl* p)
  {
    return p;
  }

  HOSTRPC_ANNOTATE server_counters get_counters() { return Counter::get(); }


};

template <typename BufferElementT, typename WordT, typename SZT, typename Counter = counters::server>
struct server : public server_impl<BufferElementT, WordT, SZT, Counter>
{
  using base = server_impl<BufferElementT, WordT, SZT, Counter>;
  using base::server_impl;
  template <unsigned I, unsigned O>
  using typed_port_t = typename base::template typed_port_t<I, O>;
  template <unsigned S>
  using partial_port_t = typename base::template partial_port_t<S>;

  static_assert(cxx::is_trivially_copyable<base>()/*::value*/, "");

  // rpc_handle return true if it handled one task, does not attempt multiple.

  // Requires Operate argument.
  // If passed Clear, will call it on garbage collection
  // If passed location, will use it to round robin across slots

  template <typename Operate, typename Clear>
  HOSTRPC_ANNOTATE bool rpc_handle(Operate&& op, Clear&& cl,
                                   uint32_t* location) noexcept
  {
    auto active_threads = platform::active_threads();
    return rpc_handle_impl<Operate, Clear, true>(
        active_threads, cxx::forward<Operate>(op), cxx::forward<Clear>(cl),
        location);
  }

  template <typename Operate>
  HOSTRPC_ANNOTATE bool rpc_handle(Operate&& op, uint32_t* location) noexcept
  {
    auto active_threads = platform::active_threads();
    struct Clear
    {
      HOSTRPC_ANNOTATE void operator()(uint32_t, BufferElementT*){};
    };
    return rpc_handle_impl<Operate, Clear, false>(
        active_threads, cxx::forward<Operate>(op), Clear{}, location);
  }

  // Default location to always start from zero
  template <typename Operate, typename Clear>
  HOSTRPC_ANNOTATE bool rpc_handle(Operate&& op, Clear&& cl) noexcept
  {
    uint32_t location = 0;
    return rpc_handle<Operate, Clear>(cxx::forward<Operate>(op),
                                      cxx::forward<Clear>(cl), &location);
  }

  template <typename Operate>
  HOSTRPC_ANNOTATE bool rpc_handle(Operate&& op) noexcept
  {
    uint32_t location = 0;
    return rpc_handle<Operate>(cxx::forward<Operate>(op), &location);
  }

 private:
  template <typename Operate, typename Clear, bool have_precondition,
            typename T>
  HOSTRPC_ANNOTATE bool rpc_handle_impl(T active_threads, Operate&& op,
                                        Clear&& cl,
                                        uint32_t* location_arg) noexcept
  {
    bool result = false;
    // rpc_handle only reports 'true' on operate, garbage collection isn't
    // counted

    typename partial_port_t<1>::maybe maybe_port =
        base::template rpc_try_open_partial_port(active_threads, *location_arg);
    if (maybe_port)  // else do nothing, result initialised to false
      {
        partial_port_t<1> pport = maybe_port.value();
        constexpr bool OutboxGuess = false;
        if (pport.outbox_state() == OutboxGuess)
          {
            typename typed_port_t<0, 0>::maybe mport =
                base::template partial_to_typed<OutboxGuess>(active_threads,
                                                             cxx::move(pport));
            if (mport)
              {
                typed_port_t<0, 0> port = mport.value();
                typed_port_t<0, 1> res = base::template rpc_port_apply(
                    active_threads, cxx::move(port), cxx::forward<Operate>(op));
                *location_arg = 1 + static_cast<uint32_t>(res);
                base::template rpc_close_port(active_threads, cxx::move(res));
              }
            else
              {
                __builtin_unreachable();
              }
            mport.consumed();
          }
        else
          {
            typename typed_port_t<1, 1>::maybe mport =
                base::template partial_to_typed<!OutboxGuess>(active_threads,
                                                              cxx::move(pport));
            if (mport)
              {
                typed_port_t<1, 1> port = mport.value();
                typed_port_t<1, 0> res = base::template rpc_port_apply(
                    active_threads, cxx::move(port), cxx::forward<Clear>(cl));
                *location_arg = 1 + static_cast<uint32_t>(res);
                base::template rpc_close_port(active_threads, cxx::move(res));
              }
            else
              {
                __builtin_unreachable();
              }
            mport.consumed();
          }
        pport.consumed();
      }
    maybe_port.consumed();

    return result;
  }
};

}  // namespace hostrpc
#endif
