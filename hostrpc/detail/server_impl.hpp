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

template <typename WordT, typename SZT, typename Counter = counters::server>
struct server_impl : public state_machine_impl<WordT, SZT, Counter,
                                               true>
{
  using base =
    state_machine_impl<WordT, SZT, Counter, true>;
  using typename base::state_machine_impl;

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
                               outbox_t outbox,
                               page_t* shared_buffer)
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

  template <typename Clear, typename T>
  HOSTRPC_ANNOTATE port_t rpc_open_port(T active_threads, Clear&& cl,
                                        uint32_t* location_arg)
  {
    return rpc_open_port_impl<Clear, false>(
        active_threads, cxx::forward<Clear>(cl), location_arg);
  }

  template <typename T>
  HOSTRPC_ANNOTATE port_t rpc_open_port(T active_threads,
                                        uint32_t* location_arg)
  {
    struct Clear
    {
      HOSTRPC_ANNOTATE void operator()(uint32_t, hostrpc::page_t*){};
    };
    return rpc_open_port_impl<Clear, false>(active_threads, Clear{},
                                            location_arg);
  }

  // default location_arg to zero and discard returned hint
  template <typename Clear, typename T>
  HOSTRPC_ANNOTATE port_t rpc_open_port(T active_threads, Clear&& cl)
  {
    uint32_t location_arg = 0;
    return rpc_open_port<Clear>(active_threads, cxx::forward<Clear>(cl),
                                &location_arg);
  }

  template <typename T>
  HOSTRPC_ANNOTATE port_t rpc_open_port(T active_threads)
  {
    uint32_t location_arg = 0;
    return rpc_open_port(active_threads, &location_arg);
  }

  HOSTRPC_ANNOTATE bool lock_held(port_t port)
  {
    const uint32_t element = index_to_element<Word>(port);
    const uint32_t subindex = index_to_subindex<Word>(port);
    return bits::nthbitset(base::active.load_word(size(), element), subindex);
  }

  template <typename Operate, typename T>
  HOSTRPC_ANNOTATE void rpc_port_operate_given_available(T active_threads,
                                                         Operate&& op,
                                                         port_t port)
  {
    (void)active_threads;
    (void)op;
    (void)port;

    (void)active_threads;
    assert(port != port_t::unavailable);

    base::rpc_port_apply_lo(active_threads, port, cxx::forward<Operate>(op));

    // claim

    assert(lock_held(port));
  }

 protected:
  template <typename Clear, bool have_precondition, typename T>
  HOSTRPC_ANNOTATE port_t rpc_open_port_impl(T active_threads, Clear&& cl,
                                             uint32_t* location_arg)
  {
    typename base::port_state ps;
    Clear& clref = cl;

  try_again:;
    port_t p = base::template rpc_open_port(active_threads, *location_arg, &ps);
    if (p == port_t::unavailable)
      {
        return p;
      }

    if (ps == base::port_state::high_values)
      {
        base::rpc_port_apply_hi(active_threads, p, clref);
        base::rpc_close_port(active_threads, p);
        goto try_again;
      }

    assert(ps == base::port_state::low_values);
    *location_arg = static_cast<uint32_t>(p) + 1;
    return p;
  }
};

template <typename WordT, typename SZT, typename Counter = counters::server>
struct server : public server_impl<WordT, SZT, Counter>
{
  using base = server_impl<WordT, SZT, Counter>;
  using base::server_impl;
  template <unsigned I, unsigned O>
  using typed_port_t = typename base::template typed_port_t<I, O>;

  static_assert(cxx::is_trivially_copyable<base>::value, "");

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
      HOSTRPC_ANNOTATE void operator()(uint32_t, hostrpc::page_t*){};
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
  template <typename T, typename Op, unsigned IandO, bool retValue>
  struct noLambdasInOpenCL
  {
    server* self;
    bool* result;
    Op& op;

    HOSTRPC_ANNOTATE noLambdasInOpenCL(server* self, bool* result, Op& op)
        : self(self), result(result), op(op)
    {
    }

    HOSTRPC_ANNOTATE typed_port_t<IandO, !IandO> operator()(
        T active_threads, typed_port_t<IandO, IandO>&& port)
    {
      typed_port_t<IandO, !IandO> r = self->rpc_port_apply(
          active_threads, cxx::move(port), cxx::forward<Op>(op));
      *result = retValue;
      return cxx::move(r);
    }
  };

  template <typename Operate, typename Clear, bool have_precondition,
            typename T>
  HOSTRPC_ANNOTATE bool rpc_handle_impl(T active_threads, Operate&& op,
                                        Clear&& cl,
                                        uint32_t* location_arg) noexcept
  {
    bool result = false;
    // rpc_handle only reports 'true' on operate, garbage collection isn't
    // counted
    auto On00 = noLambdasInOpenCL<T, Operate, 0, true>(this, &result, op);
    auto On11 = noLambdasInOpenCL<T, Clear, 1, false>(this, &result, cl);
    struct None
    {
      /*result initialised to false*/
      HOSTRPC_ANNOTATE void operator()(T) {}
    };

    *location_arg = 1 + base::template rpc_with_opened_port(
                            active_threads, *location_arg, cxx::move(On00),
                            cxx::move(On11), None{});

    return result;
  }
};

}  // namespace hostrpc
#endif
