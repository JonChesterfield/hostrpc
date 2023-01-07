#ifndef HOSTRPC_SERVER_HPP_INCLUDED
#define HOSTRPC_SERVER_HPP_INCLUDED

#include "../platform/detect.hpp"
#include "common.hpp"
#include "cxx.hpp"
#include "state_machine.hpp"

namespace hostrpc
{

template <typename BufferElementT, typename WordT, typename SZT>
using server =
  state_machine_impl<BufferElementT, WordT, SZT, true>;

template <typename BufferElementT, typename WordT, typename SZT,
          typename Operate, typename Clear>
HOSTRPC_ANNOTATE bool rpc_handle(server<BufferElementT, WordT, SZT>* server,
                                 Operate&& op, Clear&& cl,
                                 uint32_t* location_arg) noexcept
{
  auto active_threads = platform::active_threads();
  bool result = false;
  // rpc_handle only reports 'true' on operate, garbage collection isn't
  // counted

  using server_t = typename hostrpc::server<BufferElementT, WordT, SZT>;

  typename server_t::template partial_port_t<1>::maybe maybe_port =
      server->rpc_try_open_partial_port(active_threads, *location_arg);
  if (maybe_port)  // else do nothing, result initialised to false
    {
      auto pport = maybe_port.value();
      constexpr bool OutboxGuess = false;
      if (pport.outbox_state() == OutboxGuess)
        {
          auto mport = server->template partial_to_typed<OutboxGuess>(
              active_threads, cxx::move(pport));
          if (mport)
            {
              auto port = mport.value();
              auto res = server->rpc_port_apply(active_threads, cxx::move(port),
                                                cxx::forward<Operate>(op));
              *location_arg = 1 + static_cast<uint32_t>(res);
              server->rpc_close_port(active_threads, cxx::move(res));
            }
          else
            {
              __builtin_unreachable();
            }
          mport.consumed();
        }
      else
        {
          auto mport = server->template partial_to_typed<!OutboxGuess>(
              active_threads, cxx::move(pport));
          if (mport)
            {
              auto port = mport.value();
              auto res = server->rpc_port_apply(active_threads, cxx::move(port),
                                                cxx::forward<Clear>(cl));
              *location_arg = 1 + static_cast<uint32_t>(res);
              server->rpc_close_port(active_threads, cxx::move(res));
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

template <typename BufferElementT, typename WordT, typename SZT,
          typename Operate>
HOSTRPC_ANNOTATE bool rpc_handle(server<BufferElementT, WordT, SZT>* server,
                                 Operate&& op, uint32_t* location) noexcept
{
  auto active_threads = platform::active_threads();
  struct Clear
  {
    HOSTRPC_ANNOTATE void operator()(uint32_t, BufferElementT*){};
  };
  return rpc_handle(server, active_threads, cxx::forward<Operate>(op), Clear{},
                    location);
}

// Default location to always start from zero
template <typename BufferElementT, typename WordT, typename SZT,
          typename Operate, typename Clear>
HOSTRPC_ANNOTATE bool rpc_handle(server<BufferElementT, WordT, SZT>* server,
                                 Operate&& op, Clear&& cl) noexcept
{
  uint32_t location = 0;
  return rpc_handle(server, cxx::forward<Operate>(op), cxx::forward<Clear>(cl),
                    &location);
}

template <typename BufferElementT, typename WordT, typename SZT,
          typename Operate>
HOSTRPC_ANNOTATE bool rpc_handle(server<BufferElementT, WordT, SZT>* server,
                                 Operate&& op) noexcept
{
  uint32_t location = 0;
  return rpc_handle(server, cxx::forward<Operate>(op), &location);
}

}  // namespace hostrpc
#endif
