#ifndef SERVER_THREAD_STATE_HPP_INCLUDED
#define SERVER_THREAD_STATE_HPP_INCLUDED

#include "detail/platform.hpp"

namespace hostrpc
{
// make_thread on &instance
template <typename Server, typename Operate, typename Clear>
struct server_thread_state
{
  Server* server;
  HOSTRPC_ATOMIC(uint32_t) * control;
  Operate op;
  Clear cl;

  server_thread_state() {}

  server_thread_state(Server* server, HOSTRPC_ATOMIC(uint32_t) * control,
                      Operate op, Clear cl)
      : server(server), control(control), op(op), cl(cl)
  {
  }

  void operator()()
  {
    uint32_t location = 0;
    auto serv_func_busy = [&]() {
      bool r = true;
      while (r)
        {
          r = server->template rpc_handle<Operate, Clear>(op, cl, &location);
        }
    };

    for (;;)
      {
        serv_func_busy();

        // ran out of work, has client set control to cease?
        uint32_t ctrl =
            platform::atomic_load<uint32_t, __ATOMIC_ACQUIRE,
                                  __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
                control);

        if (ctrl == 0)
          {
            // client called to cease, empty any clear jobs in the pipeline
            serv_func_busy();
            break;
          }

        // nothing to do, but not told to stop. spin.
        for (unsigned j = 0; j < 1000; j++)
          {
            platform::sleep();
          }
      }
  }
};

template <typename Server, typename Operate, typename Clear>
server_thread_state<Server, Operate, Clear> make_server_thread_state(
    Server* server, HOSTRPC_ATOMIC(uint32_t) * control, Operate op, Clear cl)
{
  return server_thread_state<Server, Operate, Clear>(server, control, op, cl);
}

}  // namespace hostrpc
#endif
