#ifndef HOSTRPC_PRINTF_ENABLE_HPP_INCLUDED
#define HOSTRPC_PRINTF_ENABLE_HPP_INCLUDED

#include "detail/platform/detect.hpp"

#include <stddef.h>
#include <stdint.h>

#if (HOSTRPC_HOST)
#include "hsa.h"

#include "hostrpc_printf_server.hpp"
#include "hostrpc_thread.hpp"
#include "server_thread_state.hpp"

#include <memory>
#include <stdlib.h>

namespace
{
template <typename server_client_type>
struct wrap_server_state
{
  using sts_ty =
      hostrpc::server_thread_state<typename server_client_type::server_type,
                                   operate, clear>;

  std::unique_ptr<server_client_type> p;
  HOSTRPC_ATOMIC(uint32_t) server_control;

  sts_ty server_state;
  std::unique_ptr<hostrpc::thread<sts_ty>> thrd;

  std::unique_ptr<print_buffer_t> print_buffer;

  wrap_server_state() = delete;

  wrap_server_state(wrap_server_state &&) = delete;
  wrap_server_state &operator=(wrap_server_state &&) = delete;

  wrap_server_state(std::unique_ptr<server_client_type> &&p_, uint32_t size)
      : p(std::move(p_))
  {
    platform::atomic_store<uint32_t, __ATOMIC_RELEASE,
                           __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
        &server_control, 1);

    print_buffer = std::make_unique<print_buffer_t>();
    print_buffer->resize(size);

    operate op(print_buffer.get());
    server_state = sts_ty(&p->server, &server_control, op, clear{});

    thrd =
        std::make_unique<hostrpc::thread<sts_ty>>(make_thread(&server_state));

    if (!thrd->valid())
      {
        fprintf(stderr, "Failed to spawn thread\n");
        exit(1);
      }
  }

  ~wrap_server_state()
  {
    platform::atomic_store<uint32_t, __ATOMIC_RELEASE,
                           __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
        &server_control, 0);

    thrd->join();
  }
};
}  // namespace

extern "C"
{
  int hostrpc_print_enable_on_hsa_agent(hsa_executable_t ex,
                                        hsa_agent_t kernel_agent);
}
#endif

#endif
