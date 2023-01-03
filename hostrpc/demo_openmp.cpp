
#pragma omp declare target

#include "detail/client_impl.hpp"
#include "detail/server_impl.hpp"
#include "memory.hpp"

struct cacheline_t
{
  alignas(64) uint64_t element[8];
};
static_assert(sizeof(cacheline_t) == 64, "");

struct BufferElement
{
  enum
  {
    width = 64
  };
  alignas(4096) cacheline_t cacheline[width];
};
static_assert(sizeof(BufferElement) == 4096, "");

inline void *operator new(size_t, BufferElement *p) { return p; }

using WordType = uint32_t;

enum
{
  slots = 128,  // number of slots, usually in bits
  slots_bytes = slots / 8,
  slots_words = slots_bytes / sizeof(WordType),
};

using demo_client =
    hostrpc::client<BufferElement, uint32_t, hostrpc::size_compiletime<slots>>;
using demo_server =
    hostrpc::server<BufferElement, uint32_t, hostrpc::size_compiletime<slots>>;

#pragma omp end declare target
demo_client client;
#pragma omp declare target to(client)
demo_server server;

#pragma omp begin declare target  // device_type(nohost)

size_t __printf_strlen(const char *str)
{
  // unreliable at O0
  if (__builtin_constant_p(str))
    {
      return __builtin_strlen(str);
    }
  else
    {
      for (size_t i = 0;; i++)
        {
          if (str[i] == '\0')
            {
              return i;
            }
        }
    }
}

bool write_to_stderr(const char *str)
{
  auto active_threads =
      platform::active_threads();  // warning, not valid on volta
  // uint64_t L = __printf_strlen(str);

  if (auto maybe = client.template rpc_try_open_typed_port(active_threads))
    {
      auto send = client.template rpc_port_send(
          active_threads, maybe.value(),
          [=](hostrpc::port_t, BufferElement *data) {
            auto me = platform::get_lane_id();
            enum
            {
              width = 48
            };

            data->cacheline[me].element[0] = 1;
            data->cacheline[me].element[7] = 0;

            __builtin_memcpy(&data->cacheline[me].element[1], str, width);
          });

      client.template rpc_close_port(active_threads, hostrpc::cxx::move(send));

      return true;
    }
  else
    {
      return false;
    }
}

#pragma omp end declare target

#include <stdio.h>

#include <omp.h>

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void *llvm_omp_target_alloc_host(size_t, int);

int main()
{
#define FORCE_ONTO_HOST 1

  // GPU locks

  // todo: force these things to be properly aligned (or fail if they aren't)
#if FORCE_ONTO_HOST
  void *gpu_locks = aligned_alloc(slots_bytes, 0);
#else
  void *gpu_locks = omp_target_alloc(slots_bytes, 0);
#endif
  void *host_locks = aligned_alloc(64, slots_bytes);

  void *client_inbox = llvm_omp_target_alloc_host(slots_bytes, 0);
  void *client_outbox = llvm_omp_target_alloc_host(slots_bytes, 0);

  void *shared_buffer =
      llvm_omp_target_alloc_host(slots * sizeof(BufferElement), 0);

  memset(host_locks, 0, slots_bytes);
  memset(client_inbox, 0, slots_bytes);
  memset(client_outbox, 0, slots_bytes);
  memset(shared_buffer, 0, slots_bytes);

  client = demo_client(
      {},
      hostrpc::careful_cast_to_bitmap<demo_client::lock_t>(gpu_locks,
                                                           slots_words),
      hostrpc::careful_cast_to_bitmap<demo_client::inbox_t>(client_inbox,
                                                            slots_words),
      hostrpc::careful_cast_to_bitmap<demo_client::outbox_t>(client_outbox,
                                                             slots_words),
      hostrpc::careful_array_cast<BufferElement>(shared_buffer, slots));
#pragma omp target update to(client)

  server = demo_server(
      {},
      hostrpc::careful_cast_to_bitmap<demo_server::lock_t>(host_locks,
                                                           slots_words),
      hostrpc::careful_cast_to_bitmap<demo_server::inbox_t>(client_outbox,
                                                            slots_words),
      hostrpc::careful_cast_to_bitmap<demo_server::outbox_t>(client_inbox,
                                                             slots_words),
      hostrpc::careful_array_cast<BufferElement>(shared_buffer, slots));

#pragma omp parallel num_threads(2)
  {
    unsigned id = omp_get_thread_num();
    printf("on the host, thread %u\n", id);

    if (id == 0)
      {
#if FORCE_ONTO_HOST
#else
#pragma omp target
#endif

        {
#if 0
          auto thrds = platform::active_threads();
          bool r = client.rpc_invoke_noapply(
               thrds,
               [](hostrpc::port_t, BufferElement *data) {
                 auto me = platform::get_lane_id();
                 data->cacheline[me].element[0] = me * me + 5;
               },
               [](hostrpc::port_t, BufferElement *data){});
#else
          write_to_stderr("some string literal");
#endif
        }
      }
    else
      {
        bool got_work = false;
        bool got_cleanup = false;

      again:;
        bool r = server.rpc_handle(
            [&](hostrpc::port_t, BufferElement *data) {
              fprintf(stderr, "Server got work to do:\n");
              got_work = true;
              for (unsigned i = 0; i < 64; i++)
                {
                  auto ith = data->cacheline[i];
                  fprintf(stderr, "data[%u] = {%lu, %lu...}\n", i,
                          ith.element[0], ith.element[1]);
                }
            },
            [&](hostrpc::port_t, BufferElement *data) {
              fprintf(stderr, "Server cleaning up\n");
              got_cleanup = true;
              for (unsigned i = 0; i < 64; i++)
                {
                  data->cacheline[i].element[0] = 0;
                }
            });

        if (!r)
          {
            for (unsigned i = 0; i < 10000; i++) platform::sleep_briefly();
            fprintf(stderr, "Server [%u][%u] no work\n", got_work, got_cleanup);
            if (got_work && got_cleanup)
              {
              }
            else
              {
                goto again;
              }
          }
        else
          {
            fprintf(stderr, "Server [%u][%u] returned true\n", got_work,
                    got_cleanup);
          }
      }
  }

#if FORCE_ONTO_HOST
  free(gpu_locks);
#else
  omp_target_free(gpu_locks, 0);
#endif
  free(host_locks);
  omp_target_free(client_inbox, 0);
  omp_target_free(client_outbox, 0);
  omp_target_free(shared_buffer, 0);
}
