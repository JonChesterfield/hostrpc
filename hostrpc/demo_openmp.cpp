
#pragma omp declare target

#include "detail/client_impl.hpp"
#include "detail/server_impl.hpp"
#include "memory.hpp"

struct BufferElement
{
  uint64_t data[64];
};
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

#include <stdio.h>

#include <omp.h>

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void *llvm_omp_target_alloc_host(size_t, int);

int main()
{
  // GPU locks

  // todo: force these things to be properly aligned (or fail if they aren't)
  void *gpu_locks = omp_target_alloc(slots_bytes, 0);
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
#pragma omp target
        {
          auto thrds = platform::active_threads();

          bool r = client.rpc_invoke_noapply(
              thrds, [](hostrpc::port_t, BufferElement *data) {
                auto me = platform::get_lane_id();
                data->data[me] = me * me + 5;
              });
        }
      }
    else
      {
        bool got_work = false;
        bool got_cleanup = false;
          
      again:;
        bool r = server.rpc_handle(
            [](hostrpc::port_t, BufferElement *data) {
              fprintf(stderr, "Server got work to do:\n");
              got_work = true;
              for (unsigned i = 0; i < 64; i++)
                {
                  fprintf(stderr, "data[%u] = %lu\n", i, data->data[i]);
                }
            },
            [](hostrpc::port_t, BufferElement *data) {
              fprintf(stderr, "Server cleaning up");
              got_cleanup = true;
              for (unsigned i = 0; i < 64; i++)
                {
                  data->data[i] = 0;
                }
            });

        if (!r)
          {
            for (unsigned i = 0; i < 10000; i++) platform::sleep_briefly();
            fprintf(stderr, "Sever [%u][%u ]no work\n", got_work, got_cleanup);
            goto again;
          }
        else
          {
            fprintf(stderr, "Server [%u][%u]returned true\n", got_work, got_cleanup);
          }
      }
  }

  omp_target_free(gpu_locks, 0);
  free(host_locks);
  omp_target_free(client_inbox, 0);
  omp_target_free(client_outbox, 0);
  omp_target_free(shared_buffer, 0);
}
