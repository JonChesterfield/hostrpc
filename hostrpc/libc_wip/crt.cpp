#include "crt.hpp"


demo_client client;

static void init_client(void * gpu_locks,
                 void * client_inbox,
                 void * client_outbox,
                 void * shared_buffer)
{
  client = demo_client(
      {},
      hostrpc::careful_cast_to_bitmap<demo_client::lock_t>(gpu_locks,
                                                           slots_words),
      hostrpc::careful_cast_to_bitmap<demo_client::inbox_t>(client_inbox,
                                                            slots_words),
      hostrpc::careful_cast_to_bitmap<demo_client::outbox_t>(client_outbox,
                                                             slots_words),
      hostrpc::careful_array_cast<BufferElement>(shared_buffer, slots));
}


extern "C"
int main(int, char**);

extern "C"
void __libc_write_stderr(const char* str) {
  auto active_threads =   platform::active_threads(); 

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
    }


}

__attribute__((amdgpu_kernel))
__attribute__((visibility("default")) )
extern "C"
void __start(void) // int, void*, int*
{
  
  __attribute__((address_space(4))) void * ptr = __builtin_amdgcn_dispatch_ptr();
  enum {kernarg_address_offset = 40};

  char* kernarg_address = (char*)ptr + kernarg_address_offset;
  char*kernarg;
  __builtin_memcpy(&kernarg, kernarg_address, 8);

  void * rpc_pointers[4];
  __builtin_memcpy(&rpc_pointers, kernarg, sizeof(rpc_pointers));
  kernarg+= sizeof(rpc_pointers);

  init_client(rpc_pointers[0],
              rpc_pointers[1],
              rpc_pointers[2],
              rpc_pointers[3]);

  
  int argc;
  __builtin_memcpy(&argc, kernarg, 4);
  kernarg+= 4;

  // padding
  kernarg+= 4;

  // pointer into mutable memory as argv strings are mutable
  char** argv;
  __builtin_memcpy(&argv, kernarg, 8);
  kernarg += 8;

  // also a pointer into mutable memory, wavefront_size wide
  int* result;
  __builtin_memcpy(&argv, kernarg, 8);
    kernarg += 8;
  
  int rc = main(argc, argv);
  (void)rc;
}
