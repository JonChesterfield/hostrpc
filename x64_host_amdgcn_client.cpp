#include "x64_host_amdgcn_client.hpp"

#if defined(__AMDGCN__)
__attribute__((visibility("default")))
hostrpc::x64_amdgcn_client<hostrpc::x64_host_amdgcn_array_size>
    client_singleton;

void hostcall_client_async(uint64_t data[8])
{
  // slightly easier to tell if it's running if the code is synchronous
  client_singleton.rpc_invoke<true>(static_cast<void *>(&data[0]));
}

// copied from hsa for now
typedef struct hsa_signal_s
{
  uint64_t handle;
} hsa_signal_t;

__attribute__((visibility("default"))) hsa_signal_t signal_singleton;

void hostcall_client_kick_signal()
{
  // handle is a pointer to an amd_signal_t
  // that's a complex type, from which we need 'value' to hit the atomic
  // there's a uint32_t event_id and a uint64_t event_mailbox_ptr
  // try to get this working roughly first to avoid working out how to
  // link in the ockl stuff
  // see hsaqs.cl
  char *ptr = (char *)signal_singleton.handle;
  // kind is first 8 bytes, then a union containing value in next 8 bytes
  _Atomic(uint64_t) *event_mailbox_ptr = (_Atomic(uint64_t) *)(ptr + 16);
  uint32_t *event_id = (uint32_t *)(ptr + 24);

  assert(event_mailbox_ptr);  // I don't hink this should be null

  if (platform::is_master_lane())
    {
      if (event_mailbox_ptr)
        {
#define AS(P, V, O, S) __opencl_atomic_store(P, V, O, S)
          uint32_t id = *event_id;
          AS(event_mailbox_ptr, id, __ATOMIC_RELEASE,
             __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);
#undef AS
          __builtin_amdgcn_s_sendmsg(1 | (0 << 4),
                                     __builtin_amdgcn_readfirstlane(id) & 0xff);
        }
    }
}

#else

const char* hostcall_client_symbol() { return "client_singleton"; }
const char* hostcall_signal_symbol() { return "signal_singleton"; }

// #include "hsa.hpp" // probably can't use this directly
hostrpc::x64_amdgcn_server<hostrpc::x64_host_amdgcn_array_size>
    server_singleton;

void* hostcall_server_init(hsa_region_t fine, void* client_address,
                           void* signal_address)
{
  hostrpc::x64_amdgcn_pair<hostrpc::x64_host_amdgcn_array_size>* res =
      new hostrpc::x64_amdgcn_pair<hostrpc::x64_host_amdgcn_array_size>(fine);

  hsa_status_t sig_rc = hsa_signal_create(UINT64_MAX, 0, NULL, &res->signal);
  assert(sig_rc == HSA_STATUS_SUCCESS);  // todo

  {
    size_t sz = res->client.serialize_size();
    uint64_t bytes[sz];
    res->client.serialize(bytes);
    memcpy(client_address, bytes, sz * sizeof(uint64_t));
  }
  {
    size_t sz = res->server.serialize_size();
    uint64_t bytes[sz];
    res->server.serialize(bytes);
    server_singleton.deserialize(bytes);
  }

  memcpy(signal_address, &res->signal, 8);

  return static_cast<void*>(res);
}

void hostcall_server_dtor(void* arg)
{
  hostrpc::x64_amdgcn_pair<hostrpc::x64_host_amdgcn_array_size>* res =
      static_cast<
          hostrpc::x64_amdgcn_pair<hostrpc::x64_host_amdgcn_array_size>*>(arg);
  hsa_signal_destroy(res->signal);
  delete (res);
}

bool hostcall_server_handle_one_packet(void* arg)
{
  hostrpc::x64_amdgcn_pair<hostrpc::x64_host_amdgcn_array_size>* res =
      static_cast<
          hostrpc::x64_amdgcn_pair<hostrpc::x64_host_amdgcn_array_size>*>(arg);

  const bool verbose = false;
  if (verbose)
    {
      printf("Client\n");
      res->client.inbox.dump();
      res->client.outbox.dump();
      res->client.active.dump();

      printf("Server\n");
      res->server.inbox.dump();
      res->server.outbox.dump();
      res->server.active.dump();
    }
#if 0
  hsa_signal_wait_acquire(res->signal, HSA_SIGNAL_CONDITION_NE, UINT64_MAX,
                          1024 * 1024, HSA_WAIT_STATE_BLOCKED);
#endif

  bool r = server_singleton.rpc_handle(nullptr);

  if (verbose)
    {
      printf(" --------------\n");
    }

  return r;
}

#endif
