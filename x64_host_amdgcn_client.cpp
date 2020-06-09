#include "x64_host_amdgcn_client.hpp"

#if defined(__AMDGCN__)
hostrpc::x64_amdgcn_client<hostrpc::x64_host_amdgcn_array_size>
    client_singleton;

void hostrpc::hostcall_client_async(uint64_t data[8])
{
  client_singleton.rpc_invoke<false>(static_cast<void*>(&data[0]));
}

#else

// #include "hsa.hpp" // probably can't use this directly
hostrpc::x64_amdgcn_server<hostrpc::x64_host_amdgcn_array_size>
    server_singleton;

void hostcall_server_init() {}
void hostcall_server_dtor() {}
void hostcall_server_handle_one_packet() {}

#endif
