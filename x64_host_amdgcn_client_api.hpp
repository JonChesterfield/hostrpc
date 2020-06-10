#ifndef HOSTRPC_X64_HOST_AMDGCN_CLIENT_API_HPP_INCLUDED
#define HOSTRPC_X64_HOST_AMDGCN_CLIENT_API_HPP_INCLUDED

#if defined(__AMDGCN__)
// todo: wire up a host alternative?
#include <stdint.h>
void hostcall_client_async(uint64_t data[8]);
void hostcall_client_kick_signal();
#else
#include "hsa.h"
const char *hostcall_client_symbol();
const char *hostcall_signal_symbol();
void *hostcall_server_init(hsa_region_t fine, void *client_address, void * signal_address);
void hostcall_server_dtor(void *);
void hostcall_server_handle_one_packet(void*);
#endif

#endif
