#ifndef HOSTRPC_X64_GCN_TYPE_HPP_INCLUDED
#define HOSTRPC_X64_GCN_TYPE_HPP_INCLUDED

#include "client_server_pair.hpp"

namespace hostrpc
{
template <typename SZ>
using x64_gcn_type = client_server_pair_t<SZ, arch::x64, arch::gcn>;

}  // namespace hostrpc

#endif
