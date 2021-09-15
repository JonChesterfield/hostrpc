#ifndef HOSTRPC_OPENMP_PLUGINS_HPP_INCLUDED
#define HOSTRPC_OPENMP_PLUGINS_HPP_INCLUDED

namespace hostrpc
{
void disable_amdgcn();
void disable_nvptx();

struct plugins
{
  bool amdgcn = false;
  bool nvptx = false;
};

plugins find_plugins();  // lazily evaluated then cached
}  // namespace hostrpc

#endif
