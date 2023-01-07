#ifndef HOSTRPC_PLATFORM_HPP_INCLUDED
#error "Expected to include platform_atomic.inc from platform.hpp"
#endif

#ifndef HOSTRPC_ATOMIC
#define "Require definition for HOSTRPC_ATOMIC"
#endif

#ifndef HOSTRPC_ANNOTATE
#error "Require definition for HOSTRPC_ANNOTATE"
#endif

#ifndef HOSTRPC_PLATFORM_ATOMIC_ADDRSPACE_ATTRIBUTE
#error "Require definition for HOSTRPC_PLATFORM_ATOMIC_ADDRSPACE_ATTRIBUTE"
#endif

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE T atomic_load(
    HOSTRPC_PLATFORM_ATOMIC_ADDRSPACE_ATTRIBUTE HOSTRPC_ATOMIC(T) const *addr)
{
  static_assert(platform::detail::atomic_params_load<memorder, scope>(), "");
  return __opencl_atomic_load(addr, memorder, scope);
}

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE void atomic_store(
    HOSTRPC_PLATFORM_ATOMIC_ADDRSPACE_ATTRIBUTE HOSTRPC_ATOMIC(T) * addr,
    T value)
{
  static_assert(platform::detail::atomic_params_store<memorder, scope>(), "");
  return __opencl_atomic_store(addr, value, memorder, scope);
}

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE T atomic_fetch_add(
    HOSTRPC_PLATFORM_ATOMIC_ADDRSPACE_ATTRIBUTE HOSTRPC_ATOMIC(T) * addr,
    T value)
{
  static_assert(
      platform::detail::atomic_params_readmodifywrite<memorder, scope>(), "");
  return __opencl_atomic_fetch_add(addr, value, memorder, scope);
}

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE T atomic_fetch_sub(
    HOSTRPC_PLATFORM_ATOMIC_ADDRSPACE_ATTRIBUTE HOSTRPC_ATOMIC(T) * addr,
    T value)
{
  static_assert(
      platform::detail::atomic_params_readmodifywrite<memorder, scope>(), "");
  return __opencl_atomic_fetch_sub(addr, value, memorder, scope);
}

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE T atomic_fetch_and(
    HOSTRPC_PLATFORM_ATOMIC_ADDRSPACE_ATTRIBUTE HOSTRPC_ATOMIC(T) * addr,
    T value)
{
  static_assert(
      platform::detail::atomic_params_readmodifywrite<memorder, scope>(), "");
  return __opencl_atomic_fetch_and(addr, value, memorder, scope);
}

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE T atomic_fetch_or(
    HOSTRPC_PLATFORM_ATOMIC_ADDRSPACE_ATTRIBUTE HOSTRPC_ATOMIC(T) * addr,
    T value)
{
  static_assert(
      platform::detail::atomic_params_readmodifywrite<memorder, scope>(), "");
  return __opencl_atomic_fetch_or(addr, value, memorder, scope);
}

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE T atomic_fetch_xor(
    HOSTRPC_PLATFORM_ATOMIC_ADDRSPACE_ATTRIBUTE HOSTRPC_ATOMIC(T) * addr,
    T value)
{
  static_assert(
      platform::detail::atomic_params_readmodifywrite<memorder, scope>(), "");
  return __opencl_atomic_fetch_xor(addr, value, memorder, scope);
}

#undef HOSTRPC_PLATFORM_ATOMIC_ADDRSPACE_ATTRIBUTE
