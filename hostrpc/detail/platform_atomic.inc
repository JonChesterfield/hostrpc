#ifndef HOSTRPC_PLATFORM_HPP_INCLUDED
#error "Expected to include platform_atomic.inc from platform.hpp"
#endif

#ifndef HOSTRPC_ATOMIC
#define "Require definition for HOSTRPC_ATOMIC"
#endif

#ifndef HOSTRPC_PLATFORM_ATOMIC_FUNCTION_ATTRIBUTE
#error "Require definition for HOSTRPC_PLATFORM_ATOMIC_FUNCTION_ATTRIBUTE"
#endif

#ifndef HOSTRPC_PLATFORM_ATOMIC_ADDRSPACE_ATTRIBUTE
#error "Require definition for HOSTRPC_PLATFORM_ATOMIC_ADDRSPACE_ATTRIBUTE"
#endif

template <typename T, size_t memorder, size_t scope>
HOSTRPC_PLATFORM_ATOMIC_FUNCTION_ATTRIBUTE T atomic_load(
    HOSTRPC_PLATFORM_ATOMIC_ADDRSPACE_ATTRIBUTE HOSTRPC_ATOMIC(T) const *addr)
{
  static_assert(detail::atomic_params_load<memorder, scope>(), "");
  return __opencl_atomic_load(addr, memorder, scope);
}

template <typename T, size_t memorder, size_t scope>
HOSTRPC_PLATFORM_ATOMIC_FUNCTION_ATTRIBUTE void atomic_store(
    HOSTRPC_PLATFORM_ATOMIC_ADDRSPACE_ATTRIBUTE HOSTRPC_ATOMIC(T) * addr,
    T value)
{
  static_assert(detail::atomic_params_store<memorder, scope>(), "");
  return __opencl_atomic_store(addr, value, memorder, scope);
}

template <typename T, size_t memorder, size_t scope>
HOSTRPC_PLATFORM_ATOMIC_FUNCTION_ATTRIBUTE T atomic_fetch_add(
    HOSTRPC_PLATFORM_ATOMIC_ADDRSPACE_ATTRIBUTE HOSTRPC_ATOMIC(T) * addr,
    T value)
{
  static_assert(detail::atomic_params_readmodifywrite<memorder, scope>(), "");
  return __opencl_atomic_fetch_add(addr, value, memorder, scope);
}

template <typename T, size_t memorder, size_t scope>
HOSTRPC_PLATFORM_ATOMIC_FUNCTION_ATTRIBUTE T atomic_fetch_and(
    HOSTRPC_PLATFORM_ATOMIC_ADDRSPACE_ATTRIBUTE HOSTRPC_ATOMIC(T) * addr,
    T value)
{
  static_assert(detail::atomic_params_readmodifywrite<memorder, scope>(), "");
  return __opencl_atomic_fetch_and(addr, value, memorder, scope);
}

template <typename T, size_t memorder, size_t scope>
HOSTRPC_PLATFORM_ATOMIC_FUNCTION_ATTRIBUTE T atomic_fetch_or(
    HOSTRPC_PLATFORM_ATOMIC_ADDRSPACE_ATTRIBUTE HOSTRPC_ATOMIC(T) * addr,
    T value)
{
  static_assert(detail::atomic_params_readmodifywrite<memorder, scope>(), "");
  return __opencl_atomic_fetch_or(addr, value, memorder, scope);
}

template <typename T, size_t memorder, size_t scope>
HOSTRPC_PLATFORM_ATOMIC_FUNCTION_ATTRIBUTE bool atomic_compare_exchange_weak(
    HOSTRPC_PLATFORM_ATOMIC_ADDRSPACE_ATTRIBUTE HOSTRPC_ATOMIC(T) * addr,
    T expected, T desired, T *loaded)
{
  static_assert(detail::atomic_params_readmodifywrite<memorder, scope>(), "");
  bool r = __opencl_atomic_compare_exchange_weak(addr, &expected, desired,
                                                 memorder, memorder, scope);
  *loaded = expected;
  return r;
}

#undef HOSTRPC_PLATFORM_ATOMIC_FUNCTION_ATTRIBUTE
#undef HOSTRPC_PLATFORM_ATOMIC_ADDRSPACE_ATTRIBUTE
