#ifndef CXX_ATEXIT_HPP_INCLUDED
#define CXX_ATEXIT_HPP_INCLUDED

// Toolchain doesn't seem totally set up. Atexit gets called from global
// constructors, but global constructors don't actually get run on amdgcn/hsa.
// 'Implementing' here for now.

#if !defined __OPENCL__
#if defined(__AMDGCN__)
// The toolchain shouldn't be emitting undef symbols to this.
// Work around here for now.
extern "C"
{
  __attribute__((weak)) int __cxa_atexit(void (*)(void *), void *, void *)
  {
    return 0;
  }
}
#endif
#endif
#endif
