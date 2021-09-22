#ifndef HOSTRPC_PRINTF_API_MACRO_H_INCLUDED
#define HOSTRPC_PRINTF_API_MACRO_H_INCLUDED

#ifdef __attribute__
#warning "__attribute__ is a macro, missing freestanding?"
#endif

// printf implementation macros, noinline is convenient for reading IR
#define __PRINTF_API_EXTERNAL_ HOSTRPC_ANNOTATE __attribute__((noinline))
#define __PRINTF_API_INTERNAL_ \
  HOSTRPC_ANNOTATE static inline __attribute__((unused))

#ifdef __cplusplus
#define __PRINTF_API_EXTERNAL __PRINTF_API_EXTERNAL_ extern "C"
#define __PRINTF_API_INTERNAL __PRINTF_API_INTERNAL_
#else
#define __PRINTF_API_EXTERNAL __PRINTF_API_EXTERNAL_
#define __PRINTF_API_INTERNAL \
  __PRINTF_API_INTERNAL_ __attribute__((overloadable))
#endif


#endif
