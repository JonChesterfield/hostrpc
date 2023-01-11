#ifndef TYPESTATE_H_INCLUDED
#define TYPESTATE_H_INCLUDED

#define HOSTRPC_USE_TYPESTATE 1

#if defined(__OPENCL_C_VERSION__)
// May be able to make this work, need work out why opencl is upset about
// the definition of cxx::move. Looks like tagging __private or __generic
// on some of the types is done inconsistently
#undef HOSTRPC_USE_TYPESTATE
#define HOSTRPC_USE_TYPESTATE 0
#endif

#if HOSTRPC_USE_TYPESTATE
#define HOSTRPC_CONSUMABLE_CLASS __attribute__((consumable(unconsumed)))

#define HOSTRPC_RETURN_CONSUMED __attribute__((return_typestate(consumed)))
#define HOSTRPC_RETURN_UNKNOWN __attribute__((return_typestate(unknown)))
#define HOSTRPC_CREATED_RES __attribute__((return_typestate(unconsumed)))

#define HOSTRPC_CONSUMED_ARG                   \
  __attribute__((param_typestate(unconsumed))) \
      __attribute__((return_typestate(consumed)))
#define HOSTRPC_CONST_REF_ARG                  \
  __attribute__((param_typestate(unconsumed))) \
      __attribute__((return_typestate(unconsumed)))

#define HOSTRPC_CALL_ON_LIVE __attribute__((callable_when(unconsumed)))
#define HOSTRPC_CALL_ON_DEAD __attribute__((callable_when(consumed)))
#define HOSTRPC_CALL_ON_UNKNOWN __attribute__((callable_when(unknown)))

#define HOSTRPC_SET_TYPESTATE(X) __attribute__((set_typestate(X)))
#define HOSTRPC_TEST_TYPESTATE(X) __attribute__((test_typestate(X)))
#else
#define HOSTRPC_CONSUMABLE_CLASS
#define HOSTRPC_RETURN_CONSUMED
#define HOSTRPC_RETURN_UNKNOWN
#define HOSTRPC_CREATED_RES
#define HOSTRPC_CONSUMED_ARG
#define HOSTRPC_CONST_REF_ARG
#define HOSTRPC_CALL_ON_LIVE
#define HOSTRPC_CALL_ON_DEAD
#define HOSTRPC_CALL_ON_UNKNOWN
#define HOSTRPC_SET_TYPESTATE(X)
#define HOSTRPC_TEST_TYPESTATE(X)
#endif

#endif
