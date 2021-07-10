#if defined (__AMDGCN__)
#define EVILUNIT_USE_STDIO 0
#define EVILUNIT_HAVE_PRINTF 1
#include "hostrpc_printf.h"
#define printf(...) __hostrpc_printf(__VA_ARGS__)
#else
#ifndef EVILUNIT_USE_STDIO
#define EVILUNIT_USE_STDIO 1
#endif
#endif
#include "../../EvilUnit/EvilUnit.h"
