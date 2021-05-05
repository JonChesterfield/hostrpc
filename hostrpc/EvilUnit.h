#if defined (__AMDGCN__)
#define EVILUNIT_USE_STDIO 0
#define EVILUNIT_HAVE_PRINTF 1
#include "hostrpc_printf.h"
#else
#ifndef EVILUNIT_USE_STDIO
#define EVILUNIT_USE_STDIO 1
#endif
#endif
#include "../../EvilUnit/EvilUnit.h"
