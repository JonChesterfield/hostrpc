#include "run_on_hsa.hpp"
#include "hsa.h"
#include <stdio.h>


namespace hostrpc
{
void run_on_hsa(void *arg, size_t len, const char *name)
{
  (void)arg;
  fprintf(stderr, "Call run_on_hsa, %zu bytes, %s\n", len, name);
}
}

