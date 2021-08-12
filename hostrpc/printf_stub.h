
// Functions that will be implemented via hostrpc, currently stubbed out
_Static_assert(sizeof(int32_t) == sizeof(int), "");
_Static_assert((sizeof(int64_t) == sizeof(long)) ||
                   (sizeof(int64_t) == sizeof(long long)),
               "");
_Static_assert(sizeof(uint32_t) == sizeof(unsigned int), "");
_Static_assert((sizeof(uint64_t) == sizeof(unsigned long)) ||
                   (sizeof(uint64_t) == sizeof(unsigned long long)),
               "");

uint32_t __printf_print_start(const char *fmt)
{
  (printf)("Start: %s ", fmt);
  return 101;
}

int __printf_print_end(uint32_t port)
{
  (void)port;
  (printf)("\n");
  return 0;
}

void __printf_pass_element_int32(uint32_t port, int32_t x)
{
  (void)port;
  (printf)("%d", x);
}
void __printf_pass_element_uint32(uint32_t port, uint32_t x)
{
  (void)port;
  (printf)("%u", x);
}

void __printf_pass_element_int64(uint32_t port, int64_t x)
{
  (void)port;
  if (sizeof(long) == sizeof(int64_t))
    {
      (printf)("%ld", (long)x);
    }
  else if (sizeof(long long) == sizeof(int64_t))
    {
      (printf)("%lld", (long long)x);
    }
}

void __printf_pass_element_uint64(uint32_t port, uint64_t x)
{
  (void)port;
  if (sizeof(unsigned long) == sizeof(uint64_t))
    {
      (printf)("%lu", (unsigned long)x);
    }
  else if (sizeof(unsigned long long) == sizeof(uint64_t))
    {
      (printf)("%llu", (unsigned long long)x);
    }
}

void __printf_pass_element_double(uint32_t port, double x)
{
  (void)port;
  (printf)("%f", x);
}

void __printf_pass_element_cstr(uint32_t port, const char *x)
{
  (void)port;
  (printf)("%s", x);
}

void __printf_pass_element_void(uint32_t port, const void *x)
{
  (void)port;
  (printf)("%p", x);
}

void __printf_pass_element_write_int32(uint32_t port, int32_t *x)
{
  (void)port;
  (printf)("%n", x);
}

void __printf_pass_element_write_int64(uint32_t port, int64_t *x)
{
  (void)port;
  if (sizeof(long) == sizeof(int64_t))
    {
      (printf)("%ln", (long*)x);
    }
  else if (sizeof(long long) == sizeof(int64_t))
    {
      (printf)("%lln", (long long*)x);
    }
}
