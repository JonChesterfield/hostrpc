
static int strcmp(const char *l, const char *r)
{
  // from musl
  for (; *l == *r && *l; l++, r++)
    ;
  return *(unsigned char *)l - *(unsigned char *)r;
}

// Example.
extern "C" __attribute__((visibility("default"))) int main(int argc,
                                                           char **argv)
{
  for (int i = 0; i < argc; i++)
    {
      if (strcmp(argv[i], "arguments") == 0)
        {
          return i;
        }
    }

  return argc;
}
