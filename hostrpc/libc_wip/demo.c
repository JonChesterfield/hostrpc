
static unsigned get_lane_id(void)
{
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}

void __libc_write_stderr(const char*);
int main(int argc, char **argv)
{
  (void)argc;
  (void)argv;
  __libc_write_stderr("Testing, testing from the gpu\n");
  return 0;
}
