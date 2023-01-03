void __libc_write_stderr(const char*);
int main(int argc, char **argv)
{
  (void)argc;
  (void)argv;
  __libc_write_stderr("testing, testing\n");
  return 0;
}
