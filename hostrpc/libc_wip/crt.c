
int main(int, char**);
void __libc_write_stderr(const char* str) {(void)str; /*stub*/}

__attribute__((amdgpu_kernel))
__attribute__((visibility("default")) )            
void __start(void) // int, void*, int*
{
  
  __attribute__((address_space(4))) void * ptr = __builtin_amdgcn_dispatch_ptr();
  enum {kernarg_address_offset = 40};

  char* kernarg_address = (char*)ptr + kernarg_address_offset;
  char*kernarg;
  __builtin_memcpy(&kernarg, kernarg_address, 8);

  int argc;
  __builtin_memcpy(&argc, kernarg, 4);
  kernarg+= 4;

  // padding
  kernarg+= 4;

  char** argv;
  __builtin_memcpy(&argv, kernarg, 8);
  kernarg += 8;

  // todo: probably put this before argc
  // maybe want an array of length number threads? unsure
  char* result;
  __builtin_memcpy(&argv, kernarg, 8);
    kernarg += 8;
  
  int rc = main(argc, argv);
  (void)rc;
}
