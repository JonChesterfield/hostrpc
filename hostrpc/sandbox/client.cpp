#define _GNU_SOURCE 1
#include <sys/mman.h>


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>





       #define errExit(msg)    do { perror(msg); exit(1); \
                               } while (0)


int main(int argc, char**argv)
{
  printf("I am the client. argc = %d\n", argc);

  for (int i = 0; i < argc; i++)
    {
      printf("argv[%d] = %s\n", i, argv[i]);
    }

  int fd;
  sscanf(argv[2], "%d", &fd); // uh...

  printf("fd = %d\n", fd);

  void *mapped = mmap(NULL, 4096, PROT_READ, MAP_SHARED, fd, 0);
  if (mapped == MAP_FAILED) errExit("mmap");

  printf("%s", (char*)mapped);
  
  return 1;
}
