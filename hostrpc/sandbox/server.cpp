#define _GNU_SOURCE 1
#include <fcntl.h>
#include <sys/mman.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define errExit(msg) \
  do                 \
    {                \
      perror(msg);   \
      exit(1);       \
    }                \
  while (0)

struct memfd
{
  operator int() { return filedescriptor; }

  memfd(size_t req) : size(req)
  {
    if (req == 0) return;

    int fd = memfd_create("f", MFD_ALLOW_SEALING);
    if (fd == -1) return;

    // Set size to requested
    if (ftruncate(fd, req) == -1)
      {
        close(fd);
        return;
      }

    struct stat fd_stat;
    if (fstat(fd, &fd_stat) != 0)
      {
        close(fd);
        return;
      }

    if (req != fd_stat.st_size)
      {
        // todo: must it be exact?
        close(fd);
        return;
      }

    // No more changing the size
    if (fcntl(fd, F_ADD_SEALS, F_SEAL_GROW | F_SEAL_SHRINK) != 0)
      {
        close(fd);
        return;
      }

    filedescriptor = fd;
  }

  void *map_writable() { return map(PROT_READ | PROT_WRITE); }

  void *map_readable() { return map(PROT_READ); }

  int no_more_map_writable()
  {
    if (filedescriptor != -1)
      return fcntl(filedescriptor, F_ADD_SEALS, F_SEAL_FUTURE_WRITE);
    else
      return 0;  // may as well claim success, map accessors won't work anyway
  }

  ~memfd()
  {
    if (filedescriptor != -1) close(filedescriptor);
  }

 private:
  void *map(size_t prot)
  {
    if (filedescriptor == -1)
      {
        return MAP_FAILED;
      }
    else
      {
        return mmap(NULL, size, prot, MAP_SHARED, filedescriptor, 0);
      }
  }

  size_t size;
  int filedescriptor = -1;
};

int main()
{
  printf("I am the server\n");

  auto fd_h = memfd(256);
  int fd = fd_h;
  if (fd == -1) errExit("memfd");

  printf("server fd %d\n", fd);

  void *mapped = fd_h.map_writable();
  if (mapped == MAP_FAILED) errExit("mmap");

  // Can create no more writable maps into it, but the above one is still fine
  if (fd_h.no_more_map_writable() != 0) errExit("add write seal");

  sprintf((char *)mapped, "%s", "Have some bytes\n");

  char buf[64];
  sprintf(buf, "%d", fd);

  const char *arr[] = {
      "./client", "foo", buf, "bar", NULL,
  };
  enum
  {
    N = sizeof(arr) / sizeof(arr[0])
  };

  char *cparr[N];
  for (unsigned i = 0; i < N; i++)
    {
      if (arr[i] == NULL)
        cparr[i] = NULL;
      else
        cparr[i] = strdup(arr[i]);
    }

  pid_t pid = fork();
  if (pid == -1)
    {
      errExit("can't fork");
    }

  if (pid > 0)
    {
      printf("still server, child pid is %ld\n", (long)pid);

      int status;
      waitpid(pid, &status, 0);
      printf("waitpid returned\n");
    }
  else
    {
      printf("going to be the client\n");
      int rc = execv(cparr[0], cparr);
      if (rc == -1)
        {
          errExit("couldnt execv");
        }
    }

  return 0;
}
