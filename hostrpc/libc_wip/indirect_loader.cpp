#include "crt.hpp"

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

#include <vector>


#define errExit(msg) \
  do                 \
    {                \
      perror(msg);   \
      exit(1);       \
    }                \
  while (0)


#include "../hsa.hpp"
#include "../launch.hpp"
#include "../raiifile.hpp"

#include "crt.hpp"
#include <string.h>

static __libc_rpc_server server;


struct memfd
{
  operator int() { return filedescriptor; }

  memfd(size_t req) : size(req)
  {
    if (req == 0) return;

    int fd = memfd_create("f", 0);
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

    if ((long long)req != fd_stat.st_size)
      {
        // todo: must it be exact?
        close(fd);
        return;
      }

    // No more changing the size
#if 0
    if (fcntl(fd, F_ADD_SEALS, F_SEAL_GROW | F_SEAL_SHRINK) != 0)
      {
        close(fd);
        return;
      }
#endif
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


struct handles_t
{
  handles_t()
    : server_inbox(slots_bytes),
      server_outbox(slots_bytes),
      shared(slots * sizeof(BufferElement)) {}


  bool valid() {
    return server_inbox != -1 &&
      server_outbox != -1 &&
      shared != -1;
  }
  memfd server_inbox;
  memfd server_outbox;
  memfd shared;
};

int main(int argc, char ** argv)
{
  
  handles_t handles;
  if (!handles.valid()) { return 1; }
  


  void *host_locks = aligned_alloc(64, slots_bytes);  // todo: stop leaking this

  void *server_inbox_ptr_host = handles.server_inbox.map_writable();
  void *server_outbox_ptr_host = handles.server_outbox.map_writable();
  void *shared_buffer_ptr_host = handles.shared.map_writable();

  assert(host_locks);
  assert(server_inbox_ptr_host != MAP_FAILED);
  assert(server_outbox_ptr_host != MAP_FAILED);
  assert(shared_buffer_ptr_host != MAP_FAILED);

  memset(host_locks, 0, slots_bytes);
  memset(server_inbox_ptr_host, 0, slots_bytes);
  memset(server_outbox_ptr_host, 0, slots_bytes);
  memset(shared_buffer_ptr_host, 0, slots_bytes * sizeof(BufferElement));


  server = __libc_rpc_server(
      {},
      hostrpc::careful_cast_to_bitmap<__libc_rpc_server::lock_t>(host_locks,
                                                                 slots_words),
      hostrpc::careful_cast_to_bitmap<__libc_rpc_server::inbox_t>(
          server_inbox_ptr_host, slots_words),
      hostrpc::careful_cast_to_bitmap<__libc_rpc_server::outbox_t>(
          server_outbox_ptr_host, slots_words),
      hostrpc::careful_array_cast<BufferElement>(shared_buffer_ptr_host,
                                                 slots));


  
  char buf[128];
  sprintf(buf, "(%d %d %d)", (int)handles.server_outbox, (int)handles.shared, (int)handles.server_inbox);

  std::vector<char *> arr;
  arr.push_back(buf); // file handle hackery going in first
  arr.push_back(strdup("./libc_wip/amdgcn_loader.exe"));
  for (int i = 1; i < argc; i++)
    {
      arr.push_back(argv[i]);
    }

  arr.push_back(nullptr);

  
  pid_t pid = fork();
  if (pid == -1)
    {
      errExit("can't fork");
    }

  if (pid == 0){
    printf("going to be the client\n");
    int rc = execv(arr[1], arr.data());
    if (rc == -1)
      {
        errExit("couldnt execv");
      }
  }
  
  for(;;)
  {


      bool r =  // server.
          rpc_handle(
              &server,
              [&](hostrpc::port_t, BufferElement *data) {
                fprintf(stderr, "Indirect server got work to do:\n");

                for (unsigned i = 0; i < 64; i++)
                  {
                    auto ith = data->cacheline[i];
                    uint64_t opcode = ith.element[0];
                    switch (opcode)
                      {
                        case no_op:
                        default:
                          continue;
                        case print_to_stderr:
                          enum
                          {
                            w = 7 * 8
                          };
                          char buf[w];
                          memcpy(buf, &ith.element[1], w);
                          buf[w - 1] = '\0';
                          fprintf(stderr, "%s", buf);
                          break;
                      }
                  }
              },
              [&](hostrpc::port_t, BufferElement *data) {
                fprintf(stderr, "Server cleaning up\n");
                for (unsigned i = 0; i < 64; i++)
                  {
                    data->cacheline[i].element[0] = no_op;
                  }
              });

      if (r)
        {
          // did something
          printf("server did something\n");
        }
      else
        {
          // found no work, could sleep here
          printf("no work found\n");
          for (unsigned i = 0; i < 10000; i++) platform::sleep_briefly();
        }

    
  int status;
      pid_t res = waitpid(pid, &status, WNOHANG);
      if (res == -1)
        {
          errExit("waitpid failed");
        }
      else
      if (res == 0)
        {
          // keep going
          continue;
          
        }
      else
        {
          //done
          printf("waitpid says child is done\n");
          break;
        }
  
  }

  return 0;
}
