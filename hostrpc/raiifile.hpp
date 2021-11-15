#ifndef RAIIFILE_HPP_INCLUDED
#define RAIIFILE_HPP_INCLUDED

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>

struct raiifile
{
  void *mmapped_bytes = nullptr;
  size_t mmapped_length = 0;

  raiifile(const char *filename)
  {
    errno = 0;
    FILE *handle = fopen(filename, "rb");
    if (!handle)
      {
        fprintf(stderr, "Failed to open %s, errno %d %s\n", filename, errno,
                strerror(errno));
        return;
      }

    int n = fileno(handle);
    struct stat buf;
    int rc = fstat(n, &buf);
    if (rc == 0)
      {
        size_t l = buf.st_size;
        void *m = mmap(NULL, l, PROT_READ, MAP_PRIVATE, n, 0);
        if (m != MAP_FAILED)
          {
            mmapped_bytes = m;
            mmapped_length = l;
          }
      }

    fclose(handle);
  }

  ~raiifile()
  {
    if (mmapped_bytes)
      {
        munmap(mmapped_bytes, mmapped_length);
      }
  }
};

#endif
