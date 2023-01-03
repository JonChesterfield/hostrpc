#ifndef LIBC_CRT_H_INCLUDED
#define LIBC_CRT_H_INCLUDED

#include "../detail/client_impl.hpp"
#include "../detail/server_impl.hpp"
#include "../memory.hpp"


enum opcodes {
              no_op = 0,
              print_to_stderr = 1,
};

struct cacheline_t
{
  alignas(64) uint64_t element[8];
};
static_assert(sizeof(cacheline_t) == 64, "");

struct BufferElement
{
  enum
  {
    width = 64
  };
  alignas(4096) cacheline_t cacheline[width];
};
static_assert(sizeof(BufferElement) == 4096, "");

inline void *operator new(size_t, BufferElement *p) { return p; }

using WordType = uint32_t;

// todo: width of this structure should be a function of architecture
enum
{
  slots = 128,  // number of slots, usually in bits
  slots_bytes = slots / 8,
  slots_words = slots_bytes / sizeof(WordType),
};

enum {rpc_buffer_kernarg_size = 4 * 8}; // bytes used to pass pointers

using __libc_rpc_client =
    hostrpc::client<BufferElement, uint32_t, hostrpc::size_compiletime<slots>>;
using __libc_rpc_server =
    hostrpc::server<BufferElement, uint32_t, hostrpc::size_compiletime<slots>>;


#endif
