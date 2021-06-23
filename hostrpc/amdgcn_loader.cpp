#include "hsa.hpp"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

#include <unistd.h>

#include <elf.h>
#include <libelf.h>

#include "hostcall.hpp"
#include "hostcall_hsa.hpp"
#include "raiifile.hpp"

#include "hostrpc_printf.h"
#include "launch.hpp"
#undef printf

// written for rtl.cpp
namespace
{
Elf64_Shdr *find_only_SHT_HASH(Elf *elf)
{
  size_t N;
  int rc = elf_getshdrnum(elf, &N);
  if (rc != 0)
    {
      return nullptr;
    }

  Elf64_Shdr *result = nullptr;
  for (size_t i = 0; i < N; i++)
    {
      Elf_Scn *scn = elf_getscn(elf, i);
      if (scn)
        {
          Elf64_Shdr *shdr = elf64_getshdr(scn);
          if (shdr)
            {
              if (shdr->sh_type == SHT_HASH)
                {
                  if (result == nullptr)
                    {
                      result = shdr;
                    }
                  else
                    {
                      // multiple SHT_HASH sections not handled
                      return nullptr;
                    }
                }
            }
        }
    }
  return result;
}

const Elf64_Sym *elf_lookup(Elf *elf, char *base, Elf64_Shdr *section_hash,
                            const char *symname)
{
  assert(section_hash);
  size_t section_symtab_index = section_hash->sh_link;
  Elf64_Shdr *section_symtab =
      elf64_getshdr(elf_getscn(elf, section_symtab_index));
  size_t section_strtab_index = section_symtab->sh_link;

  const Elf64_Sym *symtab =
      reinterpret_cast<const Elf64_Sym *>(base + section_symtab->sh_offset);

  const uint32_t *hashtab =
      reinterpret_cast<const uint32_t *>(base + section_hash->sh_offset);

  // Layout:
  // nbucket
  // nchain
  // bucket[nbucket]
  // chain[nchain]
  uint32_t nbucket = hashtab[0];
  const uint32_t *bucket = &hashtab[2];
  const uint32_t *chain = &hashtab[nbucket + 2];

  const size_t max = strlen(symname) + 1;
  const uint32_t hash = elf_hash(symname);
  for (uint32_t i = bucket[hash % nbucket]; i != 0; i = chain[i])
    {
      char *n = elf_strptr(elf, section_strtab_index, symtab[i].st_name);
      if (strncmp(symname, n, max) == 0)
        {
          return &symtab[i];
        }
    }

  return nullptr;
}

struct symbol_info
{
  void *addr = nullptr;
  uint32_t size = UINT32_MAX;
  uint32_t sh_type = SHT_NULL;
};

int get_symbol_info_without_loading(Elf *elf, char *base, const char *symname,
                                    symbol_info *res)
{
  if (elf_kind(elf) != ELF_K_ELF)
    {
      return 1;
    }

  Elf64_Shdr *section_hash = find_only_SHT_HASH(elf);
  if (!section_hash)
    {
      return 1;
    }

  const Elf64_Sym *sym = elf_lookup(elf, base, section_hash, symname);
  if (!sym)
    {
      return 1;
    }

  if (sym->st_size > UINT32_MAX)
    {
      return 1;
    }

  if (sym->st_shndx == SHN_UNDEF)
    {
      return 1;
    }

  Elf_Scn *section = elf_getscn(elf, sym->st_shndx);
  if (!section)
    {
      return 1;
    }

  Elf64_Shdr *header = elf64_getshdr(section);
  if (!header)
    {
      return 1;
    }

  res->addr = sym->st_value + base;
  res->size = static_cast<uint32_t>(sym->st_size);
  res->sh_type = header->sh_type;
  return 0;
}

int get_symbol_info_without_loading(char *base, size_t img_size,
                                    const char *symname, symbol_info *res)
{
  Elf *elf = elf_memory(base, img_size);
  if (elf)
    {
      int rc = get_symbol_info_without_loading(elf, base, symname, res);
      elf_end(elf);
      return rc;
    }
  return 1;
}
}  // namespace

namespace
{
std::vector<size_t> offsets_into_strtab(int argc, char **argv)
{
  std::vector<size_t> res;
  unsigned offset = 0;
  for (int i = 0; i < argc; i++)
    {
      char *arg = argv[i];
      size_t sz = strlen(arg) + 1;
      res.push_back(offset);
      offset += sz;
    }

  // And an end element for the total size
  res.push_back(offset);
  return res;
}

static const char *const kernel_entry = "__device_start.kd";

uint64_t find_entry_address(hsa::executable &ex)
{
  hsa_executable_symbol_t symbol = ex.get_symbol_by_name(kernel_entry);
  if (symbol.handle == hsa::sentinel())
    {
      fprintf(stderr, "HSA failed to find kernel %s\n", kernel_entry);
      exit(1);
    }

  hsa_symbol_kind_t kind = hsa::symbol_get_info_type(symbol);
  if (kind != HSA_SYMBOL_KIND_KERNEL)
    {
      fprintf(stderr, "Symbol %s is not a kernel\n", kernel_entry);
      exit(1);
    }

  return hsa::symbol_get_info_kernel_object(symbol);
}

}  // namespace

uint64_t read_symbol(raiifile *file, const char *name, uint64_t fallback)
{
  symbol_info sym;
  if (0 ==
      get_symbol_info_without_loading(static_cast<char *>(file->mmapped_bytes),
                                      file->mmapped_length, name, &sym))
    {
      if (0)
        fprintf(stderr, "Found symbol %s, %p / %u / %u\n", name, sym.addr,
                sym.size, sym.sh_type);
      if (sym.size == 8)
        {
          uint64_t r;
          memcpy(&r, sym.addr, sizeof(r));
          return r;
        }
      if (sym.size == 4)
        {
          uint32_t r = 0;
          memcpy(&r, sym.addr, sizeof(r));
          return r;
        }
      if (sym.size == 2)
        {
          uint16_t r = 0;
          memcpy(&r, sym.addr, sizeof(r));
          return r;
        }
      if (sym.size == 1)
        {
          uint8_t r = 0;
          memcpy(&r, sym.addr, sizeof(r));
          return r;
        }
    }

  return fallback;
}

static void callbackQueue(hsa_status_t status, hsa_queue_t *source, void *)
{
  if (status != HSA_STATUS_SUCCESS)
    {
      // may only be called with status other than success
      const char *msg = "UNKNOWN ERROR";
      hsa_status_string(status, &msg);
      fprintf(stderr, "Queue at %p inactivated due to async error:\n\t%s\n",
              source, msg);

      abort();
    }
}

static int main_with_hsa(int argc, char **argv)
{
  const bool verbose = false;
  if (argc < 2)
    {
      fprintf(stderr, "Require at least one argument\n");
      return 1;
    }

  hsa_agent_t kernel_agent = hsa::find_a_gpu_or_exit();

  raiifile file(argv[1]);
  if (!file.mmapped_bytes)
    {
      fprintf(stderr, "Failed to open file %s\n", argv[1]);
      return 1;
    }

  hsa::executable ex(kernel_agent, file.mmapped_bytes, file.mmapped_length);
  if (!ex.valid())
    {
      fprintf(stderr, "HSA failed to load contents of %s\n", argv[1]);
      return 1;
    }

  // probably need to populate some of the implicit args for intrinsics to work
  hsa_region_t kernarg_region = hsa::region_kernarg(kernel_agent);
  hsa_region_t fine_grained_region = hsa::region_fine_grained(kernel_agent);
  hsa_region_t coarse_grained_region = hsa::region_coarse_grained(kernel_agent);
  {
    uint64_t fail = reinterpret_cast<uint64_t>(nullptr);
    if (kernarg_region.handle == fail || fine_grained_region.handle == fail ||
        coarse_grained_region.handle == fail)
      {
        fprintf(stderr, "Failed to find allocation region on kernel agent\n");
        exit(1);
      }
  }

  // Drop the loader name from the forwarded argc/argv
  int app_argc = argc - 1;
  char **app_argv = &argv[1];

  // arguments must be in kernarg memory, which is constant
  // opencl doesn't accept char** as a type and returns void
  // combined, choosing to pass arguments as:
  // int argc
  // int padding
  // void * to_argv
  // int * to_result

  // there's also a number of implicit arguments, where those passed by atmi
  // don't match those I see from an opencl kernel. The first 24 bytes are
  // consistently used for offset_x, offset_y, offset_z. Zero those.
  // opencl and atmi both think the implicit structure is 80 long.

  size_t implicit_offset_size = 24;
  size_t extra_implicit_size = 80 - implicit_offset_size;

  // implicit offset needs to be 8 byte aligned, which it w/ 24 bytes explicit
  size_t bytes_for_kernarg = 24 + implicit_offset_size + extra_implicit_size;

  auto offsets = offsets_into_strtab(app_argc, app_argv);
  size_t bytes_for_argv = 8 * app_argc;
  size_t bytes_for_strtab = (offsets.back() + 3) & ~size_t{3};
  size_t number_return_values = 64;  // max number waves
  size_t bytes_for_return = sizeof(int) * number_return_values;

  // Always allocates > 0 because of the return slot
  auto mutable_alloc =
      hsa::allocate(fine_grained_region,
                    bytes_for_argv + bytes_for_strtab + bytes_for_return);

  const char *strtab_start =
      static_cast<char *>(mutable_alloc.get()) + bytes_for_argv;
  const char *result_location = static_cast<char *>(mutable_alloc.get()) +
                                bytes_for_argv + bytes_for_strtab;

  auto kernarg_alloc = hsa::allocate(kernarg_region, bytes_for_kernarg);
  if (!mutable_alloc || !kernarg_alloc)
    {
      fprintf(stderr, "Failed to allocate %zu bytes for kernel arguments\n",
              bytes_for_argv + bytes_for_strtab + bytes_for_kernarg);
      exit(1);
    }

  // Populate argv array, immediately followed by string table
  char *argv_array = static_cast<char *>(mutable_alloc.get());
  for (int i = 0; i < app_argc; i++)
    {
      const char *loc = strtab_start + offsets[i];
      memcpy(argv_array, &loc, 8);
      argv_array += 8;
    }
  for (int i = 0; i < app_argc; i++)
    {
      char *arg = app_argv[i];
      size_t sz = strlen(arg) + 1;
      memcpy(argv_array, arg, sz);
      argv_array += sz;
    }

  for (unsigned i = 0; i < bytes_for_strtab - offsets.back(); i++)
    {
      // alignment padding for the return value
      char z = 0;
      memcpy(argv_array, &z, 1);
      argv_array += 1;
    }

  // init the return value. not strictly necessary
  {
    assert(argv_array == result_location);
    for (size_t i = 0; i < number_return_values; i++)
      {
        unsigned z = 0xdead;
        memcpy(argv_array, &z, 4);
        argv_array += 4;
      }
  }

  // Set up kernel arguments
  {
    char *kernarg = (char *)kernarg_alloc.get();

    // argc
    memcpy(kernarg, &app_argc, 4);
    kernarg += 4;

    // padding
    memset(kernarg, 0, 4);
    kernarg += 4;

    // argv
    void *raw_mutable_alloc = mutable_alloc.get();
    memcpy(kernarg, &raw_mutable_alloc, 8);
    kernarg += 8;

    // result
    memcpy(kernarg, &result_location, 8);
    kernarg += 8;

    // x, y, z implicit offsets
    memset(kernarg, 0, implicit_offset_size);
    kernarg += implicit_offset_size;

    // remaining implicit gets scream. I don't think the kernels are
    // using it, but if they do, -1 is relatively obvious in the dump
    memset(kernarg, 0xff, extra_implicit_size);
    kernarg += extra_implicit_size;
  }

  if (verbose)
    {
      printf("Spawn queue\n");
    }
  
  hsa_queue_t *queue;
  {
    hsa_status_t rc = hsa_queue_create(
        kernel_agent /* make the queue on this agent */,
        hsa::agent_get_info_queue_max_size(kernel_agent),
        HSA_QUEUE_TYPE_MULTI /* baseline */,
        callbackQueue /* called on every async event? */,
        NULL /* data passed to previous */,
        // If sizes exceed these values, things are supposed to work slowly
        UINT32_MAX /* private_segment_size, 32_MAX is unknown */,
        UINT32_MAX /* group segment size, as above */, &queue);
    if (rc != HSA_STATUS_SUCCESS)
      {
        fprintf(stderr, "Failed to create queue\n");
        exit(1);
      }
  }

  if (hostrpc_print_enable_on_hsa_agent(ex, kernel_agent) != 0)
    {
      fprintf(stderr, "Failed to create host printf thread\n");
      exit(1);
    }

  hostcall hc(kernel_agent);
  if (!hc.valid())
    {
      fprintf(stderr, "Failed to create hostcall\n");
      exit(1);
    }

  if (hc.enable_executable(ex) != 0)
    {
      fprintf(stderr, "Warning: Failed to enable hostcall on executable\n");
    }
  else
    {
      if (hc.enable_queue(queue) != 0)
        {
          fprintf(stderr, "Failed to enable queue\n");
          exit(1);
        }
      for (unsigned r = 0; r < 2; r++)
        {
          if (hc.spawn_worker(queue) != 0)
            {
              fprintf(stderr, "Failed to spawn worker\n");
              exit(1);
            }
        }
    }

  // Claim a packet
  uint64_t packet_id = hsa::acquire_available_packet_id(queue);

  const uint32_t mask = queue->size - 1;  // %
  hsa_kernel_dispatch_packet_t *packet =
      (hsa_kernel_dispatch_packet_t *)queue->base_address + (packet_id & mask);

  hsa::initialize_packet_defaults(packet);

  packet->workgroup_size_x =
      read_symbol(&file, "main_workgroup_size_x", packet->workgroup_size_x);
  packet->grid_size_x =
      read_symbol(&file, "main_grid_size_x", packet->workgroup_size_x);

  uint64_t kernel_address = find_entry_address(ex);
  packet->kernel_object = kernel_address;

  {
    void *raw_kernarg_alloc = kernarg_alloc.get();
    memcpy(&packet->kernarg_address, &raw_kernarg_alloc, 8);
  }

  auto rc = hsa_signal_create(1, 0, NULL, &packet->completion_signal);
  if (rc != HSA_STATUS_SUCCESS)
    {
      printf("Can't make signal\n");
      exit(1);
    }

  auto m = ex.get_kernel_info();

  // todo: use hsa::launch?
  auto it = m.find(std::string(kernel_entry));
  if (it != m.end())
    {
      packet->private_segment_size = it->second.private_segment_fixed_size;
      packet->group_segment_size = it->second.group_segment_fixed_size;
    }
  else
    {
      printf("Error: get_kernel_info failed for kernel %s\n", kernel_entry);
      exit(1);
    }

  hsa::packet_store_release((uint32_t *)packet,
                            hsa::header(HSA_PACKET_TYPE_KERNEL_DISPATCH),
                            hsa::kernel_dispatch_setup());

  hsa_signal_store_release(queue->doorbell_signal, packet_id);

  if (verbose)
    {
      printf("Launch kernel\n");
    }
  do
    {
      // TODO: Polling is better than waiting here as it lets the initial
      // dispatch spawn a graph
    }
  while (hsa_signal_wait_acquire(packet->completion_signal,
                                 HSA_SIGNAL_CONDITION_EQ, 0, 5000 /*000000*/,
                                 HSA_WAIT_STATE_ACTIVE) != 0);

  if (verbose)
    {
      printf("Kernel signalled\n");
    }
  int result[number_return_values];
  memcpy(&result, result_location, sizeof(int) * number_return_values);

  hsa_signal_destroy(packet->completion_signal);
  hsa_queue_destroy(queue);

  bool results_match = true;
  {
    int res = result[0];
    for (size_t i = 1; i < number_return_values; i++)
      {
        if (result[i] != res)
          {
            results_match = false;
          }
      }
  }

  if (!results_match)
    {
      fprintf(stderr, "Warning: Non-uniform return values\n");

      printf("Queue in x64: %lx\n", (uint64_t)queue);
      uint64_t v = ((uint64_t)result[0] & 0x00000000FFFFFFFFull) |
                   (((uint64_t)result[1] & 0x00000000FFFFFFFFull) << 32u);
      printf("Queue: %lx\n", v);
      for (size_t i = 0; i < number_return_values; i++)
        {
          fprintf(stderr, "rc[%zu] = %x\n", i, result[i]);
        }
    }

  if (verbose)
    {
      printf("Result[0] %d\n", result[0]);
    }
  return result[0];
}

extern "C" int amdgcn_loader_main(int argc, char **argv)
{
  // valgrind thinks this is leaking slightly
  hsa::init hsa_state;  // Need to destroy this last
  return main_with_hsa(argc, argv);
}

__attribute__((weak)) extern "C" int main(int argc, char **argv)
{
  return amdgcn_loader_main(argc, argv);
}
