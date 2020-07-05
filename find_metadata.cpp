#include "find_metadata.hpp"

// system includes libelf.h as well. Have to link against -lelf
#include <gelf.h>

#include <cstdint>
#include <cstring> // memcmp

typedef unsigned char *address;
typedef struct {
  uint32_t n_namesz; /* Length of note's name. */
  uint32_t n_descsz; /* Length of note's value. */
  uint32_t n_type;   /* Type of note. */
  // then name
  // then padding, optional
  // then desc, at 4 byte alignment (not 8, despite being elf64)
} Elf_Note;

namespace core {
template <typename T>
inline T alignDown(T value, size_t alignment) {
  return (T)(value & ~(alignment - 1));
}


template <typename T>
inline T alignUp(T value, size_t alignment) {
  return alignDown((T)(value + alignment - 1), alignment);
}

}
std::pair<unsigned char *, unsigned char *>
find_metadata(void *binary, size_t binSize) {
  std::pair<unsigned char *, unsigned char *> failure = {nullptr, nullptr};

  Elf *e = elf_memory(static_cast<char *>(binary), binSize);
  if (elf_kind(e) != ELF_K_ELF) {
    return failure;
  }

  size_t numpHdrs;
  if (elf_getphdrnum(e, &numpHdrs) != 0) {
    return failure;
  }

  for (size_t i = 0; i < numpHdrs; ++i) {
    GElf_Phdr pHdr;
    if (gelf_getphdr(e, i, &pHdr) != &pHdr) {
      continue;
    }
    // Look for the runtime metadata note
    if (pHdr.p_type == PT_NOTE && pHdr.p_align >= sizeof(int)) {
      // Iterate over the notes in this segment
      address ptr = (address)binary + pHdr.p_offset;
      address segmentEnd = ptr + pHdr.p_filesz;

      while (ptr < segmentEnd) {
        Elf_Note *note = reinterpret_cast<Elf_Note *>(ptr);
        address name = (address)&note[1];

        if (note->n_type == 7 || note->n_type == 8) {
          return failure;
        } else if (note->n_type == 10 /* NT_AMD_AMDGPU_HSA_METADATA */ &&
                   note->n_namesz == sizeof "AMD" &&
                   !memcmp(name, "AMD", note->n_namesz)) {
          // code object v2 uses yaml metadata, no longer supported
          return failure;
        } else if (note->n_type == 32 /* NT_AMDGPU_METADATA */ &&
                   note->n_namesz == sizeof "AMDGPU" &&
                   !memcmp(name, "AMDGPU", note->n_namesz)) {

          // n_descsz = 485
          // value is padded to 4 byte alignment, may want to move end up to
          // match
          size_t offset = sizeof(uint32_t) * 3 /* fields */
                          + sizeof("AMDGPU")   /* name */
                          + 1 /* padding to 4 byte alignment */;

          // Including the trailing padding means both pointers are 4 bytes
          // aligned, which may be useful later.
          unsigned char *metadata_start = (unsigned char *)ptr + offset;
          unsigned char *metadata_end =
              metadata_start + core::alignUp(note->n_descsz, 4);
          return {metadata_start, metadata_end};
        }
        ptr += sizeof(*note) + core::alignUp(note->n_namesz, sizeof(int)) +
               core::alignUp(note->n_descsz, sizeof(int));
      }
    }
  }

  return failure;
}

