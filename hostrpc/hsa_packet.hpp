#ifndef HSA_PACKET_H_INCLUDED
#define HSA_PACKET_H_INCLUDED

// code that is valid on host and gpu. Data types, common code.
#include <stdint.h>

// todo: saner printf handling
#include "detail/platform.hpp"

#if HOSTRPC_HOST
#include <stdio.h>
#else
#undef printf
#include "hostrpc_printf.h"
#define printf(...) __hostrpc_printf(__VA_ARGS__)
#endif

namespace hsa_packet
{
// magic numbers are in hsa.h, undecided how to handle
enum
{
  default_header = (1 << 16u) | (5122u)
};

struct hsa_kernel_dispatch_packet
{
  uint16_t header;
  uint16_t setup;
  uint16_t workgroup_size_x;
  uint16_t workgroup_size_y;
  uint16_t workgroup_size_z;
  uint16_t reserved0;
  uint32_t grid_size_x;
  uint32_t grid_size_y;
  uint32_t grid_size_z;
  uint32_t private_segment_size;
  uint32_t group_segment_size;
  uint64_t kernel_object;
  uint64_t kernarg_address;
  uint64_t reserved2;
  uint64_t completion_signal;
};
static_assert(sizeof(hsa_kernel_dispatch_packet) == 64, "");

struct kernel_descriptor
{
  uint32_t group_segment_fixed_size;
  uint32_t private_segment_fixed_size;
  uint32_t kernarg_size;
  uint8_t reserved0[4];
  int64_t kernel_code_entry_byte_offset;
  uint8_t reserved1[24];
  uint32_t compute_pgm_rsrc1;
  uint32_t compute_pgm_rsrc2;
  uint16_t kernel_code_properties;
  uint8_t reserved2[6];
};
static_assert(sizeof(kernel_descriptor) == 64, "");

inline void initialize_packet_defaults(unsigned char* out)
{
  hsa_kernel_dispatch_packet packet;
  // Reserved fields, private and group memory, and completion signal are all
  // set to 0.
  __builtin_memset(((uint8_t*)&packet) + 4, 0,
                   sizeof(hsa_kernel_dispatch_packet) - 4);
  // These values should probably be read from the kernel
  // Currently they're copied from documentation
  // Launching a single wavefront makes for easier debugging
  packet.workgroup_size_x = 64;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 64;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;
  __builtin_memcpy(out, &packet, sizeof(packet));
}

inline void write_from_kd_into_hsa(const unsigned char* kd,
                                   unsigned char* packet)
{
  __builtin_memcpy(packet + offsetof(hsa_kernel_dispatch_packet, kernel_object),
                   &kd, 8);

  __builtin_memcpy(
      packet + offsetof(hsa_kernel_dispatch_packet, group_segment_size),
      kd + offsetof(kernel_descriptor, group_segment_fixed_size), 4);

  __builtin_memcpy(
      packet + offsetof(hsa_kernel_dispatch_packet, private_segment_size),
      kd + offsetof(kernel_descriptor, private_segment_fixed_size), 4);
}

constexpr inline uint32_t packet_header(uint16_t header, uint16_t rest)
{
  return (uint32_t)header | ((uint32_t)rest << 16u);
}

inline void packet_store_release(uint32_t* packet, uint16_t header,
                                 uint16_t rest)
{
  __atomic_store_n(packet, packet_header(header, rest), __ATOMIC_RELEASE);
}

inline void dump_kernel(const unsigned char* kernel)
{
  hsa_kernel_dispatch_packet inst;
  __builtin_memcpy(&inst, kernel, 64);

  printf("  header:               %lu\n", (unsigned long)inst.header);
  printf("  setup:                %lu\n", (unsigned long)inst.setup);
  printf("  workgroup_size_x:     %lu\n", (unsigned long)inst.workgroup_size_x);
  printf("  workgroup_size_y:     %lu\n", (unsigned long)inst.workgroup_size_y);
  printf("  workgroup_size_z:     %lu\n", (unsigned long)inst.workgroup_size_z);
  printf("  reserved0:            %lu\n", (unsigned long)inst.reserved0);
  printf("  grid_size_x:          %lu\n", (unsigned long)inst.grid_size_x);
  printf("  grid_size_y:          %lu\n", (unsigned long)inst.grid_size_y);
  printf("  grid_size_z:          %lu\n", (unsigned long)inst.grid_size_z);
  printf("  private_segment_size: %lu\n",
         (unsigned long)inst.private_segment_size);
  printf("  group_segment_size:   %lu\n",
         (unsigned long)inst.group_segment_size);
  printf("  kernel_object:        0x%lx\n", (unsigned long)inst.kernel_object);
  printf("  kernarg_address:      0x%lx\n",
         (unsigned long)inst.kernarg_address);
  printf("  reserved2:            0x%lx\n", (unsigned long)inst.reserved2);
  printf("  completion_signal:    0x%lx\n",
         (unsigned long)inst.completion_signal);
}

inline void dump_descriptor(const unsigned char* kd)
{
  kernel_descriptor inst;
  __builtin_memcpy(&inst, kd, sizeof(inst));

  printf("  group_segment_fixed_size:  %lu\n",
         (unsigned long)inst.group_segment_fixed_size);
  printf("  private_segment_fixed_size:  %lu\n",
         (unsigned long)inst.private_segment_fixed_size);
  printf("  kernarg_size:  %lu\n", (unsigned long)inst.kernarg_size);
  printf("  kernel_code_entry_byte_offset:  %lu\n",
         (unsigned long)inst.kernel_code_entry_byte_offset);
  printf("  derived kernel entry: %lu\n",
         (unsigned long)(kd + inst.kernel_code_entry_byte_offset));
  // printf("  compute_pgm_rsrc1:  %lu\n", (unsigned
  // long)inst.compute_pgm_rsrc1); printf("  compute_pgm_rsrc2:  %lu\n",
  // (unsigned long)inst.compute_pgm_rsrc2); printf("  kernel_code_properties:
  // %lu\n", (unsigned long)inst.kernel_code_properties);
}

}  // namespace hsa_packet

#endif
