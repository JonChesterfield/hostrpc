#!/bin/bash

/home/amd/rocm/aomp/bin/clang++ -Wall -Wextra -O2  --target=amdgcn-amd-amdhsa -march=gfx906 -mcpu=gfx906 -mllvm -amdgpu-fixed-function-abi -nogpulib -c client.gcn.s -o client.gcn.obj

~/rocm/aomp/bin/llvm-objcopy -O binary --only-section=.text client.gcn.obj  client.gcn.obj.text

~/rocm/aomp/bin/llvm-objdump -D --mcpu=gfx906  --section=.text client.gcn.obj &> gcn.text.s

objcopy --input binary  --output elf64-x86-64 --binary-architecture i386:x86-64 client.gcn.obj.text x64.text.o

objdump -D x64.text.o &> x64.text.s


exit

# rough sketch of looking for a x64/gfx9 polyglot

	s_waitcnt vmcnt(1) // jno 13 bytes past the end of this
        // three instructions available for gcn to branch somewhere
        s_setpc_b64 s[30:31]
        s_setpc_b64 s[30:31]
        s_setpc_b64 s[30:31]
        // 3 bytes of nop
        .byte 0x90
        // x64 lands here

        s_mov_b64 s[14:15], s[10:11]


        Payload wanted: Six words

      	s_getpc_b64 s[4:5]
	s_add_u32 s4, s4, target@rel32@lo+4
	s_addc_u32 s5, s5, target@rel32@hi+4
	s_setpc_b64 s[4:5]

        // Would like to jump over 26 bytes, 0x1a. Doesn't seem to be encodable


9090C084 | s_ashr_i64 s[16:17], 4, 64     | test %al, %al ; nop ; nop
BF8C0E71 | s_waitcnt vmcnt(1) lgkmcnt(14) | jno rip+14 ; two bytes data
BE901C00 | s_getpc_b64 s[16:17]           | Unused
8010FF10 | s_add_u32 s16, s16, 0          | Unused
00000000 | gcn reloc part 0               | Unused
BF8C0E71 | s_waitcnt vmcnt(1) lgkmcnt(14) | jno rip+14
8211FF11 | s_addc_u32 s17, s17, 0         | Unused
00000000 | gcn reloc part 1               | Unused
BE801D10 | s_setpc_b64 s[16:17]           | Unused
000000E9 | Unused data                    | jmp rip, part of x64 reloc
00000000 |                                | rest of x64 reloc
