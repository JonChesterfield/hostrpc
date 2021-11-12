#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#ifndef __AMDGCN__
#include <stdio.h>
#endif
const char specs[] = {'c', 's', 'd', 'i', 'o', 'x', 'X', 'u', 'f',
                      'F', 'e', 'E', 'a', 'A', 'g', 'G', 'n', 'p'};

bool __printf_conversion_specifier_p_reference(char c)
{
  // excluding % which is handled elsewhere

  size_t N = sizeof(specs);

  bool hit = false;
  for (size_t i = 0; i < N; i++)
    {
      hit = hit | (c == specs[i]);
    }
  return hit;
}

static bool haszero(uint32_t v)
{
  return ((v - UINT32_C(0x01010101)) & ~v & UINT32_C(0x80808080));
}

static uint32_t haystack(char a, char b, char c, char d)
{
  uint32_t res = (a << 0u) | (b << 8u) | (c << 16u) | (d << 24u);
  return res;
}

bool __printf_conversion_specifier_p_faster(char c)
{
  unsigned char uc;
  __builtin_memcpy(&uc, &c, 1);

  // 32 bit codegen is noticably better for amdgcn and ~ same as 64 bit for x64
  // Six characters to check, so checks 'h' repeatedly to avoid a zero.
  uint32_t broadcast = UINT32_C(0x01010101) * uc;
  uint32_t A = haystack('c', 's', 'd', 'i');
  uint32_t B = haystack('o', 'x', 'X', 'u');
  uint32_t C = haystack('f', 'F', 'e', 'E');
  uint32_t D = haystack('a', 'A', 'g', 'G');
  uint32_t E = haystack('n', 'p', 'n', 'p');

  // Works, but can probably be optimised further
  return haszero(broadcast ^ A) | haszero(broadcast ^ B) |
         haszero(broadcast ^ C) | haszero(broadcast ^ D) |
         haszero(broadcast ^ E);
}

int main()
{
  int differ = 0;

  size_t N = sizeof(specs);

  for (size_t i = 0; i < N; i++)
    {
      #ifndef __AMDGCN__
      printf("spec[%zu] %c = %d 0x%x\n", i, specs[i], specs[i],
             (unsigned)specs[i]);
      #endif
    }

  for (unsigned i = 0; i < 256; i++)
    {
      unsigned char u = (unsigned char)i;
      char c;
      __builtin_memcpy(&c, &u, 1);

      bool current = __printf_conversion_specifier_p_reference(c);
      bool proposed = __printf_conversion_specifier_p_faster(c);
      if (current != proposed)
        {
          differ++;
#ifndef __AMDGCN__
          printf("Differ at %u (%c), %u != %u\n", i, c, (unsigned)current,
                 (unsigned)proposed);
#endif
        }
      else
        {
          // printf("Correct at %u, both %u\n", i, (unsigned)current);
        }
    }

  return differ;
}
