#include <assert.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <x86intrin.h>

typedef __attribute__((aligned(64))) _Atomic(uint8_t) aligned_byte;

typedef __attribute__((aligned(64)))
__attribute__((may_alias)) _Atomic(uint64_t) aligned_word;

typedef unsigned char uchar8 __attribute__((ext_vector_type(8)));

uint64_t load_word(aligned_byte *d)
{
  // read 64 bytes, pack in
  (void)d;

  uint64_t w = *(aligned_word *)d;

  uint64_t r = 1;  //_pext_u64(w, UINT64_C(0x101010101010101));

  return r;
}

uint64_t expand_byte(uint8_t x)
{
  uint8_t bits[8];
  for (unsigned i = 0; i < 8; i++)
    {
      bits[i] = (x >> i) & 0x1;
    }

  uint64_t r;
  __builtin_memcpy(&r, &bits, 8);
  return r;
}

extern "C" uint8_t pack_word_reference(uint64_t x)
{
  unsigned char *p = (unsigned char *)&x;
  uint8_t res = 0;
  for (unsigned i = 0; i < 8; i++)
    {
      res |= (p[i] & UINT8_C(0x1)) << i;
    }
  return res;
}

extern "C" uint8_t pack_word_multiply(uint64_t x)
{
  x *= UINT64_C(0x102040810204080);
  return x >> 56u;
}

extern "C" uint8_t pack_word(uint64_t x)
{
#if defined(__BMI2__) && __has_builtin(__builtin_ia32_pext_di)
  return __builtin_ia32_pext_di(x, UINT64_C(0x0101010101010101));
#else
  return pack_word_multiply(x);
#endif
}

extern "C" uint64_t pack_words(aligned_byte *data)
{
  aligned_word *words = (aligned_word *)data;

  uint64_t st0 = (uint64_t)pack_word(words[0]) << UINT64_C(0);
  uint64_t st1 = (uint64_t)pack_word(words[1]) << UINT64_C(8);
  uint64_t st01 = st0 | st1;
  
  uint64_t st2 = (uint64_t)pack_word(words[2]) << UINT64_C(16);
  uint64_t st3 = (uint64_t)pack_word(words[3]) << UINT64_C(24);
  uint64_t st23 = st2 | st3;
  
  uint64_t st4 = (uint64_t)pack_word(words[4]) << UINT64_C(32);
  uint64_t st5 = (uint64_t)pack_word(words[5]) << UINT64_C(40);
  uint64_t st45 = st4 | st5;
  
  uint64_t st6 = (uint64_t)pack_word(words[6]) << UINT64_C(48);
  uint64_t st7 = (uint64_t)pack_word(words[7]) << UINT64_C(56);
  uint64_t st67 = st6 | st7;

  uint64_t st0123 = st01 | st23;

  uint64_t st4567 = st45 | st67;

  return st0123 | st4567;
  
  uint8_t st[8];
  for (unsigned i = 0; i < 8; i++)
    {
      st[i] = pack_word(words[i]);
    }

  uint64_t res;
  __builtin_memcpy(&res, &st, 8);

  return res;
}

#define FMT_BUF_SIZE (CHAR_BIT * sizeof(uintmax_t) + 1)
char *binary_fmt(uintmax_t x, char buf[FMT_BUF_SIZE])
{
  char *s = buf + FMT_BUF_SIZE;
  *--s = 0;
  if (!x) *--s = '0';
  for (; x; x /= 2) *--s = '0' + x % 2;
  return s;
}

void round_trip()
{
  for (unsigned i = 0; i < 256; i++)
    {
      uint64_t word = expand_byte(i);
      uint32_t byte = pack_word(word);

      if (byte != i)
        {
          printf("%u => %lu => %u\n", i, word, byte);
          // exit(1);
        }
    }
}

int main()
{
#if defined(__BMI2__) && __has_builtin(__builtin_ia32_pext_di)
  printf("have builtin\n");
#else
  printf("no pext\n");
#endif

  round_trip();

  return 0;

  char tmp[FMT_BUF_SIZE];
  for (unsigned i = 0; i < 256; i++)
    {
      printf("0b%s\n", binary_fmt(expand_byte(i), tmp));
    }
}
