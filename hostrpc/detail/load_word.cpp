#include <assert.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <x86intrin.h>

typedef __attribute__((aligned(64))) HOSTRPC_ATOMIC(uint8_t) aligned_byte;

typedef __attribute__((aligned(64))) __attribute__((may_alias))
HOSTRPC_ATOMIC(uint64_t) aligned_word;

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
      printf("i[%u]: %u |= (%u & UINT8_C(0x1)) << %u\n", i, res, p[i], i);
      res |= (p[i] & UINT8_C(0x1)) << i;
    }
  return res;
}

extern "C" uint8_t pack_word_multiply(uint64_t x)
{
  // x = 0000000h 0000000g 0000000f 0000000e 0000000d 0000000c 0000000b 0000000a
  uint64_t m = x * UINT64_C(0x102040810204080);
  // m = hgfedcba -------- -------- -------- -------- -------- -------- --------
  uint64_t r = m >> 56u;
  // r = 00000000 00000000 00000000 00000000 00000000 00000000 00000000 hgfedcba
  return r;
}

extern "C" uint8_t pack_word(uint64_t x)
{
  // default to this for now
  return pack_word_multiply(x);

#if defined(__BMI2__) && __has_builtin(__builtin_ia32_pext_di)
  return __builtin_ia32_pext_di(x, UINT64_C(0x0101010101010101));
#else
  return pack_word_multiply(x);
#endif
}

void expand_words_reference(uint64_t x, aligned_byte *data)
{
  aligned_word *words = (aligned_word *)data;
  unsigned char *byte = (unsigned char *)&x;

  for (unsigned i = 0; i < 8; i++)
    {
      words[i] = expand_byte(byte[i]);
    }
}

extern "C" uint64_t pack_words_reference(aligned_byte *data)
{
  aligned_word *words = (aligned_word *)data;
  uint64_t res = 0;
  for (unsigned i = 0; i < 8; i++)
    {
      res |= ((uint64_t)pack_word(words[i]) & UINT8_C(0xff)) << 8 * i;
    }
  return res;
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

extern "C" uint64_t pack_words_shuf(aligned_byte *data)
{
  aligned_word *words = (aligned_word *)data;
  uint8_t x0 = pack_word(words[0]);
  uint8_t x1 = pack_word(words[1]);
  uint8_t x2 = pack_word(words[2]);
  uint8_t x3 = pack_word(words[3]);
  uint8_t x4 = pack_word(words[4]);
  uint8_t x5 = pack_word(words[5]);
  uint8_t x6 = pack_word(words[6]);
  uint8_t x7 = pack_word(words[7]);

  // uses vpins
  uchar8 vec = {x0, x1, x2, x3, x4, x5, x6, x7};
  uint64_t res;
  __builtin_memcpy(&res, &vec, 8);
  return res;
}

#if 0
extern "C" uint64_t pack_words_avx(aligned_byte *data)
{
  // avx2
  __m256i tmp = _mm256_loadu_si256((const __m256i*)data);
  tmp =  _mm256_cmpgt_epi8(tmp, _mm256_setzero_si256());
  uint32_t low =  _mm256_movemask_epi8(tmp);

  data += 256;

  tmp = _mm256_loadu_si256((const __m256i*)data);
  tmp =  _mm256_cmpgt_epi8(tmp, _mm256_setzero_si256());
  uint32_t high =  _mm256_movemask_epi8(tmp);
  

  return  ((uint64_t)high << 32u) | low;
}
#endif

typedef unsigned char vec __attribute__((__vector_size__(64), __aligned__(64)))
__attribute__((may_alias));

// Want to hit this IR, which is roughly what the avx implementation builds, but
// having trouble getting clang to represent the i1 vector
#if 0
; Function Attrs: norecurse noreturn nounwind readnone uwtable
define dso_local i64 @pack_words_vec(i8* nocapture readnone %data) local_unnamed_addr #5 {
entry:
  %0 = bitcast i8* %data to <64 x i8>*
  %val = load <64 x i8>, <64 x i8>* %0, align 64 
  %cmp = icmp ugt <64 x i8> %val, zeroinitializer ; ne?
  %cast = bitcast <64 x i1> %cmp to i64
  ret i64 %cast
}
#endif

extern "C" uint64_t pack_words_vec(aligned_byte *data)
{
  vec tmp = *(const vec *)data;
  vec zero = {0};
  vec lt = tmp > zero;
  lt &= (vec){
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  };

  return ((uint64_t)lt[63] << UINT64_C(63)) |
         ((uint64_t)lt[62] << UINT64_C(62)) |
         ((uint64_t)lt[61] << UINT64_C(61)) |
         ((uint64_t)lt[60] << UINT64_C(60)) |
         ((uint64_t)lt[59] << UINT64_C(59)) |
         ((uint64_t)lt[58] << UINT64_C(58)) |
         ((uint64_t)lt[57] << UINT64_C(57)) |
         ((uint64_t)lt[56] << UINT64_C(56)) |
         ((uint64_t)lt[55] << UINT64_C(55)) |
         ((uint64_t)lt[54] << UINT64_C(54)) |
         ((uint64_t)lt[53] << UINT64_C(53)) |
         ((uint64_t)lt[52] << UINT64_C(52)) |
         ((uint64_t)lt[51] << UINT64_C(51)) |
         ((uint64_t)lt[50] << UINT64_C(50)) |
         ((uint64_t)lt[49] << UINT64_C(49)) |
         ((uint64_t)lt[48] << UINT64_C(48)) |
         ((uint64_t)lt[47] << UINT64_C(47)) |
         ((uint64_t)lt[46] << UINT64_C(46)) |
         ((uint64_t)lt[45] << UINT64_C(45)) |
         ((uint64_t)lt[44] << UINT64_C(44)) |
         ((uint64_t)lt[43] << UINT64_C(43)) |
         ((uint64_t)lt[42] << UINT64_C(42)) |
         ((uint64_t)lt[41] << UINT64_C(41)) |
         ((uint64_t)lt[40] << UINT64_C(40)) |
         ((uint64_t)lt[39] << UINT64_C(39)) |
         ((uint64_t)lt[38] << UINT64_C(38)) |
         ((uint64_t)lt[37] << UINT64_C(37)) |
         ((uint64_t)lt[36] << UINT64_C(36)) |
         ((uint64_t)lt[35] << UINT64_C(35)) |
         ((uint64_t)lt[34] << UINT64_C(34)) |
         ((uint64_t)lt[33] << UINT64_C(33)) |
         ((uint64_t)lt[32] << UINT64_C(32)) |
         ((uint64_t)lt[31] << UINT64_C(31)) |
         ((uint64_t)lt[30] << UINT64_C(30)) |
         ((uint64_t)lt[29] << UINT64_C(29)) |
         ((uint64_t)lt[28] << UINT64_C(28)) |
         ((uint64_t)lt[27] << UINT64_C(27)) |
         ((uint64_t)lt[26] << UINT64_C(26)) |
         ((uint64_t)lt[25] << UINT64_C(25)) |
         ((uint64_t)lt[24] << UINT64_C(24)) |
         ((uint64_t)lt[23] << UINT64_C(23)) |
         ((uint64_t)lt[22] << UINT64_C(22)) |
         ((uint64_t)lt[21] << UINT64_C(21)) |
         ((uint64_t)lt[20] << UINT64_C(20)) |
         ((uint64_t)lt[19] << UINT64_C(19)) |
         ((uint64_t)lt[18] << UINT64_C(18)) |
         ((uint64_t)lt[17] << UINT64_C(17)) |
         ((uint64_t)lt[16] << UINT64_C(16)) |
         ((uint64_t)lt[15] << UINT64_C(15)) |
         ((uint64_t)lt[14] << UINT64_C(14)) |
         ((uint64_t)lt[13] << UINT64_C(13)) |
         ((uint64_t)lt[12] << UINT64_C(12)) |
         ((uint64_t)lt[11] << UINT64_C(11)) |
         ((uint64_t)lt[10] << UINT64_C(10)) | ((uint64_t)lt[9] << UINT64_C(9)) |
         ((uint64_t)lt[8] << UINT64_C(8)) | ((uint64_t)lt[7] << UINT64_C(7)) |
         ((uint64_t)lt[6] << UINT64_C(6)) | ((uint64_t)lt[5] << UINT64_C(5)) |
         ((uint64_t)lt[4] << UINT64_C(4)) | ((uint64_t)lt[3] << UINT64_C(3)) |
         ((uint64_t)lt[2] << UINT64_C(2)) | ((uint64_t)lt[1] << UINT64_C(1)) |
         ((uint64_t)lt[0] << UINT64_C(0));
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

void round_trip_word()
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

void zero(aligned_byte *d)
{
  for (unsigned i = 0; i < 64; i++)
    {
      d[i] = 0;
    }
}

void dump(aligned_byte *d)
{
  for (unsigned i = 0; i < 64; i++)
    {
      printf(" %u", (unsigned)d[i]);
    }
  printf("\n");
}

void round_trip_words()
{
  alignas(64) HOSTRPC_ATOMIC(uint8_t) tmp[64];

  for (unsigned s = 0; s < 64; s++)
    {
      uint64_t x = UINT64_C(1) << s;

      zero(tmp);
      expand_words_reference(x, tmp);
      uint64_t y = pack_words_reference(tmp);
      if (x != y)
        {
          dump(tmp);
          printf("%lx => %lx\n", x, y);
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

  round_trip_word();

  round_trip_words();

  return 0;

  char tmp[FMT_BUF_SIZE];
  for (unsigned i = 0; i < 256; i++)
    {
      printf("0b%s\n", binary_fmt(expand_byte(i), tmp));
    }
}
