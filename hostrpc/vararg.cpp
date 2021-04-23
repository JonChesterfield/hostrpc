#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

enum func_type : uint8_t
{
  func_print_nop = 0,
  func_print_uuu = 1,
  func_print_start = 2,
  func_print_finish = 3,
  func_print_append_str = 4,

  func_piecewise_print_start = 5,
  func_piecewise_print_end = 6,
  func_piecewise_pass_element_cstr = 7,

  func_piecewise_pass_element_scalar = 8,

  func_piecewise_pass_element_uint32,
  func_piecewise_pass_element_uint64,
  func_piecewise_pass_element_double,
  func_piecewise_pass_element_void,
  func_piecewise_pass_element_write_int32,
  func_piecewise_pass_element_write_int64,

};

struct field
{
  field() = default;

  // 32 bit integers
  field(uint32_t x) : tag(func_piecewise_pass_element_uint32)
  {
    u64_ = (uint64_t)x;
  }
  field(int32_t x) : tag(func_piecewise_pass_element_uint32)
  {
    int64_t tmp = x;
    __builtin_memcpy(&u64_, &tmp, 8);
  }

  field(uint64_t x) : tag(func_piecewise_pass_element_uint64) { u64_ = x; }
  field(int64_t x) : tag(func_piecewise_pass_element_uint64)
  {
    __builtin_memcpy(&u64_, &x, 8);
  }

  field(double x) : tag(func_piecewise_pass_element_double) { dbl_ = x; }

  field(void* x) : tag(func_piecewise_pass_element_void) { ptr_ = x; }

  static field cstr()
  {
    field r;
    r.tag = func_piecewise_pass_element_cstr;
    return r;
  }

  template <size_t N>
  void append_cstr(const char* s)
  {
    assert(tag == func_piecewise_pass_element_cstr);
    cstr_.insert(cstr_.end(), s, s + N);
  }

  uint64_t tag = func_print_nop;
  uint64_t u64_;
  double dbl_;
  void* ptr_;
  std::vector<char> cstr_;
};

template <typename T>
struct dump;

template <>
struct dump<int32_t>
{
  static const char* str() { return " int32_t"; }
  dump() { printf("%s", str()); }
};

template <>
struct dump<uint32_t>
{
  static const char* str() { return " uint32_t"; }
  dump() { printf("%s", str()); }
};

template <>
struct dump<int64_t>
{
  static const char* str() { return " int64_t"; }
  dump() { printf("%s", str()); }
};

template <>
struct dump<uint64_t>
{
  static const char* str() { return " uint64_t"; }
  dump() { printf("%s", str()); }
};

template <>
struct dump<double>
{
  static const char* str() { return " double"; }
  dump() { printf("%s", str()); }
};

template <>
struct dump<const char*>
{
  static const char* str() { return " const-char-*"; }
  dump() { printf("%s", str()); }
};

template <>
struct dump<void*>
{
  static const char* str() { return " void*"; }
  dump() { printf("%s", str()); }
};

// c++14, courtesy of dietmar
template <typename T, std::size_t N, std::size_t... I>
constexpr std::array<T, N + 1> append_aux(std::array<T, N> a, T t,
                                          std::index_sequence<I...>)
{
  return std::array<T, N + 1>{a[I]..., t};
}
template <typename T, std::size_t N>
constexpr std::array<T, N + 1> append(std::array<T, N> a, T t)
{
  return append_aux(a, t, std::make_index_sequence<N>());
}

namespace
{
template <size_t ToDerive, typename asTup>
__attribute__((noinline)) int baseprint(const std::vector<field>& args)
{
  printf("Narg %zu (", ToDerive);

  std::string tmp;

  if constexpr (ToDerive > 0)
    {
      using Ty = dump<typename std::tuple_element<0, asTup>::type>;
      tmp += std::string(Ty::str());
    }
  if constexpr (ToDerive > 1)
    {
      using Ty = dump<typename std::tuple_element<1, asTup>::type>;
      tmp += std::string(Ty::str());
    }
  if constexpr (ToDerive > 2)
    {
      using Ty = dump<typename std::tuple_element<2, asTup>::type>;
      tmp += std::string(Ty::str());
    }
  if constexpr (ToDerive > 3)
    {
      using Ty = dump<typename std::tuple_element<3, asTup>::type>;
      tmp += std::string(Ty::str());
    }
  if constexpr (ToDerive > 4)
    {
      using Ty = dump<typename std::tuple_element<4, asTup>::type>;
      tmp += std::string(Ty::str());
    }
  return printf("%s)\n", tmp.c_str());
}

template <size_t ToDerive, size_t Derived, typename... Ts>
struct interpT
{
  static_assert(Derived < ToDerive, "");

  static void call(const std::vector<field>& args)
  {
    const field& f = args[Derived];

    constexpr size_t NextDerived = sizeof...(Ts) + 1;

    switch (f.tag)
      {
          // This is (fortunately, given compile time) not necessary
          // Not only to int32&uint32 fold, but they're passed in
          // eight byte registers. As are the pointers.
          // May need to specialise on long double (though amd64 thinks that
          // is a 80 bit value, so interop with amdgpu may be poor) or on
          // 16 bytes types

          // x64 promotes 32 to 64 bit integer, haven't checked other arch so
          // keep the distinction here

        case func_piecewise_pass_element_uint32:
          return interpT<ToDerive, NextDerived, Ts..., uint32_t>::call(args);

          // these are all 64 bit integers, assuming sizeof(void)
        case func_piecewise_pass_element_uint64:
        case func_piecewise_pass_element_void:
        case func_piecewise_pass_element_cstr:
          return interpT<ToDerive, NextDerived, Ts..., uint64_t>::call(args);

          // floating point gets promoted to double but a different path
        case func_piecewise_pass_element_double:
          return interpT<ToDerive, NextDerived, Ts..., double>::call(args);

          // return interpT<ToDerive, NextDerived, Ts..., void*>{}(args);
          // return interpT<ToDerive, NextDerived, Ts..., const char*>{}(args);
        default:
          printf("error: unhandled enum\n");
      }
  }
};

template <size_t ToDerive, typename... Ts>
struct interpT<ToDerive, ToDerive, Ts...>
{
  using asTup = std::tuple<Ts...>;
  static_assert(sizeof...(Ts) == ToDerive, "");
  static void call(const std::vector<field>& args)
  {
    baseprint<ToDerive, asTup>(args);
  }
};

template <size_t N>
void interpN(const std::vector<field>& args)
{
  interpT<N, 0>::call(args);
}

}  // namespace

extern "C" void interp_2(const std::vector<field>& args)
{
  return interpN<2>(args);
}

extern "C" void interp_3(const std::vector<field>& args)
{
  return interpN<3>(args);
}

extern "C" void interp(const std::vector<field>& args)
{
  switch (args.size())
    {
      case 0:
        return interpN<0>(args);
      case 1:
        return interpN<1>(args);
      case 2:
        return interpN<2>(args);
      case 3:
        return interpN<3>(args);
      case 4:
        return interpN<4>(args);
      case 5:
        return interpN<5>(args);
      case 6:
        return interpN<6>(args);
      case 7:
        return interpN<7>(args);
#if 0
      case 8:
        return interpN<8>(args);
      case 9:
        return interpN<9>(args);
      case 10:
        return interpN<10>(args);
      case 11:
        return interpN<11>(args);
#endif
      default:
        {
          printf("Fail, unimplemented arg size %zu\n", args.size());
          return;
        }
    }
}

int main()
{
  std::vector<field> tmp;
  interp(tmp);

  tmp.push_back(UINT32_C(32));
  interp(tmp);

  tmp.push_back(INT64_C(-32));
  interp(tmp);

  tmp.push_back(3.14);
  interp(tmp);

  {
    field f = field::cstr();
    f.append_cstr<6>("badger");
    tmp.push_back(f);
  }
  interp(tmp);

  tmp.push_back((void*)&tmp);
  interp(tmp);
}
