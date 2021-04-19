#include <cassert>
#include <cstdint>
#include <cstdio>
#include <tuple>
#include <vector>

enum func_type : uint64_t
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

  func_piecewise_pass_element_int32,
  func_piecewise_pass_element_uint32,
  func_piecewise_pass_element_int64,
  func_piecewise_pass_element_uint64,
  func_piecewise_pass_element_double,
  func_piecewise_pass_element_void,
  func_piecewise_pass_element_write_int32,
  func_piecewise_pass_element_write_int64,

};

struct field
{
  field() = default;

  field(int32_t x) : tag(func_piecewise_pass_element_int32)
  {
    int64_t tmp = x;
    __builtin_memcpy(&u64_, &tmp, 8);
  }

  field(int64_t x) : tag(func_piecewise_pass_element_int64)
  {
    __builtin_memcpy(&u64_, &x, 8);
  }

  field(uint32_t x) : tag(func_piecewise_pass_element_uint32) { u64_ = x; }

  field(uint64_t x) : tag(func_piecewise_pass_element_uint64) { u64_ = x; }

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
  dump() { printf(" int32_t"); }
};

template <>
struct dump<uint32_t>
{
  dump() { printf(" uint32_t"); }
};

template <>
struct dump<int64_t>
{
  dump() { printf(" int64_t"); }
};

template <>
struct dump<uint64_t>
{
  dump() { printf(" uint64_t"); }
};

template <>
struct dump<double>
{
  dump() { printf(" double"); }
};

template <>
struct dump<const char*>
{
  dump() { printf(" const-char-*"); }
};

template <>
struct dump<void*>
{
  dump() { printf(" void*"); }
};

template <size_t ToDerive, size_t Derived, typename... Ts>
struct interpT
{
  static_assert(Derived < ToDerive, "");

  void operator()(const std::vector<field>& args)
  {
    const field& f = args[Derived];

    constexpr size_t NextDerived = sizeof...(Ts) + 1;

    switch (f.tag)
      {
        case func_piecewise_pass_element_int32:
          using T = interpT<ToDerive, NextDerived, Ts..., int32_t>;
          return T{}(args);
        case func_piecewise_pass_element_uint32:
          return interpT<ToDerive, NextDerived, Ts..., uint32_t>{}(args);
        case func_piecewise_pass_element_int64:
          return interpT<ToDerive, NextDerived, Ts..., int64_t>{}(args);
        case func_piecewise_pass_element_uint64:
          return interpT<ToDerive, NextDerived, Ts..., uint64_t>{}(args);
        case func_piecewise_pass_element_double:
          return interpT<ToDerive, NextDerived, Ts..., double>{}(args);

        case func_piecewise_pass_element_void:
          return interpT<ToDerive, NextDerived, Ts..., void*>{}(args);

        case func_piecewise_pass_element_cstr:
          return interpT<ToDerive, NextDerived, Ts..., const char*>{}(args);
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
  void operator()(const std::vector<field>& args)
  {
    printf("Narg %zu (", ToDerive);
    if constexpr (ToDerive > 0)
      {
        dump<typename std::tuple_element<0, asTup>::type>{};
      }
    if constexpr (ToDerive > 1)
      {
        dump<typename std::tuple_element<1, asTup>::type>{};
      }
    if constexpr (ToDerive > 2)
      {
        dump<typename std::tuple_element<2, asTup>::type>{};
      }
    if constexpr (ToDerive > 3)
      {
        dump<typename std::tuple_element<3, asTup>::type>{};
      }
    if constexpr (ToDerive > 4)
      {
        dump<typename std::tuple_element<4, asTup>::type>{};
      }
    printf(")\n");
  }
};

template <size_t N>
void interpN(const std::vector<field>& args)
{
  interpT<N, 0> inst;
  inst(args);
}

void interp(const std::vector<field>& args)
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
#if 0
      case 6:
        return interpN<6>(args);
      case 7:
        return interpN<7>(args);
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
