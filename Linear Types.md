Linear Types and Clang Typestate

Jon Chesterfield

github.com/JonChesterfield/hostrpc

(ymmv, not shipping yet)

---
Linear Values
- Values that must be used exactly once
- Enforced by type system / compiler / other
- Similar to affine (value used at most once)
- Loosely related to C++ move semantics
- Decoupled from destructors
---
Emergent features
- Immutable. Only used once, so can't see changes.
- Compile time garbage collectable (like RAII).
- Not shared across threads, as can't be copied.
- Potentially annoying to print
- Interact really badly with exceptions
---
Prior art
- Baker, Lively Linear Lisp
- Rust, probably. See references
- Clang's had this since 3.7ish
	- Documentation may not be accurate
- Others?
---
Move semantics / double free
```c++
#include <stdlib.h>
#include <memory>
struct M {
    M(M &&other) = default;
    M &operator=(const M &other) = delete;
    M(int n) : ptr(malloc(n)) {}
    ~M() {free(ptr);}
    void * ptr;
};
int main() {
  M i(42);
  M j(std::move(i));
  M k(std::move(i));
}
```
https://godbolt.org/z/KK1fofEEb

clang/gcc -Wall -Wextra -Wpedantic say it's fine

---

Badness
- Unless otherwise specified, moved from object is in a valid but unspecified state
- Some functions can be called after move - std::vector::clear is probably fine
- Move can probably be called after move
- Previous struct probably gets an = nullptr in the move constructor
- Destructor will definitely still get called on it
- For N moves, likely to get N+1 destructor calls

---

Introducing Maybe. Like Optional.
```c++
using maybe_t = maybe<int>;
```

```c++
maybe_t v(42, true);
if (v) {
  assert(v.value() == 42); // OK
}
```

```c++
maybe_t v(81, false);
if (v) {
  assert(v.value() == 81); // Won't run, but OK
}
```

---

Introducing Maybe. Like Optional. Less forgiving.

Let B be true or false, same behaviour.

```c++
maybe_t v(1, B);
// implicit ~maybe_t(); // error invalid...
```

```c++
maybe_t v(2, B);
assert(v.value() == 2); // error: invalid...
```


```c++
maybe_t v(3, B);
if (!v) {
  assert(v.value() == 3); // error: invalid...
}
```

```c++
maybe_t v(4, B);
if (v) {
  assert(v.value() == 4); // OK
}
if (v) {} // error: invalid
```

---

Also quite a lot more annoying to use

```c++
template <typename T, typename U = T>
struct maybe
{
public:
  maybe(T payload, bool valid)        
    : payload(static_cast<T &&>(payload)), valid(valid)

  explicit operator bool() { return valid; }

  U value() { return static_cast<T &&>(payload); }
  operator U() { return value(); }

  ~maybe() {}
```

```c++
private:
  T payload;
  bool valid;

  // Copying or moving these types doesn't work intuitively
  maybe(const maybe &other) = delete;
  maybe(maybe &&other) = delete;
  maybe &operator=(const maybe &other) = delete;
  maybe &operator=(maybe &&other) = delete;
};
```

Note the U = T? These things don't compose well.

---

And annoying to implement, e.g.

```c++
template <typename T, typename U = T>
struct CONSUMABLE_CLASS maybe
{
  TEST_TYPESTATE(unconsumed)
  CALL_ON_UNKNOWN
  explicit operator bool() { return valid; }

  SET_TYPESTATE(consumed)
  CALL_ON_LIVE
  U value() { return static_cast<T &&>(payload); }

  CALL_ON_DEAD ~maybe() {}
};

```

  Useful for debugging
```c++
  CALL_ON_DEAD void consumed() const {}
  CALL_ON_LIVE void unconsumed() const {}
  CALL_ON_UNKNOWN void unknown() const {}
```

---

Real world - distributed state machine (on GitHub)
- `template <int state> struct port;`
- `maybe<port>open()`
- All functions take a port and statically know it's valid, no branching/asserting
- Port doesn't track if it is open, no branch in (no-op) destructor
- Port type statically encodes state of machine 
- Functions consume the old port and return a new one, sometimes with a different type
- Function type prevents calling on invalid state
- Port reliably closed

---

Clang's interface:

For identifier in consumed, unconsumed, unknown

```c++
 // Annotate classes
__attribute__((consumable(identifier)))

// On methods, requirement on *this
__attribute__((callable_when(identifier)))
```

```c++
// On methods, state after returning
__attribute__((set_typestate(identifier)))

// Magic, a boolean method that tests if state is identifier
__attribute__((test_typestate(identifier)))
```


```c++
// On functions returning one, state of the returned value
__attribute__((return_typestate(identifier)))

// On parameters, state when calling the method
__attribute__((param_typestate(identifier))

// On parameters, state after calling the method
__attribute__((return_typestate(identifier)))
```
---

How?

The annotations on parameters. Call sites track state using those. Function implementations know the required inbound and outbound state and error if the body doesn't set() it to match.

Fairly straightforward control flow analysis for a compiler.

Bunch of C++ edge cases - special cases std:: move, bakes assumptions about constructors, test() needs to be used immediately in a branch.

---

References 

https://clang.llvm.org/docs/AttributeReference.html

https://awesomekling.github.io/Catching-use-after-move-bugs-with-Clang-consumed-annotations/

Discussion of linear types with exceptions. https://borretti.me/article/linear-types-exceptions

(it's implementable if and only if you have a function that can be called implicitly to use the instance during stack unwinding)

---
More references, Rust related

http://pcwalton.github.io/2012/12/26/typestate-is-dead.html

Though it looks like rust now uses the same word to mean match on a Result type,

https://yoric.github.io/post/rust-typestate/

And Result is https://doc.rust-lang.org/std/result/

So I think Rust can deal with the unpacking neatly but can't force exactly one use of the value. See https://faultlore.com/blah/linear-rust/

---

