#include <cstdio>

template <typename Func> struct DeferImpl {
  DeferImpl(Func f) : f(f) {}
  ~DeferImpl() { f(); }
  Func f;
};

struct MakeDeferImpl {};

template <typename Func> DeferImpl<Func> operator<<(MakeDeferImpl, Func f) {
  return DeferImpl(f);
}

#define CONCAT_IMPL(a, b) a##b
#define CONCAT(a, b) CONCAT_IMPL(a, b)
#define defer auto CONCAT(defer_stmt_, __LINE__) = MakeDeferImpl() << [&]

int main() {
  defer { printf("Hello, World!"); };
  defer { printf("Hello to you to!"); };

  return 0;
}
