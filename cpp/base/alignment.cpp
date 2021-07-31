
#include<stdint.h>
#include<stddef.h>
uintptr_t AlignForwardPow2_v1(uintptr_t base, uintptr_t alignment) {
    uintptr_t mask = alignment - 1;
    return (base + mask) & ~mask;
}

uintptr_t AlignDownwardPow2_v1(uintptr_t base, uintptr_t alignment) {
    uintptr_t mask = alignment - 1;
    return base & ~mask;
}


template<uintptr_t Mask>
uintptr_t AlignForwardPow2_v2(uintptr_t base) {
    constexpr uintptr_t mask = Mask - 1;
    return (base + mask) & ~mask;
}

template<uintptr_t Mask>
uintptr_t AlignDownwardPow2_v2(uintptr_t base) {
    constexpr uintptr_t mask = Mask - 1;
    return base & ~mask;
}

template
uintptr_t AlignDownwardPow2_v2<64>(uintptr_t);

template
uintptr_t AlignDownwardPow2_v2<32>(uintptr_t);


#define AlignForwardPow2(base, alignment) AlignForwardPow2_v2<alignment>(base)
#define AlignDownwardPow2(base, alignment) AlignDownwardPow2_v2<alignment>(base)


int main() {
	return 0;
}
