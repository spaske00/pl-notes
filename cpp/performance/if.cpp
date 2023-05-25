#include<algorithm>
void f1(int* i, int left) {
    if (*i < left) {
        *i = left;
    }
}

void f2(int *i, int left) {
    *i = std::max(*i, left);
}

/*
f1(int*, int):
        cmp     DWORD PTR [rdi], esi
        jge     .L1
        mov     DWORD PTR [rdi], esi
.L1:
        ret
f2(int*, int):
        mov     eax, DWORD PTR [rdi]
        cmp     eax, esi
        cmovl   eax, esi
        mov     DWORD PTR [rdi], eax
        ret
*/
