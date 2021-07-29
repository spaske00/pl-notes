


bool slow(unsigned x, unsigned j) {
    bool result = x << j; // forces value to 0 or 1
    return result;
}

int fast(unsigned x, unsigned j) {
    int result = x << j; // doesn't force value to be 0 or 1.
    return result;
}


/*
slow(unsigned int, unsigned int):
        mov     ecx, esi
        sal     edi, cl
        test    edi, edi
        setne   al
        ret
fast(unsigned int, unsigned int):
        mov     eax, edi
        mov     ecx, esi
        sal     eax, cl
        ret
 */

