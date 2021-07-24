#include "code.h"
#include <stdio.h>
#include <dlfcn.h>
#include <assert.h>

typedef int (*f_type)(int);

int main() {
	void *libcode_so = dlopen("./libcode.so", RTLD_NOW);
	assert(libcode_so);
	f_type fp = dlsym(libcode_so, "f");
	assert(fp);
	printf("%d", fp(3));
	dlclose(libcode_so);
	return 0;
}
