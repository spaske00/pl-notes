#include <assert.h>
#include <err.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>

#define KB(x) ((x)*1024ul)
#define MB(x) (KB(x) * 1024ul)
#define GB(x) (MB(x) * 1024ul)
#define TB(x) (GB(x) * 1024ul)

namespace DynamicArray {
struct DynamicArray {
	int* data;
	size_t length;
	size_t capacity;
};
DynamicArray MakeWithCapacityOf(size_t capacity) {

  DynamicArray result;
#if 0
  result.data =
      (int *)mmap(0, capacity * sizeof(*result.data),

                  PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
#else
  int fd = open("/dev/zero", O_RDWR);
  assert(fd);
  result.data = (int *)mmap(0, capacity * sizeof(*result.data),

                            PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  close(fd);
#endif
  if (result.data == MAP_FAILED) {
    err(1, NULL);
  }
  assert(result.data != MAP_FAILED);
  result.length = 0;
  result.capacity = capacity;

  return result;
}

void Reserve(DynamicArray *arr, size_t length) {

  int result =
      mprotect(arr->data, arr->length + length, PROT_READ | PROT_WRITE);
  if (result == -1) {
    err(1, NULL);
  }

  assert(result == 0);

  arr->length += length;
}

size_t GetCapacityInBytes(DynamicArray *arr) {
  size_t result = arr->capacity * sizeof(*arr->data);
  return result;
}

size_t GetLengthInBytes(DynamicArray *arr) {
  size_t result = arr->length * sizeof(*arr->data);
  return result;
}

void ShrinkCapacityToCurrentLength(DynamicArray *arr) {
  void *result =
      mremap(arr->data, GetCapacityInBytes(arr), GetLengthInBytes(arr), 0);
  assert(result == arr->data);
}
} // namespace DynamicArray

int main() {
  rlimit limit = {.rlim_cur = RLIM_INFINITY, .rlim_max = RLIM_INFINITY};
  setrlimit(RLIMIT_AS, &limit);
  DynamicArray::DynamicArray arr = DynamicArray::MakeWithCapacityOf(TB(1ul));
  size_t offset = 0;
  size_t reserve_chunk_length = GB(1ul);
  char buffer[256] = {0};
  while (1) {
    scanf("%s", buffer);
    if (buffer[0] == 'q') {
      break;
    }
    Reserve(&arr, reserve_chunk_length);
    for (size_t i = 0; i < reserve_chunk_length; ++i) {
      arr.data[offset + i] = i;
    }
    offset += reserve_chunk_length;
  }
  printf("Out\n");
  ShrinkCapacityToCurrentLength(&arr);

  scanf("%s", buffer);
  return 0;
}
