#include "platform.h"

#include<pthread.h> // specific linux headers

static ThreadHandle global_thread_handles[4096];
static ThreadHandle global_stub_thread_handle;

struct ThreadHandle {
	// define contents of the struct specific to linux platform
	int thread_id;
};
ThreadHandle* CreateThread(void (*start)(void*data), void *data) {
	// call linux code to create a thread
	
	return global_stub_thread_handle;
}	
void DestroyThread(ThreadHandle* handle) {
	// Destory a thread on linux
}
bool IsThreadJoinable(const ThreadHandle*) {
	// check if thread is joinable
	return true;
}
void JoinThread(ThreadHandle* thread) {
	
}


