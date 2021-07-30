#include "platform.h"

#include<processthreadsapi.h> // specific windows headers

static ThreadHandle global_thread_handles[4096];
static ThreadHandle global_stub_thread_handle;

struct ThreadHandle {
	// define contents of the struct specific to windows platform
	int thread_id;
};
ThreadHandle* CreateThread(void (*start)(void*data), void *data) {
	// call windows code to create a thread
	
	return global_stub_thread_handle;
}	
void DestroyThread(ThreadHandle* handle) {
	// Destory a thread on windows
}
bool IsThreadJoinable(const ThreadHandle*) {
	// check if thread is joinable
	return true;
}
void JoinThread(ThreadHandle* thread) {
	
}


