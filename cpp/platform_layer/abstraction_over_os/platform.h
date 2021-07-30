#ifndef PLATFORM_H
#define PLATFORM_H

struct ThreadHandle;

ThreadHandle* CreateThread(void (*start)(void*data), void *data); 
void DestroyThread(ThreadHandle*);
bool IsThreadJoinable(const ThreadHandle*);
void JoinThread(ThreadHandle*);

#endif
