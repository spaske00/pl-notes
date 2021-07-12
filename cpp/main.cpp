#include<exception>
#include<iostream>
#include<vector>
#define log() std::cout << __PRETTY_FUNCTION__ << '\n';
#include<type_traits>


struct MemControlBlock {
	std::size_t size_ : 31;
	bool available_ : 1;
};

struct MemControlBlock {
	// size_ not needed because it can be computed this->next_ - this when the memory comes from the same block
	bool available_;
	MemControlBlock *prev_;
	MemControlBlock *next_;
};

// 4.4 Chunks
struct Chunk {
	void Init(std::size_t blockSize, unsigned char blocks);
	void Release();
	void *Allocate(std::size_t blockSize);
	void Deallocate(void* p, std::size_t blockSize);
	unsigned char* pData_;
	unsigned char firstAvailableBlock_;
	unsigned char blocksAvailable_;
};

void Chunk::Init(std::size_t blockSize, unsigned char blocks) {
	pData_ = new unsigned char[blockSize * blocks];
	firstAvailableBlock_ = 0;
	unsigned char i = 0;
	unsigned char* p = pData_;
	for (; i != blocks; p += blockSize) {
		*p = ++i;
	}
}

void* Chunk::Allocate(std::size_t blockSize) {
	if(!blocksAvailable_) 
		return 0;
	unsigned char* pResult = pData_ + (firstAvailableBlock_ * blockSize);
	firstAvailableBlock_ = *pResult;
	--blocksAvailable_;
	return pResult;
}

void Chunk::Deallocate(void *p, std::size_t blockSize) {
	assert(p >= pData_);
	unsigned char* toRelease = static_cast<unsigned char*>(p);

	assert((toRelease - pData_) % blockSize == 0);
	*toRelease = firstAvailableBlock_;
	firstAvailableBlock_ = static_cast<unsigned char>((toRelease - pData_) / blockSize);
	assert(firstAvailableBlock_ == (toRelease - pData_) / blockSize);
	++blocksAvailable_;
}

void Chunk::Release() {
	delete[] pData_;
	pData_ = 0;
	firstAvailableBlock_ = blocksAvailable_ = 0;
}

class FixedAllocator {
public:
	using Chunks = std::vector<Chunk>;
	FixedAllocator(std::size_t blockSize, unsigned char numBlocks)
		: blockSize_(blockSize_), numBlocks_(numBlocks) {}
	void* Allocate();
private:
	std::size_t blockSize_;
	unsigned char numBlocks_;
	Chunks chunks_;
	Chunk* allocChunk_ = nullptr;
	Chunk* deallocChunk_ = nullptr;
};

void* FixedAllocator::Allocate() {
	if (allocChunk_ == nullptr || allocChunk_->blocksAvailable_ == 0) {
		auto i = chunks_.begin();
		for(;; ++i) {
			if (i == chunks_.end()) {
				chunks_.reserve(chunks_.size() + 1);
				Chunk newChunk;
				newChunk.Init(blockSize_, numBlocks_);
				chunks_.push_back(newChunk);
				allocChunk_ = &chunks_.back();
				deallocChunk_ = &chunks_.back();
				break;
			}
			if (i->blocksAvailable_ > 0) {
				allocChunk_ = &*i;
				break;
			}
		}
	}
	assert(allocChunk_ != nullptr);
	assert(allocChunk_->blocksAvailable_ > 0);
	return allocChunk_->Allocate(blockSize_);
}
