#include<exception>
#include<iostream>
#include<vector>
#define log() std::cout << __PRETTY_FUNCTION__ << '\n';

template<typename T>
struct OpNewCreator {
	static T* Create() {
		log();
		return new T;
	}
protected:
	~OpNewCreator() {}
};


template<typename T>
struct MallocCreator {
	static T* Create() {
		log();
		void *buf = (T*)malloc(sizeof(T));
		if(!buf) return nullptr;
		return new(buf) T;
	}
protected:
	~MallocCreator() {}
};


template<typename T>
struct PrototypeCreator {
	explicit PrototypeCreator(T* pObj = nullptr) 
		: m_pPrototype(pObj) {}
	T* Create() {
		log();
		return m_pPrototype ? m_pPrototype->Clone() : nullptr;
	}
	
	T* GetPrototype() { return m_pPrototype; }
	void SetPrototype(T* pObj) { m_pPrototype = pObj; }
private:
	T* m_pPrototype;
protected:
	~PrototypeCreator() {}
};

template<typename T>
struct Cloner {
	T* Clone() const {
		return new T(*static_cast<const T*>(this));
	}
protected:
	~Cloner() {}
};

class Widget : public Cloner<Widget> {};

template <template <typename Created> class CreationPolicy>
class WidgetManager : public CreationPolicy<Widget> {
public:
	std::vector<Widget*> MakeWidgets(size_t count) {
		std::vector<Widget*> widgets;
		widgets.reserve(count);
		for (size_t i = 0; i < count; ++i) {
			auto widget = CreationPolicy<Widget>::Create();
			widgets.push_back(widget);			
		}
		return widgets;
	}
	
	/* If the user instantiates WidgetManager with a Creator policy class:
	 * that supports prototypes, SwitchPrototype can be used.
	 * that doesn't support prototypes and tries to use SwitchPrototype, a compile-time error occurs
	 * that doesn't support prototypes and does not try to use SwitchPrototype, the program is valid
	 */
	void SwitchPrototype(Widget* pNewPrototype) {
		CreationPolicy<Widget>& myPolicy = *this;
		delete myPolicy.GetPrototype();
		myPolicy.SetPrototype(pNewPrototype);
	}
};

using MyWidgetMgr = WidgetManager<PrototypeCreator>;


template<typename T> 
struct NoChecking {
	NoChecking(T*) {}
	static void Check(T*) {}
};

template<typename T>
struct EnforceNotNull {
	class NullPointerException : public std::exception {};
	EnforceNotNull(T*) {}
	static void Check(T* ptr) {
		if (!ptr) throw NullPointerException();
	}
};

template<typename T>
struct EnsureNotNull {
	EnsureNotNull(T*) {}
	static void Check(T*& ptr) {
		if (!ptr) ptr = T::GetDefaultValue();
	}
};

template<typename T>
class DefaultSmartPtrStorage {
public:
	using PointerType = T*;
	using ReferenceType = T&;
	explicit DefaultSmartPtrStorage(PointerType ptr) : ptr_(ptr) {}
protected:
	PointerType GetPointer() { return ptr_; }
	void SetPointer(PointerType ptr) { ptr_ = ptr; }
private:
	PointerType ptr_;
};

template<typename T>
class NoLock {
public:
	NoLock(T* ptr) {}
	template<typename SmartPtr>
	struct Lock {
		Lock(SmartPtr& toLock) {}
	};
};

template<
	typename T,
	template <typename> class CheckingPolicy,
	template <typename> class ThreadingModel,
	template <typename> class Storage = DefaultSmartPtrStorage
>
class SmartPtr 
: public CheckingPolicy<T>
, public ThreadingModel<T> 
, public Storage<T>
{
public:
	explicit SmartPtr(T* ptr)
		: CheckingPolicy<T>(ptr), ThreadingModel<T>(ptr), Storage<T>(ptr) {}
	T* operator->()
	{
		typename ThreadingModel<T>::Lock<SmartPtr> guard(*this);
		auto pointee = DefaultSmartPtrStorage<T>::GetPointer();
		CheckingPolicy<T>::Check(pointee);
		return pointee;
	}

};

int main() {
	MyWidgetMgr widgetManager;
	Widget widget;
	widgetManager.SetPrototype(&widget);
	auto result = widgetManager.MakeWidgets(10);


	SmartPtr<Widget, NoChecking, NoLock> ptr(&widget);
	auto clonedWidget = ptr->Clone();
	return 0;
}


