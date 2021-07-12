#include<exception>
#include<iostream>
#include<vector>
#define log() std::cout << __PRETTY_FUNCTION__ << '\n';

 // 2.1 Compile time check
template<bool> struct CompileTimeChecker {
	CompileTimeChecker(...);
};

template<> struct CompileTimeChecker<false> {};
#define STATIC_CHECK(expr, msg)\
{\
	class ERROR_##msg {};\
	(void)sizeof(CompileTimeChecker<(expr) != 0>(ERROR_##msg()));\
}


template<typename To, typename From>
To safe_reinterpret_cast(From from)
{
	STATIC_CHECK(sizeof(From) <= sizeof(To), Destination_Type_Too_Narrow);
	return reinterpret_cast<To>(from);
}

// 2.2 Partial Template Specilaization

template<typename Window, typename Controller>
class Widget { };

class ModalDialog {};
class MyController {};

template<typename Arg>
class Button {};

template<>
class Widget<ModalDialog, MyController { // Specialization for ModalDialog and MyController 
};

template<typename Window>
class Widiget<Window, MyController> {
	// Specilaization for MyController and any Window
};

template<typename ButtonArg>
class Widget<Button<ButtonArg>, MyController>
{
	// Specilization for tempalte param of Button
};

template<typename T, typename U> T Fun(U obj); // primary template
// template <typename U> void Fun<void, U>(U obj); illegal partial specilziation
template <typename T> T Fun(Window obj); // legal overloading

// 2.3 Local classes

class Interface {
public:
	virtual void Fun() = 0;
	virtual ~Interface() {};
};

template<typename T, typename P>
Interface* MakeAdapter(const T& obj, const P& arg) {
	class Local : public Interface {
	public:
		Local(const T& obj, const P& arg) : obj_(obj), arg_(arg) {}
		virtual void Fun() {
			obj_.Call(arg_);
		}
	private:
		T obj_;
		P arg_;
	};

	return new Local(obj, arg);
}

// 2.4 Mapping Integral Constants to Types

// Generates unique types for every int
// Int2Type<0> is a different type from Int2Type<1> and so on...
template <int v>
struct Int2Type
{
	enum {value = v};
};

template<typename T, bool isPolymorphic>
class NiftyContainer {
public:
	void DoSomething(T* pSomeObj) {
		// Compiler will allow this only if T has both Clone and Copy constructor
		// But if the goal is to do compile time branching based on wheter or not T isPolymorphic than the compiler 
		// wont allow this because if-else is a runtime construct and both branches need to be able to compile 
		// because it is not possible to know at compile time which branch will be taken
		if (isPolymorphic) {
			T* pNewObj = pSomeObj->Clone();
		} else {
			T* pNewObj = new T(*pSomeObj);
		}
	}
};

// Solution no 1.
template<typename T, bool isPolymorphic>
class NiftyContainer {
	void DoSomething(T* pObj, Int2Type<true>) {
		T* pNewObj = pSomeObj->Clone();
	}
	void DoSomething(T* pObj, Int2Type<false>) {
		T* pNewObj = new T(*pObj);
	}
public:
	void DoSomething(T* pObj) {
		DoSomething(pObj, Int2Type<isPolymorphic>);
	}
};

// Solution no 2. SFINAE
template<typename T, bool isPolymorphic>
class NiftyContainer;

template<typename T>
class NiftyContainer<T, true> {
public:
	void DoSomething(T* pSomeObj) {
		T* pNewObj = pSomeObj->Clone();
	}
};

template<typename T>
class NiftyContainer<T, false> {
public:
	void DoSomething(T* pSomObj) {
		T* pNewObj = new T(*pSomObj);
	}
};	

// Solution no 3. C++20 constexpr if

template<typename T>
class NiftyContainer {
public:
	void DoSomething(T* pSomObj) {
		if constexpr (requires { pSomObj->Clone(); }) {
			T* pNewObj = pSomObj->Clone();
		} else {
			T* pNewObj = new T(*pSomObj);
		}
	}	
}

// 2.4 Type2Type mapping
// How to partialy specialize a template function?
//
template<typename T>
struct Type2Type {
	using OriginalType = T;
};

template<typename T, typename U>
T* Create(const U& arg, Type2Type<T>)
{
	return new T(arg);
}

template<typename U>
Widget* Create(const U& arg, Type2Type<Widget>) {
	return new Widget(arg, -1);
}

void test() {
	Widget *pW = Create(100, Type2Type<Widget>());
	
}

// 2.6 Type selection
template<typename T, bool isPolymorphic>
struct NiftyContainerValueTraits {
	using ValueType = T*;
};

template<typename T>
struct NiftyContainerValueTraits<T, false> {
	using ValueType = T;
};

template<typename T, bool isPolymorphic>
class NiftyContainer {
	using Traits = NiftyContainerValueTraits<T, isPolymorphic>;
	using ValueType = typename Traits::ValueType;
};


// Better solution
//
template<bool flag, typename T, typename U>
struct Select {
	using Result = T;
};

template<typename T, typename U>
struct Select<false, T, U> {
	using Result = U;
};

template<typename T, bool isPolymorphic>
class NiftyContainer {
	using ValueType = typename Select<isPolymorphic, T*, T>::Result;
};

// 2.7 Detecting Convertability and Inheritance at Compile Time
// Solved by <type_traits> from C++ standrad library

// 2.8 A Wrapper around type_info
//

class TypeInfo {
public:
	TypeInfo() = default;
	TypeInfo(const std::type_info& t) : pInfo_(&t) {}
	bool before(const TypeInfo& other) const {
		return pInfo_->before(other->pInfo_);
	}	
	const char* name() const {
		return pInfo->name();
	}
private:
	const std::type_info* pInfo_;
};

bool operator==(const TypeInfo& lhs, const TypeInfo& rhs);
bool operator!=(const TypeInfo& lhs, const TypeInfo& rhs);
bool operator<(const TypeInfo& lhs, const TypeInfo& rhs);
bool operator<=(const TypeInfo& lhs, const TypeInfo& rhs);
bool operator>(const TypeInfo& lhs, const TypeInfo& rhs);
bool operator>=(const TypeInfo& lhs, const TypeInfo& rhs);

void Fun(Base* pObj) {
	TypeInfo info = typeid(Derived);
	if(typeid(*pObj) == info) {

	}
}

// 2.9 NullType and EmptyType
struct NullType {};
struct EmptyType {};

// 2.10 Type Traits
// Solved by <type_traits> in newer C++ versions
// Finding out information about types such as: Does T derive from U. Is T primitive? Is T an Enum? Is T a pointer? 
// Does T have this method ...
// All solved by newer C++ versions with #include<type_traits>, concepts, requires, if constexpr ...
template<typename InIt, typename OutIt>
OutIt Copy(InIt first, InIt last, OutIt result) {
	for (; first != last; ++first, ++result) {
		*result = *first;
	}
	return result;
}

// Prototype of BitBlast in "SIMD_Primitivies.h"
void BitBlast(const void* src, void* dest, size_t bytes);


template<typename T>
class TypeTraits {
private:
	template<typename U> struct PoitnerTraits {
		enum { result = false };
		using PointeeType = NullType;
	};


	template<typename U> struct PointerTraits<U*> {
		enum { result = true };
		using PointeeType = U;
	};


	template<typename U> struct PointerToMemberTraits {
		enum {result = false };
	};

	template<typename U, typename V> struct PointerToMemberTraits<U V::*> {
		enum { result = true };
	}

	template<typename U> struct UnConst {
		using Result = U;
	};

	template<typename U> struct UnConst<const U> {
		using Result = U;
	};
	
	template<typename U> struct SupportsBitwiseCopy {
		enum { result = TypeTraits<T>::isStdFundamental };
	};

	template<> struct SupportsBitwiseCopy<MyType> { // Solved by <type_traits> is_pod
		enum { result = true; };
	}

	
public:

	enum { 
		isPointer = PointerTraits<T>::result,
		isMemberPointer = PointerToMemberTraits<T>::result
	};

	using UnsignedInts = TYPELIST_4(unsigned char, unsigned short int, unsigned int, unsigned long int);
	using SignedInts = TYPELIST_4(signed char, short int, int, long int);
	using OtherInts = TYPELIST_3(bool, char, wchar_t);
	using Floats = TYPELIST_3(float, double, long double);

	enum { isStdUnsingedInt = TL::IndexOf<T, UnsignedInts>::value >= 0 };
	enum { isStdSignedInt = TL::IndexOf<T, SignedInts>::value >= 0 };
	enum { isStdIntegral = isStdUnsingedInt || isStdSignedInt || TL::IndexOf<T, OtherInts>::value >= 0 };
	enum { isStdFloat = TL::IndexOf<T, Floats>::Value >= 0 };
	enum { isStdArith = isStdIntegral || isStdFloat };
	enum { isStdFundamental = isStdArith || isStdFloat || Conversion<T, void>::sameType };

	
	using ParameterType = typename Select<isStdArith || isPointer || isMemberPointer, T, ReferencedType&>::Result;
	using PointeeType = typename PointerTraits<T>::PointeeType;
	using NonConstType = typename UnConst<T>::Result;
};

void test2() {
	const bool iterisPtr = TypeTraits<std::vector<int>::iterator>::isPointer;
}

enum CopyAlgoSelector { Conservative, Fast };
template<typename InIt, typename OutIt>
OutIt CopyImpl(InIt first, InIt last, OutIt result, Int2Type<Conservative>) {
	for (; first != last; ++first, ++result) {
		*result = *first;
	}
	return result;
}

template<typename InIt, typename OutIt>
OutIt CopyImpl(InIt first, InIt last, OutIt result, Int2Type<Fast>) {
	const size_t n = last - first;
	BitBlast(first, result, n * sizeof(*first));
	return result + n;
}

template<typename InIt, typename OutIt>
OutIt Copy(InIt first, InIt last, OutIt result) {
	using SrcPointee = typename TypeTratis<InIt>::PointeeType;
	using DestPointee = typename TypeTraits<OutIt>::PointeeType;
	enum { copyAlgo = TypeTraits<InIt>::isPointer && TypeTraits<OutIt>::isPointer && TypeTraits<SrcPointee>::isStdFundamental
		&& TypeTraits<DestPointee>::isStdFundamental 
		&& SupportsBitwiseCopy<SrcPointee>::result && SupportBitwiseCopy<DestPointee>::result	&& sizeof(SrcPointee) == sizeof(DestPointee) ? Fast : Conservative };
	return CopyImpl(first, last, result, Int2Type<copyAlgo>);
}

int main() {
	int a = 4;
	double c = 3.4;
	return 0;

}
