#include<exception>
#include<iostream>
#include<vector>
#define log() std::cout << __PRETTY_FUNCTION__ << '\n';
#include<type_traits>

/*xx
class WidgetFactory {
public:
	virtual Window* CreateWindow() = 0;
	virtual Button* CreateButton() = 0;
	virtual ScrollBar* CreateScrollBar() = 0;

	virtual ~WidgetFactory() = default;
};

// Makes it imposibble to do this:
template<typename T>
T* MakeRedWidget(WidgetFactory& factory) {
	T* pw = factory.CreateT(); // huh??
	pw->SetColor(RED);
	return pW;
}


// This would be preferred
template<typename T>
T* MakeRedWidget(WidgetFactory& factory) {
	T* pw = factory.Create<T>();
	pw->SetColor(RED);
	return pW;
}
*/
struct EmptyType{};
struct NullType{};

template<typename T, typename U>
struct Typelist {
	using Head = T;
	using Tail = U;
};

// Typelist algorithms
namespace TL {
	template<typename TList> struct Length;
	template<> struct Length<NullType> {
		enum { value = 0 };
	};

	template<typename T, typename U>
	struct Length< Typelist<T, U>> {
		enum { value = 1 + Length<U>::value };
	};

	template<typename TList, unsigned int index> struct TypeAt;
	
	template<typename Head, typename Tail>
	struct TypeAt<Typelist<Head, Tail>, 0> {
		using Result = Head;
	};

	template<typename Head, typename Tail, unsigned int i>
	struct TypeAt<Typelist<Head, Tail>, i> {
		using Result = typename TypeAt<Tail, i - 1>::Result;
	};
	
	template<typename TList, unsigned int index, typename DefaultType>
	struct TypeAtNonStrict;

	template<unsigned int index, typename DefaultType>
	struct TypeAtNonStrict<NullType, index, DefaultType> {
		using Result = DefaultType;
	};

	template<typename Head, typename Tail, typename DefaultType>
	struct TypeAtNonStrict<Typelist<Head, Tail>, 0, DefaultType> {
		using Result = Head;
	};

	template<typename Head, typename Tail, unsigned int index, typename DefaultType>
	struct TypeAtNonStrict<Typelist<Head, Tail>, index, DefaultType> {
		using Result = typename TypeAtNonStrict<Tail, index - 1, DefaultType>::Result;
	};

	template<typename TList, typename KeyType>
	struct IndexOf;

	template<typename T>
	struct IndexOf<NullType, T> {
		enum { value = -1 };
	};


	template<typename T, typename Tail>
	struct IndexOf<Typelist<T, Tail>, T> {
		enum { value = 0};
	};

	template<typename Head, typename Tail, typename T>
	struct IndexOf<Typelist<Head, Tail>, T> {
	private:
		enum { temp = IndexOf<Tail, T>::value };
	public:
		enum { value = temp == -1 ? -1 : 1 + temp };
	};


	template<typename TList, typename T> struct Append;

	template<>
	struct Append<NullType, NullType> {
		using Result = NullType;
	};

	template<typename Head, typename Tail>
	struct Append<NullType, Typelist<Head, Tail>> {
		using Result = Typelist<Head, Tail>;
	};
	
	template<typename Head, typename Tail, typename T>
	struct Append<Typelist<Head, Tail>, T> {
		using Result = Typelist<Head, typename Append<Tail, T>::Result>;
	};

	template<typename TList, typename T>
	struct Erase;

	template<typename T>
	struct Erase<NullType, T> {
		using Result = NullType;
	};

	template<typename T, typename Tail>
	struct Erase<Typelist<T, Tail>, T> {
		using Result = Tail;
	};

	template<typename Head, typename Tail, typename T>
	struct Erase<Typelist<Head, Tail>, T> {
		using Result = Typelist<Head, typename Erase<Tail, T>::Result>;
	};

	
	template<typename TList, typename T>
	struct EraseAll;

	template<typename T>
	struct EraseAll<NullType, T> {
		using Result = NullType;
	};

	template<typename T, typename Tail>
	struct EraseAll<Typelist<T, Tail>, T> {
		using Result = typename EraseAll<Tail, T>::Result;
	}; 

	template<typename Head, typename Tail, typename T>
	struct EraseAll<Typelist<Head, Tail>, T> {
		using Result = Typelist<Head, typename EraseAll<Tail, T>::Result>;
	};


	template<typename TList> struct NoDuplicates;

	template<>
	struct NoDuplicates<NullType> {
		using Result = NullType;
	};

	template<typename Head, typename Tail>
	struct NoDuplicates<TypeList<Head, Tail>> {
	private:
		using L1 = typename NoDuplicates<Tail>::Result;
		using L2 = typename Erase<L1, Head>::Result;
	public:
		using Result = TypeList<Head, L2>;
	};

	template<typename TList, typename T, typename U>
	struct Replace;

	template<typename T, typename U>
	struct Replace<NullType, T, U> {
		using Result = NullType;
	};

	template<typename T, typename Tail, typename U>
	struct Replace<Typelist<T, Tail>, T, U> {
		using Result = Typelist<U, Tail>;
	};

	template<typename Head, typename Tail, typename T, typename U>
	struct Replace<Typelist<Head,Tail>, T, U> {
		using Result = Typelist<Head, typename Replace<Tail, T, U>::Result>;
	};

	template<typename TList, typename T> struct MostDerived;

	template<typename T>
	struct MostDerived<NullType, T> {
		using Result = T;
	};

	template<typename Head, typename Tail, typename T>
	struct MostDerived<Typelist<Head, Tail>, T> {
	private:
		using Candidate = typename MostDerived<Tail, T>::Result;
	public:
		using Result = typename Select<std::is_base_of_v<Candidate, Head>, Head, Candidate>::Result;
	};

	template<typename T> struct DerivedToFront;

	template<>
	struct DerivedToFront<NullType> {
		using Result = NullType;
	};
	
	template<typename Head, typename Tail>
	struct DerivedToFront<Typelist<Head, Tail>> {
	private:
		using TheMostDerived = typename MostDerived<Tail, Head>::Result;
		using L = typename Replace<Tail, TheMostDerived, Head>::Result;
	public:
		using Result = Typelist<TheMostDerived, L>;
	};
}

#define TYPELIST_1(T1) Typelist<T1, NullType>
#define TYPELIST_2(T1, T2) Typelist<T1, TYPELIST_1(T2)>
#define TYPELIST_3(T1, T2, T3) Typelist< T1, TYPELIST_2(T1, T2)>
#define TYPELIST_4(T1, T2, T3, T4) Typelist<T1, TYPELIST_3(T1, T2, T3)>

template<typename TList, template<typename> class Unit>
class GenScatterHierarchy;

template<typename T1, typename T2, template<typename> class Unit>
class GenScatterHierarchy<Typelist<T1, T2>, Unit>
: public GenScatterHierarchy<T1, Unit>
, public GenScatterHierarchy<T2, Unit>
{
public:
	using TList = Typelist<T1, T2>;
	using LeftBase = GenScatterHierarchy<T1, Unit>;
	using RightBase = GenScatterHierarchy<T2, Unit>;
};

template<typename AtomicType, template<typename> class Unit>
class GenScatterHierarchy : public Unit<AtomicType> 
{
public:
	using LeftBase = Unit<AtomicType>;
};

template<template<typename> class Unit>
class GenScatterHierarchy<NullType, Unit>
{};


using CharList = Typelist<char, Typelist<signed char, unsigned char>>;
template<typename T> struct TypeOf;
int main() {
	using Ints = TYPELIST_4(short, int, long, long long);
	using Type = TL::Erase<Ints, long>;

	TypeOf<Type> t;

	return 0;
};
