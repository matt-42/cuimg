#ifndef CUDA_IMPROVED_BUILTINS
# define CUDA_IMPROVED_BUILTINS

/* # include <boost/typeof.hpp> */

/* char1, uchar1, char2, uchar2, char3, uchar3, char4, uchar4, short1,
   ushort1, short2, ushort2, short3, ushort3, short4, ushort4, int1,
   uint1, int2, uint2, int3, uint3, int4, uint4, long1, ulong1, long2,
   ulong2, long3, ulong3, long4, ulong4, longlong1, ulonglong1,
   longlong2, ulonglong2, float1, float2, float3, float4, double1,
   double2 */

#include <cuda_runtime.h>
#include "meta.h"

namespace cuimg
{
template <typename T>
struct bt_vtype;
#define bt_vtype(X) typename bt_vtype<X>::ret
#define bt_vtype_(X) bt_vtype<X>::ret

template <typename T>
struct bt_size;
#define bt_size(X) bt_size<X>::val
#define bt_size_(X) bt_size<X>::val

template <typename T, unsigned S>
struct make_bt;
#define make_bt(X, Y) typename make_bt<X, Y>::ret
#define make_bt_(X, Y) make_bt<X, Y>::ret

struct zero {};

template <typename T, unsigned N>
class improved_builtin : public make_bt(T, N)
{
 public:
  typedef make_bt(T, N) parent;
  typedef improved_builtin<T, N> self;
  typedef typename T vtype;
  enum { size = bt_size<T>::val };

  __host__ __device__ improved_builtin();

  __host__ __device__ improved_builtin(vtype x);
  __host__ __device__ improved_builtin(vtype x_, vtype y_);
  __host__ __device__ improved_builtin(vtype x_, vtype y_, vtype z_);
  __host__ __device__ improved_builtin(vtype x_, vtype y_, vtype z_, vtype w_);
  __host__ __device__ improved_builtin(const T bt);

  __host__ __device__ improved_builtin<T>& operator=(vtype x_);

  __host__ __device__ improved_builtin(const zero&);
  __host__ __device__ inline self& operator=(const zero&);

  template <typename U>
  __host__ __device__ inline improved_builtin(const improved_builtin<U>& bt);
  template <typename U>
  __host__ __device__ inline self& operator=(const improved_builtin<U>& bt);

  template <typename U, int START_POS, int OUT_START_POS>
  __host__ __device__ inline self& assign(const improved_builtin<U>& bt);

  __host__ __device__ inline vtype& operator[](unsigned i);
  __host__ __device__ inline const vtype& operator[](unsigned i) const;

  template <typename U>
  __host__ __device__ inline self& operator+=(const improved_builtin<U>& bt);
  template <typename U>
  __host__ __device__ inline self& operator-=(const improved_builtin<U>& bt);
  template <typename S>
  __host__ __device__ inline self& operator*=(const S& s);
  template <typename S>
  __host__ __device__ inline self& operator/=(const S& s);

};

template <typename CUDA_BT>
class improved_builtin : public improved_builtin<bt_vtype(CUDA_BT), bt_size(CUDA_BT)>
{
}

template <unsigned S>
struct bt_getter;

template <>
struct bt_getter<0>
{
	template<typename T>
	static __device__ __host__ inline typename improved_builtin<T>::vtype& get(T& bt)
	{
		return bt.x;
	}
	template<typename T>
	static __device__ __host__ inline const typename improved_builtin<T>::vtype& get(const T& bt)
	{
		return bt.x;
	}
};

template <>
struct bt_getter<1>
{
	template<typename T>
	static __device__ __host__ inline typename improved_builtin<T>::vtype& get(T& bt)
	{
		return bt.y;
	}
	template<typename T>
	static __device__ __host__ inline const typename improved_builtin<T>::vtype& get(const T& bt)
	{
		return bt.y;
	}
};

template <>
struct bt_getter<2>
{
	template<typename T>
	static __device__ __host__ inline typename improved_builtin<T>::vtype& get(T& bt)
	{
		return bt.z;
	}
	template<typename T>
	static __device__ __host__ inline const typename improved_builtin<T>::vtype& get(const T& bt)
	{
		return bt.z;
	}
};

template <>
struct bt_getter<3>
{
	template<typename T>
	static __device__ __host__ inline typename improved_builtin<T>::vtype& get(T& bt)
	{
		return bt.w;
	}
	template<typename T>
	static __device__ __host__ inline const typename improved_builtin<T>::vtype& get(const T& bt)
	{
		return bt.w;
	}
};



template <typename U, typename V>
__host__ __device__ inline bool operator==(const improved_builtin<U>& a, const improved_builtin<V>& b);

template <typename U, typename V>
__host__ __device__ inline bool operator!=(const improved_builtin<U>& a, const improved_builtin<V>& b);

template <typename T>
struct bt_vtype<improved_builtin<T> > { typedef typename bt_vtype<T>::ret ret; };
template <typename T>
struct bt_vtype<const T> { typedef typename bt_vtype<T>::ret ret; };

template <typename T>
struct bt_size<improved_builtin<T> > { enum {val = bt_size<T>::val }; };
template <typename T>
struct bt_size<const T> { enum {val = bt_size<T>::val }; };


#define BUILTIN_INFO(BT, VTYPE, SIZE)			\
template <> struct bt_vtype<BT> { typedef VTYPE ret; };	\
template <> struct bt_size<BT> { enum { val = SIZE }; };	\
typedef improved_builtin<BT> i_##BT;			\
template <> struct make_bt<VTYPE, SIZE> { typedef i_##BT ret; }

#define BUILTIN_INFO2(T, VTYPE)			\
BUILTIN_INFO(T##1, VTYPE, 1);			\
BUILTIN_INFO(T##2, VTYPE, 2)
#define BUILTIN_INFO4(T, VTYPE)		\
BUILTIN_INFO(T##1, VTYPE, 1);			\
BUILTIN_INFO(T##2, VTYPE, 2);			\
BUILTIN_INFO(T##3, VTYPE, 3);			\
BUILTIN_INFO(T##4, VTYPE, 4)

BUILTIN_INFO4(char, signed char);
BUILTIN_INFO4(uchar, unsigned char);
BUILTIN_INFO4(short, short);
BUILTIN_INFO4(ushort, unsigned short);
BUILTIN_INFO4(int, int);
BUILTIN_INFO4(uint, unsigned int);
BUILTIN_INFO4(long, long);
BUILTIN_INFO4(ulong, unsigned long);
BUILTIN_INFO2(longlong, long long);
BUILTIN_INFO2(ulonglong, unsigned long long);
BUILTIN_INFO4(float, float);
BUILTIN_INFO2(double, double);


#define DEF_MAKE_BT_1_COMP(BT)\
__host__ __device__ inline i_##BT make_i_##BT(i_##BT::vtype x) { return make_##BT(x); }
#define DEF_MAKE_BT_2_COMP(BT)\
__host__ __device__ inline i_##BT make_i_##BT(i_##BT::vtype x, i_##BT::vtype y) { return make_##BT(x, y); }
#define DEF_MAKE_BT_3_COMP(BT)\
__host__ __device__ inline i_##BT make_i_##BT(i_##BT::vtype x, i_##BT::vtype y, i_##BT::vtype z) { return make_##BT(x, y, z); }
#define DEF_MAKE_BT_4_COMP(BT)\
__host__ __device__ inline i_##BT make_i_##BT(i_##BT::vtype x, i_##BT::vtype y, i_##BT::vtype z, i_##BT::vtype w) { return make_##BT(x, y, z, w); }

#define DEF_MAKE_BT2(T)			\
DEF_MAKE_BT_1_COMP(T##1);			\
DEF_MAKE_BT_2_COMP(T##2);
#define DEF_MAKE_BT4(T)		\
DEF_MAKE_BT_1_COMP(T##1);			\
DEF_MAKE_BT_2_COMP(T##2);			\
DEF_MAKE_BT_3_COMP(T##3);			\
DEF_MAKE_BT_4_COMP(T##4);

DEF_MAKE_BT4(char);
DEF_MAKE_BT4(uchar);
DEF_MAKE_BT4(short);
DEF_MAKE_BT4(ushort);
DEF_MAKE_BT4(int);
DEF_MAKE_BT4(uint);
DEF_MAKE_BT4(long);
DEF_MAKE_BT4(ulong);
DEF_MAKE_BT2(longlong);
DEF_MAKE_BT2(ulonglong);
DEF_MAKE_BT4(float);
DEF_MAKE_BT2(double);

template <typename BT, typename V>
struct bt_change_vtype
{
  typedef typename make_bt<V, bt_size(BT)>::ret ret;
};
#define bt_change_vtype(X, Y) typename bt_change_vtype<X, Y>::ret
#define bt_change_vtype_(X, Y) bt_change_vtype<X, Y>::ret

}

# include <cuimg/improved_builtin.hpp>

#endif
