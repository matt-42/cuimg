#ifndef CUIMG_META_H_
# define CUIMG_META_H_

# include <cuimg/gpu/cuda.h>

# define BOOST_TYPEOF_COMPLIANT
# include <boost/typeof/typeof.hpp>
# include <boost/type_traits/remove_reference.hpp>
# include <boost/type_traits/remove_const.hpp>

namespace cuimg
{

  namespace meta
  {

    struct false_
    {
      enum { val = 0 };
    };

    struct true_
    {
      enum { val = 1 };
      typedef int checktype;
      static __device__ __host__ inline void check() {};
    };

    template <typename T>
    struct id
    {
      typedef T ret;
    };

    template <int B>
    struct bool_ : public true_ {};

    template <>
    struct bool_<0> : public false_ {};

    template <int S>
    struct int_
    {
      enum { val = S };
    };

    template <typename A, typename B>
    struct try_
    {
      typedef B ret;
    };

    template <typename A, typename B> struct equal : public false_ {};
    template <typename A>             struct equal<A, A> : public true_ {};
    template <unsigned A, unsigned B> struct equal_ : public equal<int_<A>, int_<B> > {};

    template <typename A, typename B> struct not_equal : public true_ {};
    template <typename A>             struct not_equal<A, A> : public false_ {};

    template <typename A, typename B> struct infeq  : public bool_<(A::val <= B::val)> {};
    template <unsigned A, unsigned B> struct infeq_ : public infeq<int_<A>, int_<B> > {};

    template <typename A, typename B> struct supeq  : public bool_<(A::val >= B::val)> {};
    template <unsigned A, unsigned B> struct supeq_ : public supeq<int_<A>, int_<B> > {};

    template <typename A, typename B> struct inf  : public bool_<(A::val <= B::val)> {};
    template <unsigned A, unsigned B> struct inf_ : public inf<int_<A>, int_<B> > {};

    template <typename A, typename B> struct sup  : public bool_<(A::val > B::val)> {};
    template <unsigned A, unsigned B> struct sup_ : public sup<int_<A>, int_<B> > {};

    template <typename A, typename B, int I> struct if_ : A {};
    template <typename A, typename B>        struct if_<A, B, 0> : B {};

    template <typename A, typename B> struct max : public if_<A, B, sup<A, B>::val> {};
    template <unsigned A, unsigned B> struct max_ : public max<int_<A>, int_<B> > {};

    template <typename A, typename B> struct min  : public if_<A, B, inf<A, B>::val> {};
    template <unsigned A, unsigned B> struct min_ : public min<int_<A>, int_<B> > {};

    template <typename A, typename B>
    struct type_add
    {
      typedef typename BOOST_TYPEOF( (*(typename boost::remove_reference<A>::type*)1) +
                                     (*(typename boost::remove_reference<B>::type*)1) ) ret;
    };
#define type_add(A, B) typename cuimg::meta::type_add<A, B>::ret
#define type_add_(A, B) cuimg::meta::type_add<A, B>::ret

    template <typename A, typename B>
    struct type_minus
    {
      typedef typename BOOST_TYPEOF( (*(typename boost::remove_reference<A>::type*)1) -
                                     (*(typename boost::remove_reference<B>::type*)1) ) ret;
    };
#define type_minus(A, B) typename cuimg::meta::type_minus<A, B>::ret
#define type_minus_(A, B) cuimg::meta::type_minus<A, B>::ret

    template <typename A, typename B>
    struct type_mult
    {
      typedef typename BOOST_TYPEOF( (*(typename boost::remove_reference<A>::type*)1) *
                                     (*(typename boost::remove_reference<B>::type*)1) ) ret;
    };
#define type_mult(A, B) typename cuimg::meta::type_mult<A, B>::ret
#define type_mult_(A, B) cuimg::meta::type_mult<A, B>::ret

    template <typename A, typename B>
    struct type_div
    {
      typedef typename BOOST_TYPEOF( (*(typename boost::remove_reference<A>::type*)1) /
                                     (*(typename boost::remove_reference<B>::type*)1) ) ret;
    };
#define type_div(A, B) typename cuimg::meta::type_div<A, B>::ret
#define type_div_(A, B) cuimg::meta::type_div<A, B>::ret


    template <template <int> class A, int I, int E>
    struct loop_iter
    {
      template <typename U>
      static __device__ __host__ inline void iter(U& a)
      {
        A<I>::run(a);
        loop_iter<A, I + 1, E>::iter(a);
      }
      template <typename U, typename V>
      static __device__ __host__ inline void iter(U& a, V& b)
      {
        A<I>::run(a, b);
        loop_iter<A, I + 1, E>::iter(a, b);
      }
      template <typename U, typename V, typename W>
      static __device__ __host__ inline void iter(U& a, V& b, W& c)
      {
        A<I>::run(a, b, c);
        loop_iter<A, I + 1, E>::iter(a, b, c);
      }
      template <typename U, typename V, typename W, typename X>
      static __device__ __host__ inline void iter(U& a, V& b, W& c, X& d)
      {
        A<I>::run(a, b, c, d);
        loop_iter<A, I + 1, E>::iter(a, b, c, d);
      }
    };

    template <template <int> class A, int E>
    class loop_iter<A, E, E>
    {
    public:
      template <typename U>
      static __device__ __host__ inline void iter(U& a)
      {
        A<E>::run(a);
      }
      template <typename U, typename V>
      static __device__ __host__ inline void iter(U& a, V& b)
      {
        A<E>::run(a, b);
      }
      template <typename U, typename V, typename W>
      static __device__ __host__ inline void iter(U& a, V& b, W& c)
      {
        A<E>::run(a, b, c);
      }
      template <typename U, typename V, typename W, typename X>
      static __device__ __host__ inline void iter(U& a, V& b, W& c, X& d)
      {
        A<E>::run(a, b, c, d);
      }
    };

    template <template <int> class A>
    class loop_end
    {
    public:
      template <typename U>
      static __device__ __host__ inline void iter(U& a)
      {
      }
      template <typename U, typename V>
      static __device__ __host__ inline void iter(U& a, V& b)
      {
      }
      template <typename U, typename V, typename W>
      static __device__ __host__ inline void iter(U& a, V& b, W& c)
      {
      }
      template <typename U, typename V, typename W, typename X>
      static __device__ __host__ inline void iter(U& a, V& b, W& c, X& d)
      {
      }
    };

    template <template <int> class A, int I, int E, int INFEQ>
    struct loop_ : public loop_iter<A, I, E> {};

    template <template <int> class A, int I, int E>
    struct loop_<A, I, E, 0> : public loop_end<A> {};

    template <template <int> class A, int I, int E>
    struct loop : public loop_<A, I, E, infeq<int_<I>, int_<E> >::val > {};








    template <template <int> class A, int I, int E>
    struct special_loop_iter
    {
      template <typename U>
      static __device__ __host__ inline void iter(U& a)
      {
        A<I>::run(a);
        special_loop_iter<A, I + 1, E>::iter(a);
      }
      template <typename U, typename V>
      static __device__ __host__ inline void iter(U& a, V& b)
      {
        A<I>::run(a, b);
        special_loop_iter<A, I + 1, E>::iter(a, b);
      }
      template <typename U, typename V, typename W>
      static __device__ __host__ inline void iter(U& a, V& b, W& c)
      {
        A<I>::run(a, b, c);
        special_loop_iter<A, I + 1, E>::iter(a, b, c);
      }
      template <typename U, typename V, typename W, typename X>
      static __device__ __host__ inline void iter(U& a, V& b, W& c, X& d)
      {
        A<I>::run(a, b, c, d);
        special_loop_iter<A, I + 1, E>::iter(a, b, c, d);
      }
    };

    template <template <int> class A, int E>
    class special_loop_iter<A, E, E>
    {
    public:
      template <typename U>
      static __device__ __host__ inline void iter(U& a)
      {
        A<E>::run(a);
      }
      template <typename U, typename V>
      static __device__ __host__ inline void iter(U& a, V& b)
      {
        A<E>::run(a, b);
      }
      template <typename U, typename V, typename W>
      static __device__ __host__ inline void iter(U& a, V& b, W& c)
      {
        A<E>::run(a, b, c);
      }
      template <typename U, typename V, typename W, typename X>
      static __device__ __host__ inline void iter(U& a, V& b, W& c, X& d)
      {
        A<E>::run(a, b, c, d);
      }
    };

    template <template <int> class A>
    class special_loop_end
    {
    public:
      template <typename U>
      static __device__ __host__ inline void iter(U& a)
      {
      }
      template <typename U, typename V>
      static __device__ __host__ inline void iter(U& a, V& b)
      {
      }
      template <typename U, typename V, typename W>
      static __device__ __host__ inline void iter(U& a, V& b, W& c)
      {
      }
      template <typename U, typename V, typename W, typename X>
      static __device__ __host__ inline void iter(U& a, V& b, W& c, X& d)
      {
      }
    };

    template <template <int> class A, int I, int E, int INFEQ>
    struct special_loop_ : public special_loop_iter<A, I, E> {};

    template <template <int> class A, int I, int E>
    struct special_loop_<A, I, E, 0> : public special_loop_end<A> {};

    template <template <int> class A, int I, int E>
    struct special_loop : public special_loop_<A, I, E, infeq<int_<I>, int_<E> >::val > {};

    template <typename T>
    struct remove_constref
    {
      typedef typename boost::remove_const<typename boost::remove_reference<T>::type>::type ret;
    };
  }

}

#endif
