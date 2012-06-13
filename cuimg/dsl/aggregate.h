#ifndef CUIMG_GPU_AGGREGATE_H_
# define CUIMG_GPU_AGGREGATE_H_

# include <cuimg/meta.h>
# include <cuimg/dsl/expr.h>
# include <cuimg/dsl/eval.h>
# include <cuimg/dsl/has.h>
# include <cuimg/dsl/tuple.h>
# include <cuimg/dsl/tuple_eval.h>

namespace cuimg
{

  template <typename T, typename A>
  struct aggregate_1 : public expr<aggregate_1<T, A> >
  {
    typedef int is_expr;

    aggregate_1(A a)
      : a_(a)
    {
    }

    __host__ __device__ inline
    improved_builtin<T, 1> eval(point2d<int> p) const
    {
      return improved_builtin<T, 1>(cuimg::eval(a_, p));
    }

    __host__ __device__ inline
    bool has(const point2d<int>& p)
    {
      return cuimg::has(p, a_);
    }

    typename kernel_type<A>::ret a_;
  };

  template <typename T, typename A, typename B>
  struct aggregate_2 : public expr<aggregate_2<T, A, B> >
  {
    typedef int is_expr;

    aggregate_2(A a, B b)
      : a_(a),
        b_(b)
    {
    }

    __host__ __device__ inline
    improved_builtin<T, 2> eval(point2d<int> p) const
    {
      return improved_builtin<T, 2>(cuimg::eval(a_, p), cuimg::eval(b_, p));
    }

    __host__ __device__ inline
    bool has(point2d<int> p)
    {
      return cuimg::has(p, a_, b_);
    }

    typename kernel_type<A>::ret a_;
    typename kernel_type<B>::ret b_;
  };

  template <typename T, typename A, typename B, typename C>
  struct aggregate_3 : public expr<aggregate_3<T, A, B, C> >
  {
    typedef int is_expr;

    aggregate_3(A a, B b, C c)
      : a_(a),
        b_(b),
        c_(c)
    {
    }

    __host__ __device__ inline
    improved_builtin<T, 3> eval(point2d<int> p) const
    {
      return improved_builtin<T, 3>(cuimg::eval(a_, p), cuimg::eval(b_, p), cuimg::eval(c_, p));
    }


    __host__ __device__ inline
    bool has(point2d<int> p)
    {
      return cuimg::has(p, a_, b_, c_);
    }

    typename kernel_type<A>::ret a_;
    typename kernel_type<B>::ret b_;
    typename kernel_type<C>::ret c_;
  };


  template <typename T, typename A, typename B, typename C, typename D>
  struct aggregate_4 : public expr<aggregate_4<T, A, B, C, D> >
  {
    typedef int is_expr;

    aggregate_4(A a, B b, C c, D d)
      : a_(a),
        b_(b),
        c_(c),
        d_(d)
    {
    }

    __host__ __device__ inline
    improved_builtin<T, 4> eval(point2d<int> p) const
    {
      return improved_builtin<T, 4>(cuimg::eval(a_, p), cuimg::eval(b_, p), cuimg::eval(c_, p), cuimg::eval(d_, p));
    }


    __host__ __device__ inline
    bool has(point2d<int> p)
    {
      return cuimg::has(p, a_, b_, c_, d_);
    }

    typename kernel_type<A>::ret a_;
    typename kernel_type<B>::ret b_;
    typename kernel_type<C>::ret c_;
    typename kernel_type<D>::ret d_;
  };

  template <typename T> struct comp_getters;

  template <typename T, typename ARGS>
  struct aggregate_tuple : public expr<aggregate_tuple<T, ARGS> >
  {
    typedef int is_expr;

    aggregate_tuple(ARGS t)
    : t_(t)
    {
    }

    __host__ __device__ inline
    T eval(point2d<int> p) const
    {
      typedef comp_getters<T> G;
      return tuple_eval(G::make, t_, p);
    }

    __host__ __device__ inline
    bool has(point2d<int> p)
    {
      return tuple_caller(cuimg::has, t_);
    }

    typename kernel_type<ARGS>::ret t_;
  };


  template <typename T>
  struct make
  {
    template <typename A, typename B, typename C>
    static aggregate_tuple<T, tuple<A, B, C> > run(A a, B b, C c)
    { return aggregate_tuple<T, tuple<A, B, C> >(tuple<A, B, C>(a, b, c)); }

    template <typename A, typename B, typename C, typename D>
    static aggregate_tuple<T, tuple<A, B, C, D> > run(A a, B b, C c, D d)
    { return aggregate_tuple<T, tuple<A, B, C, D> >(tuple<A, B, C, D>(a, b, c, d)); }

    template <typename A, typename B, typename C, typename D, typename E>
    static aggregate_tuple<T, tuple<A, B, C, D, E> > run(A a, B b, C c, D d, E e)
    { return aggregate_tuple<T, tuple<A, B, C, D, E> >(tuple<A, B, C, D, E>(a, b, c, d, e)); }
  };

  template <typename T, typename A, typename B, typename C, typename D>
  struct return_type<aggregate_4<T, A, B, C, D> >
  { typedef improved_builtin<T, 4> ret; };

  template <typename T, typename A, typename B>
  struct return_type<aggregate_2<T, A, B> >
  { typedef improved_builtin<T, 2> ret; };

  template <typename A, typename B, typename C, typename D>
  aggregate_4<float, A, B, C, D> aggregate_float(A a, B b, C c, D d)
  { return aggregate_4<float, A, B, C, D>(a, b, c, d); }

  template <typename T>
  struct aggregate
  {
    template <typename A>
    static aggregate_1<T, A> run(A a)
    { return aggregate_1<T, A>(a); }
    template <typename A, typename B>
    static aggregate_2<T, A, B> run(A a, B b)
    { return aggregate_2<T, A, B>(a, b); }
    template <typename A, typename B, typename C>
    static aggregate_3<T, A, B, C> run(A a, B b, C c)
    { return aggregate_3<T, A, B, C>(a, b, c); }
    template <typename A, typename B, typename C, typename D>
    static aggregate_4<T, A, B, C, D> run(A a, B b, C c, D d)
    { return aggregate_4<T, A, B, C, D>(a, b, c, d); }
  };

#define make_vec2(T, A, B) aggregate<T>::run(A, B);
#define make_vec3(T, A, B, C) aggregate<T>::run(A, B, C, D);
#define make_vec4(T, A, B, C, D) aggregate<T>::run(A, B, C, D);

}

#endif
