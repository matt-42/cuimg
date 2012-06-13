#ifndef CUIMG_GPU_HAS_H_
# define CUIMG_GPU_HAS_H_

# include <cuimg/meta.h>
# include <cuimg/dsl/expr.h>
# include <cuimg/dsl/eval.h>
# include <cuimg/dsl/typelist.h>

namespace cuimg
{



  // Check if A implements has.
  template <typename A>
  struct has_checker
  {
    // Return R
    template <typename R, typename X, bool (A::*)(const point2d<int>&) const>
    struct first
    {
      typedef R ret;
    };

    template <typename T>
    static typename first<int, T, &T::has>::ret test(int x);

    template <typename T> static char test(...);

    enum { val = sizeof(test<A>(0)) == sizeof(int) };
  };

  // Find the first type of the list A::L that implements
  // has.
  template <typename A, bool b, typename L, int acc>
  struct has_selector_;

  template <typename A, typename L, int acc>
  struct has_selector_<A, false, L, acc>
  {
    typedef typename L::head H;
    typedef typename L::tail T;
    enum { val = has_selector_<H, has_checker<H>::val, T, acc + 1>::val };
  };

  template <typename A, int acc>
  struct has_selector_<A, false, null_t, acc> {  enum { val = -1 };   };

  template <typename A, typename L, int acc>
  struct has_selector_<A, true, L, acc>       {  enum { val = acc };  };

  template <typename L>
  struct has_selector_helper
  {
    typedef typename L::head H;
    typedef typename L::tail T;

    enum { val = has_selector_<H, has_checker<H>::val, T, 0>::val };
  };

  // Call has on the ith argument.
  // Return 42 if i == -1.
  template <int i>
  struct has_;

  template <>
  struct has_<0>
  {
    template <typename A>
    static inline __host__ __device__ bool run(const point2d<int>& p, const A& a) { return a.has(p); }

    template <typename A, typename B>
    inline __host__ __device__ static bool run(const point2d<int>& p, const A& a, const B&) { return a.has(p); }

    template <typename A, typename B, typename C>
    inline __host__ __device__ static bool run(const point2d<int>& p, const A& a, const B&, const C&) { return a.has(p); }

    template <typename A, typename B, typename C, typename D>
    inline __host__ __device__ static bool run(const point2d<int>& p, const A& a, const B&, const C&, const D&) { return a.has(p); }
  };

  template <>
  struct has_<1>
  {
    template <typename A, typename B>
    inline __host__ __device__ static bool run(const point2d<int>& p, const A&, const B& b) { return b.has(p); }

    template <typename A, typename B, typename C>
    inline __host__ __device__ static bool run(const point2d<int>& p, const A&, const B& b, const C&) { return b.has(p); }

    template <typename A, typename B, typename C, typename D>
    inline __host__ __device__ static bool run(const point2d<int>& p, const A&, const B& b, const C&, const D&) { return b.has(p); }
  };

  template <>
  struct has_<2>
  {
    template <typename A, typename B, typename C>
    inline __host__ __device__ static bool run(const point2d<int>& p, const A&, const B&, const C& c) { return c.has(p); }

    template <typename A, typename B, typename C, typename D>
    inline __host__ __device__ static bool run(const point2d<int>& p, const A&, const B&, const C& c, const D&) { return c.has(p); }
  };

  template <>
  struct has_<3>
  {
    template <typename A, typename B, typename C, typename D>
    inline __host__ __device__
    static bool run(const point2d<int>& p, const A&, const B&, const C&, const D& d) { return d.has(p); }
  };

  template <>
  struct has_<-1>
  {
    template <typename A>
    static inline __host__ __device__ bool run(const point2d<int>& p, const A& a) { return true; }

    template <typename A, typename B>
    inline __host__ __device__ static bool run(const point2d<int>& p, const A& a, const B&) { return true; }

    template <typename A, typename B, typename C>
    inline __host__ __device__ static bool run(const point2d<int>& p, const A& a, const B&, const C&) { return true; }

    template <typename A, typename B, typename C, typename D>
    inline __host__ __device__ static bool run(const point2d<int>& p, const A& a, const B&, const C&, const D&) { return true; }
  };

  // Call the first element of {a, b} that implements
  // a method bool has() const. If none, return 42.
  template <typename A>
  inline __host__ __device__
  bool has(const point2d<int>& p, const A& a)
  {
    return has_<has_selector_helper<make_typelist_1(A) >::val>::run(p, a);
  }

  template <typename A, typename B>
  inline __host__ __device__
  bool has(const point2d<int>& p, const A& a, const B& b)
  {
    return has_<has_selector_helper<make_typelist_2(A, B) >::val>::run(p, a, b);
  }

  template <typename A, typename B, typename C>
  inline __host__ __device__
  bool has(const point2d<int>& p, const A& a, const B& b, const C& c)
  {
    return has_<has_selector_helper<make_typelist_3(A, B, C) >::val>::run(p, a, b, c);
  }

  template <typename A, typename B, typename C, typename D>
  inline __host__ __device__
  bool has(const point2d<int>& p, const A& a, const B& b, const C& c, const D& d)
  {
    return has_<has_selector_helper<make_typelist_4(A, B, C, D) >::val>::run(p, a, b, c, d);
  }

}

#endif
