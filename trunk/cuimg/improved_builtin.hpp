
#include <ostream>
#include <boost/type_traits/is_scalar.hpp>

#include <cuimg/meta.h>

namespace cuimg
{

  namespace internal
  {
    // --------------------------- Unrolling operators ---------------------------------
    template <int I>
    class assign
    {
    public:
      template <typename U, typename V, unsigned N, unsigned M>
      static __device__ __host__ inline void run(improved_builtin<U, N>& a, const improved_builtin<V, M>& b)
      {
        bt_getter<I>::get(a) =  bt_getter<I>::get(b);
      }
      template <typename U, typename V, unsigned N>
      static __device__ __host__ inline void run(improved_builtin<U, N>& a, const V b)
      {
        bt_getter<I>::get(a) =  bt_getter<I>::get(b);
      }
    };

    template <int I>
    class assign_explicit
    {
    public:
      template <typename U, typename V, unsigned N, unsigned M>
      static __device__ __host__ inline void run(improved_builtin<U, N>& a, const improved_builtin<V, M>& b)
      {
        bt_getter<I>::get(a) = typename improved_builtin<U, N>::vtype(bt_getter<I>::get(b));
      }
    };

    template <int I>
    class reset
    {
    public:
      template <typename U, unsigned N>
      static __device__ __host__ inline void run(improved_builtin<U, N>& a)
      {
        bt_getter<I>::get(a) = 0;
      }
    };

    template <int I>
    class self_add
    {
    public:
      template <typename U, typename V, unsigned N, unsigned M>
      static __device__ __host__ inline void run(improved_builtin<U, N>& a, const improved_builtin<V, M>& b)
      {
        bt_getter<I>::get(a) += bt_getter<I>::get(b);
      }
    };

    template <int I>
    class self_minus
    {
    public:
      template <typename U, typename V, unsigned N, unsigned M>
      static __device__ __host__ inline void run(improved_builtin<U, N>& a, const improved_builtin<V, M>& b)
      {
        bt_getter<I>::get(a) -= bt_getter<I>::get(b);
      }
    };

    template <int I>
    class self_mult
    {
    public:
      template <typename U, typename V, unsigned N>
      static __device__ __host__ inline void run(improved_builtin<U, N>& a, const V& b)
      {
        bt_getter<I>::get(a) *= b;
      }
    };

    template <int I>
    class self_div
    {
    public:
      template <typename U, typename V, unsigned N>
      static __device__ __host__ inline void run(improved_builtin<U, N>& a, const V& b)
      {
        bt_getter<I>::get(a) /= b;
      }
    };


    template <int I, int E>
    class equal
    {
    public:
      template <typename U, typename V, unsigned N, unsigned M>
      static __device__ __host__ inline bool run(const improved_builtin<U, N>& a, const improved_builtin<V, M>& b)
      {
        return bt_getter<I>::get(a) == bt_getter<I>::get(b) && equal<I + 1, E>::run(a, b);
      }
    };
    template <int E>
    class equal<E, E>
    {
    public:
      template <typename U, typename V, unsigned N, unsigned M>
      static __device__ __host__ inline bool run(const improved_builtin<U, N>& a, const improved_builtin<V, M>& b)
      {
        return bt_getter<E>::get(a) == bt_getter<E>::get(b);
      }
    };

    template <int I>
    struct add
    {
      template <typename U, typename V, typename W, unsigned US, unsigned VS, unsigned WS>
      static __device__ __host__ inline void run(const improved_builtin<U, US>& a, const improved_builtin<V, VS>& b, improved_builtin<W, WS>& out)
      {
        bt_getter<I>::get(out) = bt_getter<I>::get(a) + bt_getter<I>::get(b);
      }
    };

    template <int I>
    struct minus
    {
      template <typename U, typename V, typename W, unsigned US, unsigned VS, unsigned WS>
      static __device__ __host__ inline void run(const improved_builtin<U, US>& a, const improved_builtin<V, VS>& b, improved_builtin<W, WS>& out)
      {
        bt_getter<I>::get(out) = bt_getter<I>::get(a) - bt_getter<I>::get(b);
      }
    };

    template <int I>
    struct scalar_mult
    {
      template <typename U, typename V, typename W, unsigned US, unsigned WS>
      static __device__ __host__ inline void run(const improved_builtin<U, US>& a, const V& b, improved_builtin<W, WS>& out)
      {
        bt_getter<I>::get(out) = bt_getter<I>::get(a) * b;
      }
    };

    template <int I>
    struct scalar_div
    {
      template <typename U, typename V, typename W, unsigned US, unsigned WS>
      static __device__ __host__ inline void run(const improved_builtin<U, US>& a, const V& b, improved_builtin<W, WS>& out)
      {
        bt_getter<I>::get(out) = bt_getter<I>::get(a) / b;
      }
    };

    // --------------------------- End of unrolling operators ---------------------------------

  }

  //--------------------------- improved_builtin members. ----------------------------------

  template <typename T, unsigned N>
  improved_builtin<T, N>::improved_builtin()
  {
  }

  template <typename T, unsigned N>
  improved_builtin<T, N>::improved_builtin(const cuda_bt& bt)
  {
    meta::loop<internal::assign, 0, size - 1>::iter(*this, bt);
  }

  template <typename T, unsigned N>
  improved_builtin<T, N>::improved_builtin(vtype x_)
  {
    meta::equal<meta::int_<size>, meta::int_<1> >::check(); // Member reserved for builtins of size 1.
    this->x = x_;
  }

  template <typename T, unsigned N>
  improved_builtin<T, N>& improved_builtin<T, N>::operator=(vtype x_)
  {
    meta::equal<meta::int_<size>, meta::int_<1> >::check(); // Member reserved for builtins of size 1.
    this->x = x_;
    return *this;
  }

  template <typename T, unsigned N>
  improved_builtin<T, N>::operator vtype()
  {
    meta::equal<meta::int_<size>, meta::int_<1> >::check(); // Member reserved for builtins of size 1.
    return this->x;
  }

  template <typename T, unsigned N>
  improved_builtin<T, N>::improved_builtin(vtype x_, vtype y_)
  {
    meta::equal<meta::int_<size>, meta::int_<2> >::check(); // Member reserved for builtins of size 2.
    this->x = x_;
    this->y = y_;
  }

  template <typename T, unsigned N>
  improved_builtin<T, N>::improved_builtin(vtype x_, vtype y_, vtype z_)
  {
    meta::equal<meta::int_<size>, meta::int_<3> >::check(); // Member reserved for builtins of size 3.
    this->x = x_;
    this->y = y_;
    this->z = z_;
  }

  template <typename T, unsigned N>
  improved_builtin<T, N>::improved_builtin(vtype x_, vtype y_, vtype z_, vtype w_)
  {
    meta::equal<meta::int_<size>, meta::int_<4> >::check(); // Member reserved for builtins of size 4.
    this->x = x_;
    this->y = y_;
    this->z = z_;
    this->w = w_;
  }

  template <typename T, unsigned N>
  improved_builtin<T, N>::improved_builtin(const zero&)
  {
    meta::loop<internal::reset, 0, size - 1>::iter(*this);
  }

  template <typename T, unsigned N>
  improved_builtin<T, N>&
  improved_builtin<T, N>::operator=(const zero&)
  {
    meta::loop<internal::reset, 0, size - 1>::iter(*this);
    return *this;
  }

  template <typename T, unsigned N>
  template <typename U, unsigned US>
  improved_builtin<T, N>::improved_builtin(const improved_builtin<U, US>& bt)
  {
    meta::infeq_<size, US>::check();
    meta::loop<internal::assign_explicit, 0, size - 1>::iter(*this, bt);
  }

  template <typename T, unsigned N>
  template <typename U, unsigned US>
  improved_builtin<T, N>&
  improved_builtin<T, N>::operator=(const improved_builtin<U, US>& bt)
  {
    meta::equal<meta::int_<size>, meta::int_<US> >::check();
    meta::loop<internal::assign, 0, size - 1>::iter(*this, bt);
    return *this;
  }

//   template <typename T, unsigned N>
//   template <typename U, unsigned M, int OUT_START_POS, int IN_START_POS>
//   improved_builtin<T, N>&
//     improved_builtin<T, N>::assign(const improved_builtin<U, M>& bt)
//   {
//     for (int i; i < IN_START_POS + improved_builtin<U, M>::size &&
//       i < OUT_START_POS + size ; i++)
//       (*this)[i] = vtype(bt[i]);
//     return *this;
//   }

  template <typename T, unsigned N>
  const typename improved_builtin<T, N>::vtype&
  improved_builtin<T, N>::operator[](unsigned i) const
  {
    return *((vtype*)this + i);
  }


  template <typename T, unsigned N>
  typename improved_builtin<T, N>::vtype&
    improved_builtin<T, N>::operator[](unsigned i)
  {
    return *((vtype*)this + i);
  }

  template <typename T, unsigned N>
  template <typename U, unsigned US>
  improved_builtin<T, N>&
    improved_builtin<T, N>::operator+=(const improved_builtin<U, US>& bt)
  {
    meta::loop<internal::self_add, 0, size - 1>::iter(*this, bt);
    return *this;
  }

  template <typename T, unsigned N>
  template <typename U, unsigned US>
  improved_builtin<T, N>&
    improved_builtin<T, N>::operator-=(const improved_builtin<U, US>& bt)
  {
    meta::loop<internal::self_minus, 0, size - 1>::iter(*this, bt);
    return *this;
  }

  template <typename T, unsigned N>
  template <typename S>
  improved_builtin<T, N>&
    improved_builtin<T, N>::operator*=(const S& s)
  {
    meta::loop<internal::self_mult, 0, size - 1>::iter(*this, s);
    return *this;
  }

  template <typename T, unsigned N>
  template <typename S>
  improved_builtin<T, N>&
    improved_builtin<T, N>::operator/=(const S& s)
  {
    meta::loop<internal::self_div, 0, size - 1>::iter(*this, s);
    return *this;
  }

  template <typename U, typename V, unsigned US, unsigned VS>
  bool operator==(const improved_builtin<U, US>& a, const improved_builtin<V, VS>& b)
  {
    meta::equal<meta::int_<US>, meta::int_<VS> >::check(); // a and b must have the same size.
    return internal::equal<0, US - 1>::run(a, b);
  }

  template <typename U, typename V, unsigned US, unsigned VS>
  bool operator!=(const improved_builtin<U, US>& a, const improved_builtin<V, VS>& b)
  {
    meta::equal<meta::int_<US>, meta::int_<VS> >::check(); // a and b must have the same size.
    return !(a == b);
  }

  template <typename U, typename V>
  bool operator<(const improved_builtin<U, 1>& a, const improved_builtin<V, 1>& b)
  {
    return (a.x < b.x);
  }


  template <typename U, typename V, unsigned US, unsigned VS>
  __host__ __device__ inline
  make_bt(type_add(U, V), US)
    operator+(const improved_builtin<U, US>& a, const improved_builtin<V, VS>& b)
  {
    meta::equal<meta::int_<US>, meta::int_<VS> >::check(); // a and b must have the same size.

    make_bt(type_add(U, V), US) ret;
    meta::loop<internal::add, 0, US - 1>::iter(a, b, ret);
    return ret;
  }

  template <typename U, typename V, unsigned US, unsigned VS>
  __host__ __device__ inline
  make_bt(type_minus(U, V), US)
    operator-(const improved_builtin<U, US>& a, const improved_builtin<V, VS>& b)
  {
    meta::equal<meta::int_<US>, meta::int_<VS> >::check(); // a and b must have the same size.

    make_bt(type_minus(U, V), US) ret;
    meta::loop<internal::minus, 0, VS - 1>::iter(a, b, ret);
    return ret;
  }

  template <typename U, unsigned US, typename S>
  __host__ __device__ inline
  typename meta::try_<typename meta::equal_<boost::is_scalar<S>::value, true>::checktype,
                      make_bt(type_mult(U, S), US)>::ret
  operator*(const improved_builtin<U, US>& a,
            const S& s)
  {
    make_bt(type_mult(U, S), US) ret;
    meta::loop<internal::scalar_mult, 0, US - 1>::iter(a, s, ret);
    return ret;
  }

  template <typename U, unsigned US, typename S>
  __host__ __device__ inline
    typename meta::try_<typename meta::equal_<boost::is_scalar<S>::value, true>::checktype,
                        make_bt(type_div(U, S), US)>::ret
    operator/(const improved_builtin<U, US>& a,
    const S& s)
  {
    make_bt(type_div(U, S), US) ret;
    meta::loop<internal::scalar_div, 0, US - 1>::iter(a, s, ret);
    return ret;
  }

  template <typename U, unsigned US>
  __host__ __device__ inline
  std::ostream&
  operator<<(std::ostream& os, const improved_builtin<U, US>& a)
  {
    os << '<';
    os << a[0];
    for (unsigned i = 1; i < US; i++)
      os << ", " << a[i];
    os << '>';
    return os;
  }

}
