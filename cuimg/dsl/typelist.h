#ifndef CUIMG_TYPELIST
# define CUIMG_TYPELIST

namespace cuimg
{
  struct null_t {};

  template <typename H, typename T>
  struct list
  {
    typedef H head;
    typedef T tail;
  };

  template <typename L, unsigned i>
  struct list_get
  {
    typedef typename list_get<typename L::tail, i-1>::ret ret;
  };

  template <typename L>
  struct list_get<L, 0>
  {
    typedef typename L::head ret;
  };


  template <typename A1 = null_t, typename A2 = null_t, typename A3 = null_t,
            typename A4 = null_t, typename A5 = null_t, typename A6 = null_t>
  struct make_typelist
  {
    typedef list<A1, typename make_typelist<A2, A3, A4, A5, A6>::ret> ret;
  };

  template <> struct make_typelist<null_t, null_t, null_t, null_t, null_t, null_t> { typedef null_t ret; };


#define make_typelist_1(A) typename make_typelist<A>::ret
#define make_typelist_2(A, B) typename make_typelist<A, B>::ret
#define make_typelist_3(A, B, C) typename make_typelist<A, B, C>::ret
#define make_typelist_4(A, B, C, D) typename make_typelist<A, B, C, D>::ret

}

#endif
