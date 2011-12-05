#ifndef CUIMG_GPU_EXPR_H_
# define CUIMG_GPU_EXPR_H_

namespace cuimg
{

  template <typename E>
  struct expr
  {
    typedef int is_expr;

    __host__ __device__ E& subcast()
    {
      return *static_cast<E*>(this);
    }

    __host__ __device__ const E& subcast() const
    {
      return *static_cast<const E*>(this);
    }

  };

}

#endif
