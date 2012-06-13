#ifndef CUIMG_SWAP_H_
# define CUIMG_SWAP_H_

namespace cuimg
{

  template <typename T>
  __host__  __device__ void swap(T& a, T& b)
  {
    T tmp = a;
    a = b;
    b = tmp;
  }

}

#endif
