#ifndef CUIMG_SORT_H_
# define CUIMG_SORT_H_

# include <cuimg/improved_builtin.h>
# include <cuimg/obox2d.h>
# include <cuimg/obox3d.h>
# include <cuimg/error.h>

namespace cuimg
{
  template <typename T>
  struct basic_sort
  {
    basic_sort(T* vector);
    T& operator[](int i) { return v[i]; }
    bool sort(int i, int j)
    {
      T a = v[i];
      T b = v[j];
      if (a > b)
      {
        v[i] = b;
        v[j] = a;
        return true;
      }
      return false;
    }

    T* v;
  };

  template <typename T, unsigned size>
  __device__ void sort(T& v)
  {
    __shared__ bool change[DIMBLOCK_X];
#pragma unroll
    for (int g = 5; g >= 1; g--)
    {
      change[threadIdx.x] = 0;
      for (int i = 0; i < (size / 2) + DIMBLOCK_X; i += DIMBLOCK_X)
      {
        int ti = i + threadIdx.x;
        int x = (ti / (g + 1)) * 2 * (g + 1) + ti % (g + 1);
        if ((x + g) < size)
          change[threadIdx.x] = v.sort(x, x + g) || change[threadIdx.x];
      }
      __synthreads();
      bool notsorted = 0;
      for (int k = 0; k < DIMBLOCK_X; k++)
        notsorted = notsorted || change[k];
      if (!notsorted)
        return;
    }

    return;

    bool notsorted = 1;
    while (notsorted)
    {
      int g = 1;
      change[threadIdx.x] = 0;
      for (int i = 0; i < (size / 2) + DIMBLOCK_X; i += DIMBLOCK_X)
      {
        int ti = i + threadIdx.x;
        int x = (ti / g) * 2 * g + ti % g;
        if ((x + g) < size)
          change[threadIdx.x] = v.sort(x, x + g) || change[threadIdx.x];
      }
      __synthreads();
      bool notsorted = 0;
      for (int k = 0; k < DIMBLOCK_X; k++)
        notsorted = notsorted || change[k];
      notsorted = 1; // FIXME.
    }
  }

}

#endif
