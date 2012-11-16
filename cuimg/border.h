#ifndef CUIMG_BORDER_H_
# define CUIMG_BORDER_H_

# include <cuimg/gpu/cuda.h>
# include <cuimg/point2d.h>

namespace cuimg
{
  class border
  {
  public:

    __host__ __device__ inline border();
    __host__ __device__ inline border(int thickness)
      : thickness_(thickness)
    {
    }

    __host__ __device__ inline int thickness() const { return thickness_; }

  private:
    int thickness_;
  };

}

#endif
