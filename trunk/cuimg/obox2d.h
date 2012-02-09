#ifndef CUIMG_OBOX2D_H_
# define CUIMG_OBOX2D_H_

# include <cuda_runtime.h>
# include <cuimg/point2d.h>
# include <cuimg/obox2d.h>

namespace cuimg
{
  template <typename P>
  class obox2d
  {
  public:
    typedef point2d<int> point;

    __host__ __device__ inline obox2d();
    __host__ __device__ inline obox2d(unsigned nrows, unsigned ncols);
    __host__ __device__ inline obox2d(const obox2d<P>& d);

    __host__ __device__ inline obox2d<P>& operator=(const obox2d<P>& d);

    __host__ __device__ inline unsigned nrows() const;
    __host__ __device__ inline unsigned ncols() const;

    __host__ __device__ inline bool has(const point& p) const;

  private:
    unsigned short nrows_;
    unsigned short ncols_;
  };

  template <typename P>
  inline bool operator==(const obox2d<P>& a, const obox2d<P>& b)
  {
    return a.nrows() == b.nrows() && a.ncols() == b.ncols();
  }

}

# include <cuimg/obox2d.hpp>

#endif
