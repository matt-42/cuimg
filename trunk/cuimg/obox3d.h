#ifndef CUIMG_OBOX3D_H_
# define CUIMG_OBOX3D_H_

# include <cuimg/gpu/cuda.h>
# include <cuimg/point3d.h>
# include <cuimg/obox3d.h>

namespace cuimg
{
  template <typename P>
  class obox3d
  {
  public:
    typedef point3d<int> point;

    __host__ __device__ inline obox3d();
    __host__ __device__ inline obox3d(unsigned nslices, unsigned nrows, unsigned ncols);
    __host__ __device__ inline obox3d(const obox3d<P>& d);

    __host__ __device__ inline obox3d<P>& operator=(const obox3d<P>& d);

    __host__ __device__ inline unsigned nslices() const;
    __host__ __device__ inline unsigned nrows() const;
    __host__ __device__ inline unsigned ncols() const;

    __host__ __device__ inline bool has(const point& p) const;

  private:
    unsigned nslices_;
    unsigned nrows_;
    unsigned ncols_;
  };

  template <typename P>
  inline bool operator==(const obox3d<P>& a, const obox3d<P>& b)
  {
    return a.nslices() == b.nslices() && a.nrows() == b.nrows() && a.ncols() == b.ncols();
  }

}

# include <cuimg/obox3d.hpp>

#endif
