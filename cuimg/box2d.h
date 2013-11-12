#ifndef CUIMG_BOX2D_H_
# define CUIMG_BOX2D_H_

# include <cuimg/gpu/cuda.h>
# include <cuimg/point2d.h>

namespace cuimg
{

  class box2d;
  class obox2d;
  class box2d_iterator : public std::iterator<std::input_iterator_tag, i_int2, i_int2>
  {
    unsigned short line_start_;
    unsigned short line_end_;
    i_int2 p_;

  public:
    inline __host__ __device__ box2d_iterator(i_int2 p, box2d b);
    inline __host__ __device__ box2d_iterator(const box2d_iterator& mit);
    inline __host__ __device__ box2d_iterator& operator++();
    inline __host__ __device__ box2d_iterator operator++(int);
    inline __host__ __device__ bool operator==(const box2d_iterator& rhs);
    inline __host__ __device__ bool operator!=(const box2d_iterator& rhs);
    inline __host__ __device__ const i_int2& operator*();
  };

  class box2d
  {
  public:
    typedef box2d_iterator iterator;

    __host__ __device__ inline box2d();
    __host__ __device__ inline box2d(i_int2 a, i_int2 b);
    __host__ __device__ inline box2d(const obox2d& d);
    __host__ __device__ inline box2d(const box2d& d);

    __host__ __device__ inline box2d& operator=(const box2d& d);

    __host__ __device__ inline int nrows() const;
    __host__ __device__ inline int ncols() const;

    __host__ __device__ inline const i_int2& p1() const;
    __host__ __device__ inline const i_int2& p2() const;
    __host__ __device__ inline i_int2& p1();
    __host__ __device__ inline i_int2& p2();
    __host__ __device__ inline const i_int2 center() const;
    __host__ __device__ inline int size() const;

    __host__ __device__ inline bool has(const point2d<int>& p) const;

    __host__ __device__ inline void extend(const point2d<int>& p);
    __host__ __device__ inline void extend(const box2d& bb);

    __host__ __device__ inline iterator begin() const  { return iterator(p1(), *this); }
    __host__ __device__ inline iterator end() const { return iterator(p1() + i_int2(nrows(), 0), *this); }

  private:
    i_int2 a_;
    i_int2 b_;
  };


  inline __host__ __device__
  bool operator==(const box2d& a, const box2d& b)
  {
    return a.p1() == b.p1() && a.p2() == b.p2();
  }

  inline __host__ __device__
  bool operator!=(const box2d& a, const box2d& b)
  {
    return a.p1() != b.p1() || a.p2() != b.p2();
  }

  inline __host__ __device__
  box2d operator/(const box2d& a, const float x)
  {
    return box2d(a.p1(), a.p1() + (a.p2() - a.p1()) / x);
  }

  inline __host__ __device__
  box2d domain_div_up(const box2d& a, const float x)
  {
    return box2d(a.p1(),
		 a.p1() + i_int2(idivup(a.p2().r() - a.p1().r(), x),
				 idivup(a.p2().c() - a.p1().c(), x)));
  }

  class grid2d;
  class grid2d_iterator : public std::iterator<std::input_iterator_tag, i_int2, i_int2>
  {
    unsigned short line_start_;
    unsigned short line_end_;
    unsigned short cell_nr_;
    unsigned short cell_nc_;
    box2d cell_;

  public:
    inline __host__ __device__ grid2d_iterator(box2d cell, grid2d b);
    inline __host__ __device__ grid2d_iterator(const grid2d_iterator& mit);
    inline __host__ __device__ grid2d_iterator& operator++();
    inline __host__ __device__ grid2d_iterator operator++(int);
    inline __host__ __device__ bool operator==(const grid2d_iterator& rhs);
    inline __host__ __device__ bool operator!=(const grid2d_iterator& rhs);
    inline __host__ __device__ const box2d& operator*();
  };

  class grid2d
  {
  public:
    typedef grid2d_iterator iterator;

    grid2d(box2d b, int cell_nr, int cell_nc)
      : b_(b),
      cell_nr_(cell_nr),
      cell_nc_(cell_nc)
    {
      int nr = idivup(b.nrows(), cell_nr_);
      end_ = box2d(b.p1() + i_int2(cell_nr_ * nr, 0),
		   b.p1() + i_int2(cell_nr_ * (nr+1), cell_nc_));
    }

    const box2d& domain() const { return b_; }
    const unsigned short cell_nr() { return cell_nr_; }
    const unsigned short cell_nc() { return cell_nc_; }

    __host__ __device__ inline iterator begin() const {
      return iterator(box2d(b_.p1(), b_.p1() + i_int2(cell_nr_, cell_nc_)), *this);
    }

    __host__ __device__ inline iterator end() const { return iterator(end_, *this); }

  private:
    box2d b_;
    unsigned short cell_nr_;
    unsigned short cell_nc_;
    box2d end_;

  };
}

# include <cuimg/box2d.hpp>

#endif
