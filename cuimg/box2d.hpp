#ifndef CUIMG_BOX2D_HPP_
# define CUIMG_BOX2D_HPP_

# include <cuimg/box2d.h>
# include <cuimg/obox2d.h>

namespace cuimg
{
  box2d::box2d()
  {
  }

  box2d::box2d(i_int2 a, i_int2 b)
    : a_(a),
      b_(b)
  {
  }

  box2d::box2d(const box2d& box)
    : a_(box.a_),
      b_(box.b_)
  {
  }


  box2d::box2d(const obox2d& obox)
    : a_(0,0),
      b_(obox.nrows() - 1, obox.ncols() - 1)
  {
  }

  box2d&
  box2d::operator=(const box2d& d)
  {
    a_ = d.a_;
    b_ = d.b_;
    return *this;
  }

  int box2d::nrows() const
  {
    return b_.r() - a_.r();
  }

  int box2d::ncols() const
  {
    return b_.c() - a_.c();
  }

  const i_int2& box2d::p1() const { return a_; }
  const i_int2& box2d::p2() const { return b_; }
  i_int2& box2d::p1() { return a_; }
  i_int2& box2d::p2() { return b_; }

  int
  box2d::size() const
  {
    return nrows() * ncols();
  }

  const i_int2
  box2d::center() const
  {
    return (a_ + b_) / 2;
  }

  bool box2d::has(const point2d<int>& p) const
  {
    return p.row() >= a_.r() && p.row() <= b_.r() &&
      p.col() >= a_.c() && p.col() <= b_.c();
  }


  void box2d::extend(const point2d<int>& p)
  {
    if (p.row() < a_.r()) a_.r() = p.row();
    if (p.row() > b_.r()) b_.r() = p.row();

    if (p.col() < a_.c()) a_.c() = p.col();
    if (p.col() > b_.c()) b_.c() = p.col();
  }


  void box2d::extend(const box2d& bb)
  {
    extend(bb.a_);
    extend(bb.b_);
  }


  box2d_iterator::box2d_iterator(i_int2 p, box2d b)
    : p_(p),
      line_start_(b.p1().c()),
      line_end_(b.p2().c() + 1)
  {}

  box2d_iterator::box2d_iterator(const box2d_iterator& mit)
    : p_(mit.p_)
  {}

  box2d_iterator&
  box2d_iterator::operator++()
  {
    if (p_.c() != (line_end_ - 1))
      p_.c()++;
    else
      p_ = i_int2(p_.r() + 1, line_start_);
    return *this;
  }

  box2d_iterator
  box2d_iterator::operator++(int)
  {
    box2d_iterator tmp(*this);
    operator++();
    return tmp;
  }

  bool
  box2d_iterator::operator==(const box2d_iterator& rhs)
  {
    return p_ == rhs.p_;
  }

  bool
  box2d_iterator::operator!=(const box2d_iterator& rhs)
  {
    return p_ != rhs.p_;
  }

  const i_int2&
  box2d_iterator::operator*()
  {
    return p_;
  }


  grid2d_iterator::grid2d_iterator(box2d cell, grid2d grid)
    : cell_(cell),
      line_start_(grid.domain().p1().c()),
      line_end_(grid.domain().p2().c() + 1),
      cell_nr_(grid.cell_nr()),
      cell_nc_(grid.cell_nc())
  {}

  grid2d_iterator::grid2d_iterator(const grid2d_iterator& mit)
    : cell_(mit.cell_),
      line_start_(mit.line_start_),
      line_end_(mit.line_end_),
      cell_nr_(mit.cell_nr_),
      cell_nc_(mit.cell_nc_)
  {}

  grid2d_iterator&
  grid2d_iterator::operator++()
  {
    if (cell_.p1().c() <= (line_end_ - cell_nc_))
    {
      cell_.p1().c() += cell_nc_;
      cell_.p2().c() += cell_nc_;
    }
    else
    {
      cell_.p1().c() = line_start_;
      cell_.p2().c() = line_start_ + cell_nc_;
      cell_.p1().r() += cell_nr_;
      cell_.p2().r() += cell_nr_;
    }
    return *this;
  }

  grid2d_iterator
  grid2d_iterator::operator++(int)
  {
    grid2d_iterator tmp(*this);
    operator++();
    return tmp;
  }

  bool
  grid2d_iterator::operator==(const grid2d_iterator& rhs)
  {
    return cell_ == rhs.cell_;
  }

  bool
  grid2d_iterator::operator!=(const grid2d_iterator& rhs)
  {
    return cell_ != rhs.cell_;
  }

  const box2d&
  grid2d_iterator::operator*()
  {
    return cell_;
  }

}

#endif
