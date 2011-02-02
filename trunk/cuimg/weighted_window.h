#ifndef CUIMG_WEIGHTED_WINDOW_H_
# define CUIMG_WEIGHTED_WINDOW_H_

# include <boost/shared_array.hpp>
# include <cuimg/improved_builtin.h>
# include <cuimg/point2d.h>
# include <cuimg/error.h>

namespace cuimg
{
  class weighted_window
  {
  public:

    inline weighted_window(unsigned size)
    : dpoints_(new point2d<int>[size]),
      weights_(new float[size]),
      size_(size)
      {
      }

  inline weighted_window()
    : size_(0)
  {
  }

  inline weighted_window(const weighted_window& ww)
    : dpoints_(ww.dpoints_),
    weights_(ww.weights_),
    size_(ww.size_)
  {
  }

  inline weighted_window& operator=(const weighted_window& ww)
  {
    dpoints_ = ww.dpoints_;
    weights_ = ww.weights_;
    size_ = ww.size_;
    return *this;
  }

  inline void resize(unsigned size)
  {
    if (size_ == size)
      return;
    size_ = size;
    dpoints_.reset(new point2d<int>[size_]);
    weights_.reset(new float[size_]);
  }

  inline void fill_dpoints(point2d<int>* dps) { memcpy(dpoints_.get(), dps, size_ * sizeof(point2d<int>)); }
  inline void fill_weights(float* ws) { memcpy(weights_.get(), ws, size_ * sizeof(float)); }

  inline point2d<int> dpoints(int i)const { return dpoints_[i]; }
  inline float weights(int i) const { return weights_[i]; }
  inline point2d<int>& dpoints(int i) { return dpoints_[i]; }
  inline float& weights(int i) { return weights_[i]; }
  inline unsigned size() const { return size_; }

  private:
    boost::shared_array<point2d<int> > dpoints_;
    boost::shared_array<float> weights_;
    unsigned size_;

  };
}

#endif
