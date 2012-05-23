#ifndef CUIMG_RDEPTH_ESTIMATOR_H_
# define CUIMG_RDEPTH_ESTIMATOR_H_

# include <cuimg/obox2d.h>

namespace cuimg
{

  class rdepth_estimator
  {
  public:
    typedef obox2d D;

    rdepth_estimator();
    ~rdepth_estimator();

    template <typename V>
    inline void update(V& particles);

    void set_foe(const i_short2& p);

  private:
    i_short2 foe_;
  };

}

# include <cuimg/tracking/rdepth_estimator.hpp>

#endif // !CUIMG_RDEPTH_ESTIMATOR_H_
