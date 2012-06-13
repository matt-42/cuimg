#ifndef CUIMG_TRANSLATION_STABILIZER_H_
# define CUIMG_TRANSLATION_STABILIZER_H_

# include <cuimg/obox2d.h>

namespace cuimg
{

  class translation_stabilizer
  {
  public:
    typedef obox2d D;

    translation_stabilizer();
    ~translation_stabilizer();

    template <typename V>
    inline void      update(const V& particles);
    inline i_float2    correction() const;
    inline const host_image2d<int>& histogram() const;
    inline const int& histogram_max() const;

  private:
    i_float2 correction_;
    i_float2 speed_;
    int hmax_;
    host_image2d<int> histo;
  };

}

# include <cuimg/tracking/translation_stabilizer.hpp>

#endif // !CUIMG_TRANSLATION_STABILIZER_H_
