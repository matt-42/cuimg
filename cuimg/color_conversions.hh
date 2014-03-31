#ifndef CUIMG_COLOR_CONVERSION_HH
# define CUIMG_COLOR_CONVERSION_HH

# include <cuimg/improved_builtin.h>


namespace cuimg
{

  i_uchar3 hsv_to_rgb(int h, float s, float v)
  {
    float c = s * v;
    float h2 = h / 60.f;
    float x = c * (1 - fabs(fmod(h2, 2) - 1));

    unsigned char C = c * 255;
    unsigned char X = x * 255;
    if (h2 < 1)
      return i_uchar3(C, X, 0);
    else if (h2 < 2)
      return i_uchar3(X, C, 0);
    else if (h2 < 3)
      return i_uchar3(0, C, X);
    else if (h2 < 4)
      return i_uchar3(0, X, C);
    else if (h2 < 5)
      return i_uchar3(X, 0, C);
    else if (h2 < 6)
      return i_uchar3(C, 0, X);
    else
      return i_uchar3(0,0,0);
  }

}

#endif
