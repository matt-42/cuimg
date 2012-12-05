#ifndef CUIMG_FAST_TOOLS_H_
# define CUIMG_FAST_TOOLS_H_

namespace cuimg
{

  inline
  __device__ i_float4 int_to_color(int r)
  {
    float v = 1.f;
    float s = 0.8f;

    int h = r;
    int hi = (h / 60) % 6;
    float f =  (h / 60.f) - floor(h / 60.f);
    float p = v * (1.0 - s);
    float q = v * (1.0 - (f*s));
    float t = v * (1.0 - ((1.0 - f) * s));

    switch (hi)
    {
      case 0: return i_float4(v, t, p, 1.f);
      case 1: return i_float4(q, v, p, 1.f);
      case 2: return i_float4(p, v, t, 1.f);
      case 3: return i_float4(p, q, v, 1.f);
      case 4: return i_float4(t, p, v, 1.f);
      default: return i_float4(v, p, q, 1.f);
    }
  }

}

#endif // ! CUIMG_FAST_TOOLS_H_
