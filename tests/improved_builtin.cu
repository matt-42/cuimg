#include <cassert>
#include <cuimg/improved_builtin.h>

using namespace cuimg;

int main()
{
  {
    i_uchar4 t(make_i_float4(1,1,2,3));
    assert(t == i_uchar4(1, 1, 2, 3));
    assert(t != i_uchar4(2, 1, 2, 3));

    i_uchar4 u(make_uchar4(1,2,3,4));
    assert(u == i_uchar4(1, 2, 3, 4));
    assert(u == i_float4(1, 2, 3, 4));

    t = i_uchar4(1,2,3,4);
    assert(t == i_uchar4(1, 2, 3, 4));

    t += u;
    assert(t == i_uchar4(2, 4, 6, 8));
    t -= u;
    assert(t == i_uchar4(1, 2, 3, 4));
    t *= 2;
    assert(t == i_uchar4(2, 4, 6, 8));
    t /= 2;
    assert(t == i_uchar4(1, 2, 3, 4));

    t = u + u;
    assert(t == i_uchar4(2, 4, 6, 8));
    t = t - u;
    assert(t == i_uchar4(1, 2, 3, 4));
    t = t * 2;
    assert(t == i_uchar4(2, 4, 6, 8));
    t = t / 2;
    assert(t == i_uchar4(1, 2, 3, 4));
  }

  {
    i_uchar4 t(0,0,0,0);
    i_uchar4 u(0,0,0,0);
    i_char1 u1(41);
    i_char2 u2(41, 41);
    i_char3 u3(41, 41, 41);
    i_float4 u4(41, 41, 4, 41);

    float x = u4[2];
    assert(x == 4);

    i_float4 z = t + u4;

    i_uchar4 uchar_test = z;
    z = u4;
    z = t + u4;

    assert(z == i_float4(41, 41, 4, 41));
    z = t - u4;
    assert(z == i_float4(-41, -41, -4, -41));
    z = i_float4(2,4,6,8);
    z = z / 2;
    assert(z == i_float4(1,2,3,4));
    z = z * 2;
    assert(z == i_float4(2,4,6,8));

    assert(u4 == make_i_float4(41, 41, 4, 41));
    assert(u4 == make_i_char4(41, 41, 4, 41));
    assert(u4 != make_i_float4(41, 41, 5, 41));
    assert(u4 != make_i_char4(41, 41, 5, 41));

    assert(u4 != make_i_char4(41, 5, 41, 2));

    u.w = 4;
    t = u;
    assert(t.w == 4);
    t += make_i_char4(0,0,0,0);
    t -= make_i_char4(0,0,0,0);
    t *= 42;
    t /= 42;
  }
}
