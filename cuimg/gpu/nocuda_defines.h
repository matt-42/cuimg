#ifndef CUIMG_NOCUDA_DEFINES_H_
# define CUIMG_NOCUDA_DEFINES_H_

# define __global__
# define __host__
# define __device__

# define __align__
# define __constant__

// Builtins

#define DEFINE_BUILTIN4(CUT, T)                                         \
struct CUT##1 { T x; };                                                 \
inline CUT##1 make_##CUT##1(T x) { CUT##1 r; r.x = x; return r; }              \
struct CUT##2 { T x; T y; };                                            \
inline CUT##2 make_##CUT##2(T x, T y) { CUT##2 r; r.x = x; r.y = y; return r; } \
struct CUT##3 { T x; T y; T z; };                                       \
inline CUT##3 make_##CUT##3(T x, T y, T z) { CUT##3 r; r.x = x; r.y = y; r.z = z; return r; } \
struct CUT##4 { T x; T y; T z; T w; };                                  \
inline CUT##4 make_##CUT##4(T x, T y, T z, T w) { CUT##4 r; r.x = x; r.y = y; r.z = z; r.w = w; return r; }

#define DEFINE_BUILTIN2(CUT, T)                                         \
struct CUT##1 { T x; };                                                 \
inline CUT##1 make_##CUT##1(T x) { CUT##1 r; r.x = x; return r; }              \
struct CUT##2 { T x; T y; };                                            \
inline CUT##2 make_##CUT##2(T x, T y) { CUT##2 r; r.x = x; r.y = y; return r; }

DEFINE_BUILTIN4(char, signed char);
DEFINE_BUILTIN4(uchar, unsigned char);
DEFINE_BUILTIN4(short, short);
DEFINE_BUILTIN4(ushort, unsigned short);
DEFINE_BUILTIN4(int, int);
DEFINE_BUILTIN4(uint, unsigned int);
DEFINE_BUILTIN4(long, long);
DEFINE_BUILTIN4(ulong, unsigned long);
DEFINE_BUILTIN4(float, float);

DEFINE_BUILTIN2(longlong, long long);
DEFINE_BUILTIN2(ulonglong, unsigned long long);
DEFINE_BUILTIN2(double, double);

struct dim3
{
  dim3() {}
  inline dim3(int x_, int y_ = 1, int z_ = 1) : x (x_), y(y_), z(z_) {}

  int x;
  int y;
  int z;
};

typedef int cudaStream_t;

enum cudaTextureReadMode { cudaReadModeElementType };

typedef int textureReference;

#endif // ! CUIMG_NOCUDA_DEFINES_H_
