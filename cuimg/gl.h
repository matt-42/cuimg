#ifndef CUIMG_GL_H_
# define CUIMG_GL_H_

# include <cuimg/hd.h>
# include <cuimg/improved_builtin.h>
# include <cuimg/value_traits.h>

namespace cuimg
{

  struct gl8u;
  struct gl01f;

  struct gl8u : public i_uchar1
  {
  public:
    typedef i_uchar1 super;

    H_D_I gl8u();

    template <typename T>
    H_D_I gl8u(const T& x);
    H_D_I gl8u(const zero& x);
    explicit H_D_I gl8u(const gl8u& x);
    explicit H_D_I gl8u(const gl01f& x);

    template <typename T>
    H_D_I gl8u& operator=(const T& x);
    H_D_I gl8u& operator=(const gl8u& x);
    H_D_I gl8u& operator=(const zero& x);
    H_D_I gl8u& operator=(const gl01f& x);

    template <typename T>
    H_D_I operator T() const;
    H_D_I operator gl01f() const;
  };

  struct gl01f : public i_float1
  {
  public:
    typedef i_float1 super;

    H_D_I gl01f();

    template <typename T>
    H_D_I gl01f(const T& x);
    explicit H_D_I gl01f(const gl8u& x);
    explicit H_D_I gl01f(const gl01f& x);
    H_D_I gl01f(const zero&);

    template <typename T>
    H_D_I gl01f& operator=(const T& x);
    H_D_I gl01f& operator=(const gl01f& x);
    H_D_I gl01f& operator=(const gl8u& x);
    H_D_I gl01f& operator=(const zero& x);

    template <typename T>
    H_D_I operator T() const;
    H_D_I operator gl8u() const;
  };


  template <typename NC>
  struct change_coord_type<NC, gl8u>
  {
    typedef improved_builtin<NC, 1> ret;
  };

  template <>
  struct bt_vtype<gl8u> { typedef unsigned char ret; };


  template <>
  struct bt_vtype<gl01f> { typedef float ret; };

  template <typename NC>
  struct change_coord_type<NC, gl01f>
  {
    typedef improved_builtin<NC, 1> ret;
  };

  H_D_I gl8u::gl8u() {}

  template <typename T>
  H_D_I gl8u::gl8u(const T& x) : super(x) {}
  H_D_I gl8u::gl8u(const zero& x) : super(x) {}
  H_D_I gl8u::gl8u(const gl8u& x) : super(x.x) {}
  H_D_I gl8u::gl8u(const gl01f& x) : super(float(x) * 255.f) {}

  template <typename T>
  H_D_I gl8u& gl8u::operator=(const T& x)       { super::operator=(x);  return *this; }
  H_D_I gl8u& gl8u::operator=(const gl8u& x)    { super::operator=(x.x);  return *this; }
  H_D_I gl8u& gl8u::operator=(const gl01f& x)    { super::operator=(x.x * 255.f);  return *this; }
  H_D_I gl8u& gl8u::operator=(const zero& x)    { super::operator=(x); return *this; }

  template <typename T>
  H_D_I gl8u::operator T() const { return T(x); }
  H_D_I gl8u::operator gl01f() const { return float(x) / 255.f; }


  H_D_I gl01f::gl01f() {}
  template <typename T>
  H_D_I gl01f::gl01f(const T& x) : super(x) {}
  H_D_I gl01f::gl01f(const gl8u& x) : super(float(x) / 255.f) {}
  H_D_I gl01f::gl01f(const gl01f& x) : super(x.x) {}
  H_D_I gl01f::gl01f(const zero&) : super(0.f) {}

  template <typename T>
  H_D_I gl01f& gl01f::operator=(const T& x) { super::operator=(x); return *this; }
  H_D_I gl01f& gl01f::operator=(const gl01f& x) { super::operator=(x.x); return *this; }
  H_D_I gl01f& gl01f::operator=(const gl8u& x) { super::operator=(x.x / 255.f); return *this; }
  H_D_I gl01f& gl01f::operator=(const zero& x) { super::operator=(0.f); return *this; }

  template <typename T>
  H_D_I gl01f::operator T() const { return x; }
  H_D_I gl01f::operator gl8u() const { return float(x) * 255.f; }

}

#endif
