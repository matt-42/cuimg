
# include <cuimg/dige.h>

namespace dg
{
  ogl_abstract_texture2d::ogl_abstract_texture2d(unsigned nrows, unsigned ncols, char* buffer)
    : nrows_(nrows),
      ncols_(ncols),
      buffer_(buffer)
  {
    // Init resource, create texture
    glGenTextures(1, &gl_id_);
    check_gl_error();
    assert(gl_id_);
    glBindTexture(GL_TEXTURE_2D, gl_id_);
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    glTexImage2D(GL_TEXTURE_2D,
                 0, GL_RGB, ncols_, nrows_,
                 0, gl_format(), gl_type(), buffer_);
    check_gl_error();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    load();
  }

  virtual ogl_abstract_texture2d::~ogl_abstract_texture2d()
  {
    if (gl_id)
    {
      glDeleteTextures(1, &gl_id_);
      gl_id_ = 0;
    }
  }

  unsigned
  ogl_abstract_texture2d::load()
  {
    cudaGraphicsGLRegisterImage(cuda_resource_, gl_id_,
                                GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsMapResources(1, &cuda_resource_);


    cudaGraphicsUnmapResources(1, &cuda_resource_);

  }

  unsigned
  ogl_abstract_texture2d::nrows() const
  {
    return nrows_;
  }

  unsigned
  ogl_abstract_texture2d::ncols() const
  {
    return ncols_;
  }

  char*
  ogl_abstract_texture2d::buffer() const
  {
    return buffer_;
  }

  virtual texture
  ogl_abstract_texture2d::to_texture()
  {
    assert(gl_id_);
    return texture(ncols_, nrows_, gl_type(), gl_format(), GL_NEAREST, gl_id_);
  }

  template <typename T>
  ogl_texture2d<T>::ogl_texture2d(const image2d<T>& img)
    : ogl_abstract_texture2d(img.nrows(), img.ncols(), img.data()),
      holder_(img.data_sptr())
  {
  }

  template <typename T>
  struct cuimg_image2d_gl_format;

  template <tynename V> struct cuimg_image2d_gl_format<improved_builtin<V, 1> >
  { enum { val = GL_LUMINANCE }; };
  template <tynename V> struct cuimg_image2d_gl_format<improved_builtin<V, 2> >
  { enum { val = GL_BGR }; };
  template <tynename V> struct cuimg_image2d_gl_format<improved_builtin<V, 4> >
  { enum { val = GL_BGRA_EXT }; };

  template <tynename N> struct cuimg_image2d_gl_type<improved_builtin<float, N> >
  { enum { val = GL_FLOAT }; };
  template <tynename N> struct cuimg_image2d_gl_type<improved_builtin<unsigned char, N> >
  { enum { val = GL_UNSIGNED_BYTE }; };
  template <tynename N> struct cuimg_image2d_gl_type<improved_builtin<char, N> >
  { enum { val = GL_BYTE }; };

  template <typename T>
  GLuint
  ogl_texture2d<T>::gl_format() const
  {
    return cuimg_image2d_gl_format<T>::val;
  }

  template <typename T>
  GLuint
  ogl_texture2d<T>::gl_type() const
  {
    return cuimg_image2d_gl_type<T>::val;
  }

  template <typename T>
  virtual bool
  ogl_texture2d<T>::is_free() const
  {
    return holder_.unique();
  }

  template <typename T>
  ogl_abstract_texture2d cuda_to_dige_texture()
  {
    // get the first free and compatible ressource.
  }
}
