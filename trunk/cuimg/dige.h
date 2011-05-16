#ifndef CUIMG_DIGE_H_
# define CUIMG_DIGE_H_

# include <map>
# include <GL/glew.h>
# include <dige/image.h>

# include <cuimg/cpu/host_image2d.h>
# include <cuimg/gpu/image2d.h>
# include <cuimg/gpu/image2d_math.h>
# include <cuimg/simple_ptr.h>

namespace dg
{

  template <typename T>
  image<trait::format::bgr, T>
    adapt(const cuimg::host_image2d<cuimg::improved_builtin<T, 3> >& i)
  {
    return image<trait::format::bgr, T>
      (i.ncols(), i.nrows(), (T*)i.data());
  }

  template <typename T>
  image<trait::format::bgra, T>
    adapt(const cuimg::host_image2d<cuimg::improved_builtin<T, 4> >& i)
  {
    return image<trait::format::bgra, T>
      (i.ncols(), i.nrows(), (T*)i.data());
  }

  template <typename T>
  image<trait::format::luminance, T>
    adapt(const cuimg::host_image2d<cuimg::improved_builtin<T, 1> >& i)
  {
    return image<trait::format::luminance, T>
      (i.ncols(), i.nrows(), (T*)i.data());
  }

  namespace internal
  {
    struct cuda_texture
    {
      GLuint gl_id;
      cudaGraphicsResource_t resource;
    };

    std::map<char*, cuda_texture> textures;
  }

  template<typename V, unsigned N>
  class cuda_opengl_texture
  {
  public:
    typedef int dige_texture_type;

    template <typename T>
    struct ib_to_opengl_internal_type {};
    template <>
    struct ib_to_opengl_internal_type<cuimg::i_float4> { enum { val = GL_RGBA32F }; };
    template <>
    struct ib_to_opengl_internal_type<cuimg::i_float1> { enum { val = GL_LUMINANCE32F_ARB }; };

    template <template <class> class PT>
    cuda_opengl_texture(const cuimg::image2d<cuimg::improved_builtin<V, N>, PT>& img)
      : img_(img)
    {
    }

    void load()
    {
      typedef typename std::map<char*, internal::cuda_texture>::const_iterator iterator;
      iterator it = internal::textures.find((char*)img_.data());
      if (it == internal::textures.end())
      {
        internal::cuda_texture t;
        glGenTextures(1, &t.gl_id);
        check_gl_error();
        assert(t.gl_id);
        glBindTexture(GL_TEXTURE_2D, t.gl_id);
        glPixelStorei(GL_UNPACK_ALIGNMENT,1);
        glTexImage2D(GL_TEXTURE_2D,
                     0, ib_to_opengl_internal_type<cuimg::improved_builtin<V, N> >::val,
                     img_.ncols(), img_.nrows(),
                     0, GL_RGBA, GL_FLOAT, 0);
        check_gl_error();
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glBindTexture(GL_TEXTURE_2D, 0);

        cudaGraphicsGLRegisterImage(&t.resource, t.gl_id,
          GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);
        check_cuda_error();
        internal::textures[(char*)img_.data()] = t;
      }

      internal::cuda_texture t = internal::textures[(char*)img_.data()];
      check_cuda_error();
      cudaGraphicsMapResources(1, &t.resource);
      check_cuda_error();
      cudaArray* cuda_array;
      cudaGraphicsSubResourceGetMappedArray(&cuda_array, t.resource, 0, 0);
      check_cuda_error();
      cudaMemcpy2DToArray(cuda_array, 0, 0, img_.data(), img_.pitch(), img_.ncols() * sizeof(cuimg::improved_builtin<V, N>), img_.nrows(), cudaMemcpyDeviceToDevice);
      check_cuda_error();
      cudaGraphicsUnmapResources(1, &t.resource);
      check_cuda_error();
      gl_id_ = t.gl_id;
    }

    void unload()
    {
    }

    GLenum gl_id() const
    {
      return gl_id_;
    }

    unsigned width() const { return img_.ncols(); }
    unsigned height() const { return img_.nrows(); }

  private:
    cuimg::image2d<cuimg::improved_builtin<V, N>, cuimg::simple_ptr> img_;
    GLuint gl_id_;
  };


  template<typename V, unsigned N, template <class> class PT>
  cuda_opengl_texture<V, N>
  adapt(const cuimg::image2d<cuimg::improved_builtin<V, N>, PT>& i)
  {
    return cuda_opengl_texture<V, N>(i);
  }

  template<typename V, template <class> class PT>
  cuda_opengl_texture<V, 4>
  adapt(const cuimg::image2d<cuimg::improved_builtin<V, 4>, PT>& i)
  {
    set_alpha_channel(*const_cast<cuimg::image2d<cuimg::improved_builtin<V, 4>, PT>*>(&i));
    return cuda_opengl_texture<V, 4>(i);
  }

}

#endif
