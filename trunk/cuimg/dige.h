#ifndef CUIMG_DIGE_H_
# define CUIMG_DIGE_H_

# include <map>
# include <stack>
# include <GL/glew.h>
# include <cudaGL.h>
# include <cuda_gl_interop.h>


# include <dige/image.h>

# include <cuimg/cpu/host_image2d.h>
# include <cuimg/gpu/device_image2d.h>
# include <cuimg/gpu/device_image2d_math.h>
# include <cuimg/simple_ptr.h>
# include <cuimg/dsl/aggregate.h>
# include <cuimg/dsl/get_comp.h>
# include <cuimg/error.h>

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
  image<trait::format::rgba, T>
    adapt(const cuimg::host_image2d<cuimg::improved_builtin<T, 4> >& i)
  {
    return image<trait::format::rgba, T>
      (i.ncols(), i.nrows(), (T*)i.data());
  }

  template <typename T>
  image<trait::format::luminance, T>
    adapt(const cuimg::host_image2d<cuimg::improved_builtin<T, 1> >& i)
  {
    return image<trait::format::luminance, T>
      (i.ncols(), i.nrows(), (T*)i.data());

}
  inline
  image<trait::format::luminance, unsigned short>
    adapt(const cuimg::host_image2d<unsigned short>& i)
  {
    return image<trait::format::luminance, unsigned short>
      (i.ncols(), i.nrows(), (unsigned short*)i.data());
  }

  inline
  image<trait::format::luminance, int>
    adapt(const cuimg::host_image2d<int>& i)
  {
    return image<trait::format::luminance, int>
      (i.ncols(), i.nrows(), (int*)i.data());
  }

  namespace internal
  {
    struct cuda_texture_cat
    {
      cuda_texture_cat()
      : width(0), height(0), valuetype(0)
      {}

      cuda_texture_cat(unsigned w, unsigned h, GLuint vt)
      : width(w), height(h), valuetype(vt)
      {}

      unsigned width;
      unsigned height;
      GLuint valuetype;
    };

    struct tex_comp
    {
      bool operator()(const cuda_texture_cat& a,
        const cuda_texture_cat& b)
      {
        return (a.width < b.width) ||
          (a.height < b.height) ||
          (a.valuetype < b.valuetype);
      }
    };

    struct cuda_texture
    {
      GLuint gl_id;
      cudaGraphicsResource_t resource;
    };

    static std::map<cuda_texture_cat, std::stack<cuda_texture>, tex_comp> textures;
  }

  template <typename T>
  struct ib_to_opengl_internal_type {};
  template <>
  struct ib_to_opengl_internal_type<cuimg::i_float4>
  { enum { val = GL_RGBA32F }; };
  template <>
  struct ib_to_opengl_internal_type<cuimg::i_float1>
  { enum { val = GL_R32F }; };

  template<typename V, unsigned N>
  class cuda_opengl_texture
  {
  public:
    typedef int dige_texture_type;

    cuda_opengl_texture(const cuimg::device_image2d<cuimg::improved_builtin<V, N> >& img)
      : img_(img)
    {
    }

    void load()
    {
      internal::cuda_texture_cat texcat(img_.ncols(), img_.nrows(),
                            ib_to_opengl_internal_type<cuimg::improved_builtin<V, N> >::val);
      std::stack<internal::cuda_texture>& stack = internal::textures[texcat];

      if (stack.size() == 0)
      {
        std::cout << "Allocate texture" << std::endl;
        glGenTextures(1, &cuda_tex_.gl_id);
        check_gl_error();
        assert(cuda_tex_.gl_id);
        glBindTexture(GL_TEXTURE_2D, cuda_tex_.gl_id);
        glPixelStorei(GL_UNPACK_ALIGNMENT,1);
        glTexImage2D(GL_TEXTURE_2D,
                     0, ib_to_opengl_internal_type<cuimg::improved_builtin<V, N> >::val,
                     img_.ncols(), img_.nrows(),
                     0, GL_RGBA, GL_FLOAT, 0);
        check_gl_error();
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glBindTexture(GL_TEXTURE_2D, 0);

        cudaGraphicsGLRegisterImage(&cuda_tex_.resource, cuda_tex_.gl_id,
          GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);
        check_gl_error();
        cuimg::check_cuda_error();
      }
      else
      {
        cuda_tex_ = stack.top();
        stack.pop();
      }

      cuimg::check_cuda_error();
      cudaGraphicsMapResources(1, &cuda_tex_.resource);
      cuimg::check_cuda_error();
      cudaArray* cuda_array;
      cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_tex_.resource, 0, 0);
      cuimg::check_cuda_error();
      cudaMemcpy2DToArray(cuda_array, 0, 0, img_.data(), img_.pitch(),
                          img_.ncols() * sizeof(cuimg::improved_builtin<V, N>), img_.nrows(),
                          cudaMemcpyDeviceToDevice);
      cuimg::check_cuda_error();
      cudaGraphicsUnmapResources(1, &cuda_tex_.resource);
      cuimg::check_cuda_error();
    }

    internal::cuda_texture_cat texcat()
    {
      return internal::cuda_texture_cat(img_.ncols(), img_.nrows(),
                                        ib_to_opengl_internal_type<cuimg::improved_builtin<V, N> >::val);
    }

    void unload()
    {
      if (cuda_tex_.gl_id != 0)
      {
        internal::textures[texcat()].push(cuda_tex_);
        cuda_tex_.gl_id = 0;
      }

    }

    GLenum gl_id() const
    {
      return cuda_tex_.gl_id;
    }

    unsigned width() const { return img_.ncols(); }
    unsigned height() const { return img_.nrows(); }

  private:
    cuimg::device_image2d<cuimg::improved_builtin<V, N> > img_;
    internal::cuda_texture cuda_tex_;
  };


  template<typename V, unsigned N>
  cuda_opengl_texture<V, N>
  adapt(const cuimg::device_image2d<cuimg::improved_builtin<V, N> >& i)
  {
    return cuda_opengl_texture<V, N>(i);
  }

  template<typename V>
  cuda_opengl_texture<V, 4>
  adapt(const cuimg::device_image2d<cuimg::improved_builtin<V, 1> >& i)
  {
    cuimg::device_image2d<cuimg::i_float4> res(i.domain());
    res = cuimg::aggregate<float>::run(cuimg::get_x(i),
                                       cuimg::get_x(i),
                                       cuimg::get_x(i),
                                       1.f);
    return cuda_opengl_texture<V, 4>(res);
  }

  template<typename V>
  cuda_opengl_texture<V, 4>
  adapt(const cuimg::device_image2d<cuimg::improved_builtin<V, 4> >& i)
  {
    set_alpha_channel(*const_cast<cuimg::device_image2d<cuimg::improved_builtin<V, 4> >*>(&i));
    return cuda_opengl_texture<V, 4>(i);
  }

}

#endif
