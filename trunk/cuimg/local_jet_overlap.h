#ifndef CUIMG_LOCAL_JET_OVERLAP_H_
# define CUIMG_LOCAL_JET_OVERLAP_H_

# define BOOST_MPL_CFG_ASSERT_BROKEN_POINTER_TO_POINTER_TO_MEMBER
# include <boost/mpl/at.hpp>
# include <boost/mpl/size.hpp>

# include <cuimg/kernel_image2d.h>
# include <cuimg/image2d.h>
# include <cuimg/local_jet_static.h>

namespace cuimg
{


  template <int I, int J, int N, int SIGMA, int KERNEL_HALF_SIZE>
    struct local_jet_loop_ij
    {
      template <typename T, typename U>
      static void run(image2d<T>& in, image2d<U>& tmp, image2d<U>* out[])
      {
        std::cout << "lj[" << ((I*(I+1)) / 2 + J) << "]: " << N - I << ", "<< J << "| ";
        local_jet_static<T, U, N - I, J, SIGMA, KERNEL_HALF_SIZE>(in, *(out[(I*(I+1)) / 2 + J]), tmp);
        //local_jet_static<T, U, 0, 0, SIGMA, KERNEL_HALF_SIZE>(in, *(out[(I*(I+1)) / 2 + J]), tmp);
        //local_jet_static<T, U, 0, 0, SIGMA, KERNEL_HALF_SIZE>(in, *(out[0]), tmp);
        local_jet_loop_ij<I, J + 1, N, SIGMA, KERNEL_HALF_SIZE>::run(in, tmp, out);
      }
    };

  template <int I, int N, int SIGMA, int KERNEL_HALF_SIZE>
    struct local_jet_loop_ij<I, I, N, SIGMA, KERNEL_HALF_SIZE>
  {
    template <typename T, typename U>
      static void run(image2d<T>& in, image2d<U>& tmp, image2d<U>* out[])
    {
      std::cout << "lj[" << ((I*(I+1)) / 2 + I) << "]: " << N - I << ", "<< I << std::endl;
      local_jet_static<T, U, N - I, I, SIGMA, KERNEL_HALF_SIZE>(in, *(out[(I*(I+1)) / 2 + I]), tmp);
      //  local_jet_static<T, U, 0, 0, SIGMA, KERNEL_HALF_SIZE>(in, *(out[(I*(I+1)) / 2 + I]), tmp);
      local_jet_loop_ij<I + 1, 0, N, SIGMA, KERNEL_HALF_SIZE>::run(in, tmp, out);
    }
  };

  template <int N, int SIGMA, int KERNEL_HALF_SIZE>
    struct local_jet_loop_ij<N, N, N, SIGMA, KERNEL_HALF_SIZE>
  {
    template <typename T, typename U>
      static void run(image2d<T>& in, image2d<U>& tmp, image2d<U>* out[])
    {
      std::cout << "lj[" << ((N * (N + 1)) / 2 + N) << "]: " << 0 << ", "<< N << std::endl;
      //local_jet_static<T, U, 0, 0, SIGMA, KERNEL_HALF_SIZE>(in, *(out[(N * (N + 1)) / 2 + N]), tmp);
      local_jet_static<T, U, 0, N, SIGMA, KERNEL_HALF_SIZE>(in, *(out[(N * (N + 1)) / 2 + N]), tmp);
    }
  };

  template <int SCALE, int N, typename SIGMAS>
  struct local_jet_loop_scale
  {
    template <typename T, typename U>
    static void run(image2d<T>& in, image2d<U>& tmp, image2d<U>* out[])
    {
      local_jet_loop_ij<0, 0, N,
        boost::mpl::at_c<SIGMAS, SCALE>::type::value, 
        boost::mpl::at_c<SIGMAS, SCALE>::type::value * 5>::run(in, tmp, out + SCALE * ((N + 1) * (N + 2) / 2));

      local_jet_loop_scale<SCALE - 1, N, SIGMAS>::run(in, tmp, out);
    }
  };

  template <int N, typename SIGMAS>
  struct local_jet_loop_scale<-1, N, SIGMAS>
  {
    template <typename T, typename U>
    static void run(image2d<T>& in, image2d<U>& tmp, image2d<U>* out[])
    {
    }
  };

  template <typename T, typename U, unsigned N,
            typename SIGMAS>
  void local_jet_static_(image2d<T>& in,
  image2d<U>* images[((N+1)*(N+2)/2) * boost::mpl::size<SIGMAS>::type::value],
                        image2d<U>& tmp)
  {
    local_jet_loop_scale<boost::mpl::size<SIGMAS>::type::value - 1, N, SIGMAS>::run(in, tmp, images);
  }

}

#endif
