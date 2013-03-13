#ifndef CUIMG_DOMINANT_SPEED_HPP_
# define CUIMG_DOMINANT_SPEED_HPP_

# ifndef NO_CUDA
# include <thrust/sort.h>
# include <thrust/inner_product.h>
# include <thrust/iterator/constant_iterator.h>
# endif

# include <cuimg/improved_builtin.h>
# include <cuimg/cpu/fill.h>
# include <cuimg/profiler.h>

namespace cuimg
{

  template <typename A>
  dominant_speed_estimator<A>::dominant_speed_estimator(const obox2d& d)
    : h(d / 4)
  {
  }

  template <typename A>
  dominant_speed_estimator<A>::dominant_speed_estimator(const dominant_speed_estimator& d)
    : h(d.h.domain())
  {
    copy(d.h, h);
  }

  // dominant_speed_estimator&
  // dominant_speed_estimator<A>::operator=(const dominant_speed_estimator& d)
  // {
  //   h = host_image2d<unsigned short>(d.h.domain());
  //   copy(d.h, h);
  //   return *this;
  // }

  template <typename A>
  template <typename PI>
  i_short2
  dominant_speed_estimator<A>::estimate(const PI& pset, i_int2 prev_camera_motion, const cpu&)
  {
    SCOPE_PROF(dominant_speed_estimation);

    if (pset.size() < 100)
      return i_short2(0,0);

    typedef unsigned short US;
    fill(h, 0);
    i_int2 h_center(h.nrows() / 2, h.ncols() / 2);
    int max = 0;
    i_int2 max_bin = h_center;
    for (unsigned i = 0; i < pset.dense_particles().size(); i++)
    {
      const particle& part = pset[i];
      i_int2 bin = h_center + part.acceleration + prev_camera_motion;
      if (part.age > 2 and h.has(bin))
      {
        int c = ++h(bin);
        if (c > max)
        {
          max = c;
          max_bin = bin;
        }
      }
      // else if (part.age == 1)
      // {
      //   int c = ++h(h_center + part.speed);
      //   if (c > max)
      //   {
      //     max = c;
      //     max_bin = bin;
      //   }
      // }
    }

    return max_bin - h_center;
  }

#ifdef NVCC

  template <typename Vector1,
	    typename Vector2,
	    typename Vector3>
  void sparse_histogram(Vector1& data,
			Vector2& histogram_values,
			Vector3& histogram_counts)
  {
    typedef typename Vector1::value_type ValueType; // input value type
    typedef typename Vector3::value_type IndexType; // histogram index type

    // sort data to bring equal elements together
    thrust::sort(data.begin(), data.end());

    // number of histogram bins is equal to number of unique values (assumes data.size() > 0)
    IndexType num_bins = thrust::inner_product(data.begin(), data.end() - 1,
					       data.begin() + 1,
					       IndexType(1),
					       thrust::plus<IndexType>(),
					       thrust::not_equal_to<ValueType>());

    // resize histogram storage
    histogram_values.resize(num_bins);
    histogram_counts.resize(num_bins);

    // compact find the end of each bin of values
    thrust::reduce_by_key(data.begin(), data.end(),
			  thrust::make_constant_iterator(1),
			  histogram_values.begin(),
			  histogram_counts.begin());

  }

  template <typename P>
  __global__ void
  particle_vote(P* pset,
		int* vote_buffer_,
		unsigned size,
		kernel_image2d<int> h,
		i_int2 prev_camera_motion)
  {
    int i = thread_pos1d();
    if (i >= size) return;

    const particle& part = pset[i];
    if (part.age <= 2)
    {
      vote_buffer_[i] = -1;
      return;
    }

    i_int2 h_center(h.nrows() / 2, h.ncols() / 2);
    i_int2 bin = h_center + part.acceleration + prev_camera_motion;
    if (h.has(bin))
      vote_buffer_[i] = &h(bin) - &h(0,0);
    else
      vote_buffer_[i] = -1;
  }

  template <typename A>
  template <typename PI>
  i_short2
  dominant_speed_estimator<A>::estimate(const PI& pset, i_int2 prev_camera_motion, const cuda_gpu&)
  {
    if (pset.size() < 100)
      return i_short2(0,0);

    SCOPE_PROF(dominant_speed_estimation);

    //Vote
    vote_buffer_.resize(pset.size());
    particle_vote<<<A::dimgrid1d(pset.size()), A::dimblock1d()>>>(thrust::raw_pointer_cast(pset.dense_particles().data()),
    								  thrust::raw_pointer_cast(vote_buffer_.data()),
    								  pset.size(),
    								  h,
    								  prev_camera_motion);

    //Histogram
    sparse_histogram(vote_buffer_, histo_values_, histo_counts_);

    using thrust::device_pointer_cast;
    //Get max.
    if (histo_counts_.size() == 1) return i_short2(0,0);

    thrust::device_ptr<int> max = &*thrust::max_element(++histo_counts_.begin(), histo_counts_.end());
    int max_pos = max - &*histo_counts_.begin();
    int max_bin = histo_values_[max_pos];

    i_short2 speed = h.index_to_point(max_bin) - i_int2(h.nrows() / 2, h.ncols() / 2);
    //std::cout << speed << std::endl;
    return speed;
  }

#endif

}

#endif
