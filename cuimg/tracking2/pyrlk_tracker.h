#ifndef CUIMG_PYRLK_TRACKING_STRATEGIES_H_
# define CUIMG_PYRLK_TRACKING_STRATEGIES_H_


namespace cuimg
{
  namespace tracking_strategies
  {

    struct dummy_feature
    {
      int operator() (i_int2 p) const { return 0; }
      typedef int value_type;
    };

    template <typename DETECTOR>
    struct pyrlk_cpu
    {
    public:
      typedef pyrlk_cpu<DETECTOR> self;
      typedef DETECTOR detector_t;
      typedef particle_container<dummy_feature, particle_f, cpu> PC;
      typedef PC particles_type;
      typedef typename PC::particle_type particle_type;
      typedef cpu architecture;
      typedef typename architecture::template image2d<std::pair<int, i_float2> >::ret flow_stats_t;
      typedef typename architecture::template image2d<i_float2>::ret flow_t;
      typedef typename architecture::template image2d<i_float2>::ret gradient_t;
      typedef typename architecture::template image2d<gl8u>::ret gl8u_image2d;
      typedef typename architecture::template image2d<unsigned int>::ret uint_image2d;
      typedef gl8u_image2d input;

      pyrlk_cpu(const obox2d& o);
      inline void init() {};

      void update(const input& in, PC& pset);
      inline void match_particles(PC& pset);
      inline void new_particles(PC& pset);
      inline void filter(PC& pset);

      inline void set_upper(self* s);
      inline self* upper();

      inline self& set_detector_frequency(unsigned nframe);
      inline self& set_filtering_frequency(unsigned nframe);
      inline self& set_k(int k) { k_ = k; }


      inline detector_t& detector() { return detector_; }
      inline void clear();
      inline void create_detector_mask( PC& pset);
      inline i_float2 get_flow_at(const i_float2& p);
      inline const flow_t& flow() const { return flow_; }

      int border_needed() const;
      void set_with_merge(bool b);

      gl8u_image2d prev_input_;
      gl8u_image2d input_;
      gl8u_image2d tmp_;
      detector_t detector_;
      self* upper_;
      self* lower_;
      int frame_cpt_;

      gradient_t gradient_;
      uint_image2d contrast_;
      gl8u_image2d mask_;
      gl8u_image2d new_points_map_;
      const int flow_ratio;
      flow_stats_t flow_stats_;
      flow_t flow_;
      uint_image2d multiscale_count_;

      int detector_frequency_;
      int filtering_frequency_;
      int k_;
      bool with_merge_;
    };

  }
}

# include <cuimg/tracking2/pyrlk_tracker.hpp>

#endif

