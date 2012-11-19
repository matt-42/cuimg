#ifndef CUIMG_TRACKER2_H_
# define CUIMG_TRACKER2_H_

# include <cuimg/improved_builtin.h>

namespace cuimg
{
  template <typename F>
  class tracker
  {
  public:
    typedef typename F::input I;
    typedef typename F::particles_type particles_type;

    tracker(const obox2d& d, int nscales);
    ~tracker();


    void update_input(const I& in);
    void subsample_input(const I& in);
    void run();

    inline const obox2d& domain() const     { return input_.domain(); }
    inline const I& input() const                { return input_; }
    inline       particles_type& pset()     { return pset_; }
    inline const particles_type& pset() const { return pset_; }

    tracker<F>& scale(unsigned s) { return s && upper_tracker_ ? upper_tracker_->scale(s-1) : *this; }
  private:
    tracker(const obox2d& d, tracker<F>* lower, int nscales);

    I input_;
    particles_type pset_;
    F strategy_;
    tracker<F>* lower_tracker_;
    tracker<F>* upper_tracker_;
  };

}

# include <cuimg/tracking2/tracker.hpp>

#endif
