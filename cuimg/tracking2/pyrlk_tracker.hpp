#ifndef CUIMG_PYRLK_TRACKING_STRATEGIES_HPP_
# define CUIMG_PYRLK_TRACKING_STRATEGIES_HPP_

#include <Eigen/Dense>
#include <cuimg/copy.h>
#include <cuimg/tracking2/tracking_strategies.hpp>
#include <cuimg/tracking2/transformations.h>

namespace cuimg
{
  namespace tracking_strategies
  {

    // template <typename F, typename GD>
    // inline __host__ __device__
    // std::pair<i_short2, float> iterative_lk_match(i_short2 p, i_short2 prediction_, F A, F B, GD Ag,
    // 																							i_int2* window,
    // 																							int* offsets, int* offsets_G, int offsets_n)
    // {
    // 	assert(A.domain() == B.domain());
    // 	assert(Ag.domain() == B.domain());
    // 	assert(A.border() == B.border());
    // 	assert(Ag.border() == B.border());

    //   // Image difference.

    //   // Gradient matrix

    //   Eigen::Matrix2d G = Eigen::Matrix2d::Zero();
    //   int cpt = 0;

    // 	auto* Ag_p = &Ag(p);
    // 	auto* A_p = &A(p);

    //   for(int i = 0; i < offsets_n; i ++)
    //   {
    // 			Eigen::Matrix2d m;
    // 			double gx = Ag_p[offsets_G[i]][0] / 2.;
    // 			double gy = Ag_p[offsets_G[i]][1] / 2.;
    // 			m <<
    // 				gx * gx, gx * gy,
    // 				gx * gy, gy * gy;
    // 			G += m;
    // 			cpt++;
    // 		}
    //   }

    //   G /= cpt;

    //   //if (fabs(G.determinant()) < 0.0001)
    //   auto ev = G.eigenvalues();
    //   float min_ev = 999999;
    //   for (int i = 0; i < ev.size(); i++)
    // 		if (fabs(ev[i].real()) < min_ev) min_ev = fabs(ev[i].real());

    //   if (min_ev < 0.0001)
    //   {
    //   	return std::pair<i_short2, float>(i_short2(-1,-1), 1000);
    //   }

    //   Eigen::Matrix2d G1 = G.inverse();

    //   i_int2 v = prediction_;
    //   Eigen::Vector2d nk = Eigen::Vector2d::Ones();

    //   i_int2 gs[offsets_n];
    //   i_uchar1 as[offsets_n];
    //   for(int i = 0; i < offsets_n; i ++)
    //   {
    // 		gs[i] = Ag_p[offsets_G[i]];
    // 		as[i] = A_p[offsets[i]];
    //   }

    //   for (int k = 0; k < 3 && nk.norm() >= 1; k++)
    //   {
    // 		Eigen::Vector2d bk = Eigen::Vector2d::Zero();
    // 		// Temporal difference.
    // 		cpt = 0;
    // 		for(int i = 0; i < offsets_n; i ++)
    // 		{
    // 			//i_int2 n1 = p + i_int2(circle_r3[i]);
    // 			i_int2 n2 = v + i_int2(circle_r3[i]);
    // 			if (B.has(n2))
    // 			{
    // 				auto& g = gs[i];
    // 				double dt = (as[i] - B(n2)).x;
    // 				bk += Eigen::Vector2d(g[0] * dt, g[1] * dt);
    // 				cpt++;
    // 			}
    // 		}
    // 		bk /= cpt;

    // 		nk = G1 * bk;
    // 		v += i_int2(nk[0], nk[1]);
    //   }

    //   float err = 0;
    //   for(int i = 0; i < offsets_n; i ++)
    //   {
    // 		//i_int2 n1 = p + i_int2(circle_r3[i]);
    // 		i_int2 n2 = v + i_int2(circle_r3[i]);
    // 		if (B.has(n2))
    // 		{
    // 			err += fabs((as[i] - B(n2)).x);
    // 			cpt++;
    // 		}
    //   }

    //   //std::cout << v << std::endl;
    //   return std::pair<i_short2, float>(v, err / cpt);
    // }


    template <typename T>
    struct floatify;

    template <>
    struct floatify<gl8u>
    {
      typedef float ret;
    };

    template <typename V, unsigned N>
    struct floatify<improved_builtin<V, N> >
    {
      typedef improved_builtin<float, N> ret;
    };


    template <typename I>
    inline
    typename floatify<typename I::value_type>::ret li(const I& img, i_float2 p)
    {
      i_int2 x = p;
      float a0 = p[0] - x[0];
      float a1 = p[1] - x[1];

      //return img(x);
      const typename I::value_type* l1 = &img(x);
      const typename I::value_type* l2 = (const typename I::value_type*)(((const char*)l1) + img.pitch());
      return typename floatify<typename I::value_type>::ret(
                                                            (1 - a0) * (1 - a1) *  l1[0] +
                                                            a0 * (1 - a1) *  l2[0] +
                                                            (1 - a0) * a1 *  l1[1] +
                                                            a0 * a1 *  l2[1]);
    }

    template <typename T, unsigned N>
    Eigen::Matrix<T, N, 1> to_eigen(const improved_builtin<T, N>& x)
    {
      Eigen::Matrix<T, N, 1> res;
      for (int i = 0; i < N; i++) res[i] = x[i];
      return res;
    }


    template <typename T, unsigned N>
    improved_builtin<T, N> from_eigen(Eigen::Matrix<T, N, 1> x)
    {
      improved_builtin<T, N> res;
      for (int i = 0; i < N; i++) res[i] = x[i];
      return res;
    }

    struct iterative_lk_match_5x5
    {
      typedef translation_transform transform;

      template <typename F, typename GD>
      static inline __host__ __device__
      std::pair<i_float2, float> match(i_float2 p, translation_transform tr_prediction_, F A, F B, GD Ag)
      {
        int ws = 21;
        int hws = ws/2;

        int factor = 1;
        // Image difference.
        // Gradient matrix
        Eigen::Matrix2f G = Eigen::Matrix2f::Zero();
        int cpt = 0;
        for(int r = -hws; r <= hws; r++)
          for(int c = -hws; c <= hws; c++)
          {
            i_float2 n = p + i_int2(r, c) * factor;
            if (A.has(n))
            {
              Eigen::Matrix2f m;
              i_float2 g = li(Ag, n);
              float gx = g[0];
              float gy = g[1];
              m <<
                gx * gx, gx * gy,
                gx * gy, gy * gy;
              G += m;
              cpt++;
            }
          }

        //G /= cpt;

        //if (fabs(G.determinant()) < 0.0001)
        auto ev = G.eigenvalues();
        float min_ev = 999999;
        for (int i = 0; i < ev.size(); i++)
          if (fabs(ev[i].real()) < min_ev) min_ev = fabs(ev[i].real());

        if (min_ev < 0.0001)
          return std::pair<i_float2, float>(i_float2(-1,-1), 1000);

        Eigen::Matrix2f G1 = G.inverse();

        i_float2 prediction_ = tr_prediction_.apply_transform(p);
        i_float2 v = prediction_;
        Eigen::Vector2f nk = Eigen::Vector2f::Ones();


        i_float2 gs[ws * ws];
        i_uchar1 as[ws * ws];
        {
          for(int i = 0, r = -hws; r <= hws; r++)
          {
            for(int c = -hws; c <= hws; c++)
            {
              i_float2 n = p + i_int2(r, c) * factor;
              if (Ag.has(n))
              {
                gs[i] = li(Ag, n);
                as[i] = li(A, n);
              }
              i++;
            }
          }
        }
        auto domain = B.domain() - border(3);

        for (int k = 0; k < 30 && nk.norm() >= 0.1; k++)
        {
          Eigen::Vector2f bk = Eigen::Vector2f::Zero();
          // Temporal difference.
          cpt = 0;
          int i = 0;
          for(int r = -hws; r <= hws; r++)
          {
            for(int c = -hws; c <= hws; c++)
            {
              //i_int2 n1 = p + i_int2(r, c);
              i_float2 n2 = v + i_int2(r, c) * factor;
              //if (B.has(n2))
              {
                auto g = gs[i];
                float dt = (float(as[i].x) - li(B, n2));
                bk += Eigen::Vector2f(g[0] * dt, g[1] * dt);
                cpt++;
              }
              i++;
            }
            //i += ws;
          }
          //bk /= cpt;

          // if (bk.norm() > 1) bk /= bk.norm();
          nk = G1 * bk;
          // std::cout << nk.transpose() << std::endl;
          v += i_float2(nk[0], nk[1]);
          if (!domain.has(v))
            return std::pair<i_float2, float>(i_float2(-1,-1), 1000);
        }

        if (norml2(prediction_ - v) > ws * 2)
          return std::pair<i_float2, float>(i_float2(-1,-1), 1000);

        float err = 0;
        for(int r = -hws; r <= hws; r++)
          for(int c = -hws; c <= hws; c++)
          {
            i_float2 n2 = v + i_int2(r, c) * factor;
            int i = (r+hws) * hws + (c+hws);
            //if (B.has(n2))
            {
              err += fabs((as[i].x - li(B, n2)));
              cpt++;
            }
          }

        return std::pair<i_float2, float>(v - p, err / cpt);
      }
    };

    struct iterative_affine_lk_match_5x5
    {
      typedef affine_transform transform;

      template <typename F, typename GD>
      static inline __host__ __device__
      std::pair<transform, float>
      match(i_float2 p,
            const transform& prediction_,
            F A, F B, GD Ag)
      {
        using namespace Eigen;
        typedef Matrix<float, 6, 6> Matrix6f;
        typedef Matrix<float, 6, 1> Vector6f;
        int ws = 11;
        int hws = ws/2;

        int factor = 2;
        // Image difference.
        // Gradient matrix
        Matrix6f G = Matrix6f::Zero();
        int cpt = 0;
        for(int r = -hws; r <= hws; r++)
          for(int c = -hws; c <= hws; c++)
          {
            i_float2 n = p + i_int2(r, c) * factor;
            float x = r * factor;
            float y = c * factor;
            float x2 = x * x;
            float y2 = y * y;
            float xy = x * y;
            if (A.has(n))
            {
              Matrix6f m;
              i_float2 g = li(Ag, n);
              float gx = g[0];
              float gx2 = gx * gx;
              float gy = g[1];
              float gy2 = gy * gy;
              float gxy = gx * gy;
              m <<
                gx2, gxy, x * gx2, y * gx2, x * gxy, y * gxy,
                gxy, gy2, x * gxy, y * gxy, x * gy2, y * gy2,
                x * gx2,  x * gxy, x2 * gx2, xy * gx2, x2 * gxy, xy * gxy,
                y * gx2, y * gxy, xy * gx2, y2 * gx2, xy * gxy, y2 * gxy,
                x * gxy, x * gy2, x2 * gxy, xy * gxy, x2 * gy2, xy * gy2,
                y * gxy, y * gy2, xy * gxy, y2 * gxy, xy * gy2, y2 * gy2;

              G += m;
              cpt++;
            }
          }

        //G /= cpt;

        //if (fabs(G.determinant()) < 0.0001)
        auto ev = G.eigenvalues();
        float min_ev = 999999;
        for (int i = 0; i < ev.size(); i++)
          if (fabs(ev[i].real()) < min_ev) min_ev = fabs(ev[i].real());

        if (min_ev < 0.0001)
          return std::pair<affine_transform, float>(affine_transform::identity(), 10000);

        Matrix6f G1 = G.inverse();

        Matrix3f TA = Matrix3f::Identity();
        TA.block<2,3>(0,0) = prediction_;
        //std::cout << prediction_.block<2,1>(0,2).transpose() << std::endl;
        //TA.block<2,1>(0,2) = prediction_.block<2,1>(0,2);

        //Vector2f v(p);
        Vector6f nk = Vector6f::Ones();

        i_float2 gs[ws * ws];
        i_uchar1 as[ws * ws];
        for(int i = 0, r = -hws; r <= hws; r++)
        {
          for(int c = -hws; c <= hws; c++, i++)
          {
            Vector2f n_ = Vector2f(p.x, p.y) + (Vector3f(r * factor, c * factor, 1)).segment(0,2);
            i_float2 n(n_[0], n_[1]);
            if (Ag.has(n))
            {
              gs[i] = li(Ag, n);
              as[i] = li(A, n);
            }
          }
        }
        auto domain = B.domain() - border(ws * 1.4142/2);

        for (int k = 0; k < 30 && (k == 0 || TA.block<2, 1>(0,2).norm() >= 0.1); k++)
        {
          Vector6f bk = Vector6f::Zero();
          // Temporal difference.
          cpt = 0;
          for(int i = 0, r = -hws; r <= hws; r++)
          {
            for(int c = -hws; c <= hws; c++, i++)
            {
              //i_int2 n1 = p + i_int2(r, c);
              Vector2f n2_offset = (TA * Vector3f(r * factor, c * factor, 1)).segment(0,2);
              Vector2f n2 = to_eigen(p) + n2_offset;
              //std::cout << n2.transpose() << std::endl;
              if (B.has(i_float2(n2[0], n2[1])))
              {
                float x = n2_offset[0];
                float y = n2_offset[1];
                auto g = gs[i];
                float gx = g[0];
                float gy = g[1];

                float dt = float(as[i].x) - li(B, i_float2(n2[0],n2[1]));
                Vector6f bk_step;
                bk_step <<
                  gx * dt,
                  gy * dt,
                  x * gx * dt,
                  y * gx * dt,
                  x * gy * dt,
                  y * gy * dt;
                bk += bk_step;

                cpt++;
              }
              else
                return std::pair<affine_transform, float>(affine_transform::identity(), 1000);
            }
            //i += ws;
          }
          //bk /= cpt;

          // if (bk.norm() > 1) bk /= bk.norm();
          nk = G1 * bk;
          // std::cout << nk.transpose() << std::endl;
          // std::cout << nk.transpose() << std::endl;
          Matrix3f step;
          // step <<
          //   1 + nk[2], nk[3], nk[0],
          //   nk[4], 1 + nk[5], nk[1],
          //   0,0,1;

          step <<
            1, 0, nk[0],
            0, 1, nk[1],
            0,0,1;

          TA =  TA * step;
          Vector2f next = to_eigen(p) + TA.block<2,1>(0,2);
          // std::cout << "next: " << std::endl;
          // std::cout << next << std::endl;

          if (!domain.has(i_float2(next[0], next[1])))
            return std::pair<affine_transform, float>(affine_transform::identity(), 1000);
        }

        // std::cout << TA << std::endl;

        // if (norml2(i_float2(prediction_(0,2), prediction_(1,2))) > ws * 2)
        //   return std::pair<affine_transform, float>(affine_transform::identity(), 1000);

        float err = 0;
        for(int i = 0, r = -hws; r <= hws; r++)
          for(int c = -hws; c <= hws; c++)
          {
            Vector2f n2 = to_eigen(p) + (TA * Vector3f(r * factor, c * factor, 1)).segment(0,2);
            if (B.has(i_float2(n2[0], n2[1])))
            {
              err += fabs((as[i].x - li(B, i_float2(n2[0], n2[1]))));
              cpt++;
            }
            i++;
          }

        err = 0;
        return std::pair<affine_transform, float>(TA.block<2,3>(0,0), err / cpt);
      }

    };

    // template <typename F, typename GD>
    // inline __host__ __device__
    // std::pair<i_short2, float> iterative_lk_match2(i_short2 p, i_short2 prediction_, F A, F B, GD Ag)
    // {
    //   // Image difference.

    //   // Gradient matrix

    //   Eigen::Matrix2f G = Eigen::Matrix2f::Zero();
    //   int cpt = 0;
    //   for(int i = 0; i < 8; i ++)
    //   {
    //     i_int2 n = p + i_int2(circle_r3[i*2]);

    //     if (A.has(n))
    //     {
    //       Eigen::Matrix2f m;
    //       float gx = Ag(n)[0] / 2.;
    //       float gy = Ag(n)[1] / 2.;
    //       m <<
    //         gx * gx, gx * gy,
    //         gx * gy, gy * gy;
    //       G += m;
    //       cpt++;
    //     }
    //   }

    //   G /= cpt;

    //   //if (fabs(G.determinant()) < 0.0001)
    //   auto ev = G.eigenvalues();
    //   float min_ev = 999999;
    //   for (int i = 0; i < ev.size(); i++)
    //     if (fabs(ev[i].real()) < min_ev) min_ev = fabs(ev[i].real());

    //   if (min_ev < 0.001)
    //   {
    //   	return std::pair<i_short2, float>(i_short2(-1,-1), 1000);
    //   }

    //   Eigen::Matrix2f G1 = G.inverse();

    //   i_int2 v = prediction_;
    //   Eigen::Vector2f nk = Eigen::Vector2f::Ones();


    //   i_int2 gs[8];
    //   i_uchar1 as[8];
    //   for(int i = 0; i < 8; i ++)
    //   {
    //     i_int2 n = p + i_int2(circle_r3[i*2]);
    //     if (Ag.has(n))
    //     {
    //       gs[i] = Ag(n);
    //       as[i] = A(n);
    //     }
    //   }

    //   for (int k = 0; k < 3 && nk.norm() >= 1; k++)
    //   {
    //     Eigen::Vector2f bk = Eigen::Vector2f::Zero();
    //     // Temporal difference.
    //     cpt = 0;
    //     for(int i = 0; i < 8; i ++)
    //     {
    //       i_int2 n1 = p + i_int2(circle_r3[i*2]);
    //       i_int2 n2 = v + i_int2(circle_r3[i*2]);
    //       if (A.has(n1) and B.has(n2))
    //       {
    //         auto g = gs[i];
    //         float dt = (as[i] - B(n2)).x;
    //         bk += Eigen::Vector2f(g[0] * dt, g[1] * dt);
    //         cpt++;
    //       }
    //     }
    //     bk /= cpt;

    //     nk = G1 * bk;
    //     v += i_int2(nk[0], nk[1]);
    //   }

    //   //std::cout << v << std::endl;
    //   return std::pair<i_short2, float>(v, 0.f);
    // }



    // template <typename F, typename GD>
    // inline __host__ __device__
    // std::pair<i_short2, float> iterative_lk_match3(i_short2 p, i_short2 prediction_, F A, F B, GD Ag)
    // {
    //   // Image difference.

    //   // Gradient matrix

    //   Eigen::Matrix2f G = Eigen::Matrix2f::Zero();
    //   int cpt = 0;
    //   for(int i = 0; i < 8; i ++)
    //   {
    //     i_int2 n = p + i_int2(c8_h[i]);

    //     if (A.has(n))
    //     {
    //       Eigen::Matrix2f m;
    //       float gx = Ag(n)[0] / 2;
    //       float gy = Ag(n)[1] / 2;
    //       m <<
    //         gx * gx, gx * gy,
    //         gx * gy, gy * gy;
    //       G += m;
    //       cpt++;
    //     }
    //   }

    //   G /= cpt;

    //   //if (fabs(G.determinant()) < 0.0001)
    //   auto ev = G.eigenvalues();
    //   float min_ev = 999999;
    //   for (int i = 0; i < ev.size(); i++)
    //     if (fabs(ev[i].real()) < min_ev) min_ev = fabs(ev[i].real());

    //   if (min_ev < 0.0001)
    //   {
    //   	return std::pair<i_short2, float>(i_short2(-1,-1), 1000);
    //   }

    //   Eigen::Matrix2f G1 = G.inverse();

    //   i_int2 v = prediction_;
    //   Eigen::Vector2f nk = Eigen::Vector2f::Ones();

    //   for (int k = 0; k < 3 && nk.norm() >= 1; k++)
    //   {
    //     Eigen::Vector2f bk = Eigen::Vector2f::Zero();
    //     // Temporal difference.
    //     cpt = 0;
    //     for(int i = 0; i < 8; i ++)
    //     {
    //       i_int2 n1 = p + i_int2(c8_h[i]);
    //       i_int2 n2 = v + i_int2(c8_h[i]);
    //       if (A.has(n1) and B.has(n2))
    //       {
    //         auto g = Ag(n1);
    //         float dt = (A(n1) - B(n2)).x;
    //         bk += Eigen::Vector2f(g[0] * dt, g[1] * dt);
    //         cpt++;
    //       }
    //     }
    //     bk /= cpt;

    //     nk = G1 * bk;
    //     v += i_int2(nk[0], nk[1]);
    //   }

    //   //std::cout << v << std::endl;
    //   return std::pair<i_short2, float>(v, 0.f);
    // }

    template<typename I, typename J, typename C, typename GI, typename P, typename M>
    struct pyrlk_match_particles_kernel
    {
    private:
      typename P::kernel_type pset;
      typename I::kernel_type input;
      typename I::kernel_type prev_input;
      typename GI::kernel_type gradient;
      J* upper;
      typename C::kernel_type contrast;
      int flow_ratio;
      int k;
      box2d particle_domain;

      typedef typename M::transform transform;

    public:
      pyrlk_match_particles_kernel(P& pset_,
				   I& prev_input_,
				   I& input_,
				   GI& gradient_,
				   J* upper_,
				   C& contrast_,
				   int flow_ratio_,
                                   int k_,
                                   box2d particle_domain_)
	: pset(pset_),
	  prev_input(prev_input_),
	  input(input_),
	  gradient(gradient_),
	  upper(upper_),
	  contrast(contrast_),
	  flow_ratio(flow_ratio_),
	  k(k_),
          particle_domain(particle_domain_)
      {
      }

      inline __host__ __device__ void
      operator()(int i)
      {
        assert(i >= 0 && i < pset.size());
        particle_f<transform>& part = pset.dense_particles()[i];
        assert(pset.domain().has(part.pos));
        box2d domain = particle_domain;
        //assert(domain.has(part.pos));
        if (part.age > 0 and domain.has(part.pos))
        {
          // Prediction.
          transform pred_transform = recursive_prediction(part, upper);
          i_float2 pred = pred_transform.apply_transform(part.pos);

          // Matching.
          if (domain.has(pred))
          {
            float distance;
            std::pair<transform, float> match_res = M::match(part.pos, pred_transform,
                                                             prev_input,
                                                             input,
                                                             gradient);
            transform tr = match_res.first;
            distance = match_res.second;
            i_int2 match = tr.apply_transform(part.pos);
            // std::cout << "tr" << std::endl;
            // std::cout << tr << std::endl;
            // std::cout << "pos" << std::endl;
            // std::cout << part.pos << std::endl;
            // std::cout << "match" << std::endl;
            // std::cout << match << std::endl;
            unsigned cpt = 0;
            if (domain.has(match) and distance < k and norml2(match - pred) < std::pow(2, 5) * 5)
            {
              // if (contrast(match) < 3.f)
              // 	pset.remove(i);
              // else
              {
                pset.move(i, tr, 0);
                assert(pset.has(match));
                assert(pset.dense_particles()[i].age > 0);
              }
            }
            else
              pset.remove(i);
          }
          else
            pset.remove(i);
        }
        else
          pset.remove(i);

        // if (part.age > 0)
        //   pset.touch(i);
      }
    };

    template <typename D, typename M>
    pyrlk_cpu<D, M>::pyrlk_cpu(const obox2d& d)
      : prev_input_(d),
	tmp_(d),
	flow_ratio(2),
	detector_(d),
	upper_(0),
	frame_cpt_(0),
	gradient_(d),
	contrast_(d),
	mask_(d, 4),
	detector_frequency_(1),
	filtering_frequency_(1),
	k_(300),
	with_merge_(true)
    {
      new_points_map_ = gl8u_image2d(domain_div_up(d, 2*flow_ratio));
      flow_stats_ = flow_stats_t(domain_div_up(d, flow_ratio));
      flow_ = flow_t(domain_div_up(d, flow_ratio));
      multiscale_count_ = uint_image2d(domain_div_up(d, flow_ratio));

      particle_domain_ = d - border(3);
    }

    template <typename D, typename M>
    void
    pyrlk_cpu<D, M>::set_with_merge(bool b)
    {
      with_merge_ = b;
    }

    template <typename D, typename M>
    pyrlk_cpu<D, M>&
    pyrlk_cpu<D, M>::set_detector_frequency(unsigned nframe)
    {
      detector_frequency_ = nframe;
      if (upper_)
	upper_->set_detector_frequency(nframe);
      return *this;
    }


    template <typename D, typename M>
    pyrlk_cpu<D, M>&
    pyrlk_cpu<D, M>::set_filtering_frequency(unsigned nframe)
    {
      filtering_frequency_ = nframe;
      if (upper_)
	upper_->set_filtering_frequency(nframe);
      return *this;
    }

    template <typename D, typename M>
    int
    pyrlk_cpu<D, M>::border_needed() const
    {
      return detector_.border_needed();
    }


    template <typename D, typename M>
    void
    pyrlk_cpu<D, M>::update(const gl8u_image2d& in,
                            PC& pset)
    {
      if (frame_cpt_ > 0)
        copy(input_, prev_input_);
      else
      {
        copy(in, prev_input_);
      }

      input_ = host_image2d<gl8u>(in.domain(), in.border());
      copy(in, input_);
      // cv::Mat opencv_s1(tmp_);
      // cv::GaussianBlur(cv::Mat(in), opencv_s1, cv::Size(3, 3), 1, 1, cv::BORDER_REPLICATE);
      // // //cv::GaussianBlur(cv::Mat(in), opencv_s1, cv::Size(7, 7), 2, 2, cv::BORDER_REPLICATE);
      // copy(tmp_, input_);

      //pset.set_flow(flow_);
      if (frame_cpt_ > 0)
        match_particles(pset);

      if (!(frame_cpt_ % detector_frequency_))
      {
	create_detector_mask(pset);
	detector_.update(input_, mask_);
	new_particles(pset);
      }

      if (!(frame_cpt_ % filtering_frequency_))
      {
	filter(pset);

	START_PROF(merge_trajectories);
	if (with_merge_)
	  run_kernel1d_functor(merge_trajectories_kernel<PC>(pset),
			       pset.dense_particles().size(),
			       typename PC::architecture());
	END_PROF(merge_trajectories);
      }

      pset.tick();
      frame_cpt_++;
    }



    template <typename D, typename M>
    void
    pyrlk_cpu<D, M>::create_detector_mask(PC& pset)
    {
      memset(mask_, 255);
      run_kernel1d_functor(particle_mask_kernel<PC, gl8u_image2d>
			   (pset, mask_),
			   pset.size(), typename PC::architecture());

    }



    template <typename D, typename M>
    void
    pyrlk_cpu<D, M>::set_particle_domain(const box2d& d)
    {
      particle_domain_ = d;
    }


    template <typename D, typename M>
    void
    pyrlk_cpu<D, M>::match_particles(PC& pset)
    {
      pset.before_matching();

      // Matching
      START_PROF(matcher);

      fill(gradient_, i_float2(0,0));
      for (i_int2 p : input_.domain() - border(1))
      {
        gradient_(p)[0] =
          (1 * int(prev_input_(p + i_int2(1,-1)).x)
           + 2 * int(prev_input_(p + i_int2(1,0)).x)
           + 1 * int(prev_input_(p + i_int2(1,1)).x)
           - 1 * int(prev_input_(p + i_int2(-1,-1)).x)
           - 2 * int(prev_input_(p + i_int2(-1,0)).x)
           - 1 * int(prev_input_(p + i_int2(-1,1)).x)) / 8.;

        gradient_(p)[1] =
          (1 * int(prev_input_(p + i_int2(-1,1)).x)
           + 2 * int(prev_input_(p + i_int2(0,1)).x)
           + 1 * int(prev_input_(p + i_int2(1,1)).x)
           - 1 * int(prev_input_(p + i_int2(-1,-1)).x)
           - 2 * int(prev_input_(p + i_int2(0,-1)).x)
           - 1 * int(prev_input_(p + i_int2(1,-1)).x)) / 8.;

        // i_float2 g2;
        // g2[0] =
        //   (int(prev_input_(p + i_int2(1,0)).x) -
        //    int(prev_input_(p + i_int2(-1,0)).x)) / 2.;

        // g2[1] =
        //   (int(prev_input_(p + i_int2(0,1)).x) -
        //    int(prev_input_(p + i_int2(0,-1)).x)) / 2.;
        // gradient_(p) = g2;
        // std::cout << gradient_(p) << " " << g2 << std::endl;
      }

      pyrlk_match_particles_kernel<gl8u_image2d, self, uint_image2d, gradient_t, PC, M> func
	(pset, prev_input_, input_, gradient_, upper_, contrast_, flow_ratio, k_, particle_domain_);

      run_kernel1d_functor(func,
			   pset.dense_particles().size(),
			   typename PC::architecture());

      END_PROF(matcher);

      // Compute sparse flow.
      if (lower_)
      {
        memset(flow_stats_, 0);
        fill(flow_, transform::identity());
        memset(multiscale_count_, 0);
        run_kernel2d_functor(compute_flow_stats_kernel<PC, flow_stats_t>(pset, flow_stats_, flow_ratio),
                             flow_stats_.domain(), typename PC::architecture());
      }
      // Fusion with upper flow.
      // if (upper_)
      //   run_kernel2d_functor(flow_fusion_kernel<flow_stats_t, flow_t, uint_image2d>
      //   		     (flow_stats_, upper_->flow_stats_, flow_, upper_->flow_, multiscale_count_),
      //   		     flow_.domain(), typename PC::architecture());
      // else
      // {
      //   run_kernel2d_functor(flow_fusion_kernel_root<flow_stats_t, flow_t, uint_image2d>
      //   		     (flow_stats_, flow_, multiscale_count_),
      //   		     flow_.domain(), typename PC::architecture());
      // }

      pset.after_matching();
    }



    template <typename D, typename M>
    inline void
    pyrlk_cpu<D, M>::filter(PC& pset)
    {
      if (!(frame_cpt_ % filtering_frequency_))
      {
	// ****** Filter bad particles.
	START_PROF(filter_spacial_incoherences);

	run_kernel1d_functor(filter_bad_particles_kernel<PC, flow_t, uint_image2d>
			     (pset, flow_, multiscale_count_, flow_ratio),
			     pset.dense_particles().size(),
			     typename PC::architecture());

	END_PROF(filter_spacial_incoherences);
      }
    }

    template <typename D, typename M>
    inline void
    pyrlk_cpu<D, M>::new_particles(PC& pset)
    {
      dummy_feature f;
      detector_.new_particles(f, pset);
      pset.after_new_particles();
    }


    template <typename D, typename M>
    void
    pyrlk_cpu<D, M>::set_upper(self* u)
    {
      u->lower_ = this;
      upper_ = u;
    }

    template <typename D, typename M>
    pyrlk_cpu<D, M>*
    pyrlk_cpu<D, M>::upper()
    {
      return upper_;
    }

    template <typename D, typename M>
    void
    pyrlk_cpu<D, M>::clear()
    {
      frame_cpt_ = 0;
    }

    template <typename D, typename M>
    typename pyrlk_cpu<D, M>::transform
    pyrlk_cpu<D, M>::get_flow_at(const i_float2& p)
    {
      // return i_int2(0,0);
      // if (flow_stats_(p / flow_ratio).first)
      //   return flow_stats_(p / flow_ratio).second;
      // else
      {
	particle_f<transform> part;
	part.age = 1;
	part.pos = p;
        //return motion_based_prediction(part) - p;
	transform pred_transform = recursive_prediction(part, upper_);
        i_float2 pred = pred_transform.apply_transform(part.pos);
	if ((input_.domain() - border(3)).has(pred))
	{
	  float distance;
	  std::pair<transform, float> match_res = M::match(part.pos, pred_transform,
                                                           prev_input_,
                                                           input_,
                                                           gradient_);

	  transform tr = match_res.first;
          i_float2 match = tr.apply_transform(part.pos);
	  if (input_.domain().has(match))
	  {
	    flow_stats_(p / flow_ratio).first = 1;
	    flow_stats_(p / flow_ratio).second = tr;
	    return tr;
	  }
	}
        else
          return transform::identity();

      }

      // return i_float2(-100000,-100000);
    }

  }

}

#endif
