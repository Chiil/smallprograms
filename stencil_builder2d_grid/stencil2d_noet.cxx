// StencilBuilder example
// Copyright (c) Chiel van Heerwaarden, 2015
// chielvanheerwaarden@gmail.com

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include "StencilBuilder.h"

using namespace StencilBuilder;

// Fourth order interpolation function.
inline double interp(const double m2, const double m1, const double p1, const double p2)
{
  return (-1./16)*(m2+p2) + (9./16)*(m1+p1);
}

// Fourth order gradient function.
inline double grad(const double m2, const double m1, const double p1, const double p2)
{
  return (1./24.)*(m2-p2) + (27./24.)*(p1-m1);
}

// Test function with a similar structure as the advection operator.
void advection(double * const restrict ut,
               const double * const restrict u, const double * const restrict v,
               const int istart, const int iend,
               const int jstart, const int jend,
               const int icells)
{
  const int ii1 = 1;
  const int ii2 = 2;
  const int ii3 = 3;
  const int jj1 = 1*icells;
  const int jj2 = 2*icells;
  const int jj3 = 3*icells;

  for (int j=jstart; j<jend; ++j)
    #pragma clang loop vectorize(enable)
    #pragma GCC ivdep
    #pragma ivdep
    for (int i=istart; i<iend; ++i)
    {
      const int ijk = i + j*jj1;
      ut[ijk] += grad( interp( u[ijk-ii3], u[ijk-ii2], u[ijk-ii1], u[ijk    ] ) * interp( u[ijk-ii3], u[ijk-ii2], u[ijk-ii1], u[ijk    ] ),
                       interp( u[ijk-ii2], u[ijk-ii1], u[ijk    ], u[ijk+ii1] ) * interp( u[ijk-ii2], u[ijk-ii1], u[ijk    ], u[ijk+ii1] ),
                       interp( u[ijk-ii1], u[ijk    ], u[ijk+ii1], u[ijk+ii2] ) * interp( u[ijk-ii1], u[ijk    ], u[ijk+ii1], u[ijk+ii2] ),
                       interp( u[ijk    ], u[ijk+ii1], u[ijk+ii2], u[ijk+ii3] ) * interp( u[ijk    ], u[ijk+ii1], u[ijk+ii2], u[ijk+ii3] ))

               + grad( interp( v[ijk-ii2-jj1], v[ijk-ii1-jj1], v[ijk-jj1], v[ijk+ii1-jj1] ) * interp( u[ijk-jj3], u[ijk-jj2], u[ijk-jj1], u[ijk    ] ),
                       interp( v[ijk-ii2    ], v[ijk-ii1    ], v[ijk    ], v[ijk+ii1    ] ) * interp( u[ijk-jj2], u[ijk-jj1], u[ijk    ], u[ijk+jj1] ),
                       interp( v[ijk-ii2+jj1], v[ijk-ii1+jj1], v[ijk+jj1], v[ijk+ii1+jj1] ) * interp( u[ijk-jj1], u[ijk    ], u[ijk+jj1], u[ijk+jj2] ),
                       interp( v[ijk-ii2+jj2], v[ijk-ii1+jj2], v[ijk+jj2], v[ijk+ii1+jj2] ) * interp( u[ijk    ], u[ijk+jj1], u[ijk+jj2], u[ijk+jj3] ));
    }
}

  // Test function for time integration.
void tendency(double * const restrict at, double * const restrict a,
              const double dt,
              const int istart, const int iend,
              const int jstart, const int jend,
              const int icells)
{
  const int jj = icells;

  for (int j=jstart; j<jend; ++j)
    #pragma clang loop vectorize(enable)
    #pragma GCC ivdep
    #pragma ivdep
    for (int i=istart; i<iend; ++i)
    {
      const int ijk = i + j*jj;
      a[ijk] += dt*at[ijk];
    }

  for (int j=jstart; j<jend; ++j)
    #pragma clang loop vectorize(enable)
    #pragma GCC ivdep
    #pragma ivdep
    for (int i=istart; i<iend; ++i)
    {
      const int ijk = i + j*jj;
      at[ijk] = 0.;
    }
}

int main()
{
  // Test configuration settings.
  const int itot = 2048;
  const int jtot = 2048;
  const int gc   = 4;
  const int iter = 50;

  // Initialize the grid.
  Grid grid(itot, jtot, gc);

  // Create fields on the grid.
  Field u (grid);
  Field v (grid);
  Field ut(grid);

  // Initialize the fields.
  u.randomize();
  v.randomize();

  // Initialize a time step.
  const double dt = 1.e-3;

  // Execute the loop iter times.
  for (int ii=0; ii<iter; ++ii)
  {
    advection(ut.get_data(),
              u.get_data(), v.get_data(),
              grid.istart, grid.iend,
              grid.jstart, grid.jend,
              grid.icells);

    tendency(ut.get_data(), u.get_data(),
             dt,
             grid.istart, grid.iend,
             grid.jstart, grid.jend,
             grid.icells);
  }

  // Print a value in the middle of the field.
  std::cout << std::setprecision(8) << "u = " << u(itot/2, jtot/2) << std::endl;

  return 0;
}
