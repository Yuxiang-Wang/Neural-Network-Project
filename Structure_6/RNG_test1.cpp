//Andrew Narang
//Advanced C++ - Term Project


//testing of random number generation technique


#include <iostream>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include "SupportFunctions1.hpp"


int main()
{
  norm_rng<double> ok;


  for (int i = 0; i < 100; ++i)
    {
      std::cout << ok() << ' ';
    }

  return 0;
}
