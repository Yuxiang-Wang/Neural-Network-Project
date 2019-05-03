//Andrew Narang
//Advanced C++ - Term Project

//this file contains the implementations of the support functors for the NeuronLayer and NN class hierarchies


#ifndef SupportFunctions1_CPP
#define SupportFunctions1_CPP

#include <iostream>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include "SupportFunctions1.hpp"


//implementation of function object that generates random numbers from the normal distribution, standard normal distribution by default

template<typename T>
norm_rng<T>::norm_rng(const int& in_seed, const T& in_mean, const T& in_std_dev) : seed(in_seed), mean(in_mean), std_dev(in_std_dev), rng(new boost::mt19937(static_cast<boost::uint32_t>(in_seed))), norm(new boost::normal_distribution<>(in_mean, in_std_dev))
{
  std::cout << "norm_rng constructor called." << std::endl;
}


template<typename T>
norm_rng<T>::~norm_rng()
{
  delete norm;
  delete rng;

  std::cout << "norm_rng destructor called. " << std::endl;

}

template<typename T>
T norm_rng<T>::operator () () const
{
  return static_cast<T>( (*norm)(*rng) );
}








#endif
