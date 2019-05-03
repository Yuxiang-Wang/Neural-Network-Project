//Andrew Narang
//Advanced C++ - Term Project

//this file contains the declarations for the support functors for the NeuronLayer & NN class hierarchies.

#ifndef SupportFunctions1_HPP
#define SupportFunctions1_HPP

#include <ctime>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

//actually, we can make this a generic normal distribution random number generator - just change it so default arguments are 0 and 1 => standard normal distribution

//function object that generates random numbers from the standard normal distribution

template<typename T>
class norm_rng
{
private:
  //replace this with a smart pointer
  boost::mt19937 * rng;

  int seed;

  boost::normal_distribution<> * norm;

  T mean;

  T std_dev;
  
public:

  //seed is random
  norm_rng(const int& in_seed = std::time(0), const T& in_mean = T(0.0), const T& in_std_dev = T(1.0));

  virtual ~norm_rng();

  T operator() () const; 
  
};


//test function object
class test
{
public:

  test() {}

  ~test() {}

  double operator () () const {return 100;}
};

#ifndef SupportFunctions1_CPP
#include "SupportFunctions1.cpp"
#endif


#endif 
