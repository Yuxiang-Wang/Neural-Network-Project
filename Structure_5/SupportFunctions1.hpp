//Andrew Narang
//Advanced C++ - Term Project

//this file contains the declarations for the support functors for the NeuronLayer & NN class hierarchies.

#ifndef SupportFunctions1_HPP
#define SupportFunctions1_HPP


#include <ctime>

#include <eigen3/Eigen/Dense>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>


using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::MatrixBase;


//actually, we can make this a generic normal distribution random number generator - just change it so default arguments are 0 and 1 => standard normal distribution

//function object that generates random numbers from the standard normal distribution

template<typename T>
class norm_rng
{
private:
  //replace this with a smart pointer
  boost::mt19937 * rng;

  int seed;

  //replace this with a smart pointer
  boost::normal_distribution<> * norm;

  T mean;

  T std_dev;
  
public:

  //seed is random
  norm_rng(const int& in_seed = std::time(0), const T& in_mean = static_cast<T>(0.0), const T& in_std_dev = static_cast<T>(1.0));

  virtual ~norm_rng();

  T operator() () const; 
  
};


//activation function: sigmoid function
template<typename T>
T sigmoid(const T& input); 


//derivative of sigmoid function
template<typename T>
T dsigmoid_dx(const T& input);

//gradient function #1: designed to generate a matrix of weight changes according to the gradient of the error function with respect to the weights. This is only implementation of a matrix calculus formula that should be derived beforehand.
//this one is designed according to the formula in pg. 99 of "Make Your Own Neural Network," by Tariq Rashid
template<typename T, Derived>
Matrix<T, Dynamic, Dynamic> sigmoid_grad(const MatrixBase<Derived>& errors_k, const MatrixBase<Derived>& outputs_k, const MatrixBase<Derived>& outputs_j);

//loss functions

//loss function #1

//loss function #2

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
