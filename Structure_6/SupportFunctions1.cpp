//Andrew Narang
//Advanced C++ - Term Project

//this file contains the implementations of the support functors for the NeuronLayer and NN class hierarchies


#ifndef SupportFunctions1_CPP
#define SupportFunctions1_CPP

#include <iostream>
#include <cmath>

#include <eigen3/Eigen/Dense>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include "SupportFunctions1.hpp"


using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::MatrixBase;

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


//activation function: sigmoid function
template<typename T>
T sigmoid(const T& input)
{
  return (1 / (1 + std::exp(static_cast<T>(-1.0) * input)));
}


//derivative of sigmoid function
template<typename T>
T dsigmoid_dx(const T& input)
{
  return sigmoid<T>(input) * (1 - sigmoid<T>(input));
}

//gradient function object #1

template<typename T>
sigmoid_grad<T>::sigmoid_grad()
{
  std::cout << "sigmoid_grad constructor called." << std::endl;
}


template<typename T>
sigmoid_grad<T>::~sigmoid_grad()
{
  std::cout << "sigmoid_grad destructor called." << std::endl;
}



template<typename T>
template<typename Derived>
Matrix<T, Dynamic, Dynamic> sigmoid_grad<T>::operator () (const MatrixBase<Derived>& errors_k, const MatrixBase<Derived>& outputs_k, const MatrixBase<Derived>& outputs_j) const
{  
  Matrix<T, Dynamic, Dynamic> alpha(outputs_k.template rows(), 1);

  for (int i = 0; i < outputs_k.template rows(); ++i)
    {
      alpha(i, 0) = static_cast<T>(1.0) - outputs_k(i, 0);
    }

  return (errors_k * outputs_k * alpha * outputs_j.transpose());
}


//loss functions

//loss function #1 - implement this with unaryExpr function
//template<typename T>
//T quad_err(const T& output, const T& target)
//{
//  return ((output - target) * (output - target));
//}


//loss function object #1
template<typename T>
quad_err<T>::quad_err()
{
  std::cout << "quad_err constructor called." << std::endl;
}


template<typename T>
quad_err<T>::~quad_err()
{
  std::cout << "quad_err destructor called." << std::endl;
}

template<typename T>
template<typename Derived>
Matrix<T, Dynamic, Dynamic> quad_err<T>::operator () (const MatrixBase<Derived>& output, const MatrixBase<Derived>& target)
{
  //these are vectors so columns = 1
  Matrix<T, Dynamic, Dynamic> result(output.template rows(), 1);

  for (int i = 0; i < output.template rows(); ++i)
    {
      result(i, 0) = (output(i, 0) - target(i, 0)) * (output(i, 0) - target(i, 0));
    }

  return result; 
}



#endif

