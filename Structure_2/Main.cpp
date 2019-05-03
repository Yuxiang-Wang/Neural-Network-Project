//Andrew Narang
//Advanced C++ - Term Project


#include <iostream>
#include <eigen3/Eigen/Dense>

#include "NeuronLayer.hpp"
#include "FF_NeuronLayer.hpp"
#include "SupportFunctions1.hpp"

using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::MatrixBase;

double test0(const double& input);

double test0(const double& input)
{
  return input + 100;
}

double test1();

double test1()
{
  return 100;
}

int main()
{
  //const Matrix<double, Dynamic, Dynamic> * not_needed = nullptr;

  double (*func)() = test1;

  //const decltype(test1) * test2 = test1;
  
  NeuronLayer<double> alpha(func, 1, 1); 

  std::cout << alpha.weights << std::endl;
  
  Matrix<double, Dynamic, Dynamic> test_input(1, 1);

  test_input(0, 0) = 3;

  double (*function)(const double&) = test0;

  norm_rng<double> stripes;

  NeuronLayer<double> bravo(stripes, 5, 5);

  std::cout << bravo.weights << std::endl;

  
  try
    {  
      std::cout << alpha(test_input, function) << std::endl;
    }
  catch (...)
    {
      std::cout << "dimensions prob didn't match fam" << std::endl;
    }
  
  FF_NeuronLayer<double> g0(stripes, 4, 4); 

  std::cout << "g0 weights:\n" << g0.weights << std::endl;

  //used in backpropagation
  std::cout << "g0 error_weights:\n" << g0.error_weights << std::endl;  

  Matrix<double, Dynamic, Dynamic> target0(4, 1);

  for (int i = 0; i < 4; ++i)
    {
      target0(i, 0) = i + 3;
    }

  std::cout << "target0:\n" << target0 << std::endl;

  //used in backpropagation
  g0.node_errors(target0);
  
  std::cout << "g0 node errors:\n" << g0.errors << std::endl;


  
  return 0;
}
