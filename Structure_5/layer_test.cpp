//Andrew Narang
//Advanced C++ - Term Project

#include <iostream>
#include <eigen3/Eigen/Dense>

#include "SupportFunctions1.hpp"

#include "NeuronLayer.hpp"
#include "FF_NeuronLayer.hpp"
#include "FFNN.hpp"

using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::MatrixBase;


//template<typename T>
//T test0(const T& input);

//template<typename T>
//T test0(const T& input)
//{
//  return input * static_cast<T>(2.0);
//}

double test0(const double& input);

double test0(const double& input)
{
  return input * 2.0;
}

int main()
{
  norm_rng<double> cookie;
  
  NeuronLayer<double> vodka0(cookie, 1, 2);

  std::cout << vodka0.weights << std::endl;

  Matrix<double, Dynamic, Dynamic> weights_test(3, 4);

  for (int i = 0; i < 3; ++i)
    {
      for (int i1 = 0; i1 < 4; ++i1)
	{
	  weights_test(i, i1) = i + i1;
	}
    }

  std::cout << weights_test << std::endl;

  NeuronLayer<double> vodka1(weights_test, 0);

  std::cout << std::endl << vodka1.weights << std::endl;

  FF_NeuronLayer<double> sake0(cookie, 3, 4);

  std::cout << sake0.weights << std::endl << std::endl;

  std::cout << sake0.error_weights << std::endl;

  Matrix<double, Dynamic, Dynamic> inputs_test0(4, 1);

  //inputs_test0 << 2, 3, 4, 6;

  for (int i = 0; i < 4; ++i)
    {
      inputs_test0(i, 0) = i;
    }
  
  double (*test0_ptr)(const double&) = test0;

  std::cout << "output of vodka1(inputs_test0, test0), where vodka1 has no bias input:\n" << vodka1(inputs_test0, test0_ptr) << std::endl;
  
  std::cout << "output of sake0(inputs_test0, test0), where sake0 has a bias input:\n" << sake0(inputs_test0, test0_ptr) << std::endl;

  Matrix<double, Dynamic, Dynamic> errors_test0(3, 1);

  errors_test0 << 0.2, 0.4, 0.5;

  //I think I already tested this in the last file, but that one had no bias input.
  sake0.node_errors(errors_test0);
  
  std::cout << "result of sake0.node_errors(errors_test0), where sake0 has a bias input:\n" << sake0.errors << std::endl;

  
  //try {
  //std::cout << "output of vodka1(inputs_test, test0), where vodka1 has no bias input:\n" << vodka1(inputs_test0, test0_ptr) << "\nerror?" << std::endl;
  //}
  //catch (...)
  //  {
  //    std::cout << "wow" << std::endl;
  //  }


  //time to test FFNN!!!!!
  //we got this!!!!!!

  
  
  
  return 0;
}
