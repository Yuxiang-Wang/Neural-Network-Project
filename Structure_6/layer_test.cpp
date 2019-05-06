//Andrew Narang
//Advanced C++ - Term Project

#include <iostream>
#include <vector>

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

  //test of outputs as private member
  std::cout << "output of sake0(inputs_test0, test0), where sake0 has a bias input, via data member outputs (should match previous result):\n" << sake0.outputs << std::endl;
  
  Matrix<double, Dynamic, Dynamic> errors_test0(3, 1);

  errors_test0 << 0.2, 0.4, 0.5;

  //I think I already tested this in the last file, but that one had no bias input.
  sake0.node_errors(errors_test0);
  
  std::cout << "result of sake0.node_errors(errors_test0), where sake0 has a bias input:\n" << sake0.errors << std::endl;

  //time to test FFNN!!!!!
  //we got this!!!!!!

  double (*activ0)(const double& input) = sigmoid;

  //Matrix<double, Dynamic, Dynamic> (*grad0) (const Matrix<double, Dynamic, Dynamic>& errors_k, const Matrix<double, Dynamic, Dynamic>& outputs_k, const Matrix<double, Dynamic, Dynamic>& outputs_j) = sigmoid_grad;

  sigmoid_grad<double> grad0; 

  //double (*err0)(const double& output, const double&input) = quad_err; 

  quad_err<double> err0; 
  
  std::vector<FF_NeuronLayer<double>> FFNN_base0;
  
  FFNN_base0.push_back(FF_NeuronLayer<double>(cookie, 2, 2));

  FFNN_base0.push_back(FF_NeuronLayer<double>(cookie, 2, 2));

  //each column = a set of input data for one data point (X-vector)
  Matrix<double, Dynamic, Dynamic> input_data0(2, 3);
  
  input_data0 << 0.2, 0.4, 0.8, 0.92, 0.67, 0.32;

  //each column = a set of input data for one data point (Y-vector)
  Matrix<double, Dynamic, Dynamic> output_data0(2, 3);

  output_data0 << 0.44, 0.22, 0.33, 0.04, 0.08, 0.06;
  
  FFNN<double, decltype(activ0), decltype(err0), decltype(grad0)> FFNN_0(FFNN_base0, activ0, err0, grad0, 0.95); 

  //test input by querying

  Matrix<double, Dynamic, Dynamic> test_FFNN_input0(2, 1);

  test_FFNN_input0 << 0.05, 0.08;

  std::cout << FFNN_base0[0](test_FFNN_input0, activ0) << std::endl;
  
  std::cout << std::endl << "FFNN_0.query(test_FFNN_input0):\n" << FFNN_0.query(test_FFNN_input0) << std::endl; 

  Matrix<double, Dynamic, Dynamic> test_FFNN_output0(2, 1);

  test_FFNN_output0 << 0.9, 0.2;
  
  FFNN_0.generate_errors(test_FFNN_input0, test_FFNN_output0);

  std::cout << "\nalmost there" << std::endl;

  FFNN_0.train(input_data0, output_data0);

  std::cout << "\nFINISHED!" << std::endl;
  
  
  return 0;
}
