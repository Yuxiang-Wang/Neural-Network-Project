//Andrew Narang
//Advanced C++ - Term Structure

//Here, we implement the FF_NeuronLayer class

#ifndef FF_NeuronLayer_CPP
#define FF_NeuronLayer_CPP


#include <iostream>

#include <eigen3/Eigen/Dense>

#include "NeuronLayer.hpp"
#include "FF_NeuronLayer.hpp"

//will be useful once further updates are given
#include "SupportFunctions1.hpp"

using Eigen::Matrix;

using Eigen::Dynamic;

using Eigen::MatrixBase;



//implementation of FF_NeuronLayer class

//don't need to check for weight matrix being sized correctly - NeuronLayer constructor takes care of that

template<typename T>
template<typename rng_function>
FF_NeuronLayer<T>::FF_NeuronLayer(const rng_function& in_rng, const int& in_nodes, const int& in_inputs, const Matrix<T, Dynamic, Dynamic> * in_weights) : NeuronLayer<T>(in_rng, in_nodes, in_inputs, in_weights), error_weights(Matrix<T, Dynamic, Dynamic>(in_inputs, in_nodes)), errors(Matrix<T, Dynamic, Dynamic>(in_nodes, 1))
{
  //set up the error weight matrix - dimensions are opposite those of weights because numerators come from transpose of weights

  for (int i = 0; i < in_inputs; ++i)
    {
      for (int i1 = 0; i1 < in_nodes; ++i1)
	{
	  //using the denominator scales the weights, speeding up the training
	  
	  T denominator = 0;

	  //denominator = sum of all weights going towards a particular node, from all other nodes => hold column value steady, iterate over all rows in weights matrix
	  for (int i2 = 0; i2 < in_nodes; ++i2)
	    {
	      denominator += NeuronLayer<T>::weights(i2, i);
	    }
	  
	  error_weights(i, i1) = ((NeuronLayer<T>::weights(i1, i)) / denominator);
	}
    }

  std::cout << "FF_NeuronLayer parameterized constructor called." << std::endl;
}


template<typename T>
FF_NeuronLayer<T>::~FF_NeuronLayer()
{
  std::cout << "FF_NeuronLayer destructor called." << std::endl;
}


template<typename T>
template<typename Derived>
void FF_NeuronLayer<T>::node_errors(const MatrixBase<Derived>& next_layer_errors)
{
  errors = (error_weights * next_layer_errors);
}




#endif



