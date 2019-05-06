//Andrew Narang
//Advanced C++ - Term Structure

//Here, we implement the FF_NeuronLayer class

#ifndef FF_NeuronLayer_CPP
#define FF_NeuronLayer_CPP


#include <iostream>

#include <eigen3/Eigen/Dense>

#include "NeuronLayer.hpp"
#include "FF_NeuronLayer.hpp"

//will be more useful once further updates are given
#include "SupportFunctions1.hpp"

using Eigen::Matrix;

using Eigen::Dynamic;

using Eigen::MatrixBase;



//implementation of FF_NeuronLayer class

//constructor #1
template<typename T>
template<typename rng_function>
FF_NeuronLayer<T>::FF_NeuronLayer(const rng_function& in_rng, const int& in_nodes, const int& in_inputs, const int& in_BiasFlag) : NeuronLayer<T>(in_rng, in_nodes, in_inputs, in_BiasFlag), error_weights(in_inputs + in_BiasFlag, in_nodes), errors(in_inputs + in_BiasFlag, 1)
{
  //set up the error weight matrix - dimensions are opposite those of weights because numerators come from transpose of weights

  for (int i = 0; i < (in_inputs + in_BiasFlag); ++i)
    {
      for (int i1 = 0; i1 < in_nodes; ++i1)
	{
	  //using the denominator scales the weights, speeding up the training
	  
	  T denominator = static_cast<T>(0.0);

	  //denominator = sum of all weights going towards a particular node, from all other nodes => hold column value steady, iterate over all rows in weights matrix
	  for (int i2 = 0; i2 < in_nodes; ++i2)
	    {
	      denominator += NeuronLayer<T>::weights(i2, i);
	    }
	  
	  error_weights(i, i1) = ((NeuronLayer<T>::weights(i1, i)) / denominator);
	}
    }

  std::cout << "FF_NeuronLayer parameterized constructor #1 called." << std::endl;
}


//constructor #2
//is it better to call the cols and rows functions here or to use the data members from the base class NeuronLayer?
template<typename T>
template<typename Derived>
FF_NeuronLayer<T>::FF_NeuronLayer(const MatrixBase<Derived>& in_weights, const int& in_BiasFlag) : NeuronLayer<T>(in_weights, in_BiasFlag), error_weights(NeuronLayer<T>::inputs + in_BiasFlag, NeuronLayer<T>::nodes), errors(NeuronLayer<T>::inputs + in_BiasFlag, 1)
{
  //set up the error weight matrix - dimensions are opposite those of weights because numerators come from transpose of weights

  for (int i = 0; i < (NeuronLayer<T>::inputs + in_BiasFlag); ++i)
    {
      for (int i1 = 0; i1 < NeuronLayer<T>::nodes; ++i1)
	{
	  //using the denominator scales the weights, speeding up the training

	  //should we cast 0.0 to T - perhaps via static_cast ?
	  T denominator = static_cast<T>(0.0);

	  //denominator = sum of all weights going towards a particular node, from all other nodes => hold column value steady, iterate over all rows in weights matrix
	  for (int i2 = 0; i2 < NeuronLayer<T>::nodes; ++i2)
	    {
	      denominator += NeuronLayer<T>::weights(i2, i);
	    }
	  
	  error_weights(i, i1) = ((NeuronLayer<T>::weights(i1, i)) / denominator);
	}
    }

  std::cout << "FF_NeuronLayer parameterized constructor #2 called." << std::endl;
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



