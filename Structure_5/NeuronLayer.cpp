//Andrew Narang
//Advanced C++ - Term Project

//Implementation of NeuronLayer class.


#ifndef NeuronLayer_CPP
#define NeuronLayer_CPP


#include <iostream>
#include <eigen3/Eigen/Dense>

#include "NeuronLayer.hpp"
#include "SupportFunctions1.hpp"

using Eigen::Matrix;

using Eigen::Dynamic;

using Eigen::MatrixBase;


//template<typename T>
//template<typename Derived>
//NeuronLayer<T>::NeuronLayer(const int& in_nodes, const int& in_inputs, const MatrixBase<Derived> * in_weights) : nodes(in_nodes), inputs(in_inputs), weights(Matrix<T, Dynamic, Dynamic>(in_nodes, in_inputs))
//{
  //check if an input weights matrix has been provided
//  if (in_weights != nullptr)
// {
//    //checks if the given input weight matrix's dimensions
//    if (((*in_weights).rows() == in_nodes) && (*in_weights).cols() == in_inputs)
//	{
//	  weights = Matrix<T, Dynamic, Dynamic>((*in_weights));
//	}
//  }

//std::cout << "NeuronLayer parameterized constructor V1 called." << std::endl;
//}


/*
template<typename T>
NeuronLayer<T>::NeuronLayer(const int& in_nodes, const int& in_inputs, const Matrix<T, Dynamic, Dynamic> * in_weights) : nodes(in_nodes), inputs(in_inputs), weights(Matrix<T, Dynamic, Dynamic>(in_nodes, in_inputs))
{
  //check if an input weights matrix has been provided
  if (in_weights != nullptr)
    {
      //checks if the given input weight matrix's dimensions
      if (((*in_weights).rows() == in_nodes) && (*in_weights).cols() == in_inputs)
	{
	  weights = Matrix<T, Dynamic, Dynamic>((*in_weights));
	}
      //else {

      //standard_norm<T> finding

	
    }

  std::cout << "NeuronLayer parameterized constructor V1 called." << std::endl;
}
*/

/*
//deprecated
//constructor #3
template<typename T>
template<typename rng_function>
NeuronLayer<T>::NeuronLayer(const rng_function& in_rng, const int& in_nodes, const int& in_inputs, const Matrix<T, Dynamic, Dynamic> * in_weights) : nodes(in_nodes), inputs(in_inputs)
{
  //check if an input weights matrix has been provided
  if (in_weights != nullptr)
    {
      //checks if the given input weight matrix's dimensions
      if (((*in_weights).rows() == in_nodes) && (*in_weights).cols() == in_inputs)
	{
	  weights = Matrix<T, Dynamic, Dynamic>((*in_weights));
	}
      else {
	//if the dimensions did not match

	weights = Matrix<T, Dynamic, Dynamic>(in_nodes, in_inputs);

	for (int i = 0; i < in_nodes; ++i)
	  {
	    for (int i1 = 0; i1 < in_inputs; ++i1)
	      {
		weights(i, i1) = in_rng();
	      }
	  }
      }
    }
  else {

    weights = Matrix<T, Dynamic, Dynamic>(in_nodes, in_inputs);

    for (int i = 0; i < in_nodes; ++i)
      {
	for (int i1 = 0; i1 < in_inputs; ++i1)
	  {
	    weights(i, i1) = in_rng();
	  }
      }
    

  }

  std::cout << "NeuronLayer parameterized constructor #3 called." << std::endl;

}
*/


//the original way, where we called the constructor inside weights(), essentially makes a temporary Matrix object and then weights copy constructs it. This is inefficient. 
//constructor #1
template<typename T>
template<typename rng_function>
NeuronLayer<T>::NeuronLayer(const rng_function& in_rng, const int& in_nodes, const int& in_inputs, const int& in_BiasFlag) : nodes(in_nodes), inputs(in_inputs), BiasFlag(in_BiasFlag), weights(in_nodes, (in_inputs + in_BiasFlag)), outputs(in_nodes, 1)
{
  //randomly create weights according to your desired distribution 
  for (int i = 0; i < in_nodes; ++i)
    {
      for (int i1 = 0; i1 < (in_inputs + in_BiasFlag); ++i1)
	{
	  weights(i, i1) = in_rng();
	}
    }

  
  std::cout << "NeuronLayer parameterized constructor #1 called." << std::endl;
}


//constructor #2 - no checking for row # and column #, everything inferred from input weight matrix
//check if it works!
template<typename T>
template<typename Derived>
NeuronLayer<T>::NeuronLayer(const MatrixBase<Derived>& in_weights, const int& in_BiasFlag) : nodes(in_weights.template rows()), inputs((in_weights.template cols()) - in_BiasFlag), BiasFlag(in_BiasFlag), weights(in_weights)
{
  std::cout << "NeuronLayer parameterized constructor #2 called." << std::endl;
}
																		 
//template<typename T>
//NeuronLayer<T>::NeuronLayer(const int& in_nodes, const int& in_inputs) : nodes(in_nodes), inputs(in_//inputs), weights(Matrix<T, Dynamic, Dynamic>(in_nodes, in_inputs))
//{
//  std::cout << "basic constructor called." << std::endl;
//}



/*
//we need to be most careful when setting up the weights matrix
template<typename T>
template<typename Derived, typename rng_function>
NeuronLayer<T>::NeuronLayer(const int& in_nodes, const int& in_inputs, const MatrixBase<Derived> * in_weights, const rng_function& in_rng) : nodes(in_nodes), inputs(in_inputs)
{
  //check if an input weights matrix has been provided
  if (in_weights != nullptr)
    {
      //checks if the given input weight matrix's dimensions
      if (((*in_weights).rows() == in_nodes) && (*in_weights).cols() == in_inputs)
	{
	  weights = Matrix<T, Dynamic, Dynamic>((*in_weights));
	}
      else {
	//if the dimensions did not match

	weights = Matrix<T, Dynamic, Dynamic>(in_nodes, in_inputs);

	for (int i = 0; i < in_nodes; ++i)
	  {
	    for (int i1 = 0; i1 < in_inputs; ++i1)
	      {
		weights(i, i1) = in_rng();
	      }
	  }
      }
    }
  else {

    weights = Matrix<T, Dynamic, Dynamic>(in_nodes, in_inputs);

    for (int i = 0; i < in_nodes; ++i)
      {
	for (int i1 = 0; i1 < in_inputs; ++i1)
	  {
	    weights(i, i1) = in_rng();
	  }
      }
    

  }

  std::cout << "NeuronLayer parameterized constructor called." << std::endl;
      
}
*/


//template<typename T>
//NeuronLayer<T>::NeuronLayer() : nodes(1), inputs(1), weights(Matrix<T, Dynamic, Dynamic>(1, 1))
//{
//  std::cout << "NeuronLayer default constructor called." << std::endl;
//}


template<typename T>
NeuronLayer<T>::~NeuronLayer()
{  
  std::cout << "NeuronLayer destructor called." << std::endl;
}


/* //for private implementation

template<typename T>
const int& NeuronLayer<T>::get_nodes() const
{
  return nodes;
}


template<typename T>
void NeuronLayer<T>::set_nodes(const int& new_nodes)
{
  nodes = new_nodes;
}


template<typename T>
const int& NeuronLayer<T>::get_inputs() const
{
  return inputs;
}


template<typename T>
void NeuronLayer<T>::set_inputs(const int& new_inputs)
{
  inputs = new_inputs;
}

template<typename T>
const Matrix<T, Dynamic, Dynamic>& NeuronLayer<T>::get_weights() const
{
  return weights;
}


template<typename T>
template<typename Derived>
void set_weights(const Matrix<T, Dynamic, Dynamic>& new_weights)
{
  if ((new_weights.template rows() == nodes) && (new_weights.template cols() == inputs))
    {
      weights = new_weights;
    }

  //do nothing otherwise
}
*/


//accounting for bias inputs makes the code relatively slower, but makes the whole process much more accurate, so it is worth it.
//accomodates any input vector and any unary functor
template<typename T>
template<typename Derived, typename fun0>
Matrix<T, Dynamic, Dynamic> NeuronLayer<T>::operator () (const MatrixBase<Derived>& in_vector, const fun0& in_activ)
{
  //std::cout << weights << std::endl;

  //should we use template version here?

  if (BiasFlag)
    {
      std::cout << weights << std::endl;
      
      //we know that in_vector only has one column and has inputs rows
      Matrix<T, Dynamic, Dynamic> local((inputs + 1), 1);

      for (int i = 0; i < inputs; ++i)
	{
	  local(i, 0) = in_vector(i, 0);
	}

      local(inputs, 0) = static_cast<T>(1.0);

      return (weights * local).unaryExpr(in_activ);
      
    }

  //std::cout << "working?" << std::endl;

  //std::cout << "weights:\n" << weights << "\ninput vector:\n" << in_vector << std::endl; 

  return (weights * in_vector).unaryExpr(in_activ);
  
  
}

template<typename T>
template<typename Derived>
void NeuronLayer<T>::update_weights(const MatrixBase<Derived>& in_delta_weights)
{
  weights = weights + in_delta_weights;
}



#endif
