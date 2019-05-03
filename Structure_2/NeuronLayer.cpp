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

  std::cout << "NeuronLayer parameterized constructor called." << std::endl;

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


//accomodates any input vector and any unary functor
template<typename T>
template<typename Derived, typename fun0>
Matrix<T, Dynamic, Dynamic> NeuronLayer<T>::operator () (const MatrixBase<Derived>& in_vector, const fun0& in_activ)
{
  std::cout << weights << std::endl;

  return (weights * in_vector).unaryExpr(in_activ);
}





#endif
