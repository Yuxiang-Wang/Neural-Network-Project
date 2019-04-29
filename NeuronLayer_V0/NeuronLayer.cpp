//Andrew Narang
//Advanced C++ - Term Project

//Implementation of NeuronLayer class.

#include <iostream>
#include <eigen3/Eigen/Dense>

#include "NeuronLayer.hpp"


using Eigen::Matrix;

using Eigen::Dynamic;


//we need to be most careful when setting up the weights matrix
NeuronLayer::NeuronLayer(const int& in_nodes, const int& in_inputs, double (*in_activ)(const double& input), Matrix<double, Dynamic, Dynamic> * in_weights, double (*in_rng)()) : nodes(in_nodes), inputs(in_inputs), activ(in_activ)
{
  //check if an input weights matrix has been provided
  if (in_weights != nullptr)
    {
      //checks if the given input weight matrix's dimensions
      if (((*in_weights).rows() == in_nodes) && (*in_weights).cols() == in_inputs)
	{
	  weights = new Matrix<double, Dynamic, Dynamic>((*in_weights));
	}
      else {
	//if the dimensions did not match

	weights = new Matrix<double, Dynamic, Dynamic>(in_nodes, in_inputs);

	for (int i = 0; i < in_nodes; ++i)
	  {
	    for (int i1 = 0; i1 < in_inputs; ++i1)
	      {
		(*(weights))(i, i1) = in_rng();
	      }
	  }
      }
    }
  else {

    weights = new Matrix<double, Dynamic, Dynamic>(in_nodes, in_inputs);

    /*
    for (int i = 0; i < in_nodes; ++i)
      {
	for (int i1 = 0; i1 < in_inputs; ++i1)
	  {
	    (*(weights))(i, i1) = in_rng();
	  }
      }
    */
  }

  std::cout << "NeuronLayer parameterized constructor called." << std::endl;
  
}


NeuronLayer::~NeuronLayer()
{
  delete weights;
  
  std::cout << "NeuronLayer destructor called." << std::endl;
}


Matrix<double, Dynamic, Dynamic> NeuronLayer::operator () (const Matrix<double, Dynamic, Dynamic>& in_vector)
{
  return ((*weights) * in_vector).unaryExpr(activ); 

}
