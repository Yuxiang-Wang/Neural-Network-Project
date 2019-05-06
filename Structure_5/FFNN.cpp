//Andrew Narang
//Advanced C++ - Term Project

//Here, we implement the FFNN class.

#ifndef FFNN_CPP
#define FFNN_CPP


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



//deprecated - more efficient versions have been created. 
//update handling of in_matrix_weights
//template<typename T, typename fun0>
//template<typename rng_function>
//FFNN<T, fun0>::FFNN(const rng_function& in_rng, const fun0& in_activation, const fun0& in_loss, const fun0& in_gradient, const int& in_layer_no, const std::vector<int>& in_nodes_per_layer, const std::vector<int>& in_inputs_per_layer, const std::vector<const Matrix<T, Dynamic, Dynamic> *>& in_matrix_weights, Matrix<T, Dynamic, Dynamic> * in_data_input, Matrix<T, Dynamic, Dynamic> * in_data_output) : activation(in_activation), loss(in_loss), gradient(in_gradient), layer_no(in_layer_no), nodes_per_layer(in_nodes_per_layer), inputs_per_layer(in_inputs_per_layer), data_input(in_data_input), data_output(in_data_output), Layers()
//{
  //fill the Layers
//  for (int i = 0; i < in_layer_no; ++i)
//    {
//      Layers.push_back(FF_NeuronLayer<T>(in_rng, in_nodes_per_layer[i], in_inputs_per_layer[i]));
//    }
	 
// std::cout << "FFNN constructor called." << std::endl;
//}


//constructor #1
template<typename T, typename fun0, typename fun1, typename, fun2>
template<typename Derived>
FFNN<T, fun0, fun1, fun2>::FFNN(const std::vector<FF_NeuronLayer<T>>& in_layers, const fun0& in_activ, const fun1& in_loss, const fun2& in_grad, const T& in_lrn_rate, const MatrixBase<Derived>& in_input_data, const MatrixBase<Derived>& in_output_data) : layers(in_layers), layer_no(in_layers.size()), nodes_per_layer(in_layers.size()), inputs_per_layer(in_layers.size()), activ(in_activ), loss(in_loss), grad(in_grad), lrn_rate(in_lrn_rate), input_data(in_input_data), output_data(in_output_data)
{
  //fill nodes_per_layer
  for (int i = 0; i < layer_no; ++i)
    {
      nodes_per_layer[i] = in_layers[i].nodes;
    }

  //fill inputs_per_layer
  for (int i = 0; i < layer_no; ++i)
    {
      inputs_per_layer[i] = in_layers[i].inputs;
    } 
  
  std::cout << "FFNN constructor #1 called." << std::endl;
}





template<typename T, typename fun0>
FFNN<T, fun0>::~FFNN()
{
  std::cout << "FFNN destructor called." << std::endl;
}


//finish implementing this properly later.
/*
//add exception handling
template<typename T, typename fun0>
template<typename Derived>
int FFNN<T, fun0>::set_data_input(const MatrixBase<Derived>& new_input)
{
  if (new_input.template cols() != Layers[0].inputs)
    {
      std::cerr << "number of features != number of inputs\n";
      return 0;
    }

  if((*data_output).size())
    {
      if (new_input.template rows() != (*data_output).rows())
	{
	  std::cerr << "number of inputs != number of outputs\n";
	  return 0;
	}
    }

  data_input = new_input;

  return 1;
}


template<typename T, typename fun0>
template<typename Derived>
int FFNN<T, fun0>::set_data_output(const MatrixBase<Derived>& new_output)
{}
*/


template<typename T, typename fun0, typename fun1, typename fun2>
template<typename Derived>
Matrix<T, Dynamic, Dynamic> FFNN<T, fun0, fun1, fun2>::query(const MatrixBase<Derived>& in_arg) const
{
  Matrix<T, Dynamic, Dynamic> intermediate(layers[0](in_arg, activ));
  
  for (int i = 1; i < layer_no; ++i)
    {
      intermediate = layers[i](intermediate, activ);
    }

  return intermediate;

}

template<typename T, typename fun0, typename fun1, typename fun2>
template<typename Derived>
void FFNN<T, fun0, fun1, fun2>::generate_errors(const MatrixBase<Derived>& inputs, const MatrixBase<Derived>& target_outputs)
{
  //get the outputs given this set of inputs
  Matrix<T, Dynamic, Dynamic> outputs = query(inputs);

  //determine the errors at the output layer as per the error function fun1
  //Matrix<T, Dynamic, Dynamic> last_errors = loss(outputs, target_outputs);
  layers[layer_no - 1].errors = loss(outputs, target_outputs); 
	 
  //backpropagate the errors throughout the NN.
  for (int i = 1; i < (layer_no - 1); ++i)
    {
      layers[layer_no - i - 1].node_errors(layers[layer_no - i].errors);
    }  
  
}


//function to train the neural network given input/output data
template<typename T, typename fun0, typename fun1, typename fun2>
void FFNN<T, fun0, fun1, fun2>::train(const int& max_iterations) const
{
  for (int i = 0; i < max_iterations; ++i);
  {
    //get data as a column vector for one datapoint
    Matrix<T, Dynamic, Dynamic> local_x = (*input_data).block(0, i, inputs_per_layer[0], 1);

    Matrix<T, Dynamic, Dynamic> local_y = (*output_data).block(0, i, nodes_per_layer.back(), 1);
    //get the errors, backpropagate them through the network
    generate_errors(local_x);

    //check if the errors are small enough now
    //if ((layers[layer_no - 1].errors)

    //update weights by going backwards through the NN and applying the gradient function
    for (int i1 = 1; i1 < (layer_no - 1); ++i1)
      {
	layers[layer_no - 1 - i1].update_weights(grad(layers[layer_no - i1].errors, layers[layer_no - i1].outputs, layers[layer_no - 1 - i1].outputs));
      }

    //keep training with the same data if we reach the end before max_iterations
    if (i == (*output_data).cols())
      {
	i = 0;
      }
    
  }
}





#endif
