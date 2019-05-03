//Andrew Narang
//Advanced C++ - Term Project

//Declarations for FF_NeuronLayer class, which has been specialized to work with feed-forward neural networks.

//optimized for backpropagation.

#ifndef FF_NeuronLayer_HPP
#define FF_NeuronLayer_HPP


#include <iostream>

#include <eigen3/Eigen/Dense>

#include "NeuronLayer.hpp"

//will be useful once further updates are given
#include "SupportFunctions1.hpp"

using Eigen::Matrix;

using Eigen::Dynamic;

using Eigen::MatrixBase;




template<typename T>
class FF_NeuronLayer : public NeuronLayer<T>
{
public:

  Matrix<T, Dynamic, Dynamic> error_weights;

  Matrix<T, Dynamic, Dynamic> errors;

  //constructor, destructor

  template<typename rng_function>
  FF_NeuronLayer(const rng_function& in_rng, const int& in_nodes = 1, const int& in_inputs = 1, const Matrix<T, Dynamic, Dynamic> * in_weights = nullptr);

  ~FF_NeuronLayer();

  
  //member function to compute error at this layer

  template<typename Derived>
  void node_errors(const MatrixBase<Derived>& next_layer_errors);
  
};


#ifndef FF_NeuronLayer_CPP
#include "FF_NeuronLayer.cpp"
#endif


#endif
