//Andrew Narang
//Advanced C++ - Term Project

//Declarations for FF_NeuronLayer class, which has been specialized to work with feed-forward neural networks.

//optimized for backpropagation.

#ifndef FF_NeuronLayer_HPP
#define FF_NeuronLayer_HPP


#include <iostream>

#include <eigen3/Eigen/Dense>

#include "NeuronLayer.hpp"

//contains relevant functors
#include "SupportFunctions1.hpp"

using Eigen::Matrix;

using Eigen::Dynamic;

using Eigen::MatrixBase;




template<typename T>
class FF_NeuronLayer : public NeuronLayer<T>
{
public:

  //used to backpropagate errors when training the containing FFNN class.
  Matrix<T, Dynamic, Dynamic> error_weights;

  //this is for reference - not meaningful until we call the node_errors function below as part of the backpropagation process. 
  Matrix<T, Dynamic, Dynamic> errors;

  //constructors, destructor

  //constructor #1
  //corresponds with constructor #1 of NeuronLayer in NeuronLayer.hpp, same comments apply here
  //accounts for bias inputs
  template<typename rng_function>
  FF_NeuronLayer(const rng_function& in_rng, const int& in_nodes, const int& in_inputs, const int& in_BiasFlag = 1);


  //constructor #2
  //corresponds with constructor #2 of NeuronLayer in NeuronLayer.hpp, same comments apply here
  template<typename Derived>
  FF_NeuronLayer(const MatrixBase<Derived>& in_weights, const int& in_BiasFlag = 1); 
  
  
  //deprecated - new methods are more efficient
  //template<typename rng_function>
  //FF_NeuronLayer(const rng_function& in_rng, const int& in_nodes = 1, const int& in_inputs = 1, const Matrix<T, Dynamic, Dynamic> * in_weights = nullptr);

  ~FF_NeuronLayer();

  
  //member function to compute error at this layer
  //input argument accomodates backpropagated errors as well as error vector from the last layer, where the error function is applied before this function is called 
  template<typename Derived>
  void node_errors(const MatrixBase<Derived>& next_layer_errors);



  
};


#ifndef FF_NeuronLayer_CPP
#include "FF_NeuronLayer.cpp"
#endif


#endif
