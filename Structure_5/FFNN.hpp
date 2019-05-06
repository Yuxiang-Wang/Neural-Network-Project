//Andrew Narang
//Advanced C++ - Term Project


//Here, we define the FFNN class, which stands for feed-forward neural network.


#ifndef FFNN_HPP
#define FFNN_HPP




#include <vector>

#include <eigen3/Eigen/Dense>

#include "SupportFunctions1.hpp"
#include "NeuronLayer.hpp"
#include "FF_NeuronLayer.hpp"


using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::MatrixBase;


template<typename T, typename fun0, typename fun1, typename fun2>
class FFNN
{
private:

  //vector that stores NeuronLayer objects
  //should we make it a pointer to a vector of FFNL objects instead to avoid copying?
  std::vector<FF_NeuronLayer<T>> layers;

  //number of layers
  int layer_no;

  //to track the number of nodes at each layer
  std::vector<int> nodes_per_layer;

  //to track the number of inputs at each layer
  std::vector<int> inputs_per_layer;

  //activation function
  fun0 activ;

  //loss function
  fun1 loss;

  //gradient function
  fun2 grad;

  //idk if these are reallllly necessary but it's not like extra doubles are a big deal anyway
  //these are pointers to the input and output data - so the user knows what this FFNN object was trained for. 
  Matrix<T, Dynamic, Dynamic> * input_data;

  Matrix<T, Dynamic, Dynamic> * output_data;

  T lrn_rate;

  //not sure if this is necessary given nodes_per_layer and inputs_per_layer defined above.  
  //# of initial inputs into the neural network
  //int inputs0;

  //# of outputs from the neural network
  //int outputs0;
  
public:

  //deprecated - replace this with better ones
  //template<typename rng_function>
  //FFNN(const rng_function& in_rng, const fun0& in_activation, const fun0& in_loss, const fun0& in_gradient, const int& in_layer_no = 1, const std::vector<int>& in_nodes_per_layer = std::vector<int>(2, 2), const std::vector<int>& in_inputs_per_layer = std::vector<int>(2, 2), const std::vector<const Matrix<T, Dynamic, Dynamic> *>& in_matrix_weights = std::vector<const Matrix<T, Dynamic, Dynamic> *>(), Matrix<T, Dynamic, Dynamic> * in_data_input = nullptr, Matrix<T, Dynamic, Dynamic> * in_data_output = nullptr);


  //easiest constructor - make the layers outside of the function call - other data members are inferred in the constructor
  template<typename Derived>
  FFNN(const std::vector<FF_NeuronLayer<T>>& in_layers, const fun0& in_activ, const fun1& in_loss, const fun2& in_grad, const T& in_lrn_rate, const MatrixBase<Derived>& in_input_data, const MatrixBase<Derived>& in_output_data); 
  
  ~FFNN();

  //finish implementing these later
  //  template<typename Derived>
  //int set_data_input(const MatrixBase<Derived>& new_input);

  //  template<typename Derived>
  //int set_data_output(const MatrixBase<Derived>& new_output);

  //calculate the output of this network given a vector of inputs
  //non-recursive style
  template<typename Derived>
  Matrix<T, Dynamic, Dynamic> query(const MatrixBase<Derived>& in_arg) const;

  //should this be const?
  template<typename Derived>
  void generate_errors(const MatrixBase<Derived>& inputs, const MatrixBase<Derived>& target_outputs);

  void train(const int& max_iterations = 10) const;

  
};










#ifndef FFNN_CPP
#include "FFNN.cpp"
#endif



#endif

