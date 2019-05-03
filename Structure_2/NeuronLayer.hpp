//Andrew Narang
//Advanced C++ - Term Project

//Here, we set up the NeuronLayer class, which represents one layer of a deep-learning style, feed-forward network
//The class FFNN, which stands for Feed-Forward Neural Network, will be comprised of these NeuronLayer objects.

//This class is set up in a way that makes it easy to make derived classes from it that represent different types of NeuronLayers for different types of Neural Networks.

//Everything here is column-vector based.
//We always use Matrix<T, Dynamic, Dynamic> because matrices w/ more than 16 elements should be Dynamic rather than fixed for speed and it avoids issues with pointers and using it with other classses like STL vectors and as parts of classes.

//Changes: make it template-based, use smart pointers, make default function pointers point to sigmoid function and normal rng function, respectively in the parameterized constructor

//avoid manual memory allocation so that we can use compiler's automatically generated special member functions?

#ifndef NeuronLayer_HPP
#define NeuronLayer_HPP

#include<eigen3/Eigen/Dense>

//write functions file, include it here!
#include "SupportFunctions1.hpp"

using Eigen::Matrix;

using Eigen::Dynamic;

using Eigen::MatrixBase;




//accomodates different types of data types 
template<typename T>
class NeuronLayer
{
public:
  //private:

  int nodes;
  
  int inputs;

  //we have to use this datatype because our weight matrices are usually larger than 4x4, and the fixed size versions require constants to declare differently sized matrices as datatypes
  Matrix<T, Dynamic, Dynamic> weights;

  //pointer to activation function - we will apply this to the whole output matrix at one calculation via unaryExp() member function when the () function is called. 
  //double (*activ)(const double& input);

  //replaced this with function - accomodates function pointers and function objects this way = increased flexibility
  //function activation;

  //activation function, associated gradient function, and error function pointers have been moved to the NN (NeuralNetwork) class hierarchy as they are network-wide
  
  //public:

  //fill in the rest of the special member functions later

  //activation function should be sigmoid function by default
  //1x1 matrix by default
  //we check if there is an input weight matrix by seeing if in_weights is a null pointer. In that case, an input weight matrix is randomly generated as per distribution idea (Normal by default). If the given input weight matrix does not match the given dimensions, a random one is generated as per the distribution idea.
  //distribution idea captured via another function pointer. we can have it after the weights argument though so that the user doesn't need to specify normal just-in-case rng if they have a weights matrix in mind
  //if everything is fine, then we just make a deep copy of it
  
  //input order is specifically chosen to maximize flexibility of construction
  //NeuronLayer(const int& in_nodes = 1, const int& in_inputs = 1, double (*in_activ)(const double& input) = nullptr, Matrix<double, Dynamic, Dynamic> * in_weights = nullptr, double (*in_rng)() = nullptr);
  
  //we use const pointers for input weight matrix and rng_function arguments rather than const references so that they are optional - if one is given, the other is not needed.  
  //template<typename Derived, typename rng_function>
  //NeuronLayer(const int& in_nodes = 1, const int& in_inputs = 1, const MatrixBase<Derived> * in_weights = nullptr, const rng_function& in_rng);

  /////////////////yo yo check it out check it out:

  //apparently if you make it work with MatrixBase<Derived>, you lose the benefit of weight matrix being optional because then it cannot determine the type of the pointer, even though it will not be needed. 
  template<typename rng_function>
  NeuronLayer(const rng_function& in_rng, const int& in_nodes = 1, const int& in_inputs = 1, const Matrix<T, Dynamic, Dynamic> * in_weights = nullptr);

    
  //NeuronLayer(const int& in_nodes = 1, const int& in_inputs = 1); 


  ///////best one so far: NeuronLayer(const int& in_nodes = 1, const int& in_inputs = 1, const Matrix<T, Dynamic, Dynamic> * in_weights = nullptr);
  
  //template<typename Derived>
  //NeuronLayer(const int& in_nodes = 1, const int& in_inputs = 1, const MatrixBase<Derived> * in_weights = nullptr);
  
  virtual ~NeuronLayer();


  /*
  //setters and getters - private style

  const int& get_nodes() const;

  void set_nodes(const int& new_nodes);
  
  const int& get_inputs() const;

  void set_inputs(const int& new_inputs);
  
  const Matrix<T, Dynamic, Dynamic>& get_weights() const;

  template<typename Derived>
  void set_weights(const MatrixBase<Derived>& new_weights);
  */
  
  //operators

  //to compute the output of the NeuronLayer object given a matrix of inputs.
  //fun0 represents activation function
  template<typename Derived, typename fun0>
  Matrix<T, Dynamic, Dynamic> operator () (const MatrixBase<Derived>& in_vector, const fun0& in_activ);
  
  
};



//conditional compilation protection for inclusion of NeuronLayer.cpp. 
#ifndef NeuronLayer_CPP
#include "NeuronLayer.cpp"
#endif



#endif
