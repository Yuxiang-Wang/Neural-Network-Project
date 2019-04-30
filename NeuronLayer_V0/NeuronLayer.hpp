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

using Eigen::Matrix;

using Eigen::Dynamic;

class NeuronLayer
{
private:

  int nodes;
  
  int inputs;

  //replace this with a smart pointer
  //we have to use this datatype because our weight matrices are usually larger than 4x4, and the fixed size versions require constants to declare differently sized matrices as datatypes
  Matrix<double, Dynamic, Dynamic> * weights;

  //pointer to activation function - we will apply this to the whole output matrix at one calculation via unaryExp() member function when the () function is called. 
  double (*activ)(const double& input); 
  
public:

  //fill in the rest of the special member functions later after updating to smart pointers

  //activation function should be sigmoid function by default
  //1x1 matrix by default
  //we check if there is an input weight matrix by seeing if in_weights is a null pointer. In that case, an input weight matrix is randomly generated as per distribution idea (Normal by default). If the given input weight matrix does not match the given dimensions, a random one is generated as per the distribution idea.
  //distribution idea captured via another function pointer. we can have it after the weights argument though so that the user doesn't need to specify normal just-in-case rng if they have a weights matrix in mind
  //if everything is fine, then we just make a deep copy of it
  
  //input order is specifically chosen to maximize flexibility of construction
  NeuronLayer(const int& in_nodes = 1, const int& in_inputs = 1, double (*in_activ)(const double& input) = nullptr, Matrix<double, Dynamic, Dynamic> * in_weights = nullptr, double (*in_rng)() = nullptr); 

  virtual ~NeuronLayer();

  //operators

  //update this to smart pointer
  //to compute the output of the NeuronLayer object given a matrix of inputs.  
  Matrix<double, Dynamic, Dynamic> operator () (const Matrix<double, Dynamic, Dynamic>& in_vector);

  
  
};


#endif
