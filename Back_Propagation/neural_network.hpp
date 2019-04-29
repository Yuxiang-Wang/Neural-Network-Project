#ifndef __NEURAL_NETWORK__H
#define __NEURAL_NETWORK__H

#include<eigen3/Eigen/Core>
#include<vector>
#include<iostream>
#include<fstream>
#include<math.h>
#include<string>

using namespace Eigen;

double sigmoid(double x);

struct Layer {
	MatrixXd w, diff, delta, val, diff_total;
	Layer(MatrixXd w, MatrixXd diff, MatrixXd diff_total, MatrixXd val, MatrixXd delta);
};

class Neural_network{
	/*
	All the hidden layers have same units number in the model.
	Didn't use neuron structure. w, diff, diff_total are matrice, need numbers of units of two layers to initialize. 
	Set delta,val,diff as private class member just for convenience. save memory.
	
	delta: expected value - real output
	val: expected value
	diff: derivatives
	
	w:weights; 
	diff_total: sum of derivative of all observers.
	*/
    const int num_layer,units_hidden,units_output,units_input;
	std::vector<Layer> layers;
    float learning_rate=100;
    const int MAX_ITE=100000;
    MatrixXd data_input,data_output;

public:
    Neural_network(int num_layer,int units_input,int units_hidden,int units_output);
    void set_data_input(const MatrixXd& m);
    void set_data_output(const MatrixXd& m);
    void set_weight(const std::vector<MatrixXd>& weight);	//set weight manualy
    MatrixXd get_val() const;								//get expected output
    void forward(MatrixXd input);							//feed forward, compute expected output
    void backward(MatrixXd output);							//backward compute derivative
    void update();											//update weights	
    void training(std::string file="");						//main work, train network
};

#endif
