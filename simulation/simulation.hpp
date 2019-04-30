//
//  simulation.hpp
//  simulaiton
//
//  Created by Yu Zhong on 4/27/19.
//  Copyright © 2019 5. All rights reserved.
//

#ifndef simulation_hpp
#define simulation_hpp

#include<eigen3/Eigen/Core>
#include<vector>
#include<iostream>
#include<fstream>
#include<math.h>
#include<string>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <vector>

using namespace Eigen;

double d1(double S,double K,double r,double vol, double d,double t);
double d2(double d1, double vol, double t);
double N(double x);
double BSM_euro_call(double S,double K,double r,double vol, double d,double t);

//Can be modulized and hold different simulation samples
void simulation(const MatrixXd *inputs, const VectorXd *output, double simLen);
/*
 @brief simulate the stock paths and get input and ouput for neural network training
 @param pre-defined Eigen object to hold inputs for the neural network: S,K,r,vol,d,tau
 @param pre-defined Eigen object to hold ouputs for the neural network: BS option prices
 @param simulation time length
 */

#endif /* simulation_hpp */
