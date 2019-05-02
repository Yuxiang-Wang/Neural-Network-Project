//
//  simulation.hpp
//  simulation_v1
//
//  Created by Yu Zhong on 5/1/19.
//  Copyright © 2019 5. All rights reserved.
//

#ifndef simulation_hpp
#define simulation_hpp

#include <iostream>
#include <Eigen/Core>
#include <boost/algorithm/string.hpp>
#include<vector>
#include<fstream>
#include<math.h>
#include<string>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
using namespace Eigen;
using namespace std;

class BSM{
    public:
    double S0;
    double K;
    double r;
    double vol;
    double d;
    double t;
    
    BSM():S0{0},K{0},r{0},vol{0},d{0},t{0}{}
    BSM(const double &s,const double &k,const double &rfr,const double &volatility,const double &div,const double &len):S0{s},K{k},r{rfr},vol{volatility},d{div},t{len}{}
    
    ~BSM();
    
    void setS(const double& s){S0=s;}
    void setK(const double& k){K=k;}
    void setVol(const double& v){vol=v;}
    void setD(const double& div){d=div;}
    
    double d1();
    double d2();
    double normalCDF(const string &type);
    double BSM_euro_call();
    
};

class simulation{
    public:
    BSM Initial_set;
    //simulation length size in years
    double simLen;
    //window size in days
    int window_size;//cant be larger than simlen*252, exception handling here
    double seed_;
    double mean_;
    double stdev_;

    MatrixXd St;
    MatrixXd TAU;
    MatrixXd vol_d;
    VectorXd Option_val;
    MatrixXd input;
    
    simulation():Initial_set(BSM()),simLen{0},window_size{0},seed_{108},mean_{0},stdev_{1}{}
    simulation(const BSM &A,const double &t,const int &win,const double &seed,const double &mean, const double &stdev):Initial_set{A},simLen{t},window_size{win},seed_{seed},mean_{mean},stdev_{stdev}{}
    
    void setBSM(const BSM& s){Initial_set=s;}
    void setLen(const double& l){simLen=l;}
    
    const MatrixXd getSt() const {return St;}
    const MatrixXd getTAU() const {return TAU;}
    const MatrixXd getV_d() const {return vol_d;}
    const VectorXd getOpt() const {return Option_val;}
    const MatrixXd getInput() const {return input;}
    
    //operator=
    friend ostream& operator << (ostream& os,  simulation& l);
    /*
     @brief overload the << to output the basic info of the object
     @param output stream object
     @param simulation class object
     */
    
    void Norm_gen(vector<double> &,const double &);
    /*
     @brief generates certain number of number according to normal distribution
     @param vector that stores the numbers
     @param the number of periods
     */
    
    void simulate();
    /*
     @brief class member function for taking the object to simulate and make input and output for the neural network training
     */
};

#endif /* simulation_hpp */
