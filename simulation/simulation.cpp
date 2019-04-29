//
//  simulation.cpp
//  simulaiton
//
//  Created by Yu Zhong on 4/27/19.
//  Copyright © 2019 5. All rights reserved.
//

#include "simulation.hpp"


double d1(double S,double K,double r,double vol,double d,double t){
    return (log(S/K)+(r-d+0.5*vol*vol)*t)/(vol*sqrt(t));
}

double d2(double S,double K,double r,double vol,double d,double t){//func d1_func,
    double temp=d1(S, K, r, vol, d, t);
    return temp-vol*sqrt(vol);
}

double normalCDF(double x)
{
    return 0.5*(1+erf(x/sqrt(2)));
}

double BSM_euro_call(double S,double K,double r,double vol,double d,double t){
    double d1_=d1(S,K,r,vol,d,t);
    double d2_=d2(S,K,r,vol,d,t);
    
    return (S*exp(-d*t)*N(d1_)-K*exp(-r*t)*N(d2_));
}


void simulation(MatrixXd &inputs, VectorXf &output, const double &simLen, double S0,double K,double r,double vol0,double d){
    
    /*Employing the boost library to set the seed and random number generator through standard normal distribution, to get the numbers for simulating the stock prices path of length simlen. Parameters for inputs (S,K,r,vol0,d,tau) for neural network training would be set. And the output which is the price path of the euro call option would be set too. In this version, the K,d,r,vol are fixed. Untested since my xcode isn't able to compile Eigen lib.
     */
    
    std::vector<double>randy;
    boost::mt19937 *rng=new boost::mt19937();
    rng->seed(10);
    
    boost::normal_distribution<>norm(0.0,1.0);
    for (int i=0; i<1000; ++i) {
        float temp=norm(rng);
        
        randy.push_back(temp);
    }
    
    double S=S0;
    double T=1.0/252.0;
    double tau=T;
    //double vol=vol0;
    int periods=(int) (simLen/T);
    std::vector<double>in;
    std::vector<double>out;
    
    for (int i=1; i<periods+1; ++i) {
        std::vector<double> temp{S,K,r,vol0,d,tau};
        in.insert(in.end(), temp.begin(), temp.end());
        
        S=S*exp((r-d-0.5*vol0*vol0)*T+vol0*sqrt(T)*randy[i]);
        tau=simLen-i*T;
        
        double Vtemp=BSM_euro_call(S, K, r, vol0, d, tau);
        out.push_back(Vtemp);
        
    }
    
    MatrixXd InTemp=Map<MatrixXd>(in.data(),in.size());
    InTemp.resize(6, periods);//rows means S,K,r,vol0,d,tau; cols means different days data
    inputs=InTemp;
    
    VectorXd OutTemp=Map<VectorXd>(out.data(),out.size());
    output=OutTemp;
    
}

