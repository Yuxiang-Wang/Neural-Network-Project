//
//  simulation.cpp
//  simulation_v1
//
//  Created by Yu Zhong on 5/1/19.
//  Copyright © 2019 5. All rights reserved.
//

#include "simulation.hpp"

BSM::~BSM(){}

double BSM::d1(){
    return (log(S0/K)+(r-d+0.5*vol*vol)*t)/(vol*sqrt(t));
}

double BSM::d2(){
    double temp=this->d1();
    return temp-vol*sqrt(t);
}

double BSM::normalCDF(const string &type){
    double temp=0;
    string flag=boost::algorithm::to_lower_copy(type);
    if (flag=="d1") {
        temp=0.5*(1+erf(this->d1()/sqrt(2)));
    }
    
    if (flag=="d2") {
        temp=0.5*(1+erf(this->d2()/sqrt(2)));
    }
    
    return temp;//try handle
}

double BSM::BSM_euro_call(){
    double temp=S0*exp(-d*t)*this->normalCDF("d1")-K*exp(-r*t)*this->normalCDF("d2");
    return temp;
}

ostream& operator << (ostream& os,  simulation& l){
    os<<"S0: "<<l.Initial_set.S0<<'\n'<<"K: "<<l.Initial_set.K<<'\n'<<"r: "<<l.Initial_set.r<<'\n'<<"vol: "<<l.Initial_set.vol<<'\n'<<"d: "<<l.Initial_set.d<<'\n'<<"Maturity: "<<l.Initial_set.t<<endl;
    return os;
};

void simulation::Norm_gen(vector<double> & randy,const double &periods){
    /*==================================Set up the random number engine and seed======================================*/
    boost::mt19937 *rng=new boost::mt19937();
    rng->seed(seed_);
    
    /*=====================================Initialize the normal distribution=========================================*/
    boost::normal_distribution<>norm(mean_,stdev_);
    
    /*=================================Set up the random normal number-generator======================================*/
    boost::variate_generator<boost::mt19937, boost::normal_distribution<>> dist(*rng,norm);
    for (int i=0; i<2*periods; ++i) {
        double temp=dist();
        
        randy.push_back(temp);
    }
    
    delete rng;
}


void simulation::simulate(){
    /*Employing the boost library to set the seed and random number generator through standard normal distribution, to get the numbers for simulating the stock prices path of simulation length. Parameters for inputs (S,K,r,vol0,d,tau) for neural network training would be set. And the output which is the price path of the euro call option would be set too. In this version, the K,d,r,vol are fixed. tested.
     */
    
    /*============================Initialize parameters and temporary data holders===================================*/
    double S=Initial_set.S0;
    double K=Initial_set.K;
    double r=Initial_set.r;
    double vol0=Initial_set.vol;
    double d=Initial_set.d;
    double tau=simLen;
    double T=1.0/252.0;
    
    //double vol=vol0;
    int periods=(int) (simLen/T);
    std::vector<double>in;
    std::vector<double>out;
    std::vector<double>S_holder;
    std::vector<double>tau_holder;
    std::vector<double>V_holder;
    
    /*=====================================Set up the random normal series randy======================================*/
    std::vector<double>randy;
    Norm_gen(randy, periods);
    
    /*======================================The stock price simulation body===========================================*/
    
    for (int i=1; i<periods+1; ++i) {
        S=S*exp((r-d-0.5*vol0*vol0)*T+vol0*sqrt(T)*randy[i]);
        tau=simLen-i*T;
        S_holder.push_back(S);
        tau_holder.push_back(tau);
        
        BSM tempPrice(S,K,r,vol0,d,tau);
        double Vtemp=tempPrice.BSM_euro_call();
        V_holder.push_back(Vtemp);
        //out.push_back(Vtemp);
    }
    
    /*=====================Load and resize the inputs and ouputs for neural network training==========================*/
    
    for (int i=31; i<periods+1; ++i) {
        int j=i-31;
        in.insert(in.end(),S_holder.begin()+j,S_holder.begin()+j+30);
        in.insert(in.end(),tau_holder.begin()+j,tau_holder.begin()+j+31);
        in.push_back(vol0);
        in.push_back(d);
        in.push_back(K);
        //cout<<in.size()<<endl;
    }
    
    out.insert(out.end(),V_holder.begin()+30,V_holder.end());
    
    double *data_ptr1=&in[0];
    double *data_ptr2=&out[0];
    
    input=Eigen::Map<MatrixXd>(data_ptr1,64,periods-30);//rows means S,K,r,vol0,d,tau; cols means different days data
    //inputs.resize(5, periods);
    
    Option_val=Eigen::Map<VectorXd>(data_ptr2,periods-30);
    
}
