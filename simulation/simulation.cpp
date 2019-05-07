//
//  simulation.cpp
//  simulation_v1
//
//  Created by Yu Zhong on 5/1/19.
//  Copyright © 2019 5. All rights reserved.
//

#include "simulation.hpp"

simulation::simulation(const BSM &A,const double &t,const int &win,const double &seed,const double &mean, const double &stdev){
    if (win>252*t) {
        simException d;
        cerr<<d.what()<<endl;
        exit(-1);
    }
    
    Initial_set=A;
    simLen=t;
    window_size=win;
    seed_=seed;
    mean_=mean;
    stdev_=stdev;
}

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
    
    return temp;
}

double BSM::BSM_euro_call(){
    double temp=S0*exp(-d*t)*this->normalCDF("d1")-K*exp(-r*t)*this->normalCDF("d2");
    return temp;
}

ostream& operator << (ostream& os,  simulation& l){
    os<<"S0: "<<l.Initial_set.S0<<'\n'<<"K: "<<l.Initial_set.K<<'\n'<<"r: "<<l.Initial_set.r<<'\n'<<"vol: "<<l.Initial_set.vol<<'\n'<<"d: "<<l.Initial_set.d<<'\n'<<"Maturity: "<<l.Initial_set.t<<endl;
    return os;
};

void simulation::Norm_gen(vector<double> & randy,const double &number_of_randn){
    /*==================================Set up the random number engine and seed======================================*/
    boost::mt19937 *rng=new boost::mt19937();
    rng->seed(seed_);
    
    /*=====================================Initialize the normal distribution=========================================*/
    boost::normal_distribution<>norm(mean_,stdev_);
    
    /*=================================Set up the random normal number-generator======================================*/
    boost::variate_generator<boost::mt19937, boost::normal_distribution<>> dist(*rng,norm);
    for (int i=0; i<number_of_randn; ++i) {
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
    
    /*==================================Set up the random normal series randy=========================================*/
    std::vector<double>randy;
    Norm_gen(randy, 2*periods);
    
    /*======================================The stock price simulation body===========================================*/
    
    for (int i=1; i<periods+1; ++i) {
        S=S*exp((r-d-0.5*vol0*vol0)*T+vol0*sqrt(T)*randy[i]);
        tau=simLen-i*T;
        S_holder.push_back(S);
        tau_holder.push_back(tau);
        
        BSM tempPrice(S,K,r,vol0,d,tau);
        double Vtemp=tempPrice.BSM_euro_call();
        //cout<<Vtemp<<endl;
        V_holder.push_back(Vtemp);
        //out.push_back(Vtemp);
    }
    
    /*===================Load and resize the inputs and ouputs for neural network training============================*/
    
    vector<double>St_;
    vector<double>TAU_;
    vector<double>vd_;
    
    for (int i=window_size+1; i<periods+1; ++i) {
        int j=i-window_size-1;
        //for input matrix data
        in.insert(in.end(),S_holder.begin()+j,S_holder.begin()+j+window_size);
        in.insert(in.end(),tau_holder.begin()+j,tau_holder.begin()+j+window_size+1);
        in.push_back(vol0);
        in.push_back(d);
        in.push_back(K);
        //different parts of the input data, as Yuxiang requested
        St_.insert(St_.end(),S_holder.begin()+j,S_holder.begin()+j+window_size);//一行30个S数据
        TAU_.insert(TAU_.end(),tau_holder.begin()+j,tau_holder.begin()+j+window_size+1);//一行31个TAU数据
        vd_.push_back(vol0);
        vd_.push_back(d);
        vd_.push_back(K);
        //cout<<in.size()<<endl;
    }
    
    //cout<<S_holder[0]<<' '<<St_[0]<<endl;
    
    out.insert(out.end(),V_holder.begin()+window_size,V_holder.end());//windowsize is 30,then take the 31st element in the output; will have 222 days data
    
    /*========================================Get the general outcomes================================================*/
    double *data_ptr1=&in[0];
    double *data_ptr2=&out[0];
    
    input=Eigen::Map<MatrixXd>(data_ptr1,window_size*2+4,periods-window_size);//rows means S,K,r,vol0,d,tau; cols means different days data. 221days data，因为最后一天的没有明天的数据
    
    Option_val=Eigen::Map<VectorXd>(data_ptr2,periods-window_size);//222行
    
    
    /*=====================Get the specific outcomes S,TAU,vol and dividend, 3 matricies total========================*/
    double *S_ptr=&St_[0];
    double *TAU_ptr=&TAU_[0];
    double *VD_ptr=&vd_[0];
    
    St=Eigen::Map<MatrixXd>(S_ptr,window_size,periods-window_size-1);
    TAU=Eigen::Map<MatrixXd>(TAU_ptr,window_size+1,periods-window_size-1);
    vol_d=Eigen::Map<MatrixXd>(VD_ptr,3,periods-window_size-1);
    
    double *V_whole=&V_holder[0];
    
    opt252=Eigen::Map<VectorXd>(V_whole,periods);//252行
    
}

