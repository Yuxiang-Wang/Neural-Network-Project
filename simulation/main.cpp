//
//  main.cpp
//  simulation_v1
//
//  Created by Yu Zhong on 5/1/19.
//  Copyright © 2019 5. All rights reserved.
//

#include <iostream>
#include "simulation.hpp"

int main() {
    
    BSM Q(90,100,0.0001,0.3,0.0015,1);//make a BSM object
    
    simulation yy(Q,1,108,0,1);//take the object as initial set of values for simulation, and specify the simulation length
    
    yy.simulate();//simulate, so that the input
    MatrixXd input=yy.getInput();//get the input and output for the neural network training
    VectorXd output=yy.getOpt();
    //cout<<output<<endl;
    
    /*std::ofstream file("simulated_input.csv");
    if (file.fail()){
        cerr<<"Fail to create file for writing"<<"\n";
        return -1;
    }
    
    file<<input<<endl;
    
    file.close();
    
    std::ofstream file1("simulated_output_option_price.csv");
    if (file1.fail()){
        cerr<<"Fail to create file for writing"<<"\n";
        return -1;
    }
    
    file1<<output<<endl;*/
    
    
    
    return 0;
}
