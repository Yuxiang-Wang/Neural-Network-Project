//
//  main.cpp
//  simulation_v1
//
//  Created by Yu Zhong on 5/1/19.
//  Copyright © 2019 5. All rights reserved.
//

#include <iostream>
#include "simulation.hpp"
#include "io_simulation.hpp"

int main() {
    
    BSM Q(90,100,0.0001,0.3,0.0015,1);//make a BSM object
    
    simulation yy(Q,1,30,108,0,1);//take the object as initial set of values for simulation, and specify simulation length,window size,random seed, mean and stdev for the normal dist
    
    yy.simulate();//simulate, so that the input
    MatrixXd input=yy.getInput();//get the input and output for the neural network training
    VectorXd output=yy.getOpt();
    
    //set the format of streaming out the matrix/vector
    IOFormat csvFormat_mat(StreamPrecision, 0, ", ", "\n", "", "", "", "");
    IOFormat csvFormat_vec(StreamPrecision, 0, ", ", ", ", "", "", "", "");
    
    writeIn(input, "simulated_input.csv", csvFormat_mat);
    
    writeIn(output, "simulated_output_option_price.csv", csvFormat_vec);
    
    MatrixXd St=yy.getSt();
    MatrixXd TAU=yy.getTAU();
    MatrixXd Vol_d=yy.getV_d();
    
    writeIn(yy.getSt(), "input_S.csv", csvFormat_mat);
    writeIn(yy.getTAU(), "input_TAU.csv", csvFormat_mat);
    writeIn(yy.getV_d(), "input_Vol&d.csv", csvFormat_mat);
    
    MatrixXd AA;
    MatrixXd BB;
    MatrixXd CC;
    
    matRead(AA, "input_S.csv");
    matRead(BB, "input_TAU.csv");
    matRead(CC, "input_Vol&d.csv");
    
    cout<<AA.rows()<<' '<<AA.cols()<<endl;
    cout<<BB.rows()<<' '<<BB.cols()<<endl;
    cout<<CC.rows()<<' '<<CC.cols()<<endl;
    
    return 0;
}
