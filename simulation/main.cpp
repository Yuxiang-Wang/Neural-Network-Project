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
    
    writeIn(input, "simulated_input.csv", csvFormat_mat);//general inputs
    
    writeIn(output, "option_price222.csv", csvFormat_vec);//output:option prices
    
    MatrixXd St=yy.getSt();
    MatrixXd TAU=yy.getTAU();
    MatrixXd Vol_d=yy.getV_d();
    
    //get the parts of general inputs
    writeIn(yy.getSt(), "input_S221.csv", csvFormat_mat);
    writeIn(yy.getTAU(), "input_TAU221.csv", csvFormat_mat);
    writeIn(yy.getV_d(), "input_Vol&d221.csv", csvFormat_mat);

    
    MatrixXd AA;
    MatrixXd BB;
    MatrixXd CC;
    MatrixXd DD;
    
    matRead(AA, "input_S221.csv");
    matRead(BB, "input_TAU221.csv");
    matRead(CC, "input_Vol&d221.csv");
    matRead(DD, "option_price222.csv");
    
    cout<<AA.rows()<<' '<<AA.cols()<<endl;
    cout<<BB.rows()<<' '<<BB.cols()<<endl;
    cout<<CC.rows()<<' '<<CC.cols()<<endl;
    cout<<DD.rows()<<' '<<DD.cols()<<endl;
    
    writeIn(AA, "test.csv", csvFormat_mat);
    
    VectorXd V_all=yy.opt252;
    writeIn(V_all, "V_252.csv", csvFormat_vec);
    
    //simulation kkk(Q,1,500,108,0,1);
    
    
    
    
    //below is just for processing the real data
    /*MatrixXd real_stock;
    MatrixXd real_optionPrice;
    MatrixXd real_K;
    MatrixXd real_vol;
    
    matRead(real_stock, "amzn_stock.csv");
    MatrixXd stock_out;
    vector<double>in;
    vector<double>tempS;
    for (int i=0; i<252; ++i) {
        double temp=real_stock(i,0);
        tempS.push_back(temp);
    }
    
    for (int i=31; i<253; ++i) {
        int j=i-31;
        in.insert(in.end(),tempS.begin()+j,tempS.begin()+j+30);
    }
    
    double *d_ptr1=&in[0];

    stock_out=Map<MatrixXd>(d_ptr1,30,221);
    writeIn(stock_out, "real_stock221.csv", csvFormat_mat);
    
    //
    matRead(real_optionPrice, "amzn_optionVal.csv");
    MatrixXd opt_out;
    vector<double>tempV;
    for (int i=0; i<252; ++i) {
        double temp=real_optionPrice(i,0);
        tempV.push_back(temp);
    }
    vector<double>in1;
    in1.insert(in1.end(),tempV.begin()+30,tempV.end());
    
    double *d_ptr2=&in1[0];
    
    opt_out=Map<VectorXd>(d_ptr2,222);
    writeIn(opt_out, "real_option222.csv", csvFormat_vec);
   
    //
    
    matRead(real_K, "amzn_K.csv");
    MatrixXd K_out;
    vector<double>in2;
    vector<double>tempK;
    for (int i=0; i<252; ++i) {
        double temp=real_K(i,0);
        tempK.push_back(temp);
    }
    
    for (int i=31; i<253; ++i) {
        int j=i-31;
        in2.insert(in2.end(),tempK.begin()+j,tempK.begin()+j+30);
    }
    
    double *d_ptr3=&in2[0];
    
    K_out=Map<MatrixXd>(d_ptr3,30,221);
    writeIn(K_out, "real_K221.csv", csvFormat_mat);
    
    matRead(real_vol, "amzn_vol.csv");
    MatrixXd vol_out;
    vector<double>in3;
    vector<double>tempvol;
    for (int i=0; i<252; ++i) {
        double temp=real_vol(i,0);
        tempvol.push_back(temp);
    }
    
    for (int i=31; i<253; ++i) {
        int j=i-31;
        in3.insert(in3.end(),tempvol.begin()+j,tempvol.begin()+j+30);
    }
    
    double *d_ptr4=&in3[0];
    
    vol_out=Map<MatrixXd>(d_ptr4,30,221);
    writeIn(vol_out, "real_vol221.csv", csvFormat_mat);
    
    MatrixXd div;
    vector<double> sake(221,0.0);
    double *sake_ptr=&sake[0];
    div=Map<VectorXd>(sake_ptr,221);
    writeIn(div, "real_div221.csv", csvFormat_vec);
    
    writeIn(div, "", csvFormat_vec);*/
    
    getchar();
    getchar();
    return 0;
}
