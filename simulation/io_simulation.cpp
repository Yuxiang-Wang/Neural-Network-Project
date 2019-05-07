//
//  io_simulation.cpp
//  simulation_v1
//
//  Created by Yu Zhong on 5/1/19.
//  Copyright © 2019 5. All rights reserved.
//

#include "io_simulation.hpp"

int matRead(MatrixXd &A,const string &filename){
    ifstream infile(filename);
    if (!infile) {
        myException2 a;
        cout<<"Error: "<<a.what()<<endl;
        return -1;
    }
    
    unsigned int row=0;
    unsigned int col=0;
    vector<double>temp;
    
    
    string line,cell;
    while(getline(infile, line)){
        col=0;
        stringstream linestream(line);
        while(getline(linestream, cell,',')){
            try {
                stod(cell);
            } catch (.../*myException3 e*/) {
                myException3 e;
                cout<<"Error: "<<e.what()<<endl;
                return -1;
            }
            temp.push_back(stod(cell));
            col++;
        }
        row++;
    }
    
    double *data_ptr=&temp[0];
    A=Eigen::Map<MatrixXd>(data_ptr,col,row).transpose();
    
    infile.close();
    return 0;
}

