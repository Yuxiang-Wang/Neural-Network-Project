//
//  io_simulation.hpp
//  simulation_v1
//
//  Created by Yu Zhong on 5/1/19.
//  Copyright © 2019 5. All rights reserved.
//

#ifndef io_simulation_hpp
#define io_simulation_hpp

#include <iostream>
#include <Eigen/Core>
#include <fstream>
#include <string>
#include <exception>
#include <vector>

using namespace std;
using namespace Eigen;

//exception for creating file
class myException1:public exception{
public:
    virtual const char* what(){return "Can't create file";}
};

//exception for opening file
class myException2:public exception{
public:
    virtual const char* what(){return "Can't open file, file is posssibly not existed";}
};

class myException3:public exception{
public:
    virtual const char* what(){return "Can't transfer this cell to stod";}
};

template <typename Derived>
void writeIn(const Derived &mat,const string &filename,IOFormat &form) {
    ofstream file;
    
    try {
        file.open(filename);
    } catch (myException1 e) {
        cout<<"Error: "<<e.what()<<endl;
    }
    
    file<<mat.format(form);
    
    file.close();
}

int matRead(MatrixXd &A,const string &filename);

#endif /* io_simulation_hpp */
