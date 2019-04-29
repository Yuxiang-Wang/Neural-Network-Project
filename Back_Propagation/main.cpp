#include"neural_network.hpp"
#include<eigen3/Eigen/Core>
#include<iostream>


int main(){
    MatrixXd test_input(MatrixXd::Random(10,3));
    MatrixXd test_output;
    test_output=(test_input.rowwise().sum().array()<1).cast<double>();
    
    Neural_network nn(3,3,3,1);
    nn.set_data_input(test_input);
    nn.set_data_output(test_output);
    nn.training("wights.txt");

    MatrixXd pred(test_input.rows(),1);
    for(int i=0;i<test_input.rows();++i){
	nn.forward(test_input.row(i));
	pred(i,0)=nn.get_val()(0,0)>0.5?1:0;
    }

    MatrixXd comp(test_input.rows(),2);
    comp<<test_output,pred;
    std::cout<<'\n'<<comp<<"\n\n"
	     <<(comp.col(0).array()==comp.col(1).array())<<"\n\n";
    
	getchar();
	getchar();
    return 0;
}
