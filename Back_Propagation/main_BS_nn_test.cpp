#include"neural_network.hpp"
#include"io_simulation.hpp"
#include<eigen3/Eigen/Core>
#include<iostream>
#include<vector>
using namespace Eigen;

int main() {

	MatrixXd Sin, Tin, voldKin ,output;
	matRead(Sin, "D:/advc++/project/normed_stock.csv");
	matRead(Tin, "D:/advc++/project/input_TAU.csv");
	matRead(voldKin, "D:/advc++/project/normed_vol_d_K.csv");
	matRead(output, "D:/advc++/project/output_221.csv");

	std::cout << "Stock price : " << Sin.rows() << ',' << Sin.cols() << '\n'
		<< "Time        : " << Tin.rows() << ',' << Tin.cols() << '\n'
		<< "vol, div, K : " << voldKin.rows() << ',' << voldKin.cols() << '\n'
		<< "Option Price: " << output.rows() << ',' << output.cols() << '\n';

	int units_input = Sin.cols() + Tin.cols() + voldKin.cols(), units_output = output.cols();
	int num = Sin.rows();
	MatrixXd input(num, units_input);
	input << Sin, Tin, voldKin;
	std::cout << "------------\n"
		<< "input   : " << input.rows() << ',' << input.cols() << '\n'
		<< "output  : " << output.rows() << ',' << output.cols() << '\n';
	
	int num_layer = 5, units_hidden = 20;
	Neural_network nn(num_layer, units_input, units_hidden, units_output);
	nn.set_data_input(input);
	nn.set_data_output(output);
	nn.training("weight");

	MatrixXd pred(num, 1);
	for (int i = 0; i < input.rows(); ++i) {
		nn.forward(input.row(i));
		pred(i, 0) = nn.get_val()(0, 0) > 0.5 ? 1 : 0;
	}

	MatrixXd comp(num, 2);
	comp << output, pred;
	std::cout << "\ncorrect rate: "<< (comp.col(0).array() == comp.col(1).array()).cast<double>().sum()*1.0/num << "\n\n";

	getchar();
	getchar();
	return 0;
}
