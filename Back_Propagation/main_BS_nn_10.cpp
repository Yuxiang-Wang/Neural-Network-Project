#include"neural_network.hpp"
#include"io_simulation.hpp"
#include<eigen3/Eigen/Core>
#include<iostream>
#include<vector>
#include<fstream>
using namespace Eigen;

int main() {

	MatrixXd Sin, Tin, voldKin, output;
	matRead(Sin, "D:/advc++/project/normed_stock.csv");
	matRead(Tin, "D:/advc++/project/input_TAU.csv");
	matRead(voldKin, "D:/advc++/project/normed_vol_d_K.csv");
	matRead(output, "D:/advc++/project/out_partition_10_221.csv");
	output = Map<MatrixXd>(output.data(), output.cols(), output.rows());
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

	MatrixXd in_input(input.block(0, 0, 190, input.cols())),
		out_input(input.block(190, 0, 31, input.cols())),
		in_output(output.block(0, 0, 190, output.cols())),
		out_output(output.block(0, 0, 31, output.cols()));

	ofstream outfile("result_190_31_10");
	if (outfile.fail()) {
		std::cerr << "can't open file to write";
		exit(-1);
	}

	outfile << "num_layer units_input units_hidden units_output in_sample_nums out_sample_nums n_in n_out\n";
	int ite;
	for (int num_layer = 3; num_layer < 4; ++num_layer) {
		for (int units_hidden = 5; units_hidden < 25; ++units_hidden) {
			Neural_network nn(num_layer, units_input, units_hidden, units_output);
			nn.set_data_input(in_input);
			nn.set_data_output(in_output);
			ite = nn.training(1, "weight");
			if (ite < 3) ite = nn.training(0.1, "weight");
			if (ite < 3) ite = nn.training(0.01, "weight");
			// in sample
			MatrixXd in_pred;
			int n_in = 0;
			// compute correct numbers
			for (int i = 0; i < in_input.rows(); ++i) {
				nn.forward(in_input.row(i));
				in_pred = (nn.get_val().array() > 0.5).cast<double>();
				in_pred = Map<MatrixXd>(in_pred.data(), 1, in_output.cols());
				if ((in_pred.array() == in_output.row(i).array()).cast<int>().sum() == output.cols()) {
					++n_in;
				}
			} 
			//out sample
			MatrixXd out_pred(out_output.cols(), 1);
			int n_out = 0;
			// compute correct numbers
			for (int i = 0; i < out_input.rows(); ++i) {
				nn.forward(out_input.row(i));
				out_pred = (nn.get_val().array() > 0.5).cast<double>();
				out_pred = Map<MatrixXd>(out_pred.data(), 1, out_output.cols());
				if ((out_pred.array() == out_output.row(i).array()).cast<int>().sum() == output.cols()) {
					++n_out;
				}
			}
			std::cout << num_layer << ' ' << units_input << ' ' << units_hidden << ' ' << units_output<<' '
				<< in_input.rows() << ' ' << out_input.rows() << ' ' << n_in << ' ' << n_out << '\n';
			outfile << num_layer << ' ' << units_input << ' ' << units_hidden << ' ' << units_output<<' '
				<< in_input.rows() << ' ' << out_input.rows() << ' ' << n_in << ' ' << n_out << '\n';
		}
	}

	outfile.close();
	getchar();
	getchar();
	return 0;
}
