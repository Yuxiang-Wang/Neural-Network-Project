#include"neural_network.hpp"
#include<eigen3/Eigen/Core>
#include<iostream>
#include<vector>
using namespace Eigen;

int main() {
	int units_hidden = 10, num_layer = 4, units_input = 20, units_output = 1;
	MatrixXd test_input(MatrixXd::Random(100, units_input));
	MatrixXd test_output = (test_input.rowwise().sum().array() < 0).cast<double>();

	std::vector<MatrixXd> weight;

	weight.push_back(MatrixXd::Random(units_hidden, units_input + 1));
	for (int i = 1; i < num_layer - 2; ++i)
		weight.push_back(MatrixXd::Random(units_hidden, units_hidden + 1));
	weight.push_back(MatrixXd::Random(units_output, units_hidden + 1));

	Neural_network nn(num_layer, units_input, units_hidden, units_output);
	nn.set_data_input(test_input);
	nn.set_data_output(test_output);
	nn.set_weight(weight);
	nn.training("wights.txt");

	MatrixXd pred(test_input.rows(), 1);
	for (int i = 0; i < test_input.rows(); ++i) {
		nn.forward(test_input.row(i));
		pred(i, 0) = nn.get_val()(0, 0) > 0.5 ? 1 : 0;
	}

	MatrixXd comp(test_input.rows(), 2);
	comp << test_output, pred;
	std::cout << '\n' << comp << "\n\n"
		<< (comp.col(0).array() == comp.col(1).array()) << "\n\n";

	getchar();
	getchar();
	return 0;
}
