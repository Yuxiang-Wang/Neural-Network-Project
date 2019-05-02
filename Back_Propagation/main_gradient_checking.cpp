#include"neural_network.hpp"
#include<eigen3/Eigen/Core>
#include<iostream>
#include<vector>
using namespace Eigen;

int main(){
	MatrixXd test_input(MatrixXd::Random(20, 3));
    MatrixXd test_output=(test_input.rowwise().sum().array()<0).cast<double>();
	std::vector<MatrixXd> weight;
	int units_hidden = 3, num_layer = 4, units_input = 3, units_output = 1;
	weight.push_back(MatrixXd::Random(units_hidden, units_input + 1));
	for (int i = 1; i < num_layer - 2; ++i)
		weight.push_back(MatrixXd::Random(units_hidden, units_hidden + 1));
	weight.push_back(MatrixXd::Random(units_output, units_hidden + 1));

    Neural_network nn(num_layer,units_input,units_hidden,units_output);
    nn.set_data_input(test_input);
    nn.set_data_output(test_output);

	std::cout << "compute derivative numerically:\n";
	for (auto it = weight.begin(); it != weight.end(); ++it) {
		for (int i = 0; i < (*it).rows(); ++i) {
			for (int j = 0; j < (*it).cols(); ++j) {			
				double cost1, cost2;
				(*it)(i, j) += 0.001;
				nn.set_weight(weight);
				nn.forward(test_input.row(0));
				cost1 = cost_func(test_output.row(0),nn.get_val());
				(*it)(i, j) -= 0.002;
				nn.set_weight(weight);
				nn.forward(test_input.row(0));
				cost2 = cost_func(test_output.row(0), nn.get_val());
				std::cout << (cost1 - cost2) / 0.002 << ", ";
				(*it)(i, j) += 0.001;
				
			}
			std::cout << '\n';
		}
		std::cout << "\n\n";
	}
	
	nn.set_weight(weight);
	nn.forward(test_input.row(0));
	nn.backward(test_output.row(0));
	auto l = nn.get_layers();

	std::cout << "derivative in the model\n";
	for (auto& e : l) {
		std::cout << e.diff << "\n\n";
	}

	getchar();
	getchar();
    return 0;
}
