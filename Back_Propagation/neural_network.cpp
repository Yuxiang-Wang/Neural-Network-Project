#include"neural_network.hpp"

double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

double cost_func(MatrixXd output, const MatrixXd& x) {
	output = Map<MatrixXd>(output.data(), output.size(), 1);
	double j = 0;
	for (int i = 0; i < output.size(); ++i) {
		j += -(output(i, 0)*log(x(i, 0)) + (1 - output(i, 0))*log(1 - x(i, 0)));
	}
	return j;
}
Layer::Layer(MatrixXd w, MatrixXd diff, MatrixXd diff_total, MatrixXd val, MatrixXd delta)
	:w(w), diff(diff), delta(delta), val(val), diff_total(diff_total) {}

Neural_network::Neural_network(int num_layer, int units_input, int units_hidden, int units_output) :
	num_layer(num_layer), units_hidden(units_hidden),
	units_output(units_output), units_input(units_input) {
	// input layer
	std::srand((unsigned int)0);
	layers.push_back(Layer(MatrixXd::Random(units_hidden, units_input + 1),
		MatrixXd(units_hidden, units_input + 1),
		MatrixXd(units_hidden, units_input + 1),
		MatrixXd(units_input + 1, 1),
		MatrixXd(0, 0)));
	// hidden layers, except the last hidden layer
	for (int i = 1; i < num_layer - 2; ++i) {
		std::srand((unsigned int)i);
		layers.push_back(Layer(MatrixXd::Random(units_hidden, units_hidden + 1),
			MatrixXd(units_hidden, units_hidden + 1),
			MatrixXd(units_hidden, units_hidden + 1),
			MatrixXd(units_hidden + 1, 1),
			MatrixXd(units_hidden + 1, 1)));
	}
	// the last hidden layer
	std::srand((unsigned int)num_layer - 2);
	layers.push_back(Layer(MatrixXd::Random(units_output, units_hidden + 1),
		MatrixXd(units_output, units_hidden + 1),
		MatrixXd(units_output, units_hidden + 1),
		MatrixXd(units_hidden + 1, 1),
		MatrixXd(units_hidden + 1, 1)));
	// the output layer
	layers.push_back(Layer(MatrixXd(0, 0),
		MatrixXd(0, 0),
		MatrixXd(0, 0),
		MatrixXd(units_output, 1),
		MatrixXd(units_output, 1)));
}

void Neural_network::set_data_input(const MatrixXd& m) {
	if (m.cols() != units_input) {
		std::cerr << "nums of features not equal to units_input\n";
		exit(-1);
	}
	if (data_output.size()) {
		if (m.rows() != data_output.rows()) {
			std::cerr << "nums of input not equal to nums of output\n";
			exit(-1);
		}
	}
	data_input = m;
}

void Neural_network::set_data_output(const MatrixXd& m) {
	if (m.cols() != units_output) {
		std::cerr << "nums of features not equal to units_input, or rearange data to columns\n";
		exit(-1);
	}
	if (data_input.size()) {
		if (data_input.rows() != m.rows()) {
			std::cerr << "nums of input not equal to nums of output\n";
			exit(-1);
		}
	}
	data_output = m;
}

void Neural_network::set_weight(const std::vector<MatrixXd>& weight) {
	for (int i = 0; i < num_layer - 1; ++i)
		layers[i].w = weight[i];
}

MatrixXd Neural_network::get_val() const {
	return layers.back().val;
}

std::vector<Layer> Neural_network::get_layers() const {
	return layers;
}

void Neural_network::forward(MatrixXd input) {
	// exceptional handling for input layer and output layer
	layers[0].val << MatrixXd::Ones(1, 1)*0.1, Map<MatrixXd>(input.data(), units_input, 1);
	for (int i = 0; i < num_layer - 2; ++i) {
		layers[i + 1].val << MatrixXd::Ones(1, 1)*0.1, (layers[i].w*layers[i].val).unaryExpr(&sigmoid);
	}
	layers.back().val = (layers[num_layer - 2].w*layers[num_layer - 2].val).unaryExpr(&sigmoid);
}

void Neural_network::backward(MatrixXd output) {
	// delta
	// exceptional handling for output layer
	output = Map<MatrixXd>(output.data(), units_output, 1);
	layers.back().delta = layers.back().val - output;
	auto it = layers.rbegin() + 1;
	(*it).delta = ((*it).w.transpose()*(*(it - 1)).delta).array()*
		(*it).val.array()*(MatrixXd::Ones(units_hidden + 1, 1) - (*it).val).array();

	for (++it; it != layers.rend() - 1; ++it) {
		(*it).delta = ((*it).w.transpose()*(*(it - 1)).delta.block(1,0,units_hidden,1)).array()*
			(*it).val.array()*(MatrixXd::Ones(units_hidden + 1, 1) - (*it).val).array();
	}

	// diff
	// exceptional handling for input layer and output layer
	// transform MatrixXd to VectorXd to do outer product. didn't find a way to do it using MatrixXd
	it = layers.rbegin() + 1;
	VectorXd delta_vec(Map<VectorXd>(layers.back().delta.data(), layers.back().delta.rows()));
	VectorXd val_vec(Map<VectorXd>((*it).val.data(), (*it).val.rows()));
	(*it).diff = delta_vec * val_vec.transpose();

	for (++it; it != layers.rend(); ++it) {
		delta_vec = Map<VectorXd>((*(it - 1)).delta.data(), (*(it - 1)).delta.rows());
		val_vec = Map<VectorXd>((*it).val.data(), (*it).val.rows());
		(*it).diff = delta_vec.segment(1, delta_vec.size() - 1)*val_vec.transpose();
	}
}

void Neural_network::update(const double& learning_rate) {
	for (auto it = layers.begin(); it != layers.end() - 1; ++it) {
		(*it).w -= learning_rate * (*it).diff_total;
	}
}

int Neural_network::training(const double& learning_rate, std::string file) {
	int num = data_input.rows();

	double cost=1e5, cost_last, cost_init;
	for (int ite = 0; ite < MAX_ITE; ++ite) {
		for (auto it = layers.begin(); it != layers.end() - 1; ++it)
			(*it).diff_total.setZero();

		cost_last = cost;
		cost = 0;
		for (int i = 0; i < num; ++i) {
			forward(data_input.row(i));
			backward(data_output.row(i));
			for (auto it = layers.begin(); it != layers.end() - 1; ++it)
				(*it).diff_total += (*it).diff;
			cost += cost_func(data_output.row(i), layers.back().val);
		}
		for (auto it = layers.begin(); it != layers.end() - 1; ++it)
			(*it).diff_total /= num;
		cost /= num;

		// exit by checking value of derivative
		/*
		double max_diff = 0;
		for (auto it = layers.begin(); it != layers.end() - 1; ++it) {
			max_diff = (*it).diff_total.array().abs().maxCoeff() > max_diff ? (*it).diff_total.array().abs().maxCoeff() : max_diff;
		}
		*/
		
		if (!ite)
			cost_init = cost;
		if (cost > 2 * cost_init) {
			std::cout << "\ntraining rate too big\n";
			return 0;
		}
		if (abs(cost-cost_last) < 0.0001) {
			std::cout << "\ntraining done, total iteration " << ite << '\n';
			return ite;
		}
		update(learning_rate);

		// output for showing progress
		// output to file for store training results. 
		// don't need to re-train again incase slow converging for large scale data set
		if (!(ite % 10)) {
			std::cout << ite << ", "<<"cost: "<<cost<<" ... ";
			if (file.length()) {
				std::ofstream outfile(file);
				if (outfile.fail()) {
					std::cerr << "can't open " << file << " to write\n";
					exit(-1);
				}
				for (auto it = layers.begin(); it != layers.end() - 1; ++it)
					outfile << (*it).w << ',';
				outfile.close();
			}
		}
	}
	std::cout << "\ntraining not done, iteration times exeed maximum limit " << MAX_ITE << "\n";
}



