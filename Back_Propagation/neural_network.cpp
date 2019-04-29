#include"neural_network.hpp"

double sigmoid(double x){
    return 1/(1+exp(x));
}

Neural_network::Neural_network(int layers,int units_input,int units_hidden,int units_output):
    layers(layers),units_hidden(units_hidden),
    units_output(units_output),units_input(units_input){
    // weight init. size: layers-1, except output layer
    std::srand((unsigned int) 0);
    w.push_back(MatrixXd::Random(units_hidden,units_input+1));
    for(int i=1;i<layers-2;++i){
	std::srand((unsigned int) i);
	w.push_back(MatrixXd::Random(units_hidden,units_hidden+1));
    }
    std::srand((unsigned int) layers-2);
    w.push_back(MatrixXd::Random(units_output,units_hidden+1));
    // val init. size: layers
    val.push_back(MatrixXd(units_input+1,1));
    for(int i=1;i<layers-1;++i){
	val.push_back(MatrixXd(units_hidden+1,1));
    }
    val.push_back(MatrixXd(units_output,1));
    //diff init. size:layers-1, except output layer
    diff.push_back(MatrixXd(units_hidden,units_input+1));
    for(int i=1;i<layers-2;++i){
	diff.push_back(MatrixXd(units_hidden,units_hidden+1));
    }
    diff.push_back(MatrixXd(units_output,units_hidden+1));
    //diff_total init. size:layers-1, except output layer
    diff_total.push_back(MatrixXd(units_hidden,units_input+1));
    for(int i=1;i<layers-2;++i){
	diff_total.push_back(MatrixXd(units_hidden,units_hidden+1));
    }
    diff_total.push_back(MatrixXd(units_output,units_hidden+1));
    //delta init. size:layers-1, except input layer
    for(int i=1;i<layers-1;++i){
	delta.push_back(MatrixXd(units_hidden+1,1));
    }
    delta.push_back(MatrixXd(units_output,1));
}

void Neural_network::set_data_input(const MatrixXd& m){
    if(m.cols()!=units_input){
	std::cerr<<"nums of features not equal to units_input\n";
	exit(-1);
    }
    if(data_output.size()){
	if(m.rows()!=data_output.rows()){
	    std::cerr<<"nums of input not equal to nums of output\n";
	    exit(-1);
	}
    }
    data_input = m;
}

void Neural_network::set_data_output(const MatrixXd& m){
    if(m.cols()!=units_output){
	std::cerr<<"nums of features not equal to units_input, or rearange data to columns\n";
	exit(-1);
    }
    if(data_input.size()){
	if(data_input.rows()!=m.rows()){
	    std::cerr<<"nums of input not equal to nums of output\n";
	    exit(-1);
	}
    }
    data_output = m;
}

void Neural_network::set_weight(const std::vector<MatrixXd>& weight){
    if(weight.front().rows()!=units_hidden || weight.front().cols()!=units_input+1){
	std::cerr<<"input weights not campatible with model\n";
	exit(-1);
    }
    if(weight.back().rows()!=units_output || weight.back().cols()!=units_hidden+1){
	std::cerr<<"input weights not campatible with model\n";
	exit(-1);
    }
    for(auto it=w.begin()+1;it!=w.end()-1;++it){
	if((*it).rows()!=units_hidden || (*it).cols()!=units_hidden+1){
	    std::cerr<<"input weights not campatible with model\n";
	    exit(-1);
	}
    }
    w = weight;
}

MatrixXd Neural_network::get_val() const{
    return val.back();
}

void Neural_network::forward(MatrixXd input){
	// exceptional handling for input layer and output layer
    val[0]<<MatrixXd::Ones(1,1)*0.1,Map<MatrixXd>(input.data(),units_input,1);
    for(int i=0;i<layers-2;++i){
	val[i+1]<<MatrixXd::Ones(1,1)*0.1,(w[i]*val[i]).unaryExpr(&sigmoid);
    }
    val.back()=(w.back()*(*(val.end()-2))).unaryExpr(&sigmoid);
}

void Neural_network::backward(MatrixXd output){
	// delta
	// exceptional handling for output layer
    output = Map<MatrixXd>(output.data(),units_output,1);
    delta.back()=val.back()-output;
    auto it_delta = delta.rbegin()+1;
    auto it_w = w.rbegin();
    auto it_val = val.rbegin()+1;
    for(;it_delta!=delta.rend();++it_delta,++it_w,++it_val){
	(*it_delta)=((*it_w).transpose()*(*(it_delta-1))).array()*
	    (*it_val).array()*(MatrixXd::Ones(units_hidden+1,1)-(*it_val)).array();
    }

	// diff
	// exceptional handling for input layer and output layer
	// transform MatrixXd to VectorXd to do outer product. didn't find a way to do it using MatrixXd
    it_val = val.rbegin()+1;
    VectorXd delta_vec(Map<VectorXd>(delta.back().data(),delta.back().rows()));
    VectorXd val_vec(Map<VectorXd>((*it_val).data(),(*it_val).rows()));
    diff.back() = delta_vec*val_vec.transpose();
    auto it_diff = diff.rbegin()+1;
    it_delta = delta.rbegin()+1;
    
    for(++it_val;it_diff!=diff.rend();++it_diff,++it_val,++it_delta){
	delta_vec=Map<VectorXd>((*it_delta).data(),(*it_delta).rows());
	val_vec=Map<VectorXd>((*it_val).data(),(*it_val).rows());
	(*it_diff) = delta_vec.segment(1, delta_vec.size() - 1)*val_vec.transpose();
    }
}

void Neural_network::update(){
    auto it_w=w.begin();
    auto it_diff_total = diff_total.begin();
    for(;it_w!=w.end();++it_w,++it_diff_total){
	(*it_w)+=learning_rate*(*it_diff_total);
    }
}

void Neural_network::training(std::string file){    
    int num = data_input.rows();
    
    for(int ite=0;ite<MAX_ITE;++ite){
	for(auto it=diff_total.begin();it!=diff_total.end();++it)
	    (*it).setZero();

	for(int i=0;i<num;++i){
	    MatrixXd input(Map<MatrixXd>(data_input.row(i).data(),units_input,1));
	    MatrixXd output(Map<MatrixXd>(data_output.row(i).data(),units_output,1));
	    forward(input);
	    backward(output);
	    auto it_diff=diff.begin();
	    auto it_diff_total=diff_total.begin();
	    for(;it_diff_total!=diff_total.end();++it_diff,++it_diff_total)
		(*it_diff_total)+=(*it_diff);

	}

	// exit by checking value of derivative
	double max_diff=0;
	for(auto it=diff_total.begin();it!=diff_total.end();++it){
	    max_diff = (*it).array().abs().maxCoeff()>max_diff?
		(*it).array().abs().maxCoeff():max_diff;
	}
	if(max_diff*learning_rate<0.001){
	    std::cout<<"\ntraining done, total iteration "<<ite<<'\n';
	    return;
	}
	update();

	// output for showing progress
	// output to file for store training results. 
	// don't need to re-train again incase slow converging for large scale data set
	if(!(ite%(MAX_ITE/20))){
	    std::cout<<ite<<", ";
	    if(file.length()){
		std::ofstream outfile(file);
		if(outfile.fail()){
		    std::cerr<<"can't open "<<file<<" to write\n";
		    exit(-1);
		}
		for(auto it=w.begin();it!=w.end();++it)
		    outfile<<(*it)<<',';
		outfile.close();
	    }
	}
    }
    std::cout<<"\ntraining not done, iteration times exeed maximum limit "<<MAX_ITE<<"\n";
}		

    
    
