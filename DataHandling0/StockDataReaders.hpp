//Andrew Narang
//Advanced C++ - Assignment #4



#ifndef StockDataReaders_HPP
#define StockDataReaders_HPP

#include <fstream>
#include <vector>

#include <unordered_map>
#include <tuple>
#include <string>
#include <forward_list>
#include <map>

//dates must be a vector of 3 empty int vectors
//prices must be a vector of n_components - 1 empty double vectors
//data of source must be clean (already parsed)
//data is assumed to be of the format: date, stock0 price, stock2 price, ...
//assignment #3
void StockDataReader0(std::ifstream& source, std::vector<std::vector<int>>& dates, std::vector<std::vector<double>>& prices, const int& components_n, const char& delim, const char& date_delim);


//read in stock data and fill an unordered map - assignment #4
void StockDataReader1(std::ifstream& source, std::unordered_map<std::string, std::tuple<std::string, double, double, double, double, double>>& target, const int& components_n, const char& delim);

//read in stock data and fill a map - assignment #4
void StockDataReader2(std::ifstream& source, std::map<std::string, std::forward_list<std::string>>& target, const int& components_n, const char& delim);


#endif
