//Andrew Narang
//Advanced C++ - Term Project

//This file contains the declarations of stock data parsers.


#ifndef StockData_Parsers_HPP
#define StockData_Parsers_HPP


#include <string>

//#include <fstream>


int StockData_parser_0(const std::string& in_source, const std::string& in_output, const int& initial_alpha_components_n, const int& components_n, const char& delim);


//assumes data in format of date, number, number, number...
int StockData_parser_1(const std::string& in_source, const std::string& in_output, const int& components_n, const char& delim, const char& date_delim);



//void StockData_parser_0(std::ifstream& source, std::ofstream& output, const int& initial_alpha_components_n, const int& components_n, const char& delim);

//assumes data in format of date, number, number, number...
//void StockData_parser_1(std::ifstream& source, std::ofstream& output, const int& components_n, const char& delim, const char& date_delim);

#endif
