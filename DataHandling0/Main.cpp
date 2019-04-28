//Andrew Narang
//Advanced C++ - Term Project

//Testing of Stock Data Parsers/Readers


#include <string>
#include <iostream>

#include "StockData_Parsers.hpp"

int main()
{
  std::string source_file = "sp500_companies.csv";

  std::string output_file = "parsed_" + source_file;

  //  std::cout << output_file << std::endl;

  //testing of StockData_parser_0
  
  char delim = ',';
  int alpha_components = 2;
  int total_components = 7;

  if (StockData_parser_0(source_file, output_file, alpha_components, total_components, delim))
    {
      std::cout << "StockData_parser_0 worked for sp500_companies.csv" << std::endl;
    }

  else {
    
    std::cout << "StockData_parser_0 did not work for sp500_companies.csv" << std::endl;

  }

  //testing of StockData_parser_1

  std::string source_file_hist = "hist_prices.csv";

  std::string output_file_hist = "parsed_" + source_file_hist;
  
  int n_components = 6;
  char date_delim = '-';

  if (StockData_parser_1(source_file_hist, output_file_hist, n_components, delim, date_delim))
    {
      std::cout << "StockData_parser_1 worked for hist_prices.csv" << std::endl;
    }
  else {

    std::cout << "StockData_parser_1 did not work for hist_prices.csv" << std::endl;

  }
  
  return 0;
}
