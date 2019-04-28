//Andrew Narang
//Advanced C++ - Assignment #4


#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <iomanip>
#include <forward_list>
#include <map> 
#include <unordered_map>
#include <tuple>


//assignment #3
void StockDataReader0(std::ifstream& source, std::vector<std::vector<int>>& dates, std::vector<std::vector<double>>& prices, const int& components_n, const char& delim, const char& date_delim)
{
  //get the first line (header) from source

  std::string tank = "";

  getline(source, tank);

  while (getline(source, tank))
    {
      std::stringstream row(tank);

         
      //vector that will store everything as a string
      //std::vector<std::string> components(components_n, "");

      std::string * components = new std::string[components_n];
      
           
      for (int i = 0; i < components_n; ++i)
	{
	  getline(row, components[i], delim);
	}

      //now the array components is filled with string versions of the data in each row

      //split up the date parts into separate strings
      std::stringstream date0(components[0]);

      std::string * date_parts = new std::string[3]; 
      //std::vector<std::string> date_parts(3, "");

      for (int i = 0; i < 3; ++i)
	{
	  getline(date0, date_parts[i], date_delim);
	}

      //we have filled an array with these components. store them in dates' component vectors
      for (int i = 0; i < 3; ++i)
	{
	  dates[i].push_back(std::stoi(date_parts[i]));
	}

      delete[] date_parts;

      //now store everything else in components in the other vectors for price series
      for (int i = 1; i < components_n; ++i)
	{
	  if (std::stoi(components[i]) != -10000)
	    {
	      prices[i - 1].push_back(std::stod(components[i]));
	    }
	  else {
	    //updates with last data if there are errors in data - prices cannot be negative, so this error value works
	    prices[i - 1].push_back(prices[i - 1].back());
	  }
	}

      delete[] components;

      //std::cout << prices[0].back() << std::endl;
      
      //std::cout << "one row down" << std::endl;
      
      //move on to the next row
    }

  //std::cout << "looping done" << std::endl;

}

  

//assignment #4
void StockDataReader1(std::ifstream& source, std::unordered_map<std::string, std::tuple<std::string, double, double, double, double, double>>& target, const int& components_n, const char& delim)
{
  //get the first line (header) from source

  std::string tank = "";

  getline(source, tank);

  while (getline(source, tank))
    {
      std::stringstream row(tank);

      std::string * components = new std::string[components_n];

      for (int i = 0; i < components_n; ++i)
	{
	  getline(row, components[i], delim);
	}

      //now the array components is filled w/ string versions of the data in each row

      //insert it all according to this question's requirements
      
      target.insert(std::pair<std::string, std::tuple<std::string, double, double, double, double, double>>(components[0], std::make_tuple(components[1], std::stod(components[2]), std::stod(components[3]), std::stod(components[4]), std::stod(components[5]), std::stod(components[6])))); 
      
      delete[] components;
    }
}


//read in stock data and fill a map - assignment #4
void StockDataReader2(std::ifstream& source, std::map<std::string, std::forward_list<std::string>>& target, const int& components_n, const char& delim)
{
  std::string tank = "";

  getline(source, tank);

  while(getline(source, tank))
    {
      std::stringstream row(tank);

      std::string * components = new std::string[components_n];

      for (int i = 0; i < components_n; ++i)
	{
	  getline(row, components[i], delim);
	}

      //now the array components is filled w/ string versions of the data in each row

      //insert it all according to this question's requirements

      try {
	target.at(components[1]).push_front(components[0]);
      }
      catch (...) {
	target[components[1]] = std::forward_list<std::string>({components[0]}); 
      }
	//    target.insert(std::pair<std::string, std::tuple<std::string, double, double, double, double, double>>(components[0], std::make_tuple(components[1], std::stod(components[2]), std::stod(components[3]), std::stod(components[4]), std::stod(components[5]), std::stod(components[6])))); 
      
      delete[] components;
    }
}
