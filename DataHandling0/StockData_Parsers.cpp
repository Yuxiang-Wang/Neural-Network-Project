//Andrew Narang
//Advanced C++ - Term Project


//This file contains the implementations of stock data parsers. It can be extended to other types of data as well - it's just about the format. 

#include <fstream>
#include <string>
#include <sstream> 

#include <iostream> 

#include "StringChecking.hpp"


int StockData_parser_0(const std::string& in_source, const std::string& in_output, const int& initial_alpha_components_n, const int& components_n, const char& delim)
{
  //set up the input and output files
  
  std::ifstream source(in_source);

  if (source.fail())
    {
      std::cerr << "Error accessing " << in_source << '\n';
      return 0;
    }

  std::ofstream output(in_output);

  if (output.fail())
    {
      std::cerr << "Error creating/accessing " << in_output << '\n';
      return 0;
    }
  
  //Read in 0th line of source = header, output it to output, which is a blank output file, and then move to new line

  std::string tank = "";

  getline(source, tank);

  output << tank << std::endl;

  //now move on to actual data

  //stores each row of the source data in the string tank
  
  //first we read in each row of the data as a string. then we convert it into a stringstream object and use getline with comma as delimiting character to split up the row into a series of strings - one per piece of data. Then, we check for errors in the strings that should be numerical and we output the doubles corresponding to the numerical strings to the output file. This is how we can return a parsed result by calling the parsing function. 
  while (getline(source, tank))
    {
      //creates a stringstream object that stores the information of tank separated with commas
      std::stringstream row(tank);
      
      std::string * components = new std::string[components_n];

      for (int i = 0; i < (components_n); ++i)
	{
	  getline(row, *(components + i), delim);
	}

      //not necessary: last element in each row always has a \n at the end so we don't need a comma as delimiter
      //getline(row, *(components + (components_n - 1)), '\n'); 

      //now the array components is filled with string versions of the data in each row

      //outputs the inital alphanumeric components that do not need to be checked to output directly as strings. we preserve the delimination character in this file. 
      for (int i = 0; i < initial_alpha_components_n; ++i)
	{
	  output << components[i] << delim;
	}

      //now we iterate through the rest of the data in this row, checking if they have errors with the assumption that they are numeric

      //std::cout << components[components_n - 1] << std::endl;

      // std::cout << components[6].size() << std::endl;

      //the last element in the row always has an extra \n character that needs to be eliminated for the error parsing boolean function to properly read the strings
      components[components_n - 1].erase(components[components_n - 1].size() - 1);
      
      //std::cout << components[6] << std::endl;

      //std::cout << components[6].size() << std::endl;

      //std::cout << num_str_check(components[6]) << std::endl;

      //this one is separate from the loop to accomodate the preceding delim from the last loop
      if (num_str_check(components[initial_alpha_components_n]))
	{
	  output << components[initial_alpha_components_n];
	}
      else {
	output << -10000;
      }
      
      for (int i = initial_alpha_components_n + 1; i < components_n; ++i)
	{
	  /*
	  try {

	    output << delim << stod(components[i]);
	  }

	  catch (...)
	    {
	      output << (-1);
	    }
	  */
	
	  
	  
	  if (num_str_check(components[i]))
	  //if (isdigit((*(components + i))[0])  
	    {
	      output << delim << components[i]; 
	      //output << delim << stof(components[i]);
	    }
	  else {

	    output << delim << (-10000);

	    }

	}   
	  
      delete[] components;

      //move on to next row
      output << std::endl;
    }

  source.close();

  output.close();

  return 1;
  
}



int StockData_parser_1(const std::string& in_source, const std::string& in_output, const int& components_n, const char& delim, const char& date_delim)
{
  //set up the input and output files

  std::ifstream source(in_source);

  if (source.fail())
    {
      std::cerr << "Error accessing " << in_source << '\n';
      return 0;
    }

  std::ofstream output(in_output);

  if (output.fail())
    {
      std::cerr << "Error creating/accessing " << in_output << '\n';
      return 0;
    }
  
  //Read in 0th line of source = header, output it to output, which is a blank output file, and then move to new line

  std::string tank = "";

  getline(source, tank);

  output << tank << std::endl;

  //now move on to actual data

  //stores each row of the source data in the string tank

  //first we read in each row of the data as a string. then we convert it into a stringstream object and use getline with comma as delimiting character to split up the row into a series of strings - one per piece of data. Then, we check for errors in the strings that should be numerical and we output the doubles corresponding to the numerical strings to the output file. This is how we can return a parsed result by calling the parsing function.
  while (getline(source, tank))
    {
      //creates a stringstream object that stores the information of tank separated with commas
      std::stringstream row(tank);

      std::string * components = new std::string[components_n];

      for (int i = 0; i < (components_n); ++i)
        {
          getline(row, *(components + i), delim);
        }

      //not necessary: last element in each row always has a \n at the end so we don't need a comma as delimiter
      //getline(row, *(components + (components_n - 1)), '\n');

      //now the array components is filled with string versions of the data in each row

      //outputs the inital date component that we check is proper and then output directly as a string. we preserve the delimination character in this file.

      if (date_str_check(components[0], date_delim))
	{
	  output << components[0] << delim;
	}
      else {
	output << -10000 << delim;
      }

      //now we iterate through the rest of the data in this row, checking if they have errors with the assumption that they are numeric

      //std::cout << components[components_n - 1] << std::endl;

      // std::cout << components[6].size() << std::endl;

      //the last element in the row always has an extra \n character that needs to be eliminated for the error parsing boolean function to properly read the strings
      components[components_n - 1].erase(components[components_n - 1].size() - 1);

      //std::cout << components[6] << std::endl;

      //std::cout << components[6].size() << std::endl;

      //std::cout << num_str_check(components[6]) << std::endl;

      //this one is separate from the loop to accomodate the preceding delim from the date string output
      if (num_str_check(components[1]))
        {
          output << components[1];
        }
      else {
        output << -10000;
      }

      for (int i = 2; i < components_n; ++i)
        {
          /*
          try {

            output << delim << stod(components[i]);
          }

          catch (...)
            {
              output << (-1);
            }
          */



          if (num_str_check(components[i]))
          //if (isdigit((*(components + i))[0])
            {
              output << delim << components[i];
              //output << delim << stof(components[i]);
            }
          else {

            output << delim << (-10000);

            }

        }

      delete[] components;

      //move on to next row
      output << std::endl;
    }

  source.close();

  output.close();

  return 1;
  
}


/*

void StockData_parser_0(std::ifstream& source, std::ofstream& output, const int& initial_alpha_components_n, const int& components_n, const char& delim)
{
  //Read in 0th line of source = header, output it to output, which is a blank output file, and then move to new line

  std::string tank = "";

  getline(source, tank);

  output << tank << std::endl;

  //now move on to actual data

  //stores each row of the source data in the string tank
  
  //first we read in each row of the data as a string. then we convert it into a stringstream object and use getline with comma as delimiting character to split up the row into a series of strings - one per piece of data. Then, we check for errors in the strings that should be numerical and we output the doubles corresponding to the numerical strings to the output file. This is how we can return a parsed result by calling the parsing function. 
  while (getline(source, tank))
    {
      //creates a stringstream object that stores the information of tank separated with commas
      std::stringstream row(tank);
      
      std::string * components = new std::string[components_n];

      for (int i = 0; i < (components_n); ++i)
	{
	  getline(row, *(components + i), delim);
	}

      //not necessary: last element in each row always has a \n at the end so we don't need a comma as delimiter
      //getline(row, *(components + (components_n - 1)), '\n'); 

      //now the array components is filled with string versions of the data in each row

      //outputs the inital alphanumeric components that do not need to be checked to output directly as strings. we preserve the delimination character in this file. 
      for (int i = 0; i < initial_alpha_components_n; ++i)
	{
	  output << components[i] << delim;
	}

      //now we iterate through the rest of the data in this row, checking if they have errors with the assumption that they are numeric

      //std::cout << components[components_n - 1] << std::endl;

      // std::cout << components[6].size() << std::endl;

      //the last element in the row always has an extra \n character that needs to be eliminated for the error parsing boolean function to properly read the strings
      components[components_n - 1].erase(components[components_n - 1].size() - 1);
      
      //std::cout << components[6] << std::endl;

      //std::cout << components[6].size() << std::endl;

      //std::cout << num_str_check(components[6]) << std::endl;

      //this one is separate from the loop to accomodate the preceding delim from the last loop
      if (num_str_check(components[initial_alpha_components_n]))
	{
	  output << components[initial_alpha_components_n];
	}
      else {
	output << -10000;
      }
      
      for (int i = initial_alpha_components_n + 1; i < components_n; ++i)
	{
	  /*
	  try {

	    output << delim << stod(components[i]);
	  }

	  catch (...)
	    {
	      output << (-1);
	    }
	  */
	

/*
	  
	  if (num_str_check(components[i]))
	  //if (isdigit((*(components + i))[0])  
	    {
	      output << delim << components[i]; 
	      //output << delim << stof(components[i]);
	    }
	  else {

	    output << delim << (-10000);

	    }

	}   
	  
      delete[] components;

      //move on to next row
      output << std::endl;
    }
}
      
*/



/*
void StockData_parser_1(std::ifstream& source, std::ofstream& output, const int& components_n, const char& delim, const char& date_delim)
{
  //Read in 0th line of source = header, output it to output, which is a blank output file, and then move to new line

  std::string tank = "";

  getline(source, tank);

  output << tank << std::endl;

  //now move on to actual data

  //stores each row of the source data in the string tank

  //first we read in each row of the data as a string. then we convert it into a stringstream object and use getline with comma as delimiting character to split up the row into a series of strings - one per piece of data. Then, we check for errors in the strings that should be numerical and we output the doubles corresponding to the numerical strings to the output file. This is how we can return a parsed result by calling the parsing function.
  while (getline(source, tank))
    {
      //creates a stringstream object that stores the information of tank separated with commas
      std::stringstream row(tank);

      std::string * components = new std::string[components_n];

      for (int i = 0; i < (components_n); ++i)
        {
          getline(row, *(components + i), delim);
        }

      //not necessary: last element in each row always has a \n at the end so we don't need a comma as delimiter
      //getline(row, *(components + (components_n - 1)), '\n');

      //now the array components is filled with string versions of the data in each row

      //outputs the inital date component that we check is proper and then output directly as a string. we preserve the delimination character in this file.

      if (date_str_check(components[0], date_delim))
	{
	  output << components[0] << delim;
	}
      else {
	output << -10000 << delim;
      }

      //now we iterate through the rest of the data in this row, checking if they have errors with the assumption that they are numeric

      //std::cout << components[components_n - 1] << std::endl;

      // std::cout << components[6].size() << std::endl;

      //the last element in the row always has an extra \n character that needs to be eliminated for the error parsing boolean function to properly read the strings
      components[components_n - 1].erase(components[components_n - 1].size() - 1);

      //std::cout << components[6] << std::endl;

      //std::cout << components[6].size() << std::endl;

      //std::cout << num_str_check(components[6]) << std::endl;

      //this one is separate from the loop to accomodate the preceding delim from the date string output
      if (num_str_check(components[1]))
        {
          output << components[1];
        }
      else {
        output << -10000;
      }

      for (int i = 2; i < components_n; ++i)
        {
          /*
          try {

            output << delim << stod(components[i]);
          }

          catch (...)
            {
              output << (-1);
            }
          */

	  /*

          if (num_str_check(components[i]))
          //if (isdigit((*(components + i))[0])
            {
              output << delim << components[i];
              //output << delim << stof(components[i]);
            }
          else {

            output << delim << (-10000);

            }

        }

      delete[] components;

      //move on to next row
      output << std::endl;
    }
}
*/