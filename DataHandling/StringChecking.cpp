//Andrew Narang
//Advanced C++ - Term Project


#include <string>


bool num_str_check(const std::string& input)
{
  //checks if input string is empty
  if (input.size() == 0)
    {
      return 0;
    }

  //checks if input string is just "." => invalid
  if (input == ".")
    {
      return 0;
    }

  //checks if there are more than one '.' characters within the string - only one is allowed, as it can represent a decimal point
  int decimal_count = 0;
  
  for (int i = 0; i < input.size(); ++i)
    {
      if (input[i] == '.')
	{
	  ++decimal_count;
	}

      if (decimal_count > 1)
	{
	  return 0;
	}
    }

  //checks that every character within the string is either a . or a number
  for (int i = 0; i < input.size(); ++i)
    {
      if (!(isdigit(input[i]) || (input[i] == '.')))
	{
	  return 0;
	}
    }

  return 1; 
}


bool date_str_check(const std::string& input, const char& delim)
{
  //checks if input string is empty
  if (input.size() == 0)
    {
      return 0;
    }

  //checks if there are not two delim characters within the string - two must be there, as it sets up a date
  int delim_count = 0;

  for (int i = 0; i < input.size(); ++i)
    {
      if (input[i] == delim)
        {
          ++delim_count;
        }
    }

  if (delim_count != 2)
    {
      return 0;
    }

  for (int i = 0; i < input.size(); ++i)
    {
      if (!(isdigit(input[i]) || (input[i] == delim)))
        {
          return 0;
        }
    }

  return 1;

}
