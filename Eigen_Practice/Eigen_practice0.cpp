//Andrew Narang
//Advanced C++ - Term Project
//Eigen Practice 0

//Here, we copy the approach in Eigen's getting started part of the documentation

#include <iostream>
#include <eigen3/Eigen/Dense>

using Eigen::MatrixXd;

using Eigen::Matrix;

using Eigen::Dynamic;

using Eigen::MatrixBase;

Matrix<double, Dynamic, Dynamic> test(const Matrix<double, Dynamic, Dynamic>& a, const Matrix<double, Dynamic, Dynamic>& b);

Matrix<double, Dynamic, Dynamic> test(const Matrix<double, Dynamic, Dynamic>& a, const Matrix<double, Dynamic, Dynamic>& b)
{
  return a * b;
}

template<typename T>
Matrix<T, Dynamic, Dynamic> test1(const Matrix<T, Dynamic, Dynamic>& a, const Matrix<T, Dynamic, Dynamic>& b);

template<typename T>
Matrix<T, Dynamic, Dynamic> test1(const Matrix<T, Dynamic, Dynamic>& a, const Matrix<T, Dynamic, Dynamic>& b)
{
  return a * b;
}

//this approach is more efficient but still as flexible as the last one
template<typename T, typename Derived>
Matrix<T, Dynamic, Dynamic> test2(const MatrixBase<Derived>& a, const MatrixBase<Derived>& b);

template<typename T, typename Derived>
Matrix<T, Dynamic, Dynamic> test2(const MatrixBase<Derived>& a, const MatrixBase<Derived>& b)
{
  return (a * b);
}


template<typename Derived>
int test3(const MatrixBase<Derived>& a, const MatrixBase<Derived>& b);

template<typename Derived>
int test3(const MatrixBase<Derived>& a, const MatrixBase<Derived>& b)
{
  return ((a.template rows()) * (b.template rows()));
}


int main()
{
  MatrixXd m(2, 2);

  m(0,0) = 3;
  m(1, 0) = 2.5;
  m(0, 1) = -1;
  m(1, 1) = m(1, 0) + m(0, 1);

  std::cout << m << std::endl;


  //now, we do our own testing

  Matrix<double, 4, 4> alpha;

  //this generates a 4x4 matrix of garbage values
  std::cout << alpha << std::endl;

  std::cout << "alpha corrected:" << std::endl;

  //this does not work - comma initialization via << operator requires all elements of the matrix to be listed. This compiles, but an error (core dumped) is generated when the program is executed since there are too few elements being entered when the << operator is first used (in the first iteration of the loop). A similar error would be generated if too many elements would be entered when using the << operator to initialize the entire matrix.   
  //for (int i = 0; i < (4 * 4); ++i)
  //  {
  //    alpha << 2;
  //  }

  //equivalent method that works well:

  for (int i = 0; i < 4; ++i)
    {
      for (int i1 = 0; i1 < 4; ++i1)
	{
	  //this fills the matrix row-by-row
	  //to make it fill column by column, we switch the order to alpha(i1, i)
	  //we can use this in the project to generate random initial weights by replacing i with the output of a Boost random number generator, according to whatever distribution we believe the data comes from 
	  
	  alpha(i, i1) = i;
	}
    }

  std::cout << alpha << std::endl;

  std::cout << "alpha initiated via columns" << std::endl;

  for (int i = 0; i < 4; ++i)
    {
      for (int i1 = 0; i1 < 4; ++i1)
	{
	  //proof of concept - this fills the matrix column-by-column
     	  
	  alpha(i1, i) = i;
	}
    }

  std::cout << alpha << std::endl;

  //set up a column vector - doing the next two lines in one line does not work -> generates compiler error. Apparently << requires a Matrix object to already exist on the left-hand side.
  Matrix<double, 3, 1> column0;

  column0 << 2, 3, 4;

  std::cout << "column0:\n" << column0 << std::endl;

  //set up another column vector of the same size to test dot product

  Matrix<double, 3, 1> column1;

  column1 << 2, 2, 2;

  std::cout << "column1:\n" << column1 << std::endl;

  std::cout << "column1's transpose:\n" << column1.transpose() << std::endl;

  std::cout << "column0 . column1:\n" << column0.dot(column1) << std::endl;

  //using regular * for matrix multiplication does not work for dot product between vectors w/same #s of elements => not considered equivalent. column0 . column1 does not work because it assumes column-vector based code, and from that perspective dimensions do not match. column0 . column1.transpose() creates a 3x3 matrix with 4s in the first row, 6s in the second row, and 8s in the third row. Not sure why it's doing that. 
  
  //does it work the same way for transpose?

  std::cout << "column0 . column1's transpose:\n" << column0.dot(column1.transpose()) << std::endl;

  //yes, it does

  //zero matrix

  Matrix<double, 4, 4> a = Matrix<double, 4, 4>::Zero();

  std::cout << "4x4 zeros Matrix:\n" << a << std::endl;

  //constant matrix

  Matrix<double, 2, 3> b = Matrix<double, 2, 3>::Constant(24);

  std::cout << "2x3 24s Matrix:\n" << b << std::endl;

  //identity matrix

  Matrix<double, 5, 5> c = Matrix<double, 5, 5>::Identity();

  std::cout << "5x5 identity matrix:\n" << c << std::endl;

  //testing of non-square identity matrix

  Matrix<double, 3, 4> d = Matrix<double, 3, 4>::Identity();

  std::cout << "3x4 identity matrix:\n" << d << std::endl;
  
  //set up a matrix by combining vectors

  Matrix<double, 3, 2> e;

  e << column0, column1;

  std::cout << "3x2 matrix formed by combining column0 and column1:\n" << e << std::endl;
  
  Matrix<double, 2, 3> f;

  f << column0.transpose(), column1.transpose();

  std::cout << "transposes of column0 and column1 stacked as rows:\n" << f << std::endl;
  
  //wow Eigen is so much easier than Numpy

  //Matrix multiplication - everything is column vector-based by default

  Matrix<double, 2, 2> vodka0;

  vodka0 << 4, 3, 2, 1;

  std::cout << "vodka0:\n" << vodka0 << std::endl;

  Matrix<double, 2, 1> vodka1;

  vodka1 << 2, 2;

  std::cout << "vodka1:\n" << vodka1 << std::endl;

  std::cout << "vodka0 hits on vodka1:\n" << vodka0 * vodka1 << std::endl;

  Matrix<double, 2, 2> vodka2;

  vodka2 << 5, 6, 7, 8;

  std::cout << "vodka2:\n" << vodka2 << std::endl;

  std::cout << "vodka2 hits on vodka0:\n" << vodka2 * vodka0 << std::endl;
  
  std::cout << "vodka2 hits on vodka1:\n" << vodka2 * vodka1 << std::endl;

  ///specialized testing

  Matrix<double, Dynamic, Dynamic> * vodka3;

  vodka3 = new Matrix<double, Dynamic, Dynamic>(4, 4);

  for (int i = 0; i < 4; ++i)
    {
      for (int i1 = 0; i1 < 4; ++i1)
	{
	  (*vodka3)(i, i1) = i + i1;
	}
    }

  

  std::cout << "vodka3 on the heap:\n" << *vodka3 << std::endl;
  
  delete vodka3;

  vodka2.rows();
  
  std::cout << (vodka2.cols() == 3) << std::endl;


  Matrix<double, Dynamic, Dynamic> sake0(2, 2);

  sake0 = Matrix<double, Dynamic, Dynamic>::Random(2, 2);

  std::cout << "sake0:\n" << sake0 << std::endl;

  Matrix<double, Dynamic, Dynamic> sake1(2, 2);

  sake1 = Matrix<double, Dynamic, Dynamic>::Random(2, 2);

  std::cout << "sake1:\n" << sake1 << std::endl;

  Matrix<double, Dynamic, Dynamic> sake2;

  sake2 = sake0 * sake1;

  std::cout << "sake2:\n" << sake2 << std::endl;

  std::cout << "test(sake0, sake1):\n" << test(sake0, sake1) << std::endl;

  std::cout << "test1(sake0, sake1):\n" << test1(sake0, sake1) << std::endl;

  std::cout << "test2(sake0, sake1):\n" << test2<double>(sake0, sake1) << std::endl;

  std::cout << "test3(sake0, sake1):\n" << test3(sake0, sake1) << std::endl;

  
  //int i = 4, i1 = 2;

  //Matrix<double, i, i1> sake = Matrix<double, i, i1>::Constant(4);

  //std::cout << "sake:\n" << sake << std::endl;
  
  return 0;
}
