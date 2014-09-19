#include "PolyEvalGF2nBN.cuh"
#include "../utils.hpp"
#include <cstdlib>
#include <vector>
#include <cmath>
#include <iostream>

void get_rsh_test_parameters(sfixn& field_size, sfixn& num_coeffs, std::vector<sfixn>& coeffs, std::vector<sfixn>& x, std::vector<sfixn>& irred_poly, std::vector<sfixn>& mask); 

template <typename T>
T getNumberChunks( T length ) {
	if( !length )
		return 0;
	else
		return (length-1)/(sizeof(T)*8)+1;
}

int main(int argc, char** argv) {

	sfixn field_size;
	sfixn num_coeffs;
	std::vector<sfixn> coeffs;
	std::vector<sfixn> x;
	std::vector<sfixn> irrep_poly;
	std::vector<sfixn> mask;

	get_rsh_test_parameters(field_size, num_coeffs, coeffs, x, irrep_poly, mask);

	sfixn numberChunks = getNumberChunks(field_size + 1);

	std::vector<sfixn> result(numberChunks);

	if(false) {
		std::cout << "field_size: " << field_size << std::endl;
		std::cout << "num_coeffs: " << num_coeffs << std::endl;

		std::cout << "Coeffs: " << std::endl;
		printbin<sfixn>(&coeffs[0], numberChunks, num_coeffs);
		std::cout << "x: " << std::endl;
		printbin<sfixn>(&x[0], numberChunks, 1);
		std::cout << "irrep_poly: " << std::endl;
		printbin<sfixn>(&irrep_poly[0], numberChunks, 1);
		std::cout << "mask: " << std::endl;
		printbin<sfixn>(&mask[0], numberChunks, 1);
		std::cout << "Result before calling evaluateGF2nPolyBN:" << std::endl;
		printbin<sfixn>(&result[0], numberChunks, 1);
	}
	
	GF2nPolyBN poly(&coeffs[0], &x[0], 1, field_size, num_coeffs-1, &irrep_poly[0], &mask[0]);
	poly.evaluate(0);
	poly.getResults(&result[0]);

	if(false) {
		std::cout << "Result after calling evaluateGF2nPolyBN:" << std::endl;
		printbin<sfixn>(&result[0], numberChunks, 1);
	}
	
	return 0;
}
