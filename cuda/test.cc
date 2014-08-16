#include "../prng.hpp"
#include "../utils.hpp"
#include "PolyEvalGF2nBN.cuh"
#include <cstdlib>
#include <vector>
#include <cmath>
#include <iostream>
#include <cstdio>

void set_irrep_cuda(sfixn*, unsigned);

int main(int argc, char** argv) {

	sfixn field_size = 1023;
	sfixn num_coeffs = 1000;

	std::vector<sfixn> coeffs;
	create_randomness<sfixn>(num_coeffs*(field_size+1)/32, coeffs);
	
	std::vector<sfixn> x;
	create_randomness<sfixn>((field_size+1)/32, x);
	
	std::vector<sfixn> irrep_poly((field_size+1)/32);
	set_irrep_cuda(&irrep_poly[0], field_size);

	std::vector<sfixn> mask((field_size+1)/32);
	mask[0] = pow(2, 32-1);

	std::vector<sfixn> result((field_size+1)/32);

	std::cout << "Result before calling evaluateGF2nPolyBN:" << std::endl;
	printbincharpad<sfixn>(&result[0], (field_size+1)/32);
	
	evaluateGF2nPolyBN(&coeffs[0], &x[0], field_size, num_coeffs, &irrep_poly[0], &mask[0], &result[0]);

	std::cout << "Result after calling evaluateGF2nPolyBN:" << std::endl;
	printbincharpad<sfixn>(&result[0], (field_size+1)/32);
	
	return 0;
}
