import irreps_cuda
from random import randrange
import os
import math
import argparse
	 
def createCCodeParameterFile( filename, type_size, coeffs, num_coeffs, x, mask, irred_poly, field_size ):

	try:
		os.remove(filename)
	except:	
		None

	num_chunks = int(math.ceil((field_size+1.)/type_size))

	file = open(filename, "w")

	file.write("#include <CudaTypes.h>\n")
	file.write("#include <vector>\n")
	file.write("\n")
	file.write("void get_rsh_test_parameters(sfixn& field_size, sfixn& num_coeffs, std::vector<sfixn>& coeffs, std::vector<sfixn>& x, std::vector<sfixn>& irred_poly, std::vector<sfixn>& mask) {\n")
	file.write("\n")

	file.write("\t//Resizing vectors\n")
	file.write("\tcoeffs.resize(" + str(num_coeffs *  num_chunks) + ");\n")
	file.write("\tx.resize(" + str(num_chunks) + ");\n")
	file.write("\tmask.resize(" + str(num_chunks) + ");\n")
	file.write("\tirred_poly.resize(" + str(num_chunks) + ");\n")
	file.write("\n")

	type_sizemask = (2**type_size)-1

	file.write("\t//Setting the field size\n")
	file.write("\tfield_size = " + str(field_size) + ";\n")
	
	file.write("\n")

	file.write("\t//Setting the number of coeffs\n")
	file.write("\tnum_coeffs = " + str(num_coeffs) + ";\n")

	file.write("\n")

	# code for setting all coeffs
	for i in range(0, num_coeffs):
		file.write("\t//Setting coefficient " + str(i) + "\n")
		current_coeff = coeffs[i]
		for j in range(num_chunks-1, -1, -1):
			file.write("\tcoeffs[" + str((i * num_chunks) + j) + "] = " + str(current_coeff&type_sizemask) + ";\n")
			current_coeff >>= 32
		file.write("\n")

	file.write("\n")

	# code for setting x
	file.write("\t//Setting x\n")
	for i in range(num_chunks-1, -1, -1):
		file.write("\tx[" + str(i) + "] = " + str(x&type_sizemask) + ";\n")
		x >>= 32

	file.write("\n")

	# code for setting the irreducible polynomial
	file.write("\t//Setting the irreducible polynomial\n")
	for i in range(num_chunks-1, -1, -1):
		file.write("\tirred_poly[" + str(i) + "] = " + str(irred_poly&type_sizemask) + ";\n")
		irred_poly >>= 32

	file.write("\n")

	file.write("\t//Setting the mask\n")
	for i in range(num_chunks-1, -1, -1):
		file.write("\tmask[" + str(i) + "] = " + str(mask&type_sizemask) + ";\n")
		mask >>= 32

	file.write("\n")
	file.write("}")

	file.close()

def createPyCodeParameterFile( filename, coeffs, num_coeffs, x, mask, irred_poly, field_size ):
	
	try:
		os.remove(filename)
	except:	
		None

	file = open(filename, "w")

	file.write("def get_rsh_test_parameter_num_coeffs():\n")
	file.write("\treturn [" + str(num_coeffs) + "]\n")
	file.write("\n")

	file.write("def get_rsh_test_parameter_field_size():\n")
	file.write("\treturn [" + str(field_size) + "]\n")
	file.write("\n")

	file.write("def get_rsh_test_parameter_coeffs():\n")
	file.write("\treturn " + str(coeffs) + "\n")
	file.write("\n")

	file.write("def get_rsh_test_parameter_x():\n")
	file.write("\treturn [" + str(x) + "]\n")
	file.write("\n")

	file.write("def get_rsh_test_parameter_irred_poly():\n")
	file.write("\treturn [" + str(irred_poly) + "]\n")
	file.write("\n")

	file.write("def get_rsh_test_parameter_mask():\n")
	file.write("\treturn [" + str(mask) + "]\n")
	file.write("\n")

	file.close()

def main():

	parser = argparse.ArgumentParser(description='Creates parameter files to test the rsh extractor.')
	parser.add_argument('-f','--field_size', help='Size of the GF2n field in bits', required=True)
	parser.add_argument('-d','--degree', help='Degree of the polynomial', required=True)
	parser.add_argument('-ocf','--output_c_file', help='Output C file', required=True)
	parser.add_argument('-opf','--output_py_ile', help='Output Python file', required=True)
	parser.add_argument('-v', '--verbose', help='Print some additional output', action='store_true')

	args = vars(parser.parse_args())
	
	coeffs = [randrange((2**int(args['field_size']))-1) for _ in range(0, int(args['degree']))]
	x = randrange((2**int(args['field_size']))-1)
	mask = 2**int(args['field_size'])
	irred_poly = irreps_cuda.set_irrep(int(args['field_size']))

	if args['verbose']:
		print "Calculated coeffs:", [("{0:0" + str(int(args['field_size'])) + "b}").format(i) for i in coeffs] 
		print "Calculated x:", ("{0:0" + str(int(args['field_size'])) + "b}").format(x)
		print "Calculated mask:", ("{0:0" + str(int(args['field_size'])+1) + "b}").format(mask)
		print "Irreducible polynomial:", ("{0:0" + str(int(args['field_size'])+1) + "b}").format(irred_poly)


	createCCodeParameterFile(args['output_c_file'], 32, coeffs, int(args['degree']), x, mask, irred_poly, int(args['field_size']))
	createPyCodeParameterFile(args['output_py_ile'], coeffs, int(args['degree']), x, mask, irred_poly, int(args['field_size']))

main()