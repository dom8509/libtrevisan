from __future__ import print_function
import argparse
import rsh_test_parameters
from PolyEvalGF2n import *
from GF2n import GF2n


def loadDadaFromFile( file ):
	f = open(file, "r")
	lines = f.readlines()
	res = dict([[key.strip(), val.strip()] for key,val in [line.split("=", 1) for line in lines]])
	for key in res.keys():
		res[key] = [int(i, 2) for i in res[key].split(' ')]
	f.close()
	return res


def getWidthBinTree( n ):
	widthBinTree = 1
	while widthBinTree < n:
		widthBinTree <<= 1
	return widthBinTree


def calculatePythonResults( field, specParam ):

	# calculate tree
	resultExpandStep = parallelExpandVec(field(specParam["x"][0]), getWidthBinTree(specParam["num_coeffs"][0]))
	resultReduceStep = parallelPrefProdReduce(resultExpandStep)
	resultDownSweepStep = parallelPrefProdDownSweep(resultReduceStep)
	resultProdStep = parallelPrefProdMultiply(resultDownSweepStep, \
		([field(0)] * (getWidthBinTree(specParam["num_coeffs"][0]) - specParam["num_coeffs"][0])) + \
		[field(x) for x in specParam["coeffs"]])
	resultSumStep = parallelPrefProdSum(resultProdStep)

	# assign results
	specParam["resultExpandStep"] = [int(str(x)) for x in resultExpandStep]
	specParam["resultReduceStep"] = [int(str(x)) for x in resultReduceStep]
	specParam["resultDownSweepStep"] = [int(str(x)) for x in resultDownSweepStep]
	specParam["resultProdStep"] = [int(str(x)) for x in resultProdStep]
	specParam["resultSumStep"] = [int(str(resultSumStep))]


def printFailedResults( key, specValue, calcValue ):

	print("\n")
	print("-> Results for " + key)
	print("specified:")
	print(specValue)
	print("calculated:")
	print(calcValue)
	print("\n")


def compareResults( specParam, calcParam ):

	static_len = 12
	max_key_len = max([len(key) for key in calcParam.keys()])
	text_len = static_len + max_key_len

	for key in calcParam.keys():
		print("Checking " + key + "...", end="")
		[print(".", end="") for _ in range(3, max(0, text_len-(9+len(key))))]

		if specParam[key] == calcParam[key]:
			print("\033[92m" + "passed\n" + "\033[0m", end="")	
		else:
			print("\033[91m" + "FAILED!\n" + "\033[0m", end="")
			printFailedResults(key, specParam[key], calcParam[key])


def analyseResults( calcParam, verbose ):

	specParam = { \
		"num_coeffs" : rsh_test_parameters.get_rsh_test_parameter_num_coeffs(), \
		"field_size":rsh_test_parameters.get_rsh_test_parameter_field_size(), \
		"coeffs" : rsh_test_parameters.get_rsh_test_parameter_coeffs(), \
		"x" : rsh_test_parameters.get_rsh_test_parameter_x(), \
		"irred_poly" : rsh_test_parameters.get_rsh_test_parameter_irred_poly(), \
		"mask" : rsh_test_parameters.get_rsh_test_parameter_mask(), \
		"resultExpandStep" : 0, \
		"resultReduceStep" : 0, \
		"resultDownSweepStep" : 0, \
		"rsultProdStep" : 0, \
		"resultSumStep" : 0 \
		}

	field = GF2n(specParam["field_size"][0], specParam["irred_poly"][0])

	calculatePythonResults(field, specParam)

	compareResults(specParam, calcParam)

	if( verbose ):
		for key in specParam.keys():
			print(key + ":")
			print(specParam[key])


def main():

	parser = argparse.ArgumentParser(description='Compares the results in the passed file against a python implementation.')
	parser.add_argument('-irf','--input_result_file', help='File which contains the results of the rsh extractors', required=True)
	parser.add_argument('-v', '--verbose', help='Print some additional output', action='store_true')

	args = vars(parser.parse_args())

	file_content = loadDadaFromFile(args["input_result_file"])

	analyseResults(file_content, args["verbose"])


main()