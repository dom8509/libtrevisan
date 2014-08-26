from GF2n import GF2n


def expandVec( x, n ):
	return [x] * n


def parallelPrefProdReduce( x, widthBinTree ):
	
	offset = 1
	leaves = expandVec(x, widthBinTree)

	i = widthBinTree >> 1
	while( i > 0 ):
		for j in range(0, i):
			leaves[(offset*(2*j + 1) - offset)] = \
				leaves[(offset*(2*j + 1) - offset)] * leaves[(offset*(2*j + 2) - offset)]

		i >>= 1
		offset <<= 1

	return leaves


def parallelPrefProdDownSweep( leaves ):

	leaves[0] = (leaves[0].getField())(1)

	offset = len(leaves)
	i = 1
	while( i < len(leaves) ):
		offset >>= 1

		for j in range(0, i):
			tmp = leaves[(offset*(2*j + 2) - offset)]
			leaves[(offset*(2*j + 2) - offset)] = leaves[(offset*(2*j + 1) - offset)]
			leaves[(offset*(2*j + 1) - offset)] = leaves[(offset*(2*j + 1) - offset)] * tmp

		i *= 2 

	return leaves


def parallelPrefProdMultiply( leaves, coeffs ):
	if( len(leaves) != len(coeffs) ):
		raise Exception("Wow, there must be the same amount of coeffs as leaves! What did you think I'll do with that?")
	return [(leaves[i] * coeffs[i]) for i in range(0, len(leaves))]


def parallelPrefProdSum( leaves ):
	res = (leaves[0].getField())(0)

	for x in leaves:
		res = res + x

	return res


def checkPackage():
	f = GF2n(3, 11)
	assert [str(x) for x in expandVec(f(6), 4)] == ['6', '6', '6', '6']
	assert [str(x) for x in parallelPrefProdReduce(f(6), 4)] == ['4', '6', '2', '6']
	assert [str(x) for x in parallelPrefProdDownSweep(parallelPrefProdReduce(f(6), 4))] == ['7', '2', '6', '1']
	assert str(parallelPrefProdSum([f(4), f(3)])) == '7'


checkPackage()