from math import log


class GF2n:

	def __init__( self, field_size, irred_poly ):
		self._field_size = field_size
		self._irred_poly = irred_poly
		self._mask = 1<<field_size


	def __cmp__( self, other ):
		return self._field_size == other._field_size & \
			self._irred_poly == other._irred_poly


	def __call__( self, *args, **kwargs ):

		if len(args) != 1:
			raise Exception("Initialization needs exactly one value!!!")

		return _GF2nElement(args[0], self)



class _GF2nElement:

	def __init__( self, value, field ):
		self.__value = value
		self.__field = field


	def getField( self ):
		return self.__field


	def __add__( self, other ):
		self.__checkField(other)
		return _GF2nElement(self.__value ^ other.__value, self.__field)


	def __sub__( self, other ):
		self.__checkField(other)
		return _GF2nElement(self.__value ^ other.__value, self.__field)


	def __mul__( self, other ):
		
		self.__checkField(other)
		
		a = self.__value
		b = other.__value
		res = 0

		for i in range(0, int(log(self.__field._mask, 2))):
			if (b & 1) > 0:
				res ^= a
			
			a <<= 1
			if (a & self.__field._mask) > 0:
				a ^= self.__field._irred_poly
			b >>= 1

		return _GF2nElement(res, self.__field)


	def __div__( self, other ):
		assert 0


	def __str__( self ):
		return str(self.__value)


	def __checkField( self, other ):
		if(self.__field != other.__field):
			raise Exception("Fields do not match!")
		return True