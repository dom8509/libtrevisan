#ifndef __LIBTREVISANCUDA_H__
#define __LIBTREVISANCUDA_H__

#include "CudaTypes.h"

void evaluateGF2nPolyBN(sfixn* coeffs, sfixn* x, sfixn size_field, sfixn deg_poly, sfixn* irred_poly, sfixn* mask, sfixn* result);

#endif