/* This file is part of libtrevisan, a modular implementation of
   Trevisan's randomness extraction construction.

   Copyright (C) 2011-2012, Wolfgang Mauerer <wm@linux-kernel.net>

   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License
   as published by the Free Software Foundation; either version 2
   of the License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with libtrevisan. If not, see <http://www.gnu.org/licenses/>. */

// A 1-bit extractor based on a concatenated Reed-Solomon-Hadamard code
// Algorithmic description and parameter calculations by Christoper Portmann

#include<iostream>
#include<fstream>
#include<cstddef>
#include<cstdlib>
#include<cstdio> // stderr
#include<gmp.h>
#include<vector>
#include<NTL/tools.h>
#include<NTL/GF2E.h>
#include "GF2Es.h"
#include "timing.h"
#include "utils.hpp"
#include "irreps.h"
#include "1bitext_rsh.h"
#ifndef NO_DEBUG
#include "debug.h"
#endif

#ifdef HAVE_SSE4
#include <smmintrin.h> // popcnt
#endif

extern int debug_level;

#ifdef USE_NTL
NTL_CLIENT
#endif

using namespace std;

// NOTE: With NTL, it is not possible to work with two Galois fields
// of different sizes simultaneously.  So if both, the weak design and
// the 1-bit extractor, are based on Galois fields, we cannot use
// NTL... It is, however, possible to base one of the implementations
// on NTL. This allows for cross-checking the implementation, which is
// why we keep the code around.

bitext_rsh::~bitext_rsh() {
	#ifdef USE_CUDA
		if(irred_poly)
			delete [] irred_poly;
		if(coeffs)
			delete [] coeffs;
	#else
		if (global_rand != nullptr) {
			for (auto bignum : coeffs)
				BN_free(bignum);
		}
	#endif
}

vertex_t bitext_rsh::num_random_bits() {
	// The number of random bits required for each 1-bit extractor run
	// is 2*l. For computational efficiency, the amount needs to be easily
	// divisible into two parts, we round the amount up so that it's
	// an even multiple of 8, i.e., the number of bits per char.

	uint64_t l_rounded = l + (8-l%8);
	return 2*l_rounded;
}

/*
	Berechnet die Entropy
*/
uint64_t bitext_rsh::compute_k() {
	// Caveat: r can mean the polynomial order and the overlap in this context.
	return(static_cast<uint64_t>(bitext::r*pp.m + 4*log2(1.0/pp.eps) + 6));
}

/* 
	Berechnet die vom Extraktor benötigte Seedlänge l und den Grad
	des zu erzeugenden Polynoms r
*/
void bitext_rsh::compute_r_l() {
	l = ceil(log2((long double)pp.n) + 2*log2(2/pp.eps));
	r = ceil(pp.n/l);

	uint64_t l_rounded = l + (8-l%8); // See comment in num_random_bits
	// l_rounded ist schon die Hälfte, da durch num_random_bits ja 2*l 
	// gefordert wurde, dadurch ist l_rouded/8 = chars_per_half
	chars_per_half = l_rounded/BITS_PER_TYPE(char);

	if (debug_level >= RESULTS) {
		cerr << "RSH EXT: eps: " << pp.eps << ", n: " << pp.n << endl;
		cerr << "RSH extractor: Computed l (Galois field order):" << l 
		     << ", r (polynomial degree) " << r << endl;
		cerr << "Number of char instances in half of initial randomness: "
		     << chars_per_half << "(l_rounded=" << l_rounded << ")" << endl;
	}

	#ifdef USE_CUDA
		if(irred_poly)
			delete [] irred_poly;
		irred_poly = new sfixn[(l-1)/SIZE_CHUNK+1];
		set_irrep_cuda(irred_poly, l);
	#else
		set_irrep(irred_poly, l);
	#endif
	
	#ifdef USE_NTL
		GF2E::init(irred_poly);
	#endif
		
		if (debug_level >= INFO) {
	#ifdef USE_NTL
			cerr << "Picked rsh bit extractor irreducible polynomial "
			     << irred_poly << endl;
	#elif !defined USE_CUDA
			cerr << "Picked rsh bit extractor irreducible polynomial ";
			for (auto i : irred_poly)
				cerr << i << " ";
			cerr << endl;
	#endif
	}
}

#if !defined USE_NTL && !defined USE_CUDA
// Horner's rule for polynomial evaluation, using GF(2^n) arithmetic
// This is nearly c&p from weakdes_gf2p, but providing a unified
// version via templates would require another layer of indirection for
// the function calls, so I don't think the extra complexity pays off
// for this small amount of code.
BIGNUM *bitext_rsh::horner_poly_gf2n(BIGNUM *x) {
	BIGNUM *res; res = BN_new();
	BIGNUM tmp; BN_init(&tmp);
	BN_zero(res);

	// TODO: Ideally, This should be moved to the per-thread initialisation
	// (would require an API change that we could do during the migration
	// tp TBB). We can also allocate a new context in the copy operator
	// and the constructor to make sure each object is equipped with an
	// own one, and use one instance for each tbb thread (just provide
	// the instance as copy-by-value in the dispatcher)
	BN_CTX *ctx = BN_CTX_new();

	for(size_t i = 0; i < coeffs.size(); i++) {
		BN_GF2m_mod_mul_arr(&tmp, res, x, irred_poly, ctx);
		BN_GF2m_add(res, &tmp, coeffs[i]);
	}

	BN_CTX_free(ctx);

	return res;
}
#endif

void bitext_rsh::create_coefficients() {
	if (global_rand == NULL) {
		cerr << "Internal error: Cannot compute coefficients for uninitialised "
		     << "RSH extractor" << endl;
		exit(-1);
	}

	// wird eigentlich schon in set_input_data erledigt
	b.set_raw_data(global_rand, pp.n);
	
	// TODO: Some padding may need to be required
	// Berechnet die Anzahl der Blöcke pro Koeffizient
	// Bsp: Wenn l = 14 und Größe von chung_t = 3, so werden 5 Elemente benötigt
	uint64_t elems_per_coeff = ceil(l/(long double)BITS_PER_TYPE(chunk_t));
	vector<chunk_t> vec(elems_per_coeff);

#if !defined USE_NTL && !defined USE_CUDA
	coeffs.resize(r);
#endif
	for (uint64_t i = 0; i < r; i++) { // Iteration über alle Koeffizienten

		// Hole jeweils l Bits aus b und speichere sie in vec an der Stelle 0
		// Merke: b ist der der Eingabezufallsstring
		b.get_bit_range(i*l, (i+1)*l-1, &vec[0]);

		// TODO: In the paper, coefficient i is for the exponent r-i. This
		// is not what we do. Why are the values mingled this way? Since we
		// are creating the coefficients from randomness, their relative order
		// should not matter because the numerical values are, well, random...

		// Das einzige was hier gemacht wird ist in jeder Iteration den Wert von
		// vec in coeffs[i] als BigNumber zu speichern
#ifdef USE_NTL
		GF2Es val;
		val.setValue(vec);
		if (debug_level >= EXCESSIVE_INFO)
			cerr << "RSH Coefficient " << i << ": " << val << endl;

		SetCoeff(poly, i, val);
#else
		coeffs[i] = BN_new();
		BN_bin2bn(reinterpret_cast<const unsigned char*>(&vec[0]),
			  sizeof(chunk_t)*vec.size()/sizeof(char), coeffs[i]);
		if (debug_level >= EXCESSIVE_INFO) {
			cerr << "RSH Coefficient " << i << "(" << r << "): "
			     << BN_bn2dec(coeffs[i]) << cerr << endl;
		}
#endif
	}
}

bool bitext_rsh::extract(void *inital_rand) {
	bool res = 0;

	// Reed-Solomon part
	// Select the first l bits from the initial randomness,
	// initialise an element of the finite field with them, and
	// evaluate the polynomial on the value
#ifdef USE_NTL
	GF2Es val;
	// Assign l bits to val, the parameter for the polynomial
	GF2XFromBytes(val.LoopHole(), inital_rand, chars_per_half);

	// ... and evaluate the polynomial
	GF2E rs_res;
	eval(rs_res, poly, val); // rs_res = poly(val)
#elif USE_CUDA
	sfixn  x[chars_per_half*SIZE_CHUNK/SIZE_BYTE];
	BIGNUM *rs_res = BN_new();
#else
	BIGNUM *val = BN_new();
	BIGNUM *rs_res = BN_new();
	BN_bin2bn(reinterpret_cast<unsigned char*>(inital_rand), chars_per_half, val);

	rs_res = horner_poly_gf2n(val);
	BN_free(val);
#endif

	// Hadamard part
	// Take the second l bits of randomness, compute a logical AND with the result,
	// and compute the parity
	data_t *data;
	idx_t data_len;
#ifdef USE_NTL
	data = reinterpret_cast<_ntl_ulong*>(&rs_res.LoopHole().xrep[0]);
	data_len = (rs_res.LoopHole().xrep.length());
#else
	data = reinterpret_cast<BN_ULONG*>(rs_res->d);
	data_len = rs_res->dmax;
#endif

	// pointer vom Typ data_t, der auf den Anfang der zweiten Hälfte 
	// des Seeds zeigt
	data_t *seed_half = reinterpret_cast<data_t*>(
		reinterpret_cast<unsigned char*>(global_rand) + chars_per_half);
	bool parity = 0;
	unsigned short bitcnt;
	for (idx_t count = 0; count < data_len; count++) {
		// data wird mit zweiter Seedhälfte undiert
		*(data+count) &= *(seed_half + count);
		// If an even number of bits is set in the current subset,
		// the global parity is XORed with one.
		// TODO: Ensure that this is really alright (does always seem
		// to end up in the != case).
#ifdef HAVE_SSE4
		bitcnt = _mm_popcnt_u64(*(data+count));
#else
		// Gibt die Anzahl der Einsen zurück
		bitcnt = __builtin_popcountll(*(data+count));
#endif
		// Check if the bitcount is divisible by two (without using
		// a modulo division to speed the test up)
		if (((bitcnt >> 2) << 2) == bitcnt) {
			parity ^= 1;
		}
	}

#ifndef USE_NTL
	BN_free(rs_res);
#endif
	return parity;
}
