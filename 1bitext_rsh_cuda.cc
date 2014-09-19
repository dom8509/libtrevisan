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

// A 1-bit extractor based on a concatenated Reed-Solomon-Hadamard code implemented in cuda
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
#include "1bitext_rsh_cuda.h"
#ifndef NO_DEBUG
#include "debug.h"
#endif

#ifdef HAVE_SSE4
#include <smmintrin.h> // popcnt
#endif

extern int debug_level;

using namespace std;

// NOTE: With NTL, it is not possible to work with two Galois fields
// of different sizes simultaneously.  So if both, the weak design and
// the 1-bit extractor, are based on Galois fields, we cannot use
// NTL... It is, however, possible to base one of the implementations
// on NTL. This allows for cross-checking the implementation, which is
// why we keep the code around.

bitext_rsh_cuda::~bitext_rsh_cuda() {
	if (irred_poly)
		delete[] irred_poly;
	if (coeffs)
		delete[] coeffs;
}

vertex_t bitext_rsh_cuda::num_random_bits() {
	// The number of random bits required for each 1-bit extractor run
	// is 2*l. For computational efficiency, the amount needs to be easily
	// divisible into two parts, we round the amount up so that it's
	// an even multiple of 8, i.e., the number of bits per char.

	return 2 * l;
}

uint64_t bitext_rsh_cuda::compute_k() {
	// Caveat: r can mean the polynomial order and the overlap in this context.
	return (static_cast<uint64_t>(bitext::r * pp.m + 4 * log2(1.0 / pp.eps) + 6));
}

void bitext_rsh_cuda::compute_r_l() {
	l = ceil(log2((long double) pp.n) + 2 * log2(2 / pp.eps));
	r = ceil(pp.n / l);

	chars_per_half = l / BITS_PER_TYPE(char);

	if (debug_level >= RESULTS) {
		cerr << "RSH EXT: eps: " << pp.eps << ", n: " << pp.n << endl;
		cerr << "RSH extractor: Computed l (Galois field order):" << l << ", r (polynomial degree) " << r << endl;
		cerr << "Number of char instances in half of initial randomness: " << chars_per_half << ")" << endl;
	}

	if (irred_poly)
		delete[] irred_poly;
	uint64_t chunks_per_elems = ceil((l + 1) / (long double) BITS_PER_TYPE(chunk_t));
	irred_poly = new sfixn[chunks_per_elems];
	set_irrep_cuda(irred_poly, l);

	if (debug_level >= INFO) {
		//cerr << "Picked rsh bit extractor irreducible polynomial ";
		//for (auto i : irred_poly)
		//	cerr << i << " ";
		//cerr << endl;
	}
}

void bitext_rsh_cuda::create_coefficients() {
	if (global_rand == NULL) {
		cerr << "Internal error: Cannot compute coefficients for uninitialised " << "RSH extractor" << endl;
		exit(-1);
	}

	b.set_raw_data(global_rand, pp.n);

	// coeffs have l plus one bits one bit for the additional carry bit
	uint64_t elems_per_coeff = ceil((l + 1) / (long double) BITS_PER_TYPE(chunk_t));
	vector<chunk_t> vec(elems_per_coeff);
	vector<chunk_t> vec2(elems_per_coeff);

	if (coeffs)
		delete[] coeffs;
	coeffs = new chunk_t[elems_per_coeff * r];

	for (uint64_t i = 0; i < r; ++i) {
		unsigned char* foo = (unsigned char*) vec.data();
		unsigned char* foo2 = (unsigned char*) vec2.data();
		b.get_bit_range(i * l, (i + 1) * l - 1, &vec[0]);

		// bits spiegeln
		int lastByte = sizeof(chunk_t) * elems_per_coeff;
		int currByte = lastByte;
		for (int i = 0; i < lastByte - 1; i++) {
			foo[--currByte] = (foo2[i] * 0x0202020202ULL & 0x010884422010ULL) % 1023;
		}

		for (uint64_t j = 0; j < elems_per_coeff; ++j) {
			coeffs[i * elems_per_coeff + j] = vec[j];
		}
	}
}

bool bitext_rsh_cuda::cudaExtract(int i, vector<vector<unsigned int> > initial_rand_all, int *extracted_rand) {

	uint64_t chunks_per_elems = ceil((l + 1) / (long double) BITS_PER_TYPE(chunk_t));

//	std::cout << "extracting bit " << i << std::endl;
//	evaluateGF2nPolyBN((sfixn*) coeffs, (sfixn*) x_all.at(i).data(), l * 2, r - 1, irred_poly, NULL, (sfixn*) rs_res_all.at(i).data());
	cuda->evaluate(i);
	sfixn* rs_res_tmp = (sfixn*) malloc(sizeof(sfixn)*chunks_per_elems*2);
	cuda->getResults(rs_res_tmp);


	vector<chunk_t> rs_res(chunks_per_elems, 0);

	sfixn* foo = (sfixn*) rs_res.data();
	for (int i=0;i<chunks_per_elems*2;i++) {
		foo[i] = rs_res_tmp[i];
	}

	vector<unsigned int> initial_rand = initial_rand_all.at(i);

	// pointer vom Typ data_t, der auf den Anfang der zweiten Hälfte
	// des Seeds zeigt
	chunk_t *seed_half = reinterpret_cast<chunk_t*>(reinterpret_cast<unsigned char*>(initial_rand.data()) + chars_per_half);
	bool parity = 0;
	unsigned short bitcnt;
	for (idx_t count = 0; count < chunks_per_elems; ++count) {
		// data wird mit zweiter Seedhälfte undiert
		rs_res[count] &= *(seed_half + count);
		// If an even number of bits is set in the current subset,
		// the global parity is XORed with one.
		// TODO: Ensure that this is really alright (does always seem
		// to end up in the != case).
#ifdef HAVE_SSE4
		bitcnt = _mm_popcnt_u64(rs_res[count]);
#else
		// Gibt die Anzahl der Einsen zurück
		bitcnt = __builtin_popcountll(rs_res[count]);
#endif
		// Check if the bitcount is divisible by two (without using
		// a modulo division to speed the test up)
		if (((bitcnt >> 2) << 2) == bitcnt) {
			parity ^= 1;
		}
	}

	if (parity) {
		extracted_rand[i / BITS_PER_TYPE(int)] |= (1 << (i % BITS_PER_TYPE(int)));
	}

	return true;
}

void bitext_rsh_cuda::prepare(int m, vector<vector<unsigned int> > initial_rand_all) {
//	std::cout << "preparing" << std::endl;

	x_all.clear();
	x_all.reserve(m);

	rs_res_all.clear();
	rs_res_all.reserve(m);

	uint64_t chunks_per_elems = ceil((l + 1) / (long double) BITS_PER_TYPE(chunk_t));

	for (int k = 0; k < m; k++) {

		void *inital_rand = initial_rand_all[k].data();

		vector<chunk_t> x(chunks_per_elems, 0);
		vector<chunk_t> rs_res(chunks_per_elems, 0);

		// TODO: warum krieg ich hier l bits wenn l doch der Grad des irreduziblen Polynoms ist???
		bitfield<chunk_t, idx_t> ir((chunk_t *) inital_rand, l);
		vector<chunk_t> first_half(chunks_per_elems);

		ir.get_bit_range(0, l, first_half.data());

		unsigned char* foo = (unsigned char*) x.data();
		unsigned char* foo2 = (unsigned char*) first_half.data();
		// bits spiegeln
		int lastByte = sizeof(chunk_t) * chunks_per_elems;
		int currByte = lastByte;
		for (int i = 0; i < lastByte - 1; i++) {
			foo[--currByte] = (foo2[i] * 0x0202020202ULL & 0x010884422010ULL) % 1023;
		}

		x_all.push_back(x);
		rs_res_all.push_back(rs_res);
	}

	int chunks_per_elems2 = chunks_per_elems * (sizeof(chunk_t) / sizeof(sfixn));

	sfixn* x_all2 = (sfixn*) malloc(sizeof(sfixn) * m * chunks_per_elems2);
	for (int i = 0; i < m; i++) {
		sfixn* x_all_tmp = (sfixn*) x_all[i].data();
		for (int j = 0; j < chunks_per_elems2; j++) {
			x_all2[i * chunks_per_elems2 + j] = x_all_tmp[j];
		}
	}

	// sfixn* coeffs, sfixn* x, sfixn num_x, sfixn size_field, sfixn deg_poly, sfixn* irred_poly, sfixn* mask
	cuda = new GF2nPolyBN((sfixn*)coeffs, x_all2, m, l * 2, r - 1, irred_poly, NULL);

	// x_all2 ist ein eindimensionales array das zwei dimensionen repraesentiert
	// erste dimension ist m; zweite dimension ist chunks_per_elems2
	// ein zugriff erfolgt ueber i*chunks_per_elems2+j

}

void bitext_rsh_cuda::followUp() {
//	std::cout << "follow up" << std::endl;

	delete cuda;
}

bool bitext_rsh_cuda::extract(void *inital_rand) {
	bool res = 0;

	// Reed-Solomon part
	// Select the first l bits from the initial randomness,
	// initialise an element of the finite field with them, and
	// evaluate the polynomial on the value
	uint64_t chunks_per_elems = ceil((l + 1) / (long double) BITS_PER_TYPE(chunk_t));
	vector<chunk_t> x(chunks_per_elems);
	vector<chunk_t> rs_res(chunks_per_elems);

	// TODO: warum krieg ich hier l bits wenn l doch der Grad des irreduziblen Polynoms ist???
	bitfield<chunk_t, idx_t> ir((chunk_t *) inital_rand, l);
	vector<chunk_t> first_half(chunks_per_elems);
	ir.get_bit_range(0, l, &first_half[0]);

	unsigned char* foo = (unsigned char*) x.data();
	unsigned char* foo2 = (unsigned char*) first_half.data();
	// bits spiegeln
	int lastByte = sizeof(chunk_t) * chunks_per_elems;
	int currByte = lastByte;
	for (int i = 0; i < lastByte - 1; i++) {
		foo[--currByte] = (foo2[i] * 0x0202020202ULL & 0x010884422010ULL) % 1023;
	}
	//std::copy(std::begin(first_half), std::end(first_half), std::begin(x));

	// Evaluate the polynomial
	// TODO: mask muss erstellt werden
	//evaluateGF2nPolyBN((sfixn*) coeffs, (sfixn*) &x[0], l * 2, r - 1, irred_poly, NULL, (sfixn*) &rs_res[0]);

	// pointer vom Typ data_t, der auf den Anfang der zweiten Hälfte
	// des Seeds zeigt
	chunk_t *seed_half = reinterpret_cast<chunk_t*>(reinterpret_cast<unsigned char*>(inital_rand) + chars_per_half);
	bool parity = 0;
	unsigned short bitcnt;
	for (idx_t count = 0; count < chunks_per_elems; ++count) {
		// data wird mit zweiter Seedhälfte undiert
		rs_res[count] &= *(seed_half + count);
		// If an even number of bits is set in the current subset,
		// the global parity is XORed with one.
		// TODO: Ensure that this is really alright (does always seem
		// to end up in the != case).
#ifdef HAVE_SSE4
		bitcnt = _mm_popcnt_u64(rs_res[count]);
#else
		// Gibt die Anzahl der Einsen zurück
		bitcnt = __builtin_popcountll(rs_res[count]);
#endif
		// Check if the bitcount is divisible by two (without using
		// a modulo division to speed the test up)
		if (((bitcnt >> 2) << 2) == bitcnt) {
			parity ^= 1;
		}
	}

	return parity;
}
