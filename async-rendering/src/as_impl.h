
#ifndef _MOCHIMAZUI_AS_IMPL_H_
#define _MOCHIMAZUI_AS_IMPL_H_


//#include "rasterizer/shared/ras_define.h"
#include <cuda.h>
#include <host_defines.h>

static const int SMs = 22;  // this value must match the used GPU
static const int MOD = 256;  // do not change
static const int MM1 = MOD - 1;  // do not change
static const int Max_Dim = 32;  // do not increase

template <typename T, int factor, int dim, int order, T(*op)(T, T)>
static __global__ __launch_bounds__(1024, 2)
void ekScan(const T * const __restrict__ ginput, T * const __restrict__ goutput, const int items, volatile T * const __restrict__ gcarry, volatile int * const __restrict__ gwait)
{
	/*
	  // The following assertions need to hold but are commented out for performance reasons.
	  assert(1024 == blockDim.x);
	  assert(SMs * 2 == gridDim.x);
	  assert(64 >= gridDim.x);
	  assert(Max_Dim >= dim);
	*/
	const int chunks = (items + (1024 * factor - 1)) / (1024 * factor);
	const int tid = threadIdx.x;
	const int warp = tid >> 5;
	const int lane = tid & 31;
	const int corr = 1024 % dim;

	__shared__ T globcarry[dim][order];
	__shared__ T tempcarry[dim];
	__shared__ T sbuf[factor][32 * dim];

	for (int i = tid; i < dim * order; i += 1024) {
		globcarry[i / order][i % order] = 0;
	}

	int pos = 0;
	for (int chunk = blockIdx.x; chunk < chunks; chunk += SMs * 2) {
		const int offs = tid + chunk * (1024 * factor);
		const int firstid = offs % dim;
		const int lastid = (offs + 1024 * (factor - 1)) % dim;

		T val[factor];
		if (chunk < chunks - 1) {
			for (int i = 0; i < factor; i++) {
				if (offs==0 && i==0) {
					val[i] = 0;
				}
				else {
					val[i] = ginput[offs + 1024 * i - 1];
				}
				
			}
		}
		else {
			for (int i = 0; i < factor; i++) {
				val[i] = 0;
				if (offs + 1024 * i < items) {
					if (offs == 0 && i == 0) {
						val[i] = 0;
					}
					else {
						val[i] = ginput[offs + 1024 * i - 1];
					}
				}
			}
		}


		for (int round = 0; round < order; round++) {
			for (int i = 0; i < factor; i++) {
				for (int d = dim; d < 32; d *= 2) {
					T tmp = __shfl_up(val[i], d);
					if (lane >= d) val[i] = op(val[i], tmp);
				}
			}

			if (lane >= (32 - dim)) {
				const int tix = warp * dim;
				int id = firstid;
				for (int i = 0; i < factor; i++) {
					sbuf[i][tix + id] = val[i];
					id += corr;
					if (id >= dim) id -= dim;
				}
			}

			__syncthreads();
			if (warp < dim) {
				const int idx = (lane * dim) + warp;
				for (int i = 0; i < factor; i++) {
					T v = sbuf[i][idx];
					for (int d = 1; d < 32; d *= 2) {
						T tmp = __shfl_up(v, d);
						if (lane >= d) v = op(v, tmp);
					}
					sbuf[i][idx] = v;
				}
			}

			__syncthreads();
			if (warp > 0) {
				const int tix = warp * dim - dim;
				int id = firstid;
				for (int i = 0; i < factor; i++) {
					val[i] = op(val[i], sbuf[i][tix + id]);
					id += corr;
					if (id >= dim) id -= dim;
				}
			}

			T carry[dim];
			for (int d = 0; d < dim; d++) {
				carry[d] = 0;
			}
			int id = firstid;
			for (int i = 1; i < factor; i++) {
				for (int d = 0; d < dim; d++) {
					carry[d] = op(carry[d], sbuf[i - 1][31 * dim + d]);
				}
				id += corr;
				if (id >= dim) id -= dim;
				val[i] = op(val[i], carry[id]);
			}

			int wait = round + 1;
			if (tid > 1023 - dim) {
				gcarry[lastid * (order * MOD) + round * MOD + (chunk & MM1)] = val[factor - 1];
				gwait[round * MOD + ((chunk + (MOD - 4 * SMs)) & MM1)] = 0;
				__threadfence();
				if (tid == 1023) {
					gwait[round * MOD + (chunk & MM1)] = wait;
				}
			}

			const int tidx = pos + tid;
			if (tidx < chunk) {
				wait = gwait[round * MOD + (tidx & MM1)];
			}
			while (__syncthreads_count(wait <= round) != 0) {
				if (wait <= round) {
					wait = gwait[round * MOD + (tidx & MM1)];
				}
			}

			if (warp < dim) {
				int posx = pos + lane;
				T carry = 0;
				if (posx < chunk) {
					carry = gcarry[warp * (order * MOD) + round * MOD + (posx & MM1)];
				}
				if (SMs > 16) {
					posx += 32;
					if (posx < chunk) {
						carry = op(carry, gcarry[warp * (order * MOD) + round * MOD + (posx & MM1)]);
					}
				}
				for (int d = 1; d < 32; d *= 2) {
					carry = op(carry, __shfl_up(carry, d));
				}
				if (lane == 31) {
					T temp = op(globcarry[warp][round], carry);
					tempcarry[warp] = globcarry[warp][round] = temp;
				}
			}

			__syncthreads();
			if (tid > 1023 - dim) {
				globcarry[lastid][round] = op(globcarry[lastid][round], val[factor - 1]);
			}

			id = firstid;
			for (int i = 0; i < factor; i++) {
				val[i] = op(val[i], tempcarry[id]);
				id += corr;
				if (id >= dim) id -= dim;
			}
		} // round

		if (chunk < chunks - 1) {
			for (int i = 0; i < factor; i++) {
				goutput[offs + 1024 * i] = val[i];
			}
		}
		else {
			for (int i = 0; i < factor; i++) {
				if (offs + 1024 * i < items) {
					goutput[offs + 1024 * i] = val[i];
				}
			}
		}

		pos = chunk + 1;
	} // chunk
}

template <typename T, int factor, int dim, int order, T(*op)(T, T)>
static __global__ __launch_bounds__(1024, 2)
void ikScan(const T * const __restrict__ ginput, T * const __restrict__ goutput, const int items, volatile T * const __restrict__ gcarry, volatile int * const __restrict__ gwait)
{
	/*
	  // The following assertions need to hold but are commented out for performance reasons.
	  assert(1024 == blockDim.x);
	  assert(SMs * 2 == gridDim.x);
	  assert(64 >= gridDim.x);
	  assert(Max_Dim >= dim);
	*/
	const int chunks = (items + (1024 * factor - 1)) / (1024 * factor);
	const int tid = threadIdx.x;
	const int warp = tid >> 5;
	const int lane = tid & 31;
	const int corr = 1024 % dim;

	__shared__ T globcarry[dim][order];
	__shared__ T tempcarry[dim];
	__shared__ T sbuf[factor][32 * dim];

	for (int i = tid; i < dim * order; i += 1024) {
		globcarry[i / order][i % order] = 0;
	}

	int pos = 0;
	for (int chunk = blockIdx.x; chunk < chunks; chunk += SMs * 2) {
		const int offs = tid + chunk * (1024 * factor);
		const int firstid = offs % dim;
		const int lastid = (offs + 1024 * (factor - 1)) % dim;

		T val[factor];
		if (chunk < chunks - 1) {
			for (int i = 0; i < factor; i++) {
				val[i] = ginput[offs + 1024 * i];
			}
		}
		else {
			for (int i = 0; i < factor; i++) {
				val[i] = 0;
				if (offs + 1024 * i < items) {
					val[i] = ginput[offs + 1024 * i];
				}
			}
		}

		for (int round = 0; round < order; round++) {
			for (int i = 0; i < factor; i++) {
				for (int d = dim; d < 32; d *= 2) {
					T tmp = __shfl_up(val[i], d);
					if (lane >= d) val[i] = op(val[i], tmp);
				}
			}

			if (lane >= (32 - dim)) {
				const int tix = warp * dim;
				int id = firstid;
				for (int i = 0; i < factor; i++) {
					sbuf[i][tix + id] = val[i];
					id += corr;
					if (id >= dim) id -= dim;
				}
			}

			__syncthreads();
			if (warp < dim) {
				const int idx = (lane * dim) + warp;
				for (int i = 0; i < factor; i++) {
					T v = sbuf[i][idx];
					for (int d = 1; d < 32; d *= 2) {
						T tmp = __shfl_up(v, d);
						if (lane >= d) v = op(v, tmp);
					}
					sbuf[i][idx] = v;
				}
			}

			__syncthreads();
			if (warp > 0) {
				const int tix = warp * dim - dim;
				int id = firstid;
				for (int i = 0; i < factor; i++) {
					val[i] = op(val[i], sbuf[i][tix + id]);
					id += corr;
					if (id >= dim) id -= dim;
				}
			}

			T carry[dim];
			for (int d = 0; d < dim; d++) {
				carry[d] = 0;
			}
			int id = firstid;
			for (int i = 1; i < factor; i++) {
				for (int d = 0; d < dim; d++) {
					carry[d] = op(carry[d], sbuf[i - 1][31 * dim + d]);
				}
				id += corr;
				if (id >= dim) id -= dim;
				val[i] = op(val[i], carry[id]);
			}

			int wait = round + 1;
			if (tid > 1023 - dim) {
				gcarry[lastid * (order * MOD) + round * MOD + (chunk & MM1)] = val[factor - 1];
				gwait[round * MOD + ((chunk + (MOD - 4 * SMs)) & MM1)] = 0;
				__threadfence();
				if (tid == 1023) {
					gwait[round * MOD + (chunk & MM1)] = wait;
				}
			}

			const int tidx = pos + tid;
			if (tidx < chunk) {
				wait = gwait[round * MOD + (tidx & MM1)];
			}
			while (__syncthreads_count(wait <= round) != 0) {
				if (wait <= round) {
					wait = gwait[round * MOD + (tidx & MM1)];
				}
			}

			if (warp < dim) {
				int posx = pos + lane;
				T carry = 0;
				if (posx < chunk) {
					carry = gcarry[warp * (order * MOD) + round * MOD + (posx & MM1)];
				}
				if (SMs > 16) {
					posx += 32;
					if (posx < chunk) {
						carry = op(carry, gcarry[warp * (order * MOD) + round * MOD + (posx & MM1)]);
					}
				}
				for (int d = 1; d < 32; d *= 2) {
					carry = op(carry, __shfl_up(carry, d));
				}
				if (lane == 31) {
					T temp = op(globcarry[warp][round], carry);
					tempcarry[warp] = globcarry[warp][round] = temp;
				}
			}

			__syncthreads();
			if (tid > 1023 - dim) {
				globcarry[lastid][round] = op(globcarry[lastid][round], val[factor - 1]);
			}

			id = firstid;
			for (int i = 0; i < factor; i++) {
				val[i] = op(val[i], tempcarry[id]);
				id += corr;
				if (id >= dim) id -= dim;
			}
		} // round

		if (chunk < chunks - 1) {
			for (int i = 0; i < factor; i++) {
				goutput[offs + 1024 * i] = val[i];
			}
		}
		else {
			for (int i = 0; i < factor; i++) {
				if (offs + 1024 * i < items) {
					goutput[offs + 1024 * i] = val[i];
				}
			}
		}
		pos = chunk + 1;
	} // chunk
}

template <typename T, int factor, int dim, int order, T(*op)(T, T)>
static void erScan(const T * const __restrict__ ginput, T * const __restrict__ goutput, const int items)
{
	static int* aux = NULL;
	if (aux == NULL) {
		cudaMalloc(&aux, order * MOD * sizeof(int) + dim * order * MOD * sizeof(T));
	}
	cudaMemsetAsync(aux, 0, order * MOD * sizeof(int));
	ekScan<T, factor, dim, order, op> << <SMs * 2, 1024 >> > (ginput, goutput, items, (T *)&aux[order * MOD], aux);
}

template <typename T, int factor, int dim, int order, T(*op)(T, T)>
static void irScan(const T * const __restrict__ ginput, T * const __restrict__ goutput, const int items)
{
	static int* aux = NULL;
	if (aux == NULL) {
		cudaMalloc(&aux, order * MOD * sizeof(int) + dim * order * MOD * sizeof(T));
	}
	cudaMemsetAsync(aux, 0, order * MOD * sizeof(int));
	ikScan<T, factor, dim, order, op> << <SMs * 2, 1024 >> > (ginput, goutput, items, (T *)&aux[order * MOD], aux);
}


#define erunScan(fac) erScan<T, fac, dim, order, op>(ginput, goutput, items)

#define irunScan(fac) irScan<T, fac, dim, order, op>(ginput, goutput, items)

template <typename T, int dim, int order, T(*op)(T, T)>
static void exclusiveScan(const T * const __restrict__ ginput, T * const __restrict__ goutput, const int items)
{
	if (sizeof(T) <= 4) {
		if ((dim == 1) && (order == 1)) {
			if (items <= 1024) erunScan(1);
			else if (items <= 2048) erunScan(2);
			else if (items <= 16384) erunScan(1);
			else if (items <= 65536) erunScan(2);
			else if (items <= 131072) erunScan(4);
			else if (items <= 262144) erunScan(6);
			else if (items <= 524288) erunScan(12);
			else if (items <= 8388608) erunScan(6);
			else if (items <= 16777216) erunScan(8);
			else if (items <= 33554432) erunScan(7);
			else if (items <= 67108864) erunScan(6);
			else if (items <= 134217728) erunScan(7);
			else erunScan(8);
		}
		if ((dim == 1) && (order > 1)) {
			if (items <= 1024) erunScan(1);
			else if (items <= 2048) erunScan(2);
			else if (items <= 16384) erunScan(1);
			else if (items <= 32768) erunScan(2);
			else if (items <= 131072) erunScan(3);
			else if (items <= 262144) erunScan(6);
			else if (items <= 524288) erunScan(12);
			else if (items <= 1048576) erunScan(8);
			else if (items <= 2097152) erunScan(9);
			else if (items <= 4194304) erunScan(10);
			else erunScan(16);
		}
		if ((dim > 1) && (order == 1)) {
			if (items <= 1024) erunScan(1);
			else if (items <= 32768) erunScan(1);
			else if (items <= 65536) erunScan(3);
			else if (items <= 262144) erunScan(6);
			else if (items <= 2097152) erunScan(3);
			else erunScan(4);
		}
		if ((dim > 1) && (order > 1)) {
			if (items <= 1024) erunScan(1);
			else if (items <= 32768) erunScan(1);
			else if (items <= 65536) erunScan(2);
			else if (items <= 2097152) erunScan(3);
			else if (items <= 8388608) erunScan(4);
			else if (items <= 33554432) erunScan(5);
			else if (items <= 67108864) erunScan(6);
			else if (items <= 134217728) erunScan(5);
			else erunScan(6);
		}
	}
	else {
		if ((dim == 1) && (order == 1)) {
			if (items <= 1024) erunScan(1);
			else if (items <= 16384) erunScan(1);
			else if (items <= 65536) erunScan(2);
			else if (items <= 131072) erunScan(3);
			else if (items <= 262144) erunScan(6);
			else if (items <= 1048576) erunScan(2);
			else erunScan(3);
		}
		if ((dim == 1) && (order > 1)) {
			if (items <= 1024) erunScan(1);
			else if (items <= 16384) erunScan(1);
			else if (items <= 65536) erunScan(2);
			else if (items <= 131072) erunScan(3);
			else if (items <= 1048576) erunScan(6);
			else if (items <= 2097152) erunScan(5);
			else erunScan(6);
		}
		if ((dim > 1) && (order == 1)) {
			if (items <= 1024) erunScan(1);
			else erunScan(1);
		}
		if ((dim > 1) && (order > 1)) {
			if (items <= 1024) erunScan(1);
			else if (items <= 2048) erunScan(2);
			else if (items <= 32768) erunScan(1);
			else if (items <= 65536) erunScan(2);
			else if (items <= 131072) erunScan(3);
			else erunScan(2);
		}
	}
}

template <typename T, int dim, int order, T(*op)(T, T)>
static void inclusiveScan(const T * const __restrict__ ginput, T * const __restrict__ goutput, const int items)
{
	if (sizeof(T) <= 4) {
		if ((dim == 1) && (order == 1)) {
			if (items <= 1024) irunScan(1);
			else if (items <= 2048) irunScan(2);
			else if (items <= 16384) irunScan(1);
			else if (items <= 65536) irunScan(2);
			else if (items <= 131072) irunScan(4);
			else if (items <= 262144) irunScan(6);
			else if (items <= 524288) irunScan(12);
			else if (items <= 8388608) irunScan(6);
			else if (items <= 16777216) irunScan(8);
			else if (items <= 33554432) irunScan(7);
			else if (items <= 67108864) irunScan(6);
			else if (items <= 134217728) irunScan(7);
			else irunScan(8);
		}
		if ((dim == 1) && (order > 1)) {
			if (items <= 1024) irunScan(1);
			else if (items <= 2048) irunScan(2);
			else if (items <= 16384) irunScan(1);
			else if (items <= 32768) irunScan(2);
			else if (items <= 131072) irunScan(3);
			else if (items <= 262144) irunScan(6);
			else if (items <= 524288) irunScan(12);
			else if (items <= 1048576) irunScan(8);
			else if (items <= 2097152) irunScan(9);
			else if (items <= 4194304) irunScan(10);
			else irunScan(16);
		}
		if ((dim > 1) && (order == 1)) {
			if (items <= 1024) irunScan(1);
			else if (items <= 32768) irunScan(1);
			else if (items <= 65536) irunScan(3);
			else if (items <= 262144) irunScan(6);
			else if (items <= 2097152) irunScan(3);
			else irunScan(4);
		}
		if ((dim > 1) && (order > 1)) {
			if (items <= 1024) irunScan(1);
			else if (items <= 32768) irunScan(1);
			else if (items <= 65536) irunScan(2);
			else if (items <= 2097152) irunScan(3);
			else if (items <= 8388608) irunScan(4);
			else if (items <= 33554432) irunScan(5);
			else if (items <= 67108864) irunScan(6);
			else if (items <= 134217728) irunScan(5);
			else irunScan(6);
		}
	}
	else {
		if ((dim == 1) && (order == 1)) {
			if (items <= 1024) irunScan(1);
			else if (items <= 16384) irunScan(1);
			else if (items <= 65536) irunScan(2);
			else if (items <= 131072) irunScan(3);
			else if (items <= 262144) irunScan(6);
			else if (items <= 1048576) irunScan(2);
			else irunScan(3);
		}
		if ((dim == 1) && (order > 1)) {
			if (items <= 1024) irunScan(1);
			else if (items <= 16384) irunScan(1);
			else if (items <= 65536) irunScan(2);
			else if (items <= 131072) irunScan(3);
			else if (items <= 1048576) irunScan(6);
			else if (items <= 2097152) irunScan(5);
			else irunScan(6);
		}
		if ((dim > 1) && (order == 1)) {
			if (items <= 1024) irunScan(1);
			else irunScan(1);
		}
		if ((dim > 1) && (order > 1)) {
			if (items <= 1024) irunScan(1);
			else if (items <= 2048) irunScan(2);
			else if (items <= 32768) irunScan(1);
			else if (items <= 65536) irunScan(2);
			else if (items <= 131072) irunScan(3);
			else irunScan(2);
		}
	}
}

template <typename T, int dim, int order, T(*op)(T, T)>
static void async_inclusive_scan(const T * const __restrict__ ginput, const int items, T * const __restrict__ goutput)
{
	inclusiveScan<T, dim, order, op>(ginput, goutput, items);
}

template <typename T, int dim, int order, T(*op)(T, T)>
static void async_exclusive_scan(const T * const __restrict__ ginput, const int items, T * const __restrict__ goutput)
{
	exclusiveScan<T, dim, order, op>(ginput, goutput, items);
}

#endif
