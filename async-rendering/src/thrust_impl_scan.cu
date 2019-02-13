
#include "thrust_impl.h"

#include "as_impl.h"

#pragma warning( push, 0 ) 
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#pragma warning( pop )

#include "cuda/cuda_cached_allocator.h"

template <typename T>
__host__ __device__ T sum(T a, T b)
{
	return a + b;
}

namespace Mochimazui {

// -------- -------- -------- -------- -------- -------- -------- --------
void asyncExclusiveScan(int8_t *ibegin, uint32_t number, int8_t *obegin) {
	typedef int8_t mytype;
	const int dim = 1;
	const int order = 1;
	async_exclusive_scan<mytype, dim, order, sum<mytype> >(ibegin, number, obegin);
}

void asyncExclusiveScan(uint8_t *ibegin, uint32_t number, uint8_t *obegin) {
	typedef uint8_t mytype;
	const int dim = 1;
	const int order = 1;
	async_exclusive_scan<mytype, dim, order, sum<mytype> >(ibegin, number, obegin);
}

// -------- -------- -------- -------- -------- -------- -------- --------
void asyncExclusiveScan(int32_t *ibegin, uint32_t number, int32_t *obegin) {
	typedef int32_t mytype;
	const int dim = 1;
	const int order = 1;
	async_exclusive_scan<mytype, dim, order, sum<mytype> >(ibegin, number, obegin);
}

void asyncExclusiveScan(uint32_t *ibegin, uint32_t number, uint32_t *obegin) {
	typedef uint32_t mytype;
	const int dim = 1;
	const int order = 1;
	async_exclusive_scan<mytype, dim, order, sum<mytype> >(ibegin, number, obegin);
}

// -------- -------- -------- -------- -------- -------- -------- --------
void asyncExclusiveScan(float *ibegin, uint32_t number, float *obegin) {
	typedef float mytype;
	const int dim = 1;
	const int order = 1;
	async_exclusive_scan<mytype, dim, order, sum<mytype> >(ibegin, number, obegin);
}

// -------- -------- -------- -------- -------- -------- -------- --------
void asyncInclusiveScan(int32_t *ibegin, uint32_t number, int32_t *obegin) {
	typedef int32_t mytype;
	const int dim = 1;
	const int order = 1;
	async_inclusive_scan<mytype, dim, order, sum<mytype> >(ibegin,number,obegin);
}

void asyncInclusiveScan(uint32_t *ibegin, uint32_t number, uint32_t *obegin) {
	typedef uint32_t mytype;
	const int dim = 1;
	const int order = 1;
	async_inclusive_scan<mytype, dim, order, sum<mytype> >(ibegin, number, obegin);
}

}
