
#ifndef _MOCHIMAZUI_THRUST_IMPL_H_
#define _MOCHIMAZUI_THRUST_IMPL_H_

#include <cstdint>

namespace Mochimazui {

void asyncExclusiveScan(int8_t *ibegin, uint32_t number, int8_t *obegin);
void asyncExclusiveScan(uint8_t *ibegin, uint32_t number, uint8_t *obegin);

void asyncExclusiveScan(int32_t *ibegin, uint32_t number, int32_t *obegin);
void asyncExclusiveScan(uint32_t *ibegin, uint32_t number, uint32_t *obegin);

void asyncExclusiveScan(float *ibegin, uint32_t number, float *obegin);

void asyncInclusiveScan(int32_t *ibegin, uint32_t number, int32_t *obegin);
void asyncInclusiveScan(uint32_t *ibegin, uint32_t number, uint32_t *obegin);

}

#endif