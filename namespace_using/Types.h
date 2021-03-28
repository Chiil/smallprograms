#ifndef TYPES_H
#define TYPES_H

#ifdef SINGLE_PRECISION
using Real = float;
#else
using Real = double;
#endif

using Real_ptr = Real* const __restrict__;

#endif
