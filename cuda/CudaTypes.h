#ifndef __LIB_TREVISAN_CUDA_TYPES_H__
#define __LIB_TREVISAN_CUDA_TYPES_H__

#include <ctype.h>

#define SIZE_CHUNK 32
#define SIZE_BYTE 8

#ifdef LINUXINTEL32
#undef WINDOWS
#warning "LINUXINTEL32 selected!!"
typedef long               sfixn;
typedef long long          longfixnum;
typedef unsigned long      usfixn;
typedef unsigned long long ulongfixnum;
typedef long               int32;
typedef unsigned long      uint32;
#endif

#ifdef LINUXINTEL64
#undef WINDOWS
#warning "LINUXINTEL64 selected!!"
typedef int            sfixn;
typedef long           longfixnum;
typedef unsigned int   usfixn;
typedef unsigned long  ulongfixnum;
typedef int            int32;
typedef unsigned int   uint32;
#endif

#ifdef MAC32
#undef WINDOWS
#ifndef __CUDATYPES_H__
#define __CUDATYPES_H__

#include <stdint.h>
#warning "MAC32 selected!!"
typedef int32_t   sfixn;
typedef int64_t   longfixnum;
typedef uint32_t  usfixn;
typedef uint64_t  ulongfixnum;
typedef int32_t   int32;
typedef uint32_t  uint32;
#endif

#ifdef SOLARIS64
#undef WINDOWS
#warning "SOLARIS64 selected!!"
typedef int32_t   sfixn;
typedef int64_t   longfixnum;
typedef uint32_t  usfixn;
typedef uint64_t  ulongfixnum;
typedef int32_t   int32;
typedef uint32_t  uint32;
#endif

#ifdef PPC64
#undef WINDOWS
#warning "PPC64 selected!!"
typedef int32_t   sfixn;
typedef int64_t   longfixnum;
typedef uint32_t  usfixn;
typedef uint64_t  ulongfixnum;
typedef int32_t   int32;
typedef uint32_t  uint32;
#endif

#ifdef MAC64
#undef WINDOWS
#warning "MAC64 selected!!"
typedef int32_t   sfixn;
typedef int64_t   longfixnum;
typedef uint32_t  usfixn;
typedef uint64_t  ulongfixnum;
typedef int32_t   int32;
typedef uint32_t  uint32;
#endif

#ifdef WINDOWS
#warning "WINDOWS selected!!"
typedef _int32           sfixn;
typedef _int64           longfixnum;
typedef unsigned _int32  usfixn;
typedef unsigned _int64  ulongfixnum;
typedef _int32           int32;
typedef unsigned _int32  uint32;
#endif

#endif /* END OF INTEGER TYPES */

#endif