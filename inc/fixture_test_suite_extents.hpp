#ifndef FIXTURE_TEST_SUITE_EXTENTS_HPP_
#define FIXTURE_TEST_SUITE_EXTENTS_HPP_

// there are optimizations for the extents like  2^a × 3^b × 5^c × 7^d 
//  so examples contain (arbitrary), (2^a), (3^b|5^c|7^d)
//
#define EXTENTS_3D \
  ((63,67,65)) ((126,127,129)) ((126,255,129)) ((255,254,259)) ((511,513,509)) \
  ((64,64,64)) ((128,128,128)) ((128,256,128)) ((256,256,256)) ((512,512,512)) \
  ((64,81,51)) ((125,125,125)) ((125,243,128)) ((243,243,343)) ((625,729,343))

#define EXTENTS_2D \
  ((511,513)) ((1011,1013)) ((2047,2049)) ((4095,4097))       \
  ((512, 512)) ((1024,1024)) ((2048,2048)) ((4096,4096))      \
  ((625,343)) ((729,2187)) ((2187,2401)) ((3125,6561))

#define EXTENTS_1D \
  ((265841)) ((1048577)) ((2097153)) ((4194305)) ((8388607)) ((16777213))     \
  ((262144)) ((1048576)) ((2097152)) ((4194304)) ((8388608)) ((16777216))     \
  ((390625)) ((823543))  ((1953125)) ((5764801)) ((9765625)) ((14348907))
// dimension test cases for benchmark 
// Macro as Boost Tuple type
#ifndef EXTENTS
//#ifdef EXTENTS_SMALL // @todo cmake based editions of test size factors
#define EXTENTS EXTENTS_3D EXTENTS_2D EXTENTS_1D
//#define EXTENTS EXTENTS_1D
#endif

#endif
