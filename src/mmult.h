#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <ap_axi_sdata.h>
#include "ap_int.h"


#define N_BYTE 1
#define DIM_M 8
#define DIM_K 36
#define DIM_N 224

//typedef ap_fixed<8,8,AP_RND_CONV,AP_SAT> Dtype;
//typedef ap_fixed<16,16,AP_RND_CONV,AP_SAT> Dtype2;
//typedef ap_fixed<20,20,AP_RND_CONV,AP_SAT> Dtype_Accum;
typedef ap_int<8> Dtype;
typedef ap_int<16> Dtype2;
typedef ap_int<20> Dtype_Accum;

typedef unsigned int u32;

// function prototypes
void standalone_mmult (Dtype A[DIM_M][DIM_K], Dtype B[DIM_K][DIM_N], Dtype2 C[DIM_M][DIM_N], int shamt);
void HLS_accel (//weights
		signed char INPUT_STREAM10[DIM_M*DIM_K/N_BYTE],
				// Activations
		signed char INPUT_STREAM20[DIM_K*DIM_N/N_BYTE],
				// output
		signed char OUTPUT_STREAM10[DIM_M*DIM_N/N_BYTE],

				int shamt, int reset, int read);

// --------------------------------------------------------
// function to be accelerated in HW
template <typename T, typename T2, typename T_Accum>
void mmult_hw(T a[DIM_M][DIM_K], T b[DIM_K][DIM_N], T2 out[DIM_M][DIM_N], int shamt)
{
	const int FACT = DIM_K;
	const int FACT1 = DIM_M;
#pragma HLS array_partition variable=a block factor=FACT dim=2
#pragma HLS array_partition variable=b block factor=FACT dim=1
//#pragma HLS array_partition variable=out block factor=FACT1 dim=1

	// matrix multiplication of a A*B matrix
	LOOP0: for (int ia = 0; ia < DIM_M; ++ia){
		LOOP1: for (int ib = 0; ib < DIM_N; ++ib){
#pragma HLS PIPELINE II=1
			T_Accum sum = 0;
			LOOP2: for (int id = 0; id < DIM_K; ++id){
#pragma HLS unroll
				T2 prod;
				prod = a[ia][id] * b[id][ib];
				sum += prod;
			}


			out[ia][ib] += T2(sum);
		}
	}

	if (shamt != 0) {
		LOOP3: for (int ia = 0; ia < DIM_M; ++ia){
			LOOP4: for (int ib = 0; ib < DIM_N; ++ib){
#pragma HLS PIPELINE II=1
				if (shamt > 0)
					out[ia][ib] = out[ia][ib] >> shamt;
				else
					out[ia][ib] = out[ia][ib] << -shamt;

			}
		}
	}
}

template <typename T, typename T2, typename T_Accum>
void wrapped_mmult_hw (
	signed char in_stream10[DIM_M*DIM_K/N_BYTE],

	signed char in_stream20[DIM_K*DIM_N/N_BYTE],

	signed char out_stream10[DIM_M*DIM_N/N_BYTE],

	int shamt,
	int reset,
	int read)
{
#pragma HLS INLINE

	T a[DIM_M][DIM_K];
	T b[DIM_K][DIM_N];
	T2 out[DIM_M][DIM_N];

	if(reset==1){
		for(int i=0; i<DIM_M; i++){
			for(int j=0; j<DIM_N; j++)
			{
#pragma HLS PIPELINE II=1
				out[i][j] = 0;
			}
		}
	}else if(read==1){
		for(int i=0; i<DIM_M; ++i){
#pragma HLS unroll
			for(int j=0; j<DIM_N; j++) {
#pragma HLS PIPELINE II=1
				out_stream10[j+i*DIM_N] = T(out[i][j]);
			}
		}
	}
	else
	{
		for(int i=0; i<DIM_M; ++i){
#pragma HLS unroll
			for(int j=0; j<DIM_K; j++) {
#pragma HLS PIPELINE II=1
				a[i][j] = in_stream10[i*DIM_K + j];
			}
		}

		for(int i=0; i<DIM_K; ++i){
#pragma HLS unroll
			for(int j=0; j<DIM_N; j++) {
#pragma HLS PIPELINE II=1
				b[i][j] = in_stream20[i*DIM_N + j];
			}
		}
		///////////////////////////////////////////////

		// do the matrix multiplication
		mmult_hw<T,T2,T_Accum>(a,b,out,shamt);

	  }
	 return;
}
