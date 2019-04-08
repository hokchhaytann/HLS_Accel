#include "mmult.h"

void standalone_mmult (Dtype A[DIM_M][DIM_K], Dtype B[DIM_K][DIM_N], Dtype2 C[DIM_M][DIM_N], int shamt)
{
	for(int i=0; i<DIM_M; i++){
		for(int j=0; j<DIM_N; j++){
			C[i][j] = 0;
		}
	}
	mmult_hw<Dtype,Dtype2,Dtype_Accum>(A, B, C, shamt);
}

// THIS IS THE TOP LEVEL DESIGN THAT WILL BE SYNTHESIZED
void HLS_accel (
		//weights
		signed char INPUT_STREAM10[DIM_M*DIM_K/N_BYTE],
		// Activations
		signed char INPUT_STREAM20[DIM_K*DIM_N/N_BYTE],
		// output
		signed char OUTPUT_STREAM10[DIM_M*DIM_N/N_BYTE],
		int shamt, int reset, int read)
{
#pragma HLS INTERFACE s_axilite port=return     bundle=CONTROL_BUS
#pragma HLS INTERFACE s_axilite port=shamt      bundle=CONTROL_BUS
#pragma HLS INTERFACE s_axilite port=reset      bundle=CONTROL_BUS
#pragma HLS INTERFACE s_axilite port=read       bundle=CONTROL_BUS

#pragma HLS INTERFACE m_axi depth=72*4  port=INPUT_STREAM10  offset=slave bundle=MASTER_BUS

#pragma HLS INTERFACE m_axi depth=2016*4 port=INPUT_STREAM20 offset=slave bundle=MASTER_BUS

#pragma HLS INTERFACE m_axi depth=448*4 port=OUTPUT_STREAM10 offset=slave bundle=MASTER_BUS

	wrapped_mmult_hw <Dtype,Dtype2,Dtype_Accum>(
			INPUT_STREAM10,
			INPUT_STREAM20,
			OUTPUT_STREAM10,
			shamt,reset,read);

	return;
}


