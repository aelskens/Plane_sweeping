#include "main.cuh"

#include <cstdio>

// Those functions are an example on how to call cuda functions from the main.cpp

__global__ void dev_test_vecAdd(int* A, int* B, int* C, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;

	C[i] = A[i] + B[i];
}

void wrap_test_vectorAdd() {
	printf("Vector Add:\n");

	int N = 3;
	int a[] = { 1, 2, 3 };
	int b[] = { 1, 2, 3 };
	int c[] = { 0, 0, 0 };

	int* dev_a, * dev_b, * dev_c;

	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));

	cudaMemcpy(dev_a, a, N * sizeof(int),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int),
		cudaMemcpyHostToDevice);

	dev_test_vecAdd <<<1, N>>> (dev_a, dev_b, dev_c, N);

	cudaMemcpy(c, dev_c, N * sizeof(int),
		cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	
	for (int i = 0; i < N; ++i) {
		printf("%i + %i = %i\n", a[i], b[i], c[i]);
	}
}

//__global__ void compute_cost_naive(float* cost, float* cc, std::vector<cv::Mat> const &ref, std::vector<cv::Mat> const &cam, int* id_x, int* id_y, int N)
//{
//	int k = blockIdx.x * blockDim.x + threadIdx.x;
//	int l = blockIdx.y * blockDim.y + threadIdx.y;
//
//	if (k >= N || l >= N) return;
//
//	
//}
//
//float compute_cost_window(float* c, std::vector<cv::Mat> const &refYUV, std::vector<cv::Mat> const &camYUV, int* x, int* y) //understand what to do with ref YUV and camYUV
//{
//	printf("Naive cost compute:\n");
//
//	int N = 3;
//	float cc = 0.0f;
//
//	float* dev_cost, dev_cc;
//	int* dev_id_x, dev_id_y;
//
//	CHK(cudaSetDevice(0));
//
//	CHK(cudaMalloc((void**)&dev_cost, sizeof(float)));
//	CHK(cudaMalloc((void**)&dev_cc, sizeof(float)));
//  CHK(cudaMalloc((void**)&dev_refYUV, refYUV[0].cols * refYUV[0].rows * sizeof(float)));
//  CHK(cudaMalloc((void**)&dev_camYUV, camYUV[0].cols * camYUV[0].rows * sizeof(float)));
//	CHK(cudaMalloc((void**)&dev_id_x, sizeof(int)));
//	CHK(cudaMalloc((void**)&dev_id_y, sizeof(int)));
//
//	cudaMemcpy(dev_cost, c, sizeof(float),
//		cudaMemcpyHostToDevice);
//	cudaMemcpy(dev_cc, cc, sizeof(float),
//		cudaMemcpyHostToDevice);
//	cudaMemcpy(dev_id_x, x, sizeof(int),
//		cudaMemcpyHostToDevice);
//	cudaMemcpy(dev_id_y, y, sizeof(int),
//		cudaMemcpyHostToDevice);
//}

//float cost = 0.0f;
//float cc = 0.0f;
//for (int k = -window / 2; k <= window / 2; k++)
//{
//	for (int l = -window / 2; l <= window / 2; l++)
//	{
//		if (x + l < 0 || x + l >= ref.width)
//			continue;
//		if (y + k < 0 || y + k >= ref.height)
//			continue;
//		if (x_proj + l < 0 || x_proj + l >= cam.width)
//			continue;
//		if (y_proj + k < 0 || y_proj + k >= cam.height)
//			continue;
//
//		// Y
//		cost += fabs(ref.YUV[0].at<uint8_t>(y + k, x + l) - cam.YUV[0].at<uint8_t>((int)y_proj + k, (int)x_proj + l));
//		// U
//		// cost += fabs(ref.YUV[1].at<uint8_t >(y + k, x + l) - cam.YUV[1].at<uint8_t>((int)y_proj + k, (int)x_proj + l));
//		// V
//		// cost += fabs(ref.YUV[2].at<uint8_t >(y + k, x + l) - cam.YUV[2].at<uint8_t>((int)y_proj + k, (int)x_proj + l));
//		cc += 1.0f;
//	}
//}
//cost /= cc;