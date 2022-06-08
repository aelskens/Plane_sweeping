#include "main.cuh"

#include <cstdio>
#include <iostream>

#define N_THREADS 32
#define MI(x, y, width) ((x) + (y) * (width))
#define MI3(x, y, z, width, height) ((x) + ((y) + (z) * (height)) * (width))

#define CHK(code) \
do { \
    if ((code) != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s %s %i\n", \
                        cudaGetErrorString((code)), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

__constant__ int const_width[1];
__constant__ int const_height[1];
__constant__ float const_zi[1];
__constant__ float const_znear[1];
__constant__ float const_zfar[1];
__constant__ float const_ZPlanes[1];
__constant__ int const_half_window[1];
__constant__ float const_K[9*3];
__constant__ float const_R[9*3];
__constant__ float const_t[3*3];
__constant__ float const_inv_K[9];
__constant__ float const_inv_R[9];
__constant__ float const_inv_t[3];
__constant__ int const_cam_count[1];

// Those functions are an example on how to call cuda functions from the main.cpp

__global__ void dev_test_vecAdd(int* A, int* B, int* C, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;

	C[i] = A[i] + B[i];
}

void wrap_test_vectorAdd() {
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

	dev_test_vecAdd << <1, N >> > (dev_a, dev_b, dev_c, N);

	cudaMemcpy(c, dev_c, N * sizeof(int),
		cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
}


// width, height, zi, Znear, ZFar, ZPlanes, K, R, t, inv_K, inv_R, inv_t, window, Y_cam, Y_ref, cost_cube
//__global__ void compute_cost_naive(int* width, int* height, int* zi, float* znear, float* zfar, float* ZPlanes, int* half_window,
//	std::vector<double> const& K, std::vector<double> const& R, std::vector<double> const& t, std::vector<double> const& inv_K, std::vector<double> const& inv_R, std::vector<double> const& inv_t,
//	cv::Mat const& ref, cv::Mat const& cam_1, std::vector<float*> &const cost_cube)
//{
//	int k = blockIdx.x * blockDim.x + threadIdx.x;
//	int l = blockIdx.y * blockDim.y + threadIdx.y;
//	
//}

__global__ void compute_cost_naive_baseline(int* width, int* height, int* zi, float* znear, float* zfar, int* ZPlanes, int* half_window, double* K, double* R, double* t,
	double* inv_K, double* inv_R, double* inv_t, float* cost_cube, uint8_t* y_ref, uint8_t* y_cam)
{
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	//int l = blockIdx.y * blockDim.y + threadIdx.y;

	int x = p % *width;
	int y = p / *width;

	// Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
	double z = *znear * *zfar / (*znear + (((double)*zi / (double)*ZPlanes) * (*zfar - *znear))); //need to be in the x and y loops?

	// 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
	double X_ref = (inv_K[0] * x + inv_K[1] * y + inv_K[2]) * z;
	double Y_ref = (inv_K[3] * x + inv_K[4] * y + inv_K[5]) * z;
	double Z_ref = (inv_K[6] * x + inv_K[7] * y + inv_K[8]) * z;

	// 3D in ref camera coordinates to 3D world
	double X = inv_R[0] * X_ref + inv_R[1] * Y_ref + inv_R[2] * Z_ref - inv_t[0];
	double Y = inv_R[3] * X_ref + inv_R[4] * Y_ref + inv_R[5] * Z_ref - inv_t[1];
	double Z = inv_R[6] * X_ref + inv_R[7] * Y_ref + inv_R[8] * Z_ref - inv_t[2];

	// 3D world to projected camera 3D coordinates
	double X_proj = R[0] * X + R[1] * Y + R[2] * Z - t[0];
	double Y_proj = R[3] * X + R[4] * Y + R[5] * Z - t[1];
	double Z_proj = R[6] * X + R[7] * Y + R[8] * Z - t[2];

	// Projected camera 3D coordinates to projected camera 2D coordinates
	double x_proj = (K[0] * X_proj / Z_proj + K[1] * Y_proj / Z_proj + K[2]);
	double y_proj = (K[3] * X_proj / Z_proj + K[4] * Y_proj / Z_proj + K[5]);

	x_proj = x_proj < 0 || x_proj >= *width ? 0 : roundf(x_proj);
	y_proj = y_proj < 0 || y_proj >= *height ? 0 : roundf(y_proj);

	// (ii) calculate cost against reference
	// Calculating cost in a window
	float cost = 0.0f;
	float cc = 0.0f;
	for (int k = -(*half_window); k <= *half_window; k++)
	{
		for (int l = -(*half_window); l <= *half_window; l++)
		{
			if (x + l < 0 || x + l >= *width)
				continue;
			if (y + k < 0 || y + k >= *height)
				continue;
			if (x_proj + l < 0 || x_proj + l >= *width)
				continue;
			if (y_proj + k < 0 || y_proj + k >= *height)
				continue;

			// Y
			cost += fabsf(y_ref[(y + k) * (*width) + x + l] - y_cam[(int) ((y_proj + k) * (*width) + x_proj + l)]);
			// U
			// cost += fabs(ref.YUV[1].at<uint8_t >(y + k, x + l) - cam.YUV[1].at<uint8_t>((int)y_proj + k, (int)x_proj + l));
			// V
			// cost += fabs(ref.YUV[2].at<uint8_t >(y + k, x + l) - cam.YUV[2].at<uint8_t>((int)y_proj + k, (int)x_proj + l));
			cc += 1.0f;
		}
	}
	cost /= cc;

	//  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
	// only the minimum cost for all the cameras is stored
	cost_cube[y * *width + x] = fminf(cost_cube[y * *width + x], cost);
}

__global__ void compute_cost_naive_float(int* width, int* height, float* zi, float* znear, float* zfar, float* ZPlanes, int* half_window, float* K, float* R, float* t,
	float* inv_K, float* inv_R, float* inv_t, float* cost_cube, uint8_t* y_ref, uint8_t* y_cam)
{
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	//int l = blockIdx.y * blockDim.y + threadIdx.y;

	int x = p % *width;
	int y = p / *width;

	// Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
	float z = *znear * *zfar / (*znear + ((*zi / *ZPlanes) * (*zfar - *znear))); //need to be in the x and y loops?

	// 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
	float X_ref = (inv_K[0] * x + inv_K[1] * y + inv_K[2]) * z;
	float Y_ref = (inv_K[3] * x + inv_K[4] * y + inv_K[5]) * z;
	float Z_ref = (inv_K[6] * x + inv_K[7] * y + inv_K[8]) * z;

	// 3D in ref camera coordinates to 3D world
	float X = inv_R[0] * X_ref + inv_R[1] * Y_ref + inv_R[2] * Z_ref - inv_t[0];
	float Y = inv_R[3] * X_ref + inv_R[4] * Y_ref + inv_R[5] * Z_ref - inv_t[1];
	float Z = inv_R[6] * X_ref + inv_R[7] * Y_ref + inv_R[8] * Z_ref - inv_t[2];

	// 3D world to projected camera 3D coordinates
	float X_proj = R[0] * X + R[1] * Y + R[2] * Z - t[0];
	float Y_proj = R[3] * X + R[4] * Y + R[5] * Z - t[1];
	float Z_proj = R[6] * X + R[7] * Y + R[8] * Z - t[2];

	// Projected camera 3D coordinates to projected camera 2D coordinates
	float x_proj = (K[0] * X_proj / Z_proj + K[1] * Y_proj / Z_proj + K[2]);
	float y_proj = (K[3] * X_proj / Z_proj + K[4] * Y_proj / Z_proj + K[5]);

	int x_proj2 = x_proj < 0 || x_proj >= *width ? 0 : (int) roundf(x_proj);
	int y_proj2 = y_proj < 0 || y_proj >= *height ? 0 : (int) roundf(y_proj);

	// (ii) calculate cost against reference
	// Calculating cost in a window
	float cost = 0.0f;
	float cc = 0.0f;
	for (int k = -(*half_window); k <= *half_window; k++)
	{
		for (int l = -(*half_window); l <= *half_window; l++)
		{
			if (x + l < 0 || x + l >= *width)
				continue;
			if (y + k < 0 || y + k >= *height)
				continue;
			if (x_proj2 + l < 0 || x_proj2 + l >= *width)
				continue;
			if (y_proj2 + k < 0 || y_proj2 + k >= *height)
				continue;

			// Y
			cost += fabsf(y_ref[(y + k) * (*width) + x + l] - y_cam[((y_proj2 + k) * (*width) + x_proj2 + l)]);
			// U
			// cost += fabs(ref.YUV[1].at<uint8_t >(y + k, x + l) - cam.YUV[1].at<uint8_t>((int)y_proj + k, (int)x_proj + l));
			// V
			// cost += fabs(ref.YUV[2].at<uint8_t >(y + k, x + l) - cam.YUV[2].at<uint8_t>((int)y_proj + k, (int)x_proj + l));
			cc += 1.0f;
		}
	}
	cost /= cc;

	//  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
	// only the minimum cost for all the cameras is stored
	cost_cube[y * *width + x] = fminf(cost_cube[y * *width + x], cost);
}

__global__ void compute_cost_naive_float_2D(int* width, int* height, float* zi, float* znear, float* zfar, float* ZPlanes, int* half_window, float* K, float* R, float* t,
	float* inv_K, float* inv_R, float* inv_t, float* cost_cube, uint8_t* y_ref, uint8_t* y_cam)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= *width || y >= *height)
		return;

	// Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
	float z = *znear * *zfar / (*znear + ((*zi / *ZPlanes) * (*zfar - *znear))); //need to be in the x and y loops?

	// 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
	float X_ref = (inv_K[0] * x + inv_K[1] * y + inv_K[2]) * z;
	float Y_ref = (inv_K[3] * x + inv_K[4] * y + inv_K[5]) * z;
	float Z_ref = (inv_K[6] * x + inv_K[7] * y + inv_K[8]) * z;

	// 3D in ref camera coordinates to 3D world
	float X = inv_R[0] * X_ref + inv_R[1] * Y_ref + inv_R[2] * Z_ref - inv_t[0];
	float Y = inv_R[3] * X_ref + inv_R[4] * Y_ref + inv_R[5] * Z_ref - inv_t[1];
	float Z = inv_R[6] * X_ref + inv_R[7] * Y_ref + inv_R[8] * Z_ref - inv_t[2];

	// 3D world to projected camera 3D coordinates
	float X_proj = R[0] * X + R[1] * Y + R[2] * Z - t[0];
	float Y_proj = R[3] * X + R[4] * Y + R[5] * Z - t[1];
	float Z_proj = R[6] * X + R[7] * Y + R[8] * Z - t[2];

	// Projected camera 3D coordinates to projected camera 2D coordinates
	float x_proj = (K[0] * X_proj / Z_proj + K[1] * Y_proj / Z_proj + K[2]);
	float y_proj = (K[3] * X_proj / Z_proj + K[4] * Y_proj / Z_proj + K[5]);

	int x_proj2 = x_proj < 0 || x_proj >= *width ? 0 : (int)roundf(x_proj);
	int y_proj2 = y_proj < 0 || y_proj >= *height ? 0 : (int)roundf(y_proj);

	// (ii) calculate cost against reference
	// Calculating cost in a window
	float cost = 0.0f;
	float cc = 0.0f;
	for (int k = -(*half_window); k <= *half_window; k++)
	{
		for (int l = -(*half_window); l <= *half_window; l++)
		{
			if (x + l < 0 || x + l >= *width)
				continue;
			if (y + k < 0 || y + k >= *height)
				continue;
			if (x_proj2 + l < 0 || x_proj2 + l >= *width)
				continue;
			if (y_proj2 + k < 0 || y_proj2 + k >= *height)
				continue;

			// Y
			cost += fabsf(y_ref[(y + k) * (*width) + x + l] - y_cam[((y_proj2 + k) * (*width) + x_proj2 + l)]);
			// U
			// cost += fabs(ref.YUV[1].at<uint8_t >(y + k, x + l) - cam.YUV[1].at<uint8_t>((int)y_proj + k, (int)x_proj + l));
			// V
			// cost += fabs(ref.YUV[2].at<uint8_t >(y + k, x + l) - cam.YUV[2].at<uint8_t>((int)y_proj + k, (int)x_proj + l));
			cc += 1.0f;
		}
	}
	cost /= cc;

	//  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
	// only the minimum cost for all the cameras is stored
	cost_cube[y * *width + x] = fminf(cost_cube[y * *width + x], cost);
}

__global__ void compute_cost_partially_shared_float_2D(int* global_width, int* global_height, float* global_zi, float* global_znear, float* global_zfar,
	float* global_ZPlanes, int* global_half_window, float* global_K, float* global_R, float* global_t, float* global_inv_K, float* global_inv_R,
	float* global_inv_t, float* cost_cube, uint8_t* y_ref, uint8_t* y_cam)
{
	//if (threadIdx.x == 0 && threadIdx.y == 0) printf("I am initial block %d, %d\n", blockIdx.x, blockIdx.y);
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ int width;
	__shared__ int height;
	__shared__ float zi;
	__shared__ float znear;
	__shared__ float zfar;
	__shared__ float ZPlanes;
	__shared__ int half_window;
	__shared__ float K[9];
	__shared__ float R[9];
	__shared__ float t[3];
	__shared__ float inv_K[9];
	__shared__ float inv_R[9];
	__shared__ float inv_t[3];

	//Fill shared memory
	if (threadIdx.x == 0) {
		if (threadIdx.y == 0) width = *global_width;
		if (threadIdx.y == 1) height = *global_height;
		if (threadIdx.y == 2) zi = *global_zi;
		if (threadIdx.y == 3) znear = *global_znear;
		if (threadIdx.y == 4) zfar = *global_zfar;
		if (threadIdx.y == 5) ZPlanes = *global_ZPlanes;
		if (threadIdx.y == 6) half_window = *global_half_window;
	}
	else if (threadIdx.x == 1 && threadIdx.y < 9) K[threadIdx.y] = global_K[threadIdx.y];
	else if (threadIdx.x == 2 && threadIdx.y < 9) R[threadIdx.y] = global_R[threadIdx.y];
	else if (threadIdx.x == 3 && threadIdx.y < 3) t[threadIdx.y] = global_t[threadIdx.y];
	else if (threadIdx.x == 4 && threadIdx.y < 9) inv_K[threadIdx.y] = global_inv_K[threadIdx.y];
	else if (threadIdx.x == 5 && threadIdx.y < 9) inv_R[threadIdx.y] = global_inv_R[threadIdx.y];
	else if (threadIdx.x == 6 && threadIdx.y < 3) inv_t[threadIdx.y] = global_inv_t[threadIdx.y];

	__syncthreads();

	// Compute padding coordinates
	int padding_length = N_THREADS + 2 * half_window;
	int padding_x = half_window + threadIdx.x;
	int padding_y = half_window + threadIdx.y;

	if (x >= width || y >= height)
		return;

	// Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
	float z = znear * zfar / (znear + ((zi / ZPlanes) * (zfar - znear))); //need to be in the x and y loops?

	// 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
	float X_ref = (inv_K[0] * x + inv_K[1] * y + inv_K[2]) * z;
	float Y_ref = (inv_K[3] * x + inv_K[4] * y + inv_K[5]) * z;
	float Z_ref = (inv_K[6] * x + inv_K[7] * y + inv_K[8]) * z;

	// 3D in ref camera coordinates to 3D world
	float X = inv_R[0] * X_ref + inv_R[1] * Y_ref + inv_R[2] * Z_ref - inv_t[0];
	float Y = inv_R[3] * X_ref + inv_R[4] * Y_ref + inv_R[5] * Z_ref - inv_t[1];
	float Z = inv_R[6] * X_ref + inv_R[7] * Y_ref + inv_R[8] * Z_ref - inv_t[2];

	// 3D world to projected camera 3D coordinates
	float X_proj = R[0] * X + R[1] * Y + R[2] * Z - t[0];
	float Y_proj = R[3] * X + R[4] * Y + R[5] * Z - t[1];
	float Z_proj = R[6] * X + R[7] * Y + R[8] * Z - t[2];

	// Projected camera 3D coordinates to projected camera 2D coordinates
	float x_proj = (K[0] * X_proj / Z_proj + K[1] * Y_proj / Z_proj + K[2]);
	float y_proj = (K[3] * X_proj / Z_proj + K[4] * Y_proj / Z_proj + K[5]);

	int x_proj2 = x_proj < 0 || x_proj >= width ? 0 : (int)roundf(x_proj);
	int y_proj2 = y_proj < 0 || y_proj >= height ? 0 : (int)roundf(y_proj);

	extern __shared__ uint8_t sub_y_ref[];

	sub_y_ref[MI(padding_x, padding_y, padding_length)] = y_ref[MI(x, y, width)];

	// padding the left side of the image subsamples given to the shared memory unless at image boundaries where nothing is done
	if (threadIdx.x < half_window && x >= N_THREADS) {
		sub_y_ref[MI(threadIdx.x, padding_y, padding_length)] = y_ref[MI(x - half_window, y, width)];
	}
	// padding the right side of the image subsamples given to the shared memory unless at image boundaries where nothing is done
	else if (threadIdx.x >= N_THREADS - half_window && x < width - N_THREADS) {
		sub_y_ref[MI(padding_x + half_window, padding_y, padding_length)] = y_ref[MI(x + half_window, y, width)];
	}

	// padding the upper side of the image subsamples given to the shared memory unless at image boundaries where nothing is done
	if (threadIdx.y < half_window && y >= N_THREADS) {
		sub_y_ref[MI(padding_x, threadIdx.y, padding_length)] = y_ref[MI(x, y - half_window, width)];
	}
	// padding the lower side of the image subsamples given to the shared memory unless at image boundaries where nothing is done
	else if (threadIdx.y >= N_THREADS - half_window && y < width - N_THREADS) {
		sub_y_ref[MI(padding_x, padding_y + half_window, padding_length)] = y_ref[MI(x, y + half_window, width)];
	}

	// Inside of middle square of size window * window
	if (threadIdx.x >= (N_THREADS / 2) - half_window && threadIdx.x < (N_THREADS / 2) + half_window &&
		threadIdx.y >= (N_THREADS / 2) - half_window && threadIdx.y < (N_THREADS / 2) + half_window) {
		// padding both upper corners of the image subsamples given to the shared memory unless at image boundaries where nothing is done
		if (threadIdx.y < N_THREADS / 2 && y >= N_THREADS) {
			// padding the upper left corner of the image subsamples given to the shared memory unless at image boundaries where nothing is done
			if (threadIdx.x < N_THREADS / 2 && x >= N_THREADS) {
				sub_y_ref[MI(padding_x - N_THREADS / 2, padding_y - N_THREADS / 2, padding_length)] = y_ref[MI(x - N_THREADS / 2, y - N_THREADS / 2, width)];
			}
			// padding the upper right corner of the image subsamples given to the shared memory unless at image boundaries where nothing is done
			else if (threadIdx.x >= N_THREADS / 2 && x < width - N_THREADS) {
				sub_y_ref[MI(padding_x + N_THREADS / 2, padding_y - N_THREADS / 2, padding_length)] = y_ref[MI(x + N_THREADS / 2, y - N_THREADS / 2, width)];
			}
		}
		// padding both lower corners of the image subsamples given to the shared memory unless at image boundaries where nothing is done
		else if (threadIdx.y >= N_THREADS / 2 && y < height - N_THREADS) {
			// padding the lower left corner of the image subsamples given to the shared memory unless at image boundaries where nothing is done
			if (threadIdx.x < N_THREADS / 2 && x >= N_THREADS) {
				sub_y_ref[MI(padding_x - N_THREADS / 2, padding_y + N_THREADS / 2, padding_length)] = y_ref[MI(x - N_THREADS / 2, y + N_THREADS / 2, width)];
			}
			// padding the lower right corner of the image subsamples given to the shared memory unless at image boundaries where nothing is done
			else if (threadIdx.x >= N_THREADS / 2 && x < width - N_THREADS) {
				sub_y_ref[MI(padding_x + N_THREADS / 2, padding_y + N_THREADS / 2, padding_length)] = y_ref[MI(x + N_THREADS / 2, y + N_THREADS / 2, width)];
			}
		}
	}

	__syncthreads();

	// (ii) calculate cost against reference
	// Calculating cost in a window
	float cost = 0.0f;
	float cc = 0.0f;
	for (int k = -(half_window); k <= half_window; k++)
	{
		for (int l = -(half_window); l <= half_window; l++)
		{
			if (x + l < 0 || x + l >= width)
				continue;
			if (y + k < 0 || y + k >= height)
				continue;
			if (x_proj2 + l < 0 || x_proj2 + l >= width)
				continue;
			if (y_proj2 + k < 0 || y_proj2 + k >= height)
				continue;

			// Y
			cost += fabsf((float)sub_y_ref[MI(padding_x + l, padding_y + k, padding_length)] - (float)y_cam[MI(x_proj2 + l, y_proj2 + k, width)]);

			// U
			// cost += fabs(ref.YUV[1].at<uint8_t >(y + k, x + l) - cam.YUV[1].at<uint8_t>((int)y_proj + k, (int)x_proj + l));
			// V
			// cost += fabs(ref.YUV[2].at<uint8_t >(y + k, x + l) - cam.YUV[2].at<uint8_t>((int)y_proj + k, (int)x_proj + l));
			cc += 1.0f;
		}
	}
	cost /= cc;


	//  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
	// only the minimum cost for all the cameras is stored
	cost_cube[MI(x, y, width)] = fminf(cost_cube[MI(x, y, width)], cost);
}

__global__ void compute_cost_shared_float_2D(int* global_width, int* global_height, float* global_zi, float* global_znear, float* global_zfar, 
	float* global_ZPlanes, int* global_half_window, float* global_K, float* global_R, float* global_t, float* global_inv_K, float* global_inv_R, 
	float* global_inv_t, float* cost_cube, uint8_t* y_ref, uint8_t* y_cam)
{
	//if (threadIdx.x == 0 && threadIdx.y == 0) printf("I am initial block %d, %d\n", blockIdx.x, blockIdx.y);
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ int width;
	__shared__ int height;
	__shared__ float zi;
	__shared__ float znear;
	__shared__ float zfar;
	__shared__ float ZPlanes;
	__shared__ int half_window;
	__shared__ float K[9];
	__shared__ float R[9];
	__shared__ float t[3];
	__shared__ float inv_K[9];
	__shared__ float inv_R[9];
	__shared__ float inv_t[3];
	__shared__ int cam_x_proj[4];
	__shared__ int cam_y_proj[4];
	__shared__ uint8_t* sub_y_cam;
	__shared__ int shared_width;
	__shared__ int shared_height;

	//Fill shared memory
	if (threadIdx.x == 0) {
		if (threadIdx.y == 0) width = *global_width;
		if (threadIdx.y == 1) height = *global_height;
		if (threadIdx.y == 2) zi = *global_zi;
		if (threadIdx.y == 3) znear = *global_znear;
		if (threadIdx.y == 4) zfar = *global_zfar;
		if (threadIdx.y == 5) ZPlanes = *global_ZPlanes;
		if (threadIdx.y == 6) half_window = *global_half_window;
	}
	else if (threadIdx.x == 1 && threadIdx.y < 9) K[threadIdx.y] = global_K[threadIdx.y];
	else if (threadIdx.x == 2 && threadIdx.y < 9) R[threadIdx.y] = global_R[threadIdx.y];
	else if (threadIdx.x == 3 && threadIdx.y < 3) t[threadIdx.y] = global_t[threadIdx.y];
	else if (threadIdx.x == 4 && threadIdx.y < 9) inv_K[threadIdx.y] = global_inv_K[threadIdx.y];
	else if (threadIdx.x == 5 && threadIdx.y < 9) inv_R[threadIdx.y] = global_inv_R[threadIdx.y];
	else if (threadIdx.x == 6 && threadIdx.y < 3) inv_t[threadIdx.y] = global_inv_t[threadIdx.y];
	
	__syncthreads();

	if (x >= width || y >= height)
		return;

	// Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
	float z = znear * zfar / (znear + ((zi / ZPlanes) * (zfar - znear))); //need to be in the x and y loops?

	// 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
	float X_ref = (inv_K[0] * x + inv_K[1] * y + inv_K[2]) * z;
	float Y_ref = (inv_K[3] * x + inv_K[4] * y + inv_K[5]) * z;
	float Z_ref = (inv_K[6] * x + inv_K[7] * y + inv_K[8]) * z;

	// 3D in ref camera coordinates to 3D world
	float X = inv_R[0] * X_ref + inv_R[1] * Y_ref + inv_R[2] * Z_ref - inv_t[0];
	float Y = inv_R[3] * X_ref + inv_R[4] * Y_ref + inv_R[5] * Z_ref - inv_t[1];
	float Z = inv_R[6] * X_ref + inv_R[7] * Y_ref + inv_R[8] * Z_ref - inv_t[2];

	// 3D world to projected camera 3D coordinates
	float X_proj = R[0] * X + R[1] * Y + R[2] * Z - t[0];
	float Y_proj = R[3] * X + R[4] * Y + R[5] * Z - t[1];
	float Z_proj = R[6] * X + R[7] * Y + R[8] * Z - t[2];

	// Projected camera 3D coordinates to projected camera 2D coordinates
	float x_proj = (K[0] * X_proj / Z_proj + K[1] * Y_proj / Z_proj + K[2]);
	float y_proj = (K[3] * X_proj / Z_proj + K[4] * Y_proj / Z_proj + K[5]);
	
	int x_proj2 = x_proj < 0 || x_proj >= width ? 0 : (int)roundf(x_proj);
	int y_proj2 = y_proj < 0 || y_proj >= height ? 0 : (int)roundf(y_proj);

	// Compute projection corners
	if (threadIdx.x == 0) {
		if (threadIdx.y == 0) {
			cam_x_proj[0] = x_proj2;
			cam_y_proj[0] = y_proj2;
		}
		else if (threadIdx.y == N_THREADS - 1 || y == height - 1) {
			cam_x_proj[1] = x_proj2;
			cam_y_proj[1] = y_proj2;
		}
	}
	else if (threadIdx.x == N_THREADS - 1 || x == width - 1) {
		if (threadIdx.y == 0) {
			cam_x_proj[2] = x_proj2;
			cam_y_proj[2] = y_proj2;
		}
		else if (threadIdx.y == N_THREADS - 1 || y == height - 1) {
			cam_x_proj[3] = x_proj2;
			cam_y_proj[3] = y_proj2;
			shared_width = threadIdx.x;
			shared_height = threadIdx.y;
		}
	}

	// Compute padding coordinates
	int padding_length = N_THREADS + 2 * half_window;
	int padding_x = half_window + threadIdx.x;
	int padding_y = half_window + threadIdx.y;

	extern __shared__ uint8_t sub_y_ref[];
	
	sub_y_ref[MI(padding_x, padding_y, padding_length)] = y_ref[MI(x, y, width)];
	
	// padding the left side of the image subsamples given to the shared memory unless at image boundaries where nothing is done
	if (threadIdx.x < half_window && x >= N_THREADS) {
		sub_y_ref[MI(threadIdx.x, padding_y, padding_length)] = y_ref[MI(x - half_window, y, width)];
	}
	// padding the right side of the image subsamples given to the shared memory unless at image boundaries where nothing is done
	else if (threadIdx.x >= N_THREADS - half_window && x < width - N_THREADS) {
		sub_y_ref[MI(padding_x + half_window, padding_y, padding_length)] = y_ref[MI(x + half_window, y, width)];
	}

	// padding the upper side of the image subsamples given to the shared memory unless at image boundaries where nothing is done
	if (threadIdx.y < half_window && y >= N_THREADS) {
		sub_y_ref[MI(padding_x, threadIdx.y, padding_length)] = y_ref[MI(x, y - half_window, width)];
	}
	// padding the lower side of the image subsamples given to the shared memory unless at image boundaries where nothing is done
	else if (threadIdx.y >= N_THREADS - half_window && y < width - N_THREADS) {
		sub_y_ref[MI(padding_x, padding_y + half_window, padding_length)] = y_ref[MI(x, y + half_window, width)];
	}
	
	// Inside of middle square of size window * window
	if (threadIdx.x >= (N_THREADS / 2) - half_window && threadIdx.x < (N_THREADS / 2) + half_window && 
		threadIdx.y >= (N_THREADS / 2) - half_window && threadIdx.y < (N_THREADS / 2) + half_window) {
		// padding both upper corners of the image subsamples given to the shared memory unless at image boundaries where nothing is done
		if (threadIdx.y < N_THREADS / 2 && y >= N_THREADS) {
			// padding the upper left corner of the image subsamples given to the shared memory unless at image boundaries where nothing is done
			if (threadIdx.x < N_THREADS / 2 && x >= N_THREADS) {
				sub_y_ref[MI(padding_x - N_THREADS / 2, padding_y - N_THREADS / 2, padding_length)] = y_ref[MI(x - N_THREADS / 2, y - N_THREADS / 2, width)];
			}
			// padding the upper right corner of the image subsamples given to the shared memory unless at image boundaries where nothing is done
			else if (threadIdx.x >= N_THREADS / 2 && x < width - N_THREADS) {
				sub_y_ref[MI(padding_x + N_THREADS / 2, padding_y - N_THREADS / 2, padding_length)] = y_ref[MI(x + N_THREADS / 2, y - N_THREADS / 2, width)];
			}
		}
		// padding both lower corners of the image subsamples given to the shared memory unless at image boundaries where nothing is done
		else if (threadIdx.y >= N_THREADS / 2 && y < height - N_THREADS) {
			// padding the lower left corner of the image subsamples given to the shared memory unless at image boundaries where nothing is done
			if (threadIdx.x < N_THREADS / 2 && x >= N_THREADS) {
				sub_y_ref[MI(padding_x - N_THREADS / 2, padding_y + N_THREADS / 2, padding_length)] = y_ref[MI(x - N_THREADS / 2, y + N_THREADS / 2, width)];
			}
			// padding the lower right corner of the image subsamples given to the shared memory unless at image boundaries where nothing is done
			else if (threadIdx.x >= N_THREADS / 2 && x < width - N_THREADS) {
				sub_y_ref[MI(padding_x + N_THREADS / 2, padding_y + N_THREADS / 2, padding_length)] = y_ref[MI(x + N_THREADS / 2, y + N_THREADS / 2, width)];
			}
		}
	}
	
	__syncthreads();

	// Compute projected padding parameters
	int min_cam_x = umin(cam_x_proj[0], cam_x_proj[2]) - half_window;
	int min_cam_y = umin(cam_y_proj[0], cam_y_proj[1]) - half_window;
	int sub_y_cam_width = umax(cam_x_proj[1], cam_x_proj[3]) + half_window - min_cam_x + 1;
	int sub_y_cam_height = umax(cam_y_proj[2], cam_y_proj[3]) + half_window - min_cam_y + 1;
	int shared_memory_flag = 1;

	if (sub_y_cam_width * sub_y_cam_height > 2 * padding_length * padding_length) {
		shared_memory_flag = 0;
	}

	// fill the projected padding
	if (shared_memory_flag == 1) {
		sub_y_cam = &sub_y_ref[padding_length * padding_length];
		int p = threadIdx.x + threadIdx.y * shared_width;

		while (p < sub_y_cam_width * sub_y_cam_height) {
			int cam_x = min_cam_x + p % sub_y_cam_width;
			int cam_y = min_cam_y + p / sub_y_cam_width;
			if (cam_x < 0 || cam_y < 0 || cam_x >= width || cam_y >= height) {
				p += shared_width * shared_height;
				continue;
			}
			sub_y_cam[p] = y_cam[MI(cam_x, cam_y, width)];
			p += shared_width * shared_height;
		}
	}
	__syncthreads();

	// (ii) calculate cost against reference
	// Calculating cost in a window
	float cost = 0.0f;
	float cc = 0.0f;
	for (int k = -(half_window); k <= half_window; k++)
	{
		for (int l = -(half_window); l <= half_window; l++)
		{
			if (x + l < 0 || x + l >= width)
				continue;
			if (y + k < 0 || y + k >= height)
				continue;
			if (x_proj2 + l < 0 || x_proj2 + l >= width)
				continue;
			if (y_proj2 + k < 0 || y_proj2 + k >= height)
				continue;

			// Y
			
			if (shared_memory_flag == 1) {
				if(x_proj2 - min_cam_x + l >= 0 && x_proj2 - min_cam_x + l < sub_y_cam_width && y_proj2 - min_cam_y + k >= 0 && y_proj2 - min_cam_y + k < sub_y_cam_height){
					cost += fabsf((float)sub_y_ref[MI(padding_x + l, padding_y + k, padding_length)] - (float)sub_y_cam[MI(x_proj2 - min_cam_x + l, y_proj2 - min_cam_y + k, sub_y_cam_width)]);
				}

			}
			else {
				cost += fabsf((float)sub_y_ref[MI(padding_x + l, padding_y + k, padding_length)] - (float)y_cam[MI(x_proj2 + l, y_proj2 + k, width)]);
			}
			cc += 1.0f;
		}
	}
	cost /= cc;


	//  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
	// only the minimum cost for all the cameras is stored
	if (cost_cube[MI(x, y, width)] > cost) cost_cube[MI(x, y, width)]=cost;
}

__global__ void compute_cost_shared_full_float_2D(float* cost_cube, const float* y_ref, const float* y_cam)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	__shared__ int cam_x_proj[4];
	__shared__ int cam_y_proj[4];
	__shared__ float* sub_y_cam;
	__shared__ int shared_width;
	__shared__ int shared_height;

	if (x >= *const_width || y >= *const_height)
		return;

	// Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
	float z = *const_znear * *const_zfar / (*const_znear + ((*const_zi / *const_ZPlanes) * (*const_zfar - *const_znear))); //need to be in the x and y loops?

	// 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
	float X_ref = (const_inv_K[0] * x + const_inv_K[1] * y + const_inv_K[2]) * z;
	float Y_ref = (const_inv_K[3] * x + const_inv_K[4] * y + const_inv_K[5]) * z;
	float Z_ref = (const_inv_K[6] * x + const_inv_K[7] * y + const_inv_K[8]) * z;

	// 3D in ref camera coordinates to 3D world
	float X = const_inv_R[0] * X_ref + const_inv_R[1] * Y_ref + const_inv_R[2] * Z_ref - const_inv_t[0];
	float Y = const_inv_R[3] * X_ref + const_inv_R[4] * Y_ref + const_inv_R[5] * Z_ref - const_inv_t[1];
	float Z = const_inv_R[6] * X_ref + const_inv_R[7] * Y_ref + const_inv_R[8] * Z_ref - const_inv_t[2];

	// 3D world to projected camera 3D coordinates
	float X_proj = const_R[0] * X + const_R[1] * Y + const_R[2] * Z - const_t[0];
	float Y_proj = const_R[3] * X + const_R[4] * Y + const_R[5] * Z - const_t[1];
	float Z_proj = const_R[6] * X + const_R[7] * Y + const_R[8] * Z - const_t[2];

	// Projected camera 3D coordinates to projected camera 2D coordinates
	float x_proj = (const_K[0] * X_proj / Z_proj + const_K[1] * Y_proj / Z_proj + const_K[2]);
	float y_proj = (const_K[3] * X_proj / Z_proj + const_K[4] * Y_proj / Z_proj + const_K[5]);


	int x_proj2 = x_proj < 0 || x_proj >= *const_width ? 0 : (int)roundf(x_proj);
	int y_proj2 = y_proj < 0 || y_proj >= *const_height ? 0 : (int)roundf(y_proj);


	// Compute projection corners
	if (threadIdx.x == 0) {
		if (threadIdx.y == 0) {
			cam_x_proj[0] = x_proj2;
			cam_y_proj[0] = y_proj2;
		}
		else if (threadIdx.y == N_THREADS - 1 || y == *const_height - 1) {
			cam_x_proj[1] = x_proj2;
			cam_y_proj[1] = y_proj2;
		}
	}
	else if (threadIdx.x == N_THREADS - 1 || x == *const_width - 1) {
		if (threadIdx.y == 0) {
			cam_x_proj[2] = x_proj2;
			cam_y_proj[2] = y_proj2;
		}
		else if (threadIdx.y == N_THREADS - 1 || y == *const_height - 1) {
			cam_x_proj[3] = x_proj2;
			cam_y_proj[3] = y_proj2;
			shared_width = threadIdx.x;
			shared_height = threadIdx.y;
		}
	}

	int padding_length = N_THREADS + 2 * *const_half_window;
	int padding_x = *const_half_window + threadIdx.x;
	int padding_y = *const_half_window + threadIdx.y;

	extern __shared__ float float_sub_y_ref[];

	float_sub_y_ref[MI(padding_x, padding_y, padding_length)] = y_ref[MI(x, y, *const_width)];

	// padding the left side of the image subsamples given to the shared memory unless at image boundaries where nothing is done
	if (threadIdx.x < *const_half_window && x >= N_THREADS) {
		float_sub_y_ref[MI(threadIdx.x, padding_y, padding_length)] = y_ref[MI(x - *const_half_window, y, *const_width)];
	}
	// padding the right side of the image subsamples given to the shared memory unless at image boundaries where nothing is done
	else if (threadIdx.x >= N_THREADS - *const_half_window && x < *const_width - N_THREADS) {
		float_sub_y_ref[MI(padding_x + *const_half_window, padding_y, padding_length)] = y_ref[MI(x + *const_half_window, y, *const_width)];
	}

	// padding the upper side of the image subsamples given to the shared memory unless at image boundaries where nothing is done
	if (threadIdx.y < *const_half_window && y >= N_THREADS) {
		float_sub_y_ref[MI(padding_x, threadIdx.y, padding_length)] = y_ref[MI(x, y - *const_half_window, *const_width)];
	}
	// padding the lower side of the image subsamples given to the shared memory unless at image boundaries where nothing is done
	else if (threadIdx.y >= N_THREADS - *const_half_window && y < *const_width - N_THREADS) {
		float_sub_y_ref[MI(padding_x, padding_y + *const_half_window, padding_length)] = y_ref[MI(x, y + *const_half_window, *const_width)];
	}

	// Inside of middle square of size window * window
	if (threadIdx.x >= (N_THREADS / 2) - *const_half_window && threadIdx.x < (N_THREADS / 2) + *const_half_window &&
		threadIdx.y >= (N_THREADS / 2) - *const_half_window && threadIdx.y < (N_THREADS / 2) + *const_half_window) {
		// padding both upper corners of the image subsamples given to the shared memory unless at image boundaries where nothing is done
		if (threadIdx.y < N_THREADS / 2 && y >= N_THREADS) {
			// padding the upper left corner of the image subsamples given to the shared memory unless at image boundaries where nothing is done
			if (threadIdx.x < N_THREADS / 2 && x >= N_THREADS) {
				float_sub_y_ref[MI(padding_x - N_THREADS / 2, padding_y - N_THREADS / 2, padding_length)] = y_ref[MI(x - N_THREADS / 2, y - N_THREADS / 2, *const_width)];
			}
			// padding the upper right corner of the image subsamples given to the shared memory unless at image boundaries where nothing is done
			else if (threadIdx.x >= N_THREADS / 2 && x < *const_width - N_THREADS) {
				float_sub_y_ref[MI(padding_x + N_THREADS / 2, padding_y - N_THREADS / 2, padding_length)] = y_ref[MI(x + N_THREADS / 2, y - N_THREADS / 2, *const_width)];
			}
		}
		// padding both lower corners of the image subsamples given to the shared memory unless at image boundaries where nothing is done
		else if (threadIdx.y >= N_THREADS / 2 && y < *const_height - N_THREADS) {
			// padding the lower left corner of the image subsamples given to the shared memory unless at image boundaries where nothing is done
			if (threadIdx.x < N_THREADS / 2 && x >= N_THREADS) {
				float_sub_y_ref[MI(padding_x - N_THREADS / 2, padding_y + N_THREADS / 2, padding_length)] = y_ref[MI(x - N_THREADS / 2, y + N_THREADS / 2, *const_width)];
			}
			// padding the lower right corner of the image subsamples given to the shared memory unless at image boundaries where nothing is done
			else if (threadIdx.x >= N_THREADS / 2 && x < *const_width - N_THREADS) {
				float_sub_y_ref[MI(padding_x + N_THREADS / 2, padding_y + N_THREADS / 2, padding_length)] = y_ref[MI(x + N_THREADS / 2, y + N_THREADS / 2, *const_width)];
			}
		}
	}

	__syncthreads();

	// Compute projected padding parameters
	int min_cam_x = umin(cam_x_proj[0], cam_x_proj[2]) - *const_half_window;
	int min_cam_y = umin(cam_y_proj[0], cam_y_proj[1]) - *const_half_window;
	int sub_y_cam_width = umax(cam_x_proj[1], cam_x_proj[3]) + *const_half_window - min_cam_x + 1;
	int sub_y_cam_height = umax(cam_y_proj[2], cam_y_proj[3]) + *const_half_window - min_cam_y + 1;
	int shared_memory_flag = 1;

	if (sub_y_cam_width * sub_y_cam_height > 2 * padding_length * padding_length) {
		shared_memory_flag = 0;
	}

	// fill the projected padding
	if (shared_memory_flag == 1) {
		sub_y_cam = &float_sub_y_ref[padding_length * padding_length];
		int p = threadIdx.x + threadIdx.y * shared_width;

		while (p < sub_y_cam_width * sub_y_cam_height) {
			int cam_x = min_cam_x + p % sub_y_cam_width;
			int cam_y = min_cam_y + p / sub_y_cam_width;
			if (cam_x < 0 || cam_y < 0 || cam_x >= *const_width || cam_y >= *const_height) {
				p += shared_width * shared_height;
				continue;
			}
			sub_y_cam[p] = y_cam[MI(cam_x, cam_y, *const_width)];
			p += shared_width * shared_height;
		}
	}
	__syncthreads();

	// (ii) calculate cost against reference
	// Calculating cost in a window
	float cost = 0.0f;
	float cc = 0.0f;
	for (int k = -(*const_half_window); k <= *const_half_window; k++)
	{
		for (int l = -(*const_half_window); l <= *const_half_window; l++)
		{
			if (x + l < 0 || x + l >= *const_width)
				continue;
			if (y + k < 0 || y + k >= *const_height)
				continue;
			if (x_proj2 + l < 0 || x_proj2 + l >= *const_width)
				continue;
			if (y_proj2 + k < 0 || y_proj2 + k >= *const_height)
				continue;

			// Y

			if (shared_memory_flag == 1) {
				if (x_proj2 - min_cam_x + l >= 0 && x_proj2 - min_cam_x + l < sub_y_cam_width && y_proj2 - min_cam_y + k >= 0 && y_proj2 - min_cam_y + k < sub_y_cam_height) {
					cost += fabsf(float_sub_y_ref[MI(padding_x + l, padding_y + k, padding_length)] - sub_y_cam[MI(x_proj2 - min_cam_x + l, y_proj2 - min_cam_y + k, sub_y_cam_width)]);
				}
			}
			else {
				cost += fabsf(float_sub_y_ref[MI(padding_x + l, padding_y + k, padding_length)] - y_cam[MI(x_proj2 + l, y_proj2 + k, *const_width)]);
			}

			// U
			// cost += fabs(ref.YUV[1].at<uint8_t >(y + k, x + l) - cam.YUV[1].at<uint8_t>((int)y_proj + k, (int)x_proj + l));
			// V
			// cost += fabs(ref.YUV[2].at<uint8_t >(y + k, x + l) - cam.YUV[2].at<uint8_t>((int)y_proj + k, (int)x_proj + l));
			cc += 1.0f;
		}
	}
	cost /= cc;


	//  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
	// only the minimum cost for all the cameras is stored
	if (cost_cube[MI(x, y, *const_width)] > cost) cost_cube[MI(x, y, *const_width)] = cost;
	
}

__global__ void compute_cost_smart_naive_full_float_2D(float* cost_cube, const float* y_ref, const float* y_cam)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= *const_width || y >= *const_height)
		return;

	float z, X_ref, Y_ref, Z_ref, X, Y, Z, X_proj, Y_proj, Z_proj, x_proj, y_proj, x_proj2, y_proj2, cost, cc, k, l;
	int sub_y_cam_height;

	for (int zi = 0; zi < 256; zi++) {

		// Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
		z = *const_znear * *const_zfar / (*const_znear + ((zi / *const_ZPlanes) * (*const_zfar - *const_znear))); //need to be in the x and y loops?

		// 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
		X_ref = (const_inv_K[0] * x + const_inv_K[1] * y + const_inv_K[2]) * z;
		Y_ref = (const_inv_K[3] * x + const_inv_K[4] * y + const_inv_K[5]) * z;
		Z_ref = (const_inv_K[6] * x + const_inv_K[7] * y + const_inv_K[8]) * z;

		// 3D in ref camera coordinates to 3D world
		X = const_inv_R[0] * X_ref + const_inv_R[1] * Y_ref + const_inv_R[2] * Z_ref - const_inv_t[0];
		Y = const_inv_R[3] * X_ref + const_inv_R[4] * Y_ref + const_inv_R[5] * Z_ref - const_inv_t[1];
		Z = const_inv_R[6] * X_ref + const_inv_R[7] * Y_ref + const_inv_R[8] * Z_ref - const_inv_t[2];

		// 3D world to projected camera 3D coordinates
		X_proj = const_R[0] * X + const_R[1] * Y + const_R[2] * Z - const_t[0];
		Y_proj = const_R[3] * X + const_R[4] * Y + const_R[5] * Z - const_t[1];
		Z_proj = const_R[6] * X + const_R[7] * Y + const_R[8] * Z - const_t[2];

		// Projected camera 3D coordinates to projected camera 2D coordinates
		x_proj = (const_K[0] * X_proj / Z_proj + const_K[1] * Y_proj / Z_proj + const_K[2]);
		y_proj = (const_K[3] * X_proj / Z_proj + const_K[4] * Y_proj / Z_proj + const_K[5]);

		x_proj2 = x_proj < 0 || x_proj >= *const_width ? 0 : roundf(x_proj);
		y_proj2 = y_proj < 0 || y_proj >= *const_height ? 0 : roundf(y_proj);

		// (ii) calculate cost against reference
		// Calculating cost in a window
		cost = 0.0f;
		cc = 0.0f;
		for (k = -(*const_half_window); k <= *const_half_window; k++)
		{
			for (l = -(*const_half_window); l <= *const_half_window; l++)
			{
				if (x + l < 0.0 || x + l >= (float)*const_width)
					continue;
				if (y + k < 0.0 || y + k >= (float)*const_height)
					continue;
				if (x_proj2 + l < 0.0 || x_proj2 + l >= (float)*const_width)
					continue;
				if (y_proj2 + k < 0.0 || y_proj2 + k >= (float)*const_height)
					continue;


				cost += fabsf(y_ref[(int)MI(x + l, y + k, *const_width)] - y_cam[(int)MI(x_proj2 + l, y_proj2 + k, *const_width)]);
				cc += 1.0f;
			}
		}
		cost /= cc;


		//  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
		// only the minimum cost for all the cameras is stored
		if (cost_cube[MI3(x, y, zi, *const_width, *const_height)] > cost) cost_cube[MI3(x, y, zi, *const_width, *const_height)] = cost;
	}
}

__global__ void compute_cost_smart_shared_full_float_2D(float* cost_cube, const float* y_ref, const float* y_cam)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	__shared__ int cam_x_proj[4];
	__shared__ int cam_y_proj[4];
	__shared__ float* sub_y_cam;
	__shared__ int shared_width;
	__shared__ int shared_height;
	
	extern __shared__ float float_sub_y_ref[];

	if (x >= *const_width || y >= *const_height)
		return;

	int padding_length = N_THREADS + 2 * *const_half_window;
	int padding_x = *const_half_window + threadIdx.x;
	int padding_y = *const_half_window + threadIdx.y;


	float_sub_y_ref[MI(padding_x, padding_y, padding_length)] = y_ref[MI(x, y, *const_width)];

	// padding the left side of the image subsamples given to the shared memory unless at image boundaries where nothing is done
	if (threadIdx.x < *const_half_window && x >= N_THREADS) {
		float_sub_y_ref[MI(threadIdx.x, padding_y, padding_length)] = y_ref[MI(x - *const_half_window, y, *const_width)];
	}
	// padding the right side of the image subsamples given to the shared memory unless at image boundaries where nothing is done
	else if (threadIdx.x >= N_THREADS - *const_half_window && x < *const_width - N_THREADS) {
		float_sub_y_ref[MI(padding_x + *const_half_window, padding_y, padding_length)] = y_ref[MI(x + *const_half_window, y, *const_width)];
	}

	// padding the upper side of the image subsamples given to the shared memory unless at image boundaries where nothing is done
	if (threadIdx.y < *const_half_window && y >= N_THREADS) {
		float_sub_y_ref[MI(padding_x, threadIdx.y, padding_length)] = y_ref[MI(x, y - *const_half_window, *const_width)];
	}
	// padding the lower side of the image subsamples given to the shared memory unless at image boundaries where nothing is done
	else if (threadIdx.y >= N_THREADS - *const_half_window && y < *const_width - N_THREADS) {
		float_sub_y_ref[MI(padding_x, padding_y + *const_half_window, padding_length)] = y_ref[MI(x, y + *const_half_window, *const_width)];
	}

	// Inside of middle square of size window * window
	if (threadIdx.x >= (N_THREADS / 2) - *const_half_window && threadIdx.x < (N_THREADS / 2) + *const_half_window &&
		threadIdx.y >= (N_THREADS / 2) - *const_half_window && threadIdx.y < (N_THREADS / 2) + *const_half_window) {
		// padding both upper corners of the image subsamples given to the shared memory unless at image boundaries where nothing is done
		if (threadIdx.y < N_THREADS / 2 && y >= N_THREADS) {
			// padding the upper left corner of the image subsamples given to the shared memory unless at image boundaries where nothing is done
			if (threadIdx.x < N_THREADS / 2 && x >= N_THREADS) {
				float_sub_y_ref[MI(padding_x - N_THREADS / 2, padding_y - N_THREADS / 2, padding_length)] = y_ref[MI(x - N_THREADS / 2, y - N_THREADS / 2, *const_width)];
			}
			// padding the upper right corner of the image subsamples given to the shared memory unless at image boundaries where nothing is done
			else if (threadIdx.x >= N_THREADS / 2 && x < *const_width - N_THREADS) {
				float_sub_y_ref[MI(padding_x + N_THREADS / 2, padding_y - N_THREADS / 2, padding_length)] = y_ref[MI(x + N_THREADS / 2, y - N_THREADS / 2, *const_width)];
			}
		}
		// padding both lower corners of the image subsamples given to the shared memory unless at image boundaries where nothing is done
		else if (threadIdx.y >= N_THREADS / 2 && y < *const_height - N_THREADS) {
			// padding the lower left corner of the image subsamples given to the shared memory unless at image boundaries where nothing is done
			if (threadIdx.x < N_THREADS / 2 && x >= N_THREADS) {
				float_sub_y_ref[MI(padding_x - N_THREADS / 2, padding_y + N_THREADS / 2, padding_length)] = y_ref[MI(x - N_THREADS / 2, y + N_THREADS / 2, *const_width)];
			}
			// padding the lower right corner of the image subsamples given to the shared memory unless at image boundaries where nothing is done
			else if (threadIdx.x >= N_THREADS / 2 && x < *const_width - N_THREADS) {
				float_sub_y_ref[MI(padding_x + N_THREADS / 2, padding_y + N_THREADS / 2, padding_length)] = y_ref[MI(x + N_THREADS / 2, y + N_THREADS / 2, *const_width)];
			}
		}
	}

	__syncthreads();

	float z, X_ref, Y_ref, Z_ref, X, Y, Z, X_proj, Y_proj, Z_proj, x_proj, y_proj, x_proj2, y_proj2, cost, cc, k, l;
	int min_cam_x, min_cam_y, sub_y_cam_width, sub_y_cam_height, shared_memory_flag, p, cam_x, cam_y;

	for (int zi=0; zi<256; zi++){

		// Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
		z = *const_znear * *const_zfar / (*const_znear + ((zi / *const_ZPlanes) * (*const_zfar - *const_znear))); //need to be in the x and y loops?

		// 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
		X_ref = (const_inv_K[0] * x + const_inv_K[1] * y + const_inv_K[2]) * z;
		Y_ref = (const_inv_K[3] * x + const_inv_K[4] * y + const_inv_K[5]) * z;
		Z_ref = (const_inv_K[6] * x + const_inv_K[7] * y + const_inv_K[8]) * z;

		// 3D in ref camera coordinates to 3D world
		X = const_inv_R[0] * X_ref + const_inv_R[1] * Y_ref + const_inv_R[2] * Z_ref - const_inv_t[0];
		Y = const_inv_R[3] * X_ref + const_inv_R[4] * Y_ref + const_inv_R[5] * Z_ref - const_inv_t[1];
		Z = const_inv_R[6] * X_ref + const_inv_R[7] * Y_ref + const_inv_R[8] * Z_ref - const_inv_t[2];

		// 3D world to projected camera 3D coordinates
		X_proj = const_R[0] * X + const_R[1] * Y + const_R[2] * Z - const_t[0];
		Y_proj = const_R[3] * X + const_R[4] * Y + const_R[5] * Z - const_t[1];
		Z_proj = const_R[6] * X + const_R[7] * Y + const_R[8] * Z - const_t[2];

		// Projected camera 3D coordinates to projected camera 2D coordinates
		x_proj = (const_K[0] * X_proj / Z_proj + const_K[1] * Y_proj / Z_proj + const_K[2]);
		y_proj = (const_K[3] * X_proj / Z_proj + const_K[4] * Y_proj / Z_proj + const_K[5]);
		//float z_proj = Z_proj;

		/*x_proj2 = x_proj < 0 ? 0 : (x_proj >= *const_width ? *const_width : roundf(x_proj));
		y_proj2 = y_proj < 0 ? 0 : (y_proj >= *const_height ? *const_height : roundf(y_proj));*/

		x_proj2 = x_proj < 0 || x_proj >= *const_width ? 0 : roundf(x_proj);
		y_proj2 = y_proj < 0 || y_proj >= *const_height ? 0 : roundf(y_proj);

		if (threadIdx.x == 0) {
			if (threadIdx.y == 0) {
				cam_x_proj[0] = x_proj2;
				cam_y_proj[0] = y_proj2;
			}
			//else if ((threadIdx.y == N_THREADS - 1 && y < *const_height) || y == *const_height - 1) {
			else if (threadIdx.y == N_THREADS - 1  || y == *const_height - 1) {
				cam_x_proj[1] = x_proj2;
				cam_y_proj[1] = y_proj2;
			}
		}
			//else if ((threadIdx.x == N_THREADS - 1 && x < *const_width) || x == *const_width - 1) {
			else if (threadIdx.x == N_THREADS - 1 || x == *const_width - 1) {
			if (threadIdx.y == 0) {
				cam_x_proj[2] = x_proj2;
				cam_y_proj[2] = y_proj2;
			}
			//else if ((threadIdx.y == N_THREADS - 1 && y < *const_height) || y == *const_height - 1) {
			else if (threadIdx.y == N_THREADS - 1 || y == *const_height - 1) {
				cam_x_proj[3] = x_proj2;
				cam_y_proj[3] = y_proj2;
				shared_width = threadIdx.x;
				shared_height = threadIdx.y;
			}
		}

		__syncthreads();

		// Compute projection corners
		min_cam_x = umin(cam_x_proj[0], cam_x_proj[2]) - *const_half_window;
		min_cam_y = umin(cam_y_proj[0], cam_y_proj[1]) - *const_half_window;
		sub_y_cam_width = umax(cam_x_proj[1], cam_x_proj[3]) + *const_half_window - min_cam_x + 1;
		sub_y_cam_height = umax(cam_y_proj[2], cam_y_proj[3]) + *const_half_window - min_cam_y + 1;
		shared_memory_flag = 1;

		if (sub_y_cam_width * sub_y_cam_height > 2 * padding_length * padding_length ) {
			shared_memory_flag = 0;
		}

		// fill the projected padding
		if (shared_memory_flag == 1) {
			sub_y_cam = &float_sub_y_ref[padding_length * padding_length];
			p = threadIdx.x + threadIdx.y * shared_width;

			while (p < sub_y_cam_width * sub_y_cam_height) {
				cam_x = min_cam_x + p % sub_y_cam_width;
				cam_y = min_cam_y + p / sub_y_cam_width;
				if (cam_x < 0 || cam_y < 0 || cam_x >= *const_width || cam_y >= *const_height) {
					p += shared_width * shared_height;
					continue;
				}
				sub_y_cam[p] = y_cam[MI(cam_x, cam_y, *const_width)];
				p += shared_width * shared_height;
			}
		}
		__syncthreads();

		// (ii) calculate cost against reference
		// Calculating cost in a window
		cost = 0.0f;
		cc = 0.0f;
		for (k = -(*const_half_window); k <= *const_half_window; k++)
		{
			for (l = -(*const_half_window); l <= *const_half_window; l++)
			{
				if (x + l < 0.0 || x + l >= (float)*const_width)
					continue;
				if (y + k < 0.0 || y + k >= (float)*const_height)
					continue;
				if (x_proj2 + l < 0.0 || x_proj2 + l >= (float)*const_width)
					continue;
				if (y_proj2 + k < 0.0 || y_proj2 + k >= (float)*const_height)
					continue;


				if (shared_memory_flag == 1) {
					if (x_proj2 - (float)min_cam_x + l >= 0.0 && x_proj2 - (float)min_cam_x + l < (float)sub_y_cam_width && y_proj2 - (float)min_cam_y + k >= 0.0 && y_proj2 - (float)min_cam_y + k < (float)sub_y_cam_height) {
						cost += fabsf(float_sub_y_ref[(int)MI((float)padding_x + l, (float)padding_y + k, (float)padding_length)] - sub_y_cam[(int)MI(x_proj2 - (float)min_cam_x + l, y_proj2 - (float)min_cam_y + k, (float)sub_y_cam_width)]);

					}
				}
				else {
					cost += fabsf(float_sub_y_ref[(int)MI((float)padding_x + l, (float)padding_y + k, (float)padding_length)] - y_cam[(int)MI(x_proj2 + l, y_proj2 + k, (float)*const_width)]);
				}
				cc += 1.0f;
			}
		}
		cost /= cc;

		//  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
		// only the minimum cost for all the cameras is stored
		if (cost_cube[MI3(x, y, zi, *const_width, *const_height)] > cost) cost_cube[MI3(x, y, zi, *const_width, *const_height)] = cost;
	}
}

__global__ void compute_cost_smart_full_shared_full_float_2D(float* cost_cube, const float* y_ref, const float* y_cam)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ int width;
	__shared__ int height;
	__shared__ float znear;
	__shared__ float zfar;
	__shared__ float ZPlanes;
	__shared__ int half_window;
	__shared__ float K[9];
	__shared__ float R[9];
	__shared__ float t[3];
	__shared__ float inv_K[9];
	__shared__ float inv_R[9];
	__shared__ float inv_t[3];
	__shared__ int cam_x_proj[4];
	__shared__ int cam_y_proj[4];
	__shared__ float* sub_y_cam;
	__shared__ int shared_width;
	__shared__ int shared_height;

	extern __shared__ float float_sub_y_ref[];

	// Fill shared memory
	if (threadIdx.x == 0) {
		if (threadIdx.y == 0) width = *const_width;
		if (threadIdx.y == 1) height = *const_height;
		if (threadIdx.y == 3) znear = *const_znear;
		if (threadIdx.y == 4) zfar = *const_zfar;
		if (threadIdx.y == 5) ZPlanes = *const_ZPlanes;
		if (threadIdx.y == 6) half_window = *const_half_window;
	}
	else if (threadIdx.x == 1 && threadIdx.y < 9) K[threadIdx.y] = const_K[threadIdx.y];
	else if (threadIdx.x == 2 && threadIdx.y < 9) R[threadIdx.y] = const_R[threadIdx.y];
	else if (threadIdx.x == 3 && threadIdx.y < 3) t[threadIdx.y] = const_t[threadIdx.y];
	else if (threadIdx.x == 4 && threadIdx.y < 9) inv_K[threadIdx.y] = const_inv_K[threadIdx.y];
	else if (threadIdx.x == 5 && threadIdx.y < 9) inv_R[threadIdx.y] = const_inv_R[threadIdx.y];
	else if (threadIdx.x == 6 && threadIdx.y < 3) inv_t[threadIdx.y] = const_inv_t[threadIdx.y];

	__syncthreads();

	if (x >= width || y >= height)
		return;

	int padding_length = N_THREADS + 2 * half_window;
	int padding_x = half_window + threadIdx.x;
	int padding_y = half_window + threadIdx.y;


	float_sub_y_ref[MI(padding_x, padding_y, padding_length)] = y_ref[MI(x, y, width)];

	// padding the left side of the image subsamples given to the shared memory unless at image boundaries where nothing is done
	if (threadIdx.x < half_window && x >= N_THREADS) {
		float_sub_y_ref[MI(threadIdx.x, padding_y, padding_length)] = y_ref[MI(x - half_window, y, width)];
	}
	// padding the right side of the image subsamples given to the shared memory unless at image boundaries where nothing is done
	else if (threadIdx.x >= N_THREADS - half_window && x < width - N_THREADS) {
		float_sub_y_ref[MI(padding_x + half_window, padding_y, padding_length)] = y_ref[MI(x + half_window, y, width)];
	}

	// padding the upper side of the image subsamples given to the shared memory unless at image boundaries where nothing is done
	if (threadIdx.y < half_window && y >= N_THREADS) {
		float_sub_y_ref[MI(padding_x, threadIdx.y, padding_length)] = y_ref[MI(x, y - half_window, width)];
	}
	// padding the lower side of the image subsamples given to the shared memory unless at image boundaries where nothing is done
	else if (threadIdx.y >= N_THREADS - half_window && y < width - N_THREADS) {
		float_sub_y_ref[MI(padding_x, padding_y + half_window, padding_length)] = y_ref[MI(x, y + half_window, width)];
	}

	// Inside of middle square of size window * window
	if (threadIdx.x >= (N_THREADS / 2) - half_window && threadIdx.x < (N_THREADS / 2) + half_window &&
		threadIdx.y >= (N_THREADS / 2) - half_window && threadIdx.y < (N_THREADS / 2) + half_window) {
		// padding both upper corners of the image subsamples given to the shared memory unless at image boundaries where nothing is done
		if (threadIdx.y < N_THREADS / 2 && y >= N_THREADS) {
			// padding the upper left corner of the image subsamples given to the shared memory unless at image boundaries where nothing is done
			if (threadIdx.x < N_THREADS / 2 && x >= N_THREADS) {
				float_sub_y_ref[MI(padding_x - N_THREADS / 2, padding_y - N_THREADS / 2, padding_length)] = y_ref[MI(x - N_THREADS / 2, y - N_THREADS / 2, width)];
			}
			// padding the upper right corner of the image subsamples given to the shared memory unless at image boundaries where nothing is done
			else if (threadIdx.x >= N_THREADS / 2 && x < width - N_THREADS) {
				float_sub_y_ref[MI(padding_x + N_THREADS / 2, padding_y - N_THREADS / 2, padding_length)] = y_ref[MI(x + N_THREADS / 2, y - N_THREADS / 2, width)];
			}
		}
		// padding both lower corners of the image subsamples given to the shared memory unless at image boundaries where nothing is done
		else if (threadIdx.y >= N_THREADS / 2 && y < height - N_THREADS) {
			// padding the lower left corner of the image subsamples given to the shared memory unless at image boundaries where nothing is done
			if (threadIdx.x < N_THREADS / 2 && x >= N_THREADS) {
				float_sub_y_ref[MI(padding_x - N_THREADS / 2, padding_y + N_THREADS / 2, padding_length)] = y_ref[MI(x - N_THREADS / 2, y + N_THREADS / 2, width)];
			}
			// padding the lower right corner of the image subsamples given to the shared memory unless at image boundaries where nothing is done
			else if (threadIdx.x >= N_THREADS / 2 && x < width - N_THREADS) {
				float_sub_y_ref[MI(padding_x + N_THREADS / 2, padding_y + N_THREADS / 2, padding_length)] = y_ref[MI(x + N_THREADS / 2, y + N_THREADS / 2, width)];
			}
		}
	}

	__syncthreads();

	for (int zi = 0; zi < 256; zi++) {

		// Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
		float z = znear * zfar / (znear + ((zi / ZPlanes) * (zfar - znear))); //need to be in the x and y loops?

		// 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
		float X_ref = (inv_K[0] * x + inv_K[1] * y + inv_K[2]) * z;
		float Y_ref = (inv_K[3] * x + inv_K[4] * y + inv_K[5]) * z;
		float Z_ref = (inv_K[6] * x + inv_K[7] * y + inv_K[8]) * z;

		// 3D in ref camera coordinates to 3D world
		float X = inv_R[0] * X_ref + inv_R[1] * Y_ref + inv_R[2] * Z_ref - inv_t[0];
		float Y = inv_R[3] * X_ref + inv_R[4] * Y_ref + inv_R[5] * Z_ref - inv_t[1];
		float Z = inv_R[6] * X_ref + inv_R[7] * Y_ref + inv_R[8] * Z_ref - inv_t[2];

		// 3D world to projected camera 3D coordinates
		float X_proj = R[0] * X + R[1] * Y + R[2] * Z - t[0];
		float Y_proj = R[3] * X + R[4] * Y + R[5] * Z - t[1];
		float Z_proj = R[6] * X + R[7] * Y + R[8] * Z - t[2];

		// Projected camera 3D coordinates to projected camera 2D coordinates
		float x_proj = (K[0] * X_proj / Z_proj + K[1] * Y_proj / Z_proj + K[2]);
		float y_proj = (K[3] * X_proj / Z_proj + K[4] * Y_proj / Z_proj + K[5]);

		int x_proj2 = x_proj < 0 || x_proj >= width ? 0 : (int)roundf(x_proj);
		int y_proj2 = y_proj < 0 || y_proj >= height ? 0 : (int)roundf(y_proj);


		// Compute projection corners
		if (threadIdx.x == 0) {
			if (threadIdx.y == 0) {
				cam_x_proj[0] = x_proj2;
				cam_y_proj[0] = y_proj2;
			}
			else if (threadIdx.y == N_THREADS - 1 || y == height - 1) {
				cam_x_proj[1] = x_proj2;
				cam_y_proj[1] = y_proj2;
			}
		}
		else if (threadIdx.x == N_THREADS - 1 || x == width - 1) {
			if (threadIdx.y == 0) {
				cam_x_proj[2] = x_proj2;
				cam_y_proj[2] = y_proj2;
			}
			else if (threadIdx.y == N_THREADS - 1 || y == height - 1) {
				cam_x_proj[3] = x_proj2;
				cam_y_proj[3] = y_proj2;
				shared_width = threadIdx.x;
				shared_height = threadIdx.y;
			}
		}

		__syncthreads();

		// Compute projected padding parameters
		int min_cam_x = umin(cam_x_proj[0], cam_x_proj[2]) - half_window;
		int min_cam_y = umin(cam_y_proj[0], cam_y_proj[1]) - half_window;
		int sub_y_cam_width = umax(cam_x_proj[1], cam_x_proj[3]) + half_window - min_cam_x + 1;
		int sub_y_cam_height = umax(cam_y_proj[2], cam_y_proj[3]) + half_window - min_cam_y + 1;
		int shared_memory_flag = 1;

		if (sub_y_cam_width * sub_y_cam_height > 2 * padding_length * padding_length) {
			shared_memory_flag = 0;
		}

		// fill the projected padding
		if (shared_memory_flag == 1) {
			sub_y_cam = &float_sub_y_ref[padding_length * padding_length];
			int p = threadIdx.x + threadIdx.y * shared_width;

			while (p < sub_y_cam_width * sub_y_cam_height) {
				int cam_x = min_cam_x + p % sub_y_cam_width;
				int cam_y = min_cam_y + p / sub_y_cam_width;
				if (cam_x < 0 || cam_y < 0 || cam_x >= width || cam_y >= height) {
					p += shared_width * shared_height;
					continue;
				}
				sub_y_cam[p] = y_cam[MI(cam_x, cam_y, width)];
				p += shared_width * shared_height;
			}
		}
		__syncthreads();

		// (ii) calculate cost against reference
		// Calculating cost in a window
		float cost = 0.0f;
		float cc = 0.0f;
		for (int k = -(half_window); k <= half_window; k++)
		{
			for (int l = -(half_window); l <= half_window; l++)
			{
				if (x + l < 0 || x + l >= width)
					continue;
				if (y + k < 0 || y + k >= height)
					continue;
				if (x_proj2 + l < 0 || x_proj2 + l >= width)
					continue;
				if (y_proj2 + k < 0 || y_proj2 + k >= height)
					continue;


				if (shared_memory_flag == 1) {
					if (x_proj2 - min_cam_x + l >= 0 && x_proj2 - min_cam_x + l < sub_y_cam_width && y_proj2 - min_cam_y + k >= 0 && y_proj2 - min_cam_y + k < sub_y_cam_height) {
						cost += fabsf(float_sub_y_ref[MI(padding_x + l, padding_y + k, padding_length)] - sub_y_cam[MI(x_proj2 - min_cam_x + l, y_proj2 - min_cam_y + k, sub_y_cam_width)]);
					}
				}
				else {
					cost += fabsf(float_sub_y_ref[MI(padding_x + l, padding_y + k, padding_length)] - y_cam[MI(x_proj2 + l, y_proj2 + k, width)]);
				}
				cc += 1.0f;
			}
		}
		cost /= cc;

		//  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
		// only the minimum cost for all the cameras is stored
		if (cost_cube[MI3(x, y, zi, width, height)] > cost) cost_cube[MI3(x, y, zi, width, height)] = cost;
	}
}

__global__ void compute_all_cost_smart_full_shared_full_float_2D(float* cost_cube, const float* y_ref, const float* y_cam)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ int width;
	__shared__ int height;
	__shared__ float znear;
	__shared__ float zfar;
	__shared__ float ZPlanes;
	__shared__ int half_window;
	__shared__ float K[9];
	__shared__ float R[9];
	__shared__ float t[3];
	__shared__ float inv_K[9];
	__shared__ float inv_R[9];
	__shared__ float inv_t[3];
	__shared__ int cam_x_proj[4];
	__shared__ int cam_y_proj[4];
	__shared__ float* sub_y_cam;
	__shared__ int shared_width;
	__shared__ int shared_height;

	extern __shared__ float float_sub_y_ref[];

	if (x >= *const_width || y >= *const_height)
		return;

	//Fill shared memory
	if (threadIdx.x == 0) {
		if (threadIdx.y == 0) width = *const_width;
		if (threadIdx.y == 1) height = *const_height;
		if (threadIdx.y == 3) znear = *const_znear;
		if (threadIdx.y == 4) zfar = *const_zfar;
		if (threadIdx.y == 5) ZPlanes = *const_ZPlanes;
		if (threadIdx.y == 6) half_window = *const_half_window;
	}
	else if (threadIdx.x == 1 && threadIdx.y < 9) K[threadIdx.y] = const_K[threadIdx.y];
	else if (threadIdx.x == 2 && threadIdx.y < 9) R[threadIdx.y] = const_R[threadIdx.y];
	else if (threadIdx.x == 3 && threadIdx.y < 3) t[threadIdx.y] = const_t[threadIdx.y];
	else if (threadIdx.x == 4 && threadIdx.y < 9) inv_K[threadIdx.y] = const_inv_K[threadIdx.y];
	else if (threadIdx.x == 5 && threadIdx.y < 9) inv_R[threadIdx.y] = const_inv_R[threadIdx.y];
	else if (threadIdx.x == 6 && threadIdx.y < 3) inv_t[threadIdx.y] = const_inv_t[threadIdx.y];


	for(int cam_n = 0; cam_n < *const_cam_count; cam_n++){

		if(cam_n != 0){
			if (threadIdx.x == 1 && threadIdx.y < 9) K[threadIdx.y] = const_K[MI(threadIdx.y, cam_n, 9)];
			else if (threadIdx.x == 2 && threadIdx.y < 9) R[threadIdx.y] = const_R[MI(threadIdx.y, cam_n, 9)];
			else if (threadIdx.x == 3 && threadIdx.y < 3) t[threadIdx.y] = const_t[MI(threadIdx.y, cam_n, 3)];

			__syncthreads();
		}

		int padding_length = N_THREADS + 2 * half_window;
		int padding_x = half_window + threadIdx.x;
		int padding_y = half_window + threadIdx.y;


		float_sub_y_ref[MI(padding_x, padding_y, padding_length)] = y_ref[MI(x, y, width)];

		// padding the left side of the image subsamples given to the shared memory unless at image boundaries where nothing is done
		if (threadIdx.x < half_window && x >= N_THREADS) {
			float_sub_y_ref[MI(threadIdx.x, padding_y, padding_length)] = y_ref[MI(x - half_window, y, width)];
		}
		// padding the right side of the image subsamples given to the shared memory unless at image boundaries where nothing is done
		else if (threadIdx.x >= N_THREADS - half_window && x < width - N_THREADS) {
			float_sub_y_ref[MI(padding_x + half_window, padding_y, padding_length)] = y_ref[MI(x + half_window, y, width)];
		}

		// padding the upper side of the image subsamples given to the shared memory unless at image boundaries where nothing is done
		if (threadIdx.y < half_window && y >= N_THREADS) {
			float_sub_y_ref[MI(padding_x, threadIdx.y, padding_length)] = y_ref[MI(x, y - half_window, width)];
		}
		// padding the lower side of the image subsamples given to the shared memory unless at image boundaries where nothing is done
		else if (threadIdx.y >= N_THREADS - half_window && y < width - N_THREADS) {
			float_sub_y_ref[MI(padding_x, padding_y + half_window, padding_length)] = y_ref[MI(x, y + half_window, width)];
		}

		// Inside of middle square of size window * window
		if (threadIdx.x >= (N_THREADS / 2) - half_window && threadIdx.x < (N_THREADS / 2) + half_window &&
			threadIdx.y >= (N_THREADS / 2) - half_window && threadIdx.y < (N_THREADS / 2) + half_window) {
			// padding both upper corners of the image subsamples given to the shared memory unless at image boundaries where nothing is done
			if (threadIdx.y < N_THREADS / 2 && y >= N_THREADS) {
				// padding the upper left corner of the image subsamples given to the shared memory unless at image boundaries where nothing is done
				if (threadIdx.x < N_THREADS / 2 && x >= N_THREADS) {
					float_sub_y_ref[MI(padding_x - N_THREADS / 2, padding_y - N_THREADS / 2, padding_length)] = y_ref[MI(x - N_THREADS / 2, y - N_THREADS / 2, width)];
				}
				// padding the upper right corner of the image subsamples given to the shared memory unless at image boundaries where nothing is done
				else if (threadIdx.x >= N_THREADS / 2 && x < width - N_THREADS) {
					float_sub_y_ref[MI(padding_x + N_THREADS / 2, padding_y - N_THREADS / 2, padding_length)] = y_ref[MI(x + N_THREADS / 2, y - N_THREADS / 2, width)];
				}
			}
			// padding both lower corners of the image subsamples given to the shared memory unless at image boundaries where nothing is done
			else if (threadIdx.y >= N_THREADS / 2 && y < height - N_THREADS) {
				// padding the lower left corner of the image subsamples given to the shared memory unless at image boundaries where nothing is done
				if (threadIdx.x < N_THREADS / 2 && x >= N_THREADS) {
					float_sub_y_ref[MI(padding_x - N_THREADS / 2, padding_y + N_THREADS / 2, padding_length)] = y_ref[MI(x - N_THREADS / 2, y + N_THREADS / 2, width)];
				}
				// padding the lower right corner of the image subsamples given to the shared memory unless at image boundaries where nothing is done
				else if (threadIdx.x >= N_THREADS / 2 && x < width - N_THREADS) {
					float_sub_y_ref[MI(padding_x + N_THREADS / 2, padding_y + N_THREADS / 2, padding_length)] = y_ref[MI(x + N_THREADS / 2, y + N_THREADS / 2, width)];
				}
			}
		}

		__syncthreads();

		for (int zi = 0; zi < 256; zi++) {

			// Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
			float z = znear * zfar / (znear + ((zi / ZPlanes) * (zfar - znear))); //need to be in the x and y loops?

			// 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
			float X_ref = (inv_K[0] * x + inv_K[1] * y + inv_K[2]) * z;
			float Y_ref = (inv_K[3] * x + inv_K[4] * y + inv_K[5]) * z;
			float Z_ref = (inv_K[6] * x + inv_K[7] * y + inv_K[8]) * z;

			// 3D in ref camera coordinates to 3D world
			float X = inv_R[0] * X_ref + inv_R[1] * Y_ref + inv_R[2] * Z_ref - inv_t[0];
			float Y = inv_R[3] * X_ref + inv_R[4] * Y_ref + inv_R[5] * Z_ref - inv_t[1];
			float Z = inv_R[6] * X_ref + inv_R[7] * Y_ref + inv_R[8] * Z_ref - inv_t[2];

			// 3D world to projected camera 3D coordinates
			float X_proj = R[0] * X + R[1] * Y + R[2] * Z - t[0];
			float Y_proj = R[3] * X + R[4] * Y + R[5] * Z - t[1];
			float Z_proj = R[6] * X + R[7] * Y + R[8] * Z - t[2];

			// Projected camera 3D coordinates to projected camera 2D coordinates
			float x_proj = (K[0] * X_proj / Z_proj + K[1] * Y_proj / Z_proj + K[2]);
			float y_proj = (K[3] * X_proj / Z_proj + K[4] * Y_proj / Z_proj + K[5]);

			int x_proj2 = x_proj < 0 || x_proj >= width ? 0 : (int)roundf(x_proj);
			int y_proj2 = y_proj < 0 || y_proj >= height ? 0 : (int)roundf(y_proj);

			// Compute projection corners
			if (threadIdx.x == 0) {
				if (threadIdx.y == 0) {
					cam_x_proj[0] = x_proj2;
					cam_y_proj[0] = y_proj2;
				}
				else if (threadIdx.y == N_THREADS - 1 || y == height - 1) {
					cam_x_proj[1] = x_proj2;
					cam_y_proj[1] = y_proj2;
				}
			}
			else if (threadIdx.x == N_THREADS - 1 || x == width - 1) {
				if (threadIdx.y == 0) {
					cam_x_proj[2] = x_proj2;
					cam_y_proj[2] = y_proj2;
				}
				else if (threadIdx.y == N_THREADS - 1 || y == height - 1) {
					cam_x_proj[3] = x_proj2;
					cam_y_proj[3] = y_proj2;
					shared_width = threadIdx.x;
					shared_height = threadIdx.y;
				}
			}

			__syncthreads();

			// Compute projection corners
			int min_cam_x = umin(cam_x_proj[0], cam_x_proj[2]) - half_window;
			int min_cam_y = umin(cam_y_proj[0], cam_y_proj[1]) - half_window;
			int sub_y_cam_width = umax(cam_x_proj[1], cam_x_proj[3]) + half_window - min_cam_x + 1;
			int sub_y_cam_height = umax(cam_y_proj[2], cam_y_proj[3]) + half_window - min_cam_y + 1;
			int shared_memory_flag = 1;

			if (sub_y_cam_width * sub_y_cam_height > 2 * padding_length * padding_length) {
				shared_memory_flag = 0;
			}

			// fill the projected padding
			if (shared_memory_flag == 1) {
				sub_y_cam = &float_sub_y_ref[padding_length * padding_length];
				int p = threadIdx.x + threadIdx.y * shared_width;

				while (p < sub_y_cam_width * sub_y_cam_height) {
					int cam_x = min_cam_x + p % sub_y_cam_width;
					int cam_y = min_cam_y + p / sub_y_cam_width;
					if (cam_x < 0 || cam_y < 0 || cam_x >= width || cam_y >= height) {
						p += shared_width * shared_height;
						continue;
					}
					sub_y_cam[p] = y_cam[MI3(cam_x, cam_y, cam_n, width, height)];
					p += shared_width * shared_height;
				}
			}
			__syncthreads();

			// (ii) calculate cost against reference
			// Calculating cost in a window
			float cost = 0.0f;
			float cc = 0.0f;
			for (int k = -(half_window); k <= half_window; k++)
			{
				for (int l = -(half_window); l <= half_window; l++)
				{
					if (x + l < 0 || x + l >= width)
						continue;
					if (y + k < 0 || y + k >= height)
						continue;
					if (x_proj2 + l < 0 || x_proj2 + l >= width)
						continue;
					if (y_proj2 + k < 0 || y_proj2 + k >= height)
						continue;


					if (shared_memory_flag == 1) {
						if (x_proj2 - min_cam_x + l >= 0 && x_proj2 - min_cam_x + l < sub_y_cam_width && y_proj2 - min_cam_y + k >= 0 && y_proj2 - min_cam_y + k < sub_y_cam_height) {
							cost += fabsf(float_sub_y_ref[MI(padding_x + l, padding_y + k, padding_length)] - sub_y_cam[MI(x_proj2 - min_cam_x + l, y_proj2 - min_cam_y + k, sub_y_cam_width)]);
						}
					}
					else {
						cost += fabsf(float_sub_y_ref[MI(padding_x + l, padding_y + k, padding_length)] - y_cam[MI3(x_proj2 + l, y_proj2 + k, cam_n, width, height)]);
					}
					cc += 1.0f;
				}
			}
			cost /= cc;

			//  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
			// only the minimum cost for all the cameras is stored
			if (cost_cube[MI3(x, y, zi, width, height)] > cost) cost_cube[MI3(x, y, zi, width, height)] = cost;

			__syncthreads();
		}
	}
}

__global__ void compute_all_cost_no_fill_smart_full_shared_full_float_2D(float* cost_cube, const float* y_ref, const float* y_cam)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ int width;
	__shared__ int height;
	__shared__ float znear;
	__shared__ float zfar;
	__shared__ float ZPlanes;
	__shared__ int half_window;
	__shared__ float K[9];
	__shared__ float R[9];
	__shared__ float t[3];
	__shared__ float inv_K[9];
	__shared__ float inv_R[9];
	__shared__ float inv_t[3];
	__shared__ int cam_x_proj[4];
	__shared__ int cam_y_proj[4];
	__shared__ float* sub_y_cam;
	__shared__ int shared_width;
	__shared__ int shared_height;

	extern __shared__ float float_sub_y_ref[];

	if (x >= *const_width || y >= *const_height)
		return;

	//Fill shared memory
	if (threadIdx.x == 0) {
		if (threadIdx.y == 0) width = *const_width;
		if (threadIdx.y == 1) height = *const_height;
		if (threadIdx.y == 3) znear = *const_znear;
		if (threadIdx.y == 4) zfar = *const_zfar;
		if (threadIdx.y == 5) ZPlanes = *const_ZPlanes;
		if (threadIdx.y == 6) half_window = *const_half_window;
	}
	else if (threadIdx.x == 1 && threadIdx.y < 9) K[threadIdx.y] = const_K[threadIdx.y];
	else if (threadIdx.x == 2 && threadIdx.y < 9) R[threadIdx.y] = const_R[threadIdx.y];
	else if (threadIdx.x == 3 && threadIdx.y < 3) t[threadIdx.y] = const_t[threadIdx.y];
	else if (threadIdx.x == 4 && threadIdx.y < 9) inv_K[threadIdx.y] = const_inv_K[threadIdx.y];
	else if (threadIdx.x == 5 && threadIdx.y < 9) inv_R[threadIdx.y] = const_inv_R[threadIdx.y];
	else if (threadIdx.x == 6 && threadIdx.y < 3) inv_t[threadIdx.y] = const_inv_t[threadIdx.y];


	for (int cam_n = 0; cam_n < *const_cam_count; cam_n++) {

		if (cam_n != 0) {
			if (threadIdx.x == 1 && threadIdx.y < 9) K[threadIdx.y] = const_K[MI(threadIdx.y, cam_n, 9)];
			else if (threadIdx.x == 2 && threadIdx.y < 9) R[threadIdx.y] = const_R[MI(threadIdx.y, cam_n, 9)];
			else if (threadIdx.x == 3 && threadIdx.y < 3) t[threadIdx.y] = const_t[MI(threadIdx.y, cam_n, 3)];
		}

		__syncthreads();

		int padding_length = N_THREADS + 2 * half_window;
		int padding_x = half_window + threadIdx.x;
		int padding_y = half_window + threadIdx.y;


		float_sub_y_ref[MI(padding_x, padding_y, padding_length)] = y_ref[MI(x, y, width)];

		// padding the left side of the image subsamples given to the shared memory unless at image boundaries where nothing is done
		if (threadIdx.x < half_window && x >= N_THREADS) {
			float_sub_y_ref[MI(threadIdx.x, padding_y, padding_length)] = y_ref[MI(x - half_window, y, width)];
		}
		// padding the right side of the image subsamples given to the shared memory unless at image boundaries where nothing is done
		else if (threadIdx.x >= N_THREADS - half_window && x < width - N_THREADS) {
			float_sub_y_ref[MI(padding_x + half_window, padding_y, padding_length)] = y_ref[MI(x + half_window, y, width)];
		}

		// padding the upper side of the image subsamples given to the shared memory unless at image boundaries where nothing is done
		if (threadIdx.y < half_window && y >= N_THREADS) {
			float_sub_y_ref[MI(padding_x, threadIdx.y, padding_length)] = y_ref[MI(x, y - half_window, width)];
		}
		// padding the lower side of the image subsamples given to the shared memory unless at image boundaries where nothing is done
		else if (threadIdx.y >= N_THREADS - half_window && y < width - N_THREADS) {
			float_sub_y_ref[MI(padding_x, padding_y + half_window, padding_length)] = y_ref[MI(x, y + half_window, width)];
		}

		// Inside of middle square of size window * window
		if (threadIdx.x >= (N_THREADS / 2) - half_window && threadIdx.x < (N_THREADS / 2) + half_window &&
			threadIdx.y >= (N_THREADS / 2) - half_window && threadIdx.y < (N_THREADS / 2) + half_window) {
			// padding both upper corners of the image subsamples given to the shared memory unless at image boundaries where nothing is done
			if (threadIdx.y < N_THREADS / 2 && y >= N_THREADS) {
				// padding the upper left corner of the image subsamples given to the shared memory unless at image boundaries where nothing is done
				if (threadIdx.x < N_THREADS / 2 && x >= N_THREADS) {
					float_sub_y_ref[MI(padding_x - N_THREADS / 2, padding_y - N_THREADS / 2, padding_length)] = y_ref[MI(x - N_THREADS / 2, y - N_THREADS / 2, width)];
				}
				// padding the upper right corner of the image subsamples given to the shared memory unless at image boundaries where nothing is done
				else if (threadIdx.x >= N_THREADS / 2 && x < width - N_THREADS) {
					float_sub_y_ref[MI(padding_x + N_THREADS / 2, padding_y - N_THREADS / 2, padding_length)] = y_ref[MI(x + N_THREADS / 2, y - N_THREADS / 2, width)];
				}
			}
			// padding both lower corners of the image subsamples given to the shared memory unless at image boundaries where nothing is done
			else if (threadIdx.y >= N_THREADS / 2 && y < height - N_THREADS) {
				// padding the lower left corner of the image subsamples given to the shared memory unless at image boundaries where nothing is done
				if (threadIdx.x < N_THREADS / 2 && x >= N_THREADS) {
					float_sub_y_ref[MI(padding_x - N_THREADS / 2, padding_y + N_THREADS / 2, padding_length)] = y_ref[MI(x - N_THREADS / 2, y + N_THREADS / 2, width)];
				}
				// padding the lower right corner of the image subsamples given to the shared memory unless at image boundaries where nothing is done
				else if (threadIdx.x >= N_THREADS / 2 && x < width - N_THREADS) {
					float_sub_y_ref[MI(padding_x + N_THREADS / 2, padding_y + N_THREADS / 2, padding_length)] = y_ref[MI(x + N_THREADS / 2, y + N_THREADS / 2, width)];
				}
			}
		}

		__syncthreads();

		for (int zi = 0; zi < 256; zi++) {

			// Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
			float z = znear * zfar / (znear + ((zi / ZPlanes) * (zfar - znear))); //need to be in the x and y loops?

			// 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
			float X_ref = (inv_K[0] * x + inv_K[1] * y + inv_K[2]) * z;
			float Y_ref = (inv_K[3] * x + inv_K[4] * y + inv_K[5]) * z;
			float Z_ref = (inv_K[6] * x + inv_K[7] * y + inv_K[8]) * z;

			// 3D in ref camera coordinates to 3D world
			float X = inv_R[0] * X_ref + inv_R[1] * Y_ref + inv_R[2] * Z_ref - inv_t[0];
			float Y = inv_R[3] * X_ref + inv_R[4] * Y_ref + inv_R[5] * Z_ref - inv_t[1];
			float Z = inv_R[6] * X_ref + inv_R[7] * Y_ref + inv_R[8] * Z_ref - inv_t[2];

			// 3D world to projected camera 3D coordinates
			float X_proj = R[0] * X + R[1] * Y + R[2] * Z - t[0];
			float Y_proj = R[3] * X + R[4] * Y + R[5] * Z - t[1];
			float Z_proj = R[6] * X + R[7] * Y + R[8] * Z - t[2];

			// Projected camera 3D coordinates to projected camera 2D coordinates
			float x_proj = (K[0] * X_proj / Z_proj + K[1] * Y_proj / Z_proj + K[2]);
			float y_proj = (K[3] * X_proj / Z_proj + K[4] * Y_proj / Z_proj + K[5]);

			int x_proj2 = x_proj < 0 || x_proj >= width ? 0 : (int)roundf(x_proj);
			int y_proj2 = y_proj < 0 || y_proj >= height ? 0 : (int)roundf(y_proj);

			// Compute projection corners
			if (threadIdx.x == 0) {
				if (threadIdx.y == 0) {
					cam_x_proj[0] = x_proj2;
					cam_y_proj[0] = y_proj2;
				}
				else if (threadIdx.y == N_THREADS - 1 || y == height - 1) {
					cam_x_proj[1] = x_proj2;
					cam_y_proj[1] = y_proj2;
				}
			}
			else if (threadIdx.x == N_THREADS - 1 || x == width - 1) {
				if (threadIdx.y == 0) {
					cam_x_proj[2] = x_proj2;
					cam_y_proj[2] = y_proj2;
				}
				else if (threadIdx.y == N_THREADS - 1 || y == height - 1) {
					cam_x_proj[3] = x_proj2;
					cam_y_proj[3] = y_proj2;
					shared_width = threadIdx.x;
					shared_height = threadIdx.y;
				}
			}

			__syncthreads();

			// Compute projection corners
			int min_cam_x = umin(cam_x_proj[0], cam_x_proj[2]) - half_window;
			int min_cam_y = umin(cam_y_proj[0], cam_y_proj[1]) - half_window;
			int sub_y_cam_width = umax(cam_x_proj[1], cam_x_proj[3]) + half_window - min_cam_x + 1;
			int sub_y_cam_height = umax(cam_y_proj[2], cam_y_proj[3]) + half_window - min_cam_y + 1;
			int shared_memory_flag = 1;

			if (sub_y_cam_width * sub_y_cam_height > 2 * padding_length * padding_length) {
				shared_memory_flag = 0;
			}

			// fill the projected padding
			if (shared_memory_flag == 1) {
				sub_y_cam = &float_sub_y_ref[padding_length * padding_length];
				int p = threadIdx.x + threadIdx.y * shared_width;

				while (p < sub_y_cam_width * sub_y_cam_height) {
					int cam_x = min_cam_x + p % sub_y_cam_width;
					int cam_y = min_cam_y + p / sub_y_cam_width;
					if (cam_x < 0 || cam_y < 0 || cam_x >= width || cam_y >= height) {
						p += shared_width * shared_height;
						continue;
					}
					sub_y_cam[p] = y_cam[MI3(cam_x, cam_y, cam_n, width, height)];
					p += shared_width * shared_height;
				}
			}
			__syncthreads();

			// (ii) calculate cost against reference
			// Calculating cost in a window
			float cost = 0.0f;
			float cc = 0.0f;
			for (int k = -(half_window); k <= half_window; k++)
			{
				for (int l = -(half_window); l <= half_window; l++)
				{
					if (x + l < 0 || x + l >= width)
						continue;
					if (y + k < 0 || y + k >= height)
						continue;
					if (x_proj2 + l < 0 || x_proj2 + l >= width)
						continue;
					if (y_proj2 + k < 0 || y_proj2 + k >= height)
						continue;

					if (shared_memory_flag == 1) {
						if (x_proj2 - min_cam_x + l >= 0 && x_proj2 - min_cam_x + l < sub_y_cam_width && y_proj2 - min_cam_y + k >= 0 && y_proj2 - min_cam_y + k < sub_y_cam_height) {
							cost += fabsf(float_sub_y_ref[MI(padding_x + l, padding_y + k, padding_length)] - sub_y_cam[MI(x_proj2 - min_cam_x + l, y_proj2 - min_cam_y + k, sub_y_cam_width)]);
						}
					}
					else {
						cost += fabsf(float_sub_y_ref[MI(padding_x + l, padding_y + k, padding_length)] - y_cam[MI3(x_proj2 + l, y_proj2 + k, cam_n, width, height)]);
					}
					cc += 1.0f;
				}
			}
			cost /= cc;

			//  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
			// only the minimum cost for all the cameras is stored
			if (cam_n == 0) cost_cube[MI3(x, y, zi, width, height)] = fminf(cost, 255.0);
			else if (cost_cube[MI3(x, y, zi, width, height)] > cost) cost_cube[MI3(x, y, zi, width, height)] = cost;


			__syncthreads();
		}
	}
}

__global__ void compute_all_cost_no_fill_better_pad_smart_full_shared_full_float_2D(float* cost_cube, const float* y_ref, const float* y_cam)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ int width;
	__shared__ int height;
	__shared__ float znear;
	__shared__ float zfar;
	__shared__ float ZPlanes;
	__shared__ int half_window;
	__shared__ float K[9];
	__shared__ float R[9];
	__shared__ float t[3];
	__shared__ float inv_K[9];
	__shared__ float inv_R[9];
	__shared__ float inv_t[3];
	__shared__ int cam_x_proj[4];
	__shared__ int cam_y_proj[4];
	__shared__ float* sub_y_cam;
	__shared__ int shared_width;
	__shared__ int shared_height;

	extern __shared__ float float_sub_y_ref[];

	if (x >= *const_width || y >= *const_height)
		return;

	//Fill shared memory
	if (threadIdx.x == 0) {
		if (threadIdx.y == 0) width = *const_width;
		if (threadIdx.y == 1) height = *const_height;
		if (threadIdx.y == 3) znear = *const_znear;
		if (threadIdx.y == 4) zfar = *const_zfar;
		if (threadIdx.y == 5) ZPlanes = *const_ZPlanes;
		if (threadIdx.y == 6) half_window = *const_half_window;
	}
	else if (threadIdx.x == 1 && threadIdx.y < 9) K[threadIdx.y] = const_K[threadIdx.y];
	else if (threadIdx.x == 2 && threadIdx.y < 9) R[threadIdx.y] = const_R[threadIdx.y];
	else if (threadIdx.x == 3 && threadIdx.y < 3) t[threadIdx.y] = const_t[threadIdx.y];
	else if (threadIdx.x == 4 && threadIdx.y < 9) inv_K[threadIdx.y] = const_inv_K[threadIdx.y];
	else if (threadIdx.x == 5 && threadIdx.y < 9) inv_R[threadIdx.y] = const_inv_R[threadIdx.y];
	else if (threadIdx.x == 6 && threadIdx.y < 3) inv_t[threadIdx.y] = const_inv_t[threadIdx.y];


	for (int cam_n = 0; cam_n < *const_cam_count; cam_n++) {

		if (cam_n != 0) {
			if (threadIdx.x == 1 && threadIdx.y < 9) K[threadIdx.y] = const_K[MI(threadIdx.y, cam_n, 9)];
			else if (threadIdx.x == 2 && threadIdx.y < 9) R[threadIdx.y] = const_R[MI(threadIdx.y, cam_n, 9)];
			else if (threadIdx.x == 3 && threadIdx.y < 3) t[threadIdx.y] = const_t[MI(threadIdx.y, cam_n, 3)];
		}

		__syncthreads();

		int padding_length = N_THREADS + 2 * half_window;
		int padding_x = half_window + threadIdx.x;
		int padding_y = half_window + threadIdx.y;

		int shared_ref_width;
		int shared_ref_height;

		// Compute ref padding parameters
		if( (blockIdx.x + 1) * N_THREADS > width)
			shared_ref_width = width - blockIdx.x * N_THREADS;
		else 
			shared_ref_width = N_THREADS;
		if ((blockIdx.y + 1) * N_THREADS > height)
			shared_ref_height = height - blockIdx.y * N_THREADS;
		else
			shared_ref_height = N_THREADS;

		// Fill ref padding
		int p = threadIdx.x + threadIdx.y * shared_ref_width;
		while (p < padding_length * padding_length) {
			int cam_x = MI(p % padding_length - half_window, blockIdx.x, N_THREADS);
			int cam_y = MI(p / padding_length - half_window, blockIdx.y, N_THREADS);
			if (cam_x < 0 || cam_y < 0 || cam_x >= width || cam_y >= height) {
				p += shared_ref_width * shared_ref_height;
				continue;
			}
			float_sub_y_ref[p] = y_ref[MI(cam_x, cam_y, width)];
			p += shared_ref_width * shared_ref_height;
		}


		__syncthreads();

		for (int zi = 0; zi < 256; zi++) {

			// Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
			float z = znear * zfar / (znear + ((zi / ZPlanes) * (zfar - znear))); //need to be in the x and y loops?

			// 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
			float X_ref = (inv_K[0] * x + inv_K[1] * y + inv_K[2]) * z;
			float Y_ref = (inv_K[3] * x + inv_K[4] * y + inv_K[5]) * z;
			float Z_ref = (inv_K[6] * x + inv_K[7] * y + inv_K[8]) * z;

			// 3D in ref camera coordinates to 3D world
			float X = inv_R[0] * X_ref + inv_R[1] * Y_ref + inv_R[2] * Z_ref - inv_t[0];
			float Y = inv_R[3] * X_ref + inv_R[4] * Y_ref + inv_R[5] * Z_ref - inv_t[1];
			float Z = inv_R[6] * X_ref + inv_R[7] * Y_ref + inv_R[8] * Z_ref - inv_t[2];

			// 3D world to projected camera 3D coordinates
			float X_proj = R[0] * X + R[1] * Y + R[2] * Z - t[0];
			float Y_proj = R[3] * X + R[4] * Y + R[5] * Z - t[1];
			float Z_proj = R[6] * X + R[7] * Y + R[8] * Z - t[2];

			// Projected camera 3D coordinates to projected camera 2D coordinates
			float x_proj = (K[0] * X_proj / Z_proj + K[1] * Y_proj / Z_proj + K[2]);
			float y_proj = (K[3] * X_proj / Z_proj + K[4] * Y_proj / Z_proj + K[5]);

			int x_proj2 = x_proj < 0 || x_proj >= width ? 0 : (int)roundf(x_proj);
			int y_proj2 = y_proj < 0 || y_proj >= height ? 0 : (int)roundf(y_proj);

			// Find projected padding corners
			if (threadIdx.x == 0) {
				if (threadIdx.y == 0) {
					cam_x_proj[0] = x_proj2;
					cam_y_proj[0] = y_proj2;
				}
				else if (threadIdx.y == N_THREADS - 1 || y == height - 1) {
					cam_x_proj[1] = x_proj2;
					cam_y_proj[1] = y_proj2;
				}
			}
			else if (threadIdx.x == N_THREADS - 1 || x == width - 1) {
				if (threadIdx.y == 0) {
					cam_x_proj[2] = x_proj2;
					cam_y_proj[2] = y_proj2;
				}
				else if (threadIdx.y == N_THREADS - 1 || y == height - 1) {
					cam_x_proj[3] = x_proj2;
					cam_y_proj[3] = y_proj2;
					shared_width = threadIdx.x;
					shared_height = threadIdx.y;
				}
			}

			__syncthreads();

			// Compute projected padding parameters
			int min_cam_x = umin(cam_x_proj[0], cam_x_proj[2]) - half_window;
			int min_cam_y = umin(cam_y_proj[0], cam_y_proj[1]) - half_window;
			int sub_y_cam_width = umax(cam_x_proj[1], cam_x_proj[3]) + half_window - min_cam_x + 1;
			int sub_y_cam_height = umax(cam_y_proj[2], cam_y_proj[3]) + half_window - min_cam_y + 1;
			int shared_memory_flag = 1;

			if (sub_y_cam_width * sub_y_cam_height > 2 * padding_length * padding_length) {
				shared_memory_flag = 0;
			}

			// fill the projected padding
			if (shared_memory_flag == 1) {
				sub_y_cam = &float_sub_y_ref[padding_length * padding_length];
				int p = threadIdx.x + threadIdx.y * shared_width;

				while (p < sub_y_cam_width * sub_y_cam_height) {
					int cam_x = min_cam_x + p % sub_y_cam_width;
					int cam_y = min_cam_y + p / sub_y_cam_width;
					if (cam_x < 0 || cam_y < 0 || cam_x >= width || cam_y >= height) {
						p += shared_width * shared_height;
						continue;
					}
					sub_y_cam[p] = y_cam[MI3(cam_x, cam_y, cam_n, width, height)];
					p += shared_width * shared_height;
				}
			}
			__syncthreads();

			// (ii) calculate cost against reference
			// Calculating cost in a window
			float cost = 0.0f;
			float cc = 0.0f;
			for (int k = -(half_window); k <= half_window; k++)
			{
				for (int l = -(half_window); l <= half_window; l++)
				{
					if (x + l < 0 || x + l >= width)
						continue;
					if (y + k < 0 || y + k >= height)
						continue;
					if (x_proj2 + l < 0 || x_proj2 + l >= width)
						continue;
					if (y_proj2 + k < 0 || y_proj2 + k >= height)
						continue;

					if (shared_memory_flag == 1) {
						if (x_proj2 - min_cam_x + l >= 0 && x_proj2 - min_cam_x + l < sub_y_cam_width && y_proj2 - min_cam_y + k >= 0 && y_proj2 - min_cam_y + k < sub_y_cam_height) {
							cost += fabsf(float_sub_y_ref[MI(padding_x + l, padding_y + k, padding_length)] - sub_y_cam[MI(x_proj2 - min_cam_x + l, y_proj2 - min_cam_y + k, sub_y_cam_width)]);
						}
					}
					else {
						cost += fabsf(float_sub_y_ref[MI(padding_x + l, padding_y + k, padding_length)] - y_cam[MI3(x_proj2 + l, y_proj2 + k, cam_n, width, height)]);
					}
					cc += 1.0f;
				}
			}
			cost /= cc;

			//  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
			// only the minimum cost for all the cameras is stored
			if (cam_n == 0) cost_cube[MI3(x, y, zi, width, height)] = fminf(cost, 255.0);
			else if (cost_cube[MI3(x, y, zi, width, height)] > cost) cost_cube[MI3(x, y, zi, width, height)] = cost;


			__syncthreads();
		}
	}
}

__global__ void compute_reduced_float_all_cost_no_fill_better_pad_smart_full_shared_full_float_2D(float* cost_cube, const float* y_ref, const float* y_cam)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ int width;
	__shared__ int height;
	__shared__ float znear;
	__shared__ float zfar;
	__shared__ float ZPlanes;
	__shared__ int half_window;
	__shared__ float K[9];
	__shared__ float R[9];
	__shared__ float t[3];
	__shared__ float inv_K[9];
	__shared__ float inv_R[9];
	__shared__ float inv_t[3];
	__shared__ int cam_x_proj[4];
	__shared__ int cam_y_proj[4];
	__shared__ float* sub_y_cam;
	__shared__ int shared_width;
	__shared__ int shared_height;

	extern __shared__ float float_sub_y_ref[];

	if (x >= *const_width || y >= *const_height)
		return;

	// Fill shared memory
	if (threadIdx.x == 0) {
		if (threadIdx.y == 0) width = *const_width;
		if (threadIdx.y == 1) height = *const_height;
		if (threadIdx.y == 3) znear = *const_znear;
		if (threadIdx.y == 4) zfar = *const_zfar;
		if (threadIdx.y == 5) ZPlanes = *const_ZPlanes;
		if (threadIdx.y == 6) half_window = *const_half_window;
	}
	else if (threadIdx.x == 1 && threadIdx.y < 9) K[threadIdx.y] = const_K[threadIdx.y];
	else if (threadIdx.x == 2 && threadIdx.y < 9) R[threadIdx.y] = const_R[threadIdx.y];
	else if (threadIdx.x == 3 && threadIdx.y < 3) t[threadIdx.y] = const_t[threadIdx.y];
	else if (threadIdx.x == 4 && threadIdx.y < 9) inv_K[threadIdx.y] = const_inv_K[threadIdx.y];
	else if (threadIdx.x == 5 && threadIdx.y < 9) inv_R[threadIdx.y] = const_inv_R[threadIdx.y];
	else if (threadIdx.x == 6 && threadIdx.y < 3) inv_t[threadIdx.y] = const_inv_t[threadIdx.y];


	for (int cam_n = 0; cam_n < *const_cam_count; cam_n++) {

		if (cam_n != 0) {
			if (threadIdx.x == 1 && threadIdx.y < 9) K[threadIdx.y] = const_K[MI(threadIdx.y, cam_n, 9)];
			else if (threadIdx.x == 2 && threadIdx.y < 9) R[threadIdx.y] = const_R[MI(threadIdx.y, cam_n, 9)];
			else if (threadIdx.x == 3 && threadIdx.y < 3) t[threadIdx.y] = const_t[MI(threadIdx.y, cam_n, 3)];
		}

		__syncthreads();

		// Compute ref padding parameters
		int padding_length = N_THREADS + 2 * half_window;
		int padding_x = half_window + threadIdx.x;
		int padding_y = half_window + threadIdx.y;

		int shared_ref_width;
		int shared_ref_height;

		if ((blockIdx.x + 1) * N_THREADS > width)
			shared_ref_width = width - blockIdx.x * N_THREADS;
		else
			shared_ref_width = N_THREADS;
		if ((blockIdx.y + 1) * N_THREADS > height)
			shared_ref_height = height - blockIdx.y * N_THREADS;
		else
			shared_ref_height = N_THREADS;

		// Fill ref padding
		int p = threadIdx.x + threadIdx.y * shared_ref_width;
		while (p < padding_length * padding_length) {
			int cam_x = MI(p % padding_length - half_window, blockIdx.x, N_THREADS);
			int cam_y = MI(p / padding_length - half_window, blockIdx.y, N_THREADS);
			if (cam_x < 0 || cam_y < 0 || cam_x >= width || cam_y >= height) {
				p += shared_ref_width * shared_ref_height;
				continue;
			}
			float_sub_y_ref[p] = y_ref[MI(cam_x, cam_y, width)];
			p += shared_ref_width * shared_ref_height;
		}


		__syncthreads();

		for (int zi = 0; zi < 256; zi++) {

			// Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
			float z = znear * zfar / (znear + ((zi / ZPlanes) * (zfar - znear))); //need to be in the x and y loops?

			// 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
			float X_ref = (inv_K[0] * x + inv_K[1] * y + inv_K[2]) * z;
			float Y_ref = (inv_K[3] * x + inv_K[4] * y + inv_K[5]) * z;
			float Z_ref = (inv_K[6] * x + inv_K[7] * y + inv_K[8]) * z;

			// 3D in ref camera coordinates to 3D world
			float X = inv_R[0] * X_ref + inv_R[1] * Y_ref + inv_R[2] * Z_ref - inv_t[0];
			float Y = inv_R[3] * X_ref + inv_R[4] * Y_ref + inv_R[5] * Z_ref - inv_t[1];
			float Z = inv_R[6] * X_ref + inv_R[7] * Y_ref + inv_R[8] * Z_ref - inv_t[2];

			// 3D world to projected camera 3D coordinates
			float X_proj = R[0] * X + R[1] * Y + R[2] * Z - t[0];
			float Y_proj = R[3] * X + R[4] * Y + R[5] * Z - t[1];
			float Z_proj = R[6] * X + R[7] * Y + R[8] * Z - t[2];

			// Projected camera 3D coordinates to projected camera 2D coordinates
			float x_proj = (K[0] * X_proj / Z_proj + K[1] * Y_proj / Z_proj + K[2]);
			float y_proj = (K[3] * X_proj / Z_proj + K[4] * Y_proj / Z_proj + K[5]);

			int x_proj2 = x_proj < 0 || x_proj >= width ? 0 : (int)roundf(x_proj);
			int y_proj2 = y_proj < 0 || y_proj >= height ? 0 : (int)roundf(y_proj);

			// Find projected padding corners
			if (threadIdx.x == 0) {
				if (threadIdx.y == 0) {
					cam_x_proj[0] = x_proj2;
					cam_y_proj[0] = y_proj2;
				}
				else if (threadIdx.y == N_THREADS - 1 || y == height - 1) {
					cam_x_proj[1] = x_proj2;
					cam_y_proj[1] = y_proj2;
				}
			}
			else if (threadIdx.x == N_THREADS - 1 || x == width - 1) {
				if (threadIdx.y == 0) {
					cam_x_proj[2] = x_proj2;
					cam_y_proj[2] = y_proj2;
				}
				else if (threadIdx.y == N_THREADS - 1 || y == height - 1) {
					cam_x_proj[3] = x_proj2;
					cam_y_proj[3] = y_proj2;
					shared_width = threadIdx.x;
					shared_height = threadIdx.y;
				}
			}

			__syncthreads();

			// Compute projected padding parameters
			int min_cam_x = umin(cam_x_proj[0], cam_x_proj[2]) - half_window;
			int min_cam_y = umin(cam_y_proj[0], cam_y_proj[1]) - half_window;
			int sub_y_cam_width = umax(cam_x_proj[1], cam_x_proj[3]) + half_window - min_cam_x + 1;
			int sub_y_cam_height = umax(cam_y_proj[2], cam_y_proj[3]) + half_window - min_cam_y + 1;
			int shared_memory_flag = 1;

			if (sub_y_cam_width * sub_y_cam_height > 2 * padding_length * padding_length) {
				shared_memory_flag = 0;
			}

			// fill the projected padding
			if (shared_memory_flag == 1) {
				sub_y_cam = &float_sub_y_ref[padding_length * padding_length];
				int p = threadIdx.x + threadIdx.y * shared_width;

				while (p < sub_y_cam_width * sub_y_cam_height) {
					int cam_x = min_cam_x + p % sub_y_cam_width;
					int cam_y = min_cam_y + p / sub_y_cam_width;
					if (cam_x < 0 || cam_y < 0 || cam_x >= width || cam_y >= height) {
						p += shared_width * shared_height;
						continue;
					}
					sub_y_cam[p] = y_cam[MI3(cam_x, cam_y, cam_n, width, height)];
					p += shared_width * shared_height;
				}
			}
			__syncthreads();

			// (ii) calculate cost against reference
			// Calculating cost in a window
			float cost = 0.0f;
			float cc = 0.0f;
			for (int k = -(half_window); k <= half_window; k++)
			{
				for (int l = -(half_window); l <= half_window; l++)
				{
					if (x + l < 0 || x + l >= width)
						continue;
					if (y + k < 0 || y + k >= height)
						continue;
					if (x_proj2 + l < 0 || x_proj2 + l >= width)
						continue;
					if (y_proj2 + k < 0 || y_proj2 + k >= height)
						continue;


					if (shared_memory_flag == 1) {
						if (x_proj2 - min_cam_x + l >= 0 && x_proj2 - min_cam_x + l < sub_y_cam_width && y_proj2 - min_cam_y + k >= 0 && y_proj2 - min_cam_y + k < sub_y_cam_height) {
							cost += fabsf(float_sub_y_ref[MI(padding_x + l, padding_y + k, padding_length)] - sub_y_cam[MI(x_proj2 - min_cam_x + l, y_proj2 - min_cam_y + k, sub_y_cam_width)]);
						}
					}
					else {
						cost += fabsf(float_sub_y_ref[MI(padding_x + l, padding_y + k, padding_length)] - y_cam[MI3(x_proj2 + l, y_proj2 + k, cam_n, width, height)]);
					}
					cc += 1.0f;
				}
			}
			cost /= cc;

			//  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
			// only the minimum cost for all the cameras is stored
			if (cam_n == 0 && zi == 0) {
				if (cost < 255.0) {
					cost_cube[MI3(x, y, 0, width, height)] = (float)zi;
					cost_cube[MI3(x, y, 1, width, height)] = cost;
				}
				else {
					cost_cube[MI3(x, y, 0, width, height)] = 255.0;
					cost_cube[MI3(x, y, 1, width, height)] = 255.0;
				}
			}
			else if (cost_cube[MI3(x, y, 1, width, height)] > cost){
				cost_cube[MI3(x, y, 0, width, height)] = (float)zi;
				cost_cube[MI3(x, y, 1, width, height)] = cost;
			}
			else if (cost_cube[MI3(x, y, 1, width, height)] == cost && cost_cube[MI3(x, y, 0, width, height)] > zi) {
				cost_cube[MI3(x, y, 0, width, height)] = (float)zi;
			}

			__syncthreads();
		}
	}
}

__global__ void compute_reduced_uint8_t_all_cost_no_fill_better_pad_smart_full_shared_full_float_2D(float* cost_cube, uint8_t* depth, const float* y_ref, const float* y_cam)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ int width;
	__shared__ int height;
	__shared__ float znear;
	__shared__ float zfar;
	__shared__ float ZPlanes;
	__shared__ int half_window;
	__shared__ float K[9];
	__shared__ float R[9];
	__shared__ float t[3];
	__shared__ float inv_K[9];
	__shared__ float inv_R[9];
	__shared__ float inv_t[3];
	__shared__ int cam_x_proj[4];
	__shared__ int cam_y_proj[4];
	__shared__ float* sub_y_cam;
	__shared__ int shared_width;
	__shared__ int shared_height;

	extern __shared__ float float_sub_y_ref[];

	if (x >= *const_width || y >= *const_height)
		return;

	// Fill shared memory
	if (threadIdx.x == 0) {
		if (threadIdx.y == 0) width = *const_width;
		if (threadIdx.y == 1) height = *const_height;
		if (threadIdx.y == 3) znear = *const_znear;
		if (threadIdx.y == 4) zfar = *const_zfar;
		if (threadIdx.y == 5) ZPlanes = *const_ZPlanes;
		if (threadIdx.y == 6) half_window = *const_half_window;
	}
	else if (threadIdx.x == 1 && threadIdx.y < 9) K[threadIdx.y] = const_K[threadIdx.y];
	else if (threadIdx.x == 2 && threadIdx.y < 9) R[threadIdx.y] = const_R[threadIdx.y];
	else if (threadIdx.x == 3 && threadIdx.y < 3) t[threadIdx.y] = const_t[threadIdx.y];
	else if (threadIdx.x == 4 && threadIdx.y < 9) inv_K[threadIdx.y] = const_inv_K[threadIdx.y];
	else if (threadIdx.x == 5 && threadIdx.y < 9) inv_R[threadIdx.y] = const_inv_R[threadIdx.y];
	else if (threadIdx.x == 6 && threadIdx.y < 3) inv_t[threadIdx.y] = const_inv_t[threadIdx.y];


	for (int cam_n = 0; cam_n < *const_cam_count; cam_n++) {

		if (cam_n != 0) {
			if (threadIdx.x == 1 && threadIdx.y < 9) K[threadIdx.y] = const_K[MI(threadIdx.y, cam_n, 9)];
			else if (threadIdx.x == 2 && threadIdx.y < 9) R[threadIdx.y] = const_R[MI(threadIdx.y, cam_n, 9)];
			else if (threadIdx.x == 3 && threadIdx.y < 3) t[threadIdx.y] = const_t[MI(threadIdx.y, cam_n, 3)];
		}

		__syncthreads();

		// Compute ref padding parameters
		int padding_length = N_THREADS + 2 * half_window;
		int padding_x = half_window + threadIdx.x;
		int padding_y = half_window + threadIdx.y;

		int shared_ref_width;
		int shared_ref_height;

		if ((blockIdx.x + 1) * N_THREADS > width)
			shared_ref_width = width - blockIdx.x * N_THREADS;
		else
			shared_ref_width = N_THREADS;
		if ((blockIdx.y + 1) * N_THREADS > height)
			shared_ref_height = height - blockIdx.y * N_THREADS;
		else
			shared_ref_height = N_THREADS;

		// Fill ref padding
		int p = threadIdx.x + threadIdx.y * shared_ref_width;

		while (p < padding_length * padding_length) {
			int cam_x = MI(p % padding_length - half_window, blockIdx.x, N_THREADS);
			int cam_y = MI(p / padding_length - half_window, blockIdx.y, N_THREADS);
			if (cam_x < 0 || cam_y < 0 || cam_x >= width || cam_y >= height) {
				p += shared_ref_width * shared_ref_height;
				continue;
			}
			float_sub_y_ref[p] = y_ref[MI(cam_x, cam_y, width)];
			p += shared_ref_width * shared_ref_height;
		}


		__syncthreads();

		for (int zi = 0; zi < 256; zi++) {

			// Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
			float z = znear * zfar / (znear + ((zi / ZPlanes) * (zfar - znear))); //need to be in the x and y loops?

			// 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
			float X_ref = (inv_K[0] * x + inv_K[1] * y + inv_K[2]) * z;
			float Y_ref = (inv_K[3] * x + inv_K[4] * y + inv_K[5]) * z;
			float Z_ref = (inv_K[6] * x + inv_K[7] * y + inv_K[8]) * z;

			// 3D in ref camera coordinates to 3D world
			float X = inv_R[0] * X_ref + inv_R[1] * Y_ref + inv_R[2] * Z_ref - inv_t[0];
			float Y = inv_R[3] * X_ref + inv_R[4] * Y_ref + inv_R[5] * Z_ref - inv_t[1];
			float Z = inv_R[6] * X_ref + inv_R[7] * Y_ref + inv_R[8] * Z_ref - inv_t[2];

			// 3D world to projected camera 3D coordinates
			float X_proj = R[0] * X + R[1] * Y + R[2] * Z - t[0];
			float Y_proj = R[3] * X + R[4] * Y + R[5] * Z - t[1];
			float Z_proj = R[6] * X + R[7] * Y + R[8] * Z - t[2];

			// Projected camera 3D coordinates to projected camera 2D coordinates
			float x_proj = (K[0] * X_proj / Z_proj + K[1] * Y_proj / Z_proj + K[2]);
			float y_proj = (K[3] * X_proj / Z_proj + K[4] * Y_proj / Z_proj + K[5]);

			int x_proj2 = x_proj < 0 || x_proj >= width ? 0 : (int)roundf(x_proj);
			int y_proj2 = y_proj < 0 || y_proj >= height ? 0 : (int)roundf(y_proj);

			// Find projected padding corners
			if (threadIdx.x == 0) {
				if (threadIdx.y == 0) {
					cam_x_proj[0] = x_proj2;
					cam_y_proj[0] = y_proj2;
				}
				else if (threadIdx.y == N_THREADS - 1 || y == height - 1) {
					cam_x_proj[1] = x_proj2;
					cam_y_proj[1] = y_proj2;
				}
			}
			else if (threadIdx.x == N_THREADS - 1 || x == width - 1) {
				if (threadIdx.y == 0) {
					cam_x_proj[2] = x_proj2;
					cam_y_proj[2] = y_proj2;
				}
				else if (threadIdx.y == N_THREADS - 1 || y == height - 1) {
					cam_x_proj[3] = x_proj2;
					cam_y_proj[3] = y_proj2;
					shared_width = threadIdx.x;
					shared_height = threadIdx.y;
				}
			}

			__syncthreads();

			// Compute projected padding parameters
			int min_cam_x = umin(cam_x_proj[0], cam_x_proj[2]) - half_window;
			int min_cam_y = umin(cam_y_proj[0], cam_y_proj[1]) - half_window;
			int sub_y_cam_width = umax(cam_x_proj[1], cam_x_proj[3]) + half_window - min_cam_x + 1;
			int sub_y_cam_height = umax(cam_y_proj[2], cam_y_proj[3]) + half_window - min_cam_y + 1;
			int shared_memory_flag = 1;

			if (sub_y_cam_width * sub_y_cam_height > 2 * padding_length * padding_length) {
				shared_memory_flag = 0;
			}

			// fill the projected padding
			if (shared_memory_flag == 1) {
				sub_y_cam = &float_sub_y_ref[padding_length * padding_length];
				int p = threadIdx.x + threadIdx.y * shared_width;

				while (p < sub_y_cam_width * sub_y_cam_height) {
					int cam_x = min_cam_x + p % sub_y_cam_width;
					int cam_y = min_cam_y + p / sub_y_cam_width;
					if (cam_x < 0 || cam_y < 0 || cam_x >= width || cam_y >= height) {
						p += shared_width * shared_height;
						continue;
					}
					sub_y_cam[p] = y_cam[MI3(cam_x, cam_y, cam_n, width, height)];
					p += shared_width * shared_height;
				}
			}
			__syncthreads();

			// (ii) calculate cost against reference
			// Calculating cost in a window
			float cost = 0.0f;
			float cc = 0.0f;
			for (int k = -(half_window); k <= half_window; k++)
			{
				for (int l = -(half_window); l <= half_window; l++)
				{
					if (x + l < 0 || x + l >= width)
						continue;
					if (y + k < 0 || y + k >= height)
						continue;
					if (x_proj2 + l < 0 || x_proj2 + l >= width)
						continue;
					if (y_proj2 + k < 0 || y_proj2 + k >= height)
						continue;


					if (shared_memory_flag == 1) {
						if (x_proj2 - min_cam_x + l >= 0 && x_proj2 - min_cam_x + l < sub_y_cam_width && y_proj2 - min_cam_y + k >= 0 && y_proj2 - min_cam_y + k < sub_y_cam_height) {
							cost += fabsf(float_sub_y_ref[MI(padding_x + l, padding_y + k, padding_length)] - sub_y_cam[MI(x_proj2 - min_cam_x + l, y_proj2 - min_cam_y + k, sub_y_cam_width)]);
						}
					}
					else {
						cost += fabsf(float_sub_y_ref[MI(padding_x + l, padding_y + k, padding_length)] - y_cam[MI3(x_proj2 + l, y_proj2 + k, cam_n, width, height)]);
					}
					cc += 1.0f;
				}
			}
			cost /= cc;

			//  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
			// only the minimum cost for all the cameras is stored
			if (cam_n == 0 && zi == 0) {
				if (cost < 255.0) {
					depth[MI(x, y, width)] = zi;
					cost_cube[MI(x, y, width)] = cost;
				}
				else {
					depth[MI(x, y, width)] = 255;
					cost_cube[MI(x, y, width)] = 255.0;
				}
			}
			else if (cost_cube[MI(x, y, width)] > cost) {
				depth[MI(x, y, width)] = zi;
				cost_cube[MI(x, y, width)] = cost;
			}
			else if (cost_cube[MI(x, y, width)] == cost && depth[MI(x, y, width)] > zi) {
				depth[MI(x, y, width)] = zi;
			}

			__syncthreads();
		}
	}
}

__global__ void compute_reduced_uint8_t_all_cost_no_fill_better_pad_less_global_smart_full_shared_full_float_2D(float* cost_cube, uint8_t* depth, const float* y_ref, const float* y_cam)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ int width;
	__shared__ int height;
	__shared__ float znear;
	__shared__ float zfar;
	__shared__ float ZPlanes;
	__shared__ int half_window;
	__shared__ float K[9];
	__shared__ float R[9];
	__shared__ float t[3];
	__shared__ float inv_K[9];
	__shared__ float inv_R[9];
	__shared__ float inv_t[3];
	__shared__ int cam_x_proj[4];
	__shared__ int cam_y_proj[4];
	__shared__ float* sub_y_cam;
	__shared__ int shared_width;
	__shared__ int shared_height;

	extern __shared__ float float_sub_y_ref[];

	if (x >= *const_width || y >= *const_height)
		return;

	float best_cost = 255.0;
	float best_depth = 0.0; 

	// Fill shared memory
	if (threadIdx.x == 0) {
		if (threadIdx.y == 0) width = *const_width;
		if (threadIdx.y == 1) height = *const_height;
		if (threadIdx.y == 3) znear = *const_znear;
		if (threadIdx.y == 4) zfar = *const_zfar;
		if (threadIdx.y == 5) ZPlanes = *const_ZPlanes;
		if (threadIdx.y == 6) half_window = *const_half_window;
	}
	else if (threadIdx.x == 1 && threadIdx.y < 9) K[threadIdx.y] = const_K[threadIdx.y];
	else if (threadIdx.x == 2 && threadIdx.y < 9) R[threadIdx.y] = const_R[threadIdx.y];
	else if (threadIdx.x == 3 && threadIdx.y < 3) t[threadIdx.y] = const_t[threadIdx.y];
	else if (threadIdx.x == 4 && threadIdx.y < 9) inv_K[threadIdx.y] = const_inv_K[threadIdx.y];
	else if (threadIdx.x == 5 && threadIdx.y < 9) inv_R[threadIdx.y] = const_inv_R[threadIdx.y];
	else if (threadIdx.x == 6 && threadIdx.y < 3) inv_t[threadIdx.y] = const_inv_t[threadIdx.y];


	for (int cam_n = 0; cam_n < *const_cam_count; cam_n++) {

		if (cam_n != 0) {
			if (threadIdx.x == 1 && threadIdx.y < 9) K[threadIdx.y] = const_K[MI(threadIdx.y, cam_n, 9)];
			else if (threadIdx.x == 2 && threadIdx.y < 9) R[threadIdx.y] = const_R[MI(threadIdx.y, cam_n, 9)];
			else if (threadIdx.x == 3 && threadIdx.y < 3) t[threadIdx.y] = const_t[MI(threadIdx.y, cam_n, 3)];
		}

		__syncthreads();

		int padding_length = N_THREADS + 2 * half_window;
		int padding_x = half_window + threadIdx.x;
		int padding_y = half_window + threadIdx.y;

		int shared_ref_width;
		int shared_ref_height;

		// Compute ref padding parameters
		if ((blockIdx.x + 1) * N_THREADS > width)
			shared_ref_width = width - blockIdx.x * N_THREADS;
		else
			shared_ref_width = N_THREADS;
		if ((blockIdx.y + 1) * N_THREADS > height)
			shared_ref_height = height - blockIdx.y * N_THREADS;
		else
			shared_ref_height = N_THREADS;

		// Fill ref padding
		int p = threadIdx.x + threadIdx.y * shared_ref_width;

		while (p < padding_length * padding_length) {
			int cam_x = MI(p % padding_length - half_window, blockIdx.x, N_THREADS);
			int cam_y = MI(p / padding_length - half_window, blockIdx.y, N_THREADS);
			if (cam_x < 0 || cam_y < 0 || cam_x >= width || cam_y >= height) {
				p += shared_ref_width * shared_ref_height;
				continue;
			}
			float_sub_y_ref[p] = y_ref[MI(cam_x, cam_y, width)];
			p += shared_ref_width * shared_ref_height;
		}


		__syncthreads();

		for (int zi = 0; zi < 256; zi++) {

			// Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
			float z = znear * zfar / (znear + ((zi / ZPlanes) * (zfar - znear))); //need to be in the x and y loops?

			// 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
			float X_ref = (inv_K[0] * x + inv_K[1] * y + inv_K[2]) * z;
			float Y_ref = (inv_K[3] * x + inv_K[4] * y + inv_K[5]) * z;
			float Z_ref = (inv_K[6] * x + inv_K[7] * y + inv_K[8]) * z;

			// 3D in ref camera coordinates to 3D world
			float X = inv_R[0] * X_ref + inv_R[1] * Y_ref + inv_R[2] * Z_ref - inv_t[0];
			float Y = inv_R[3] * X_ref + inv_R[4] * Y_ref + inv_R[5] * Z_ref - inv_t[1];
			float Z = inv_R[6] * X_ref + inv_R[7] * Y_ref + inv_R[8] * Z_ref - inv_t[2];

			// 3D world to projected camera 3D coordinates
			float X_proj = R[0] * X + R[1] * Y + R[2] * Z - t[0];
			float Y_proj = R[3] * X + R[4] * Y + R[5] * Z - t[1];
			float Z_proj = R[6] * X + R[7] * Y + R[8] * Z - t[2];

			// Projected camera 3D coordinates to projected camera 2D coordinates
			float x_proj = (K[0] * X_proj / Z_proj + K[1] * Y_proj / Z_proj + K[2]);
			float y_proj = (K[3] * X_proj / Z_proj + K[4] * Y_proj / Z_proj + K[5]);

			int x_proj2 = x_proj < 0 || x_proj >= width ? 0 : (int)roundf(x_proj);
			int y_proj2 = y_proj < 0 || y_proj >= height ? 0 : (int)roundf(y_proj);

			// Find projected padding corners
			if (threadIdx.x == 0) {
				if (threadIdx.y == 0) {
					cam_x_proj[0] = x_proj2;
					cam_y_proj[0] = y_proj2;
				}
				else if (threadIdx.y == N_THREADS - 1 || y == height - 1) {
					cam_x_proj[1] = x_proj2;
					cam_y_proj[1] = y_proj2;
				}
			}
			else if (threadIdx.x == N_THREADS - 1 || x == width - 1) {
				if (threadIdx.y == 0) {
					cam_x_proj[2] = x_proj2;
					cam_y_proj[2] = y_proj2;
				}
				else if (threadIdx.y == N_THREADS - 1 || y == height - 1) {
					cam_x_proj[3] = x_proj2;
					cam_y_proj[3] = y_proj2;
					shared_width = threadIdx.x;
					shared_height = threadIdx.y;
				}
			}

			__syncthreads();

			// Compute projected padding parameters
			int min_cam_x = umin(cam_x_proj[0], cam_x_proj[2]) - half_window;
			int min_cam_y = umin(cam_y_proj[0], cam_y_proj[1]) - half_window;
			int sub_y_cam_width = umax(cam_x_proj[1], cam_x_proj[3]) + half_window - min_cam_x + 1;
			int sub_y_cam_height = umax(cam_y_proj[2], cam_y_proj[3]) + half_window - min_cam_y + 1;
			int shared_memory_flag = 1;

			if (sub_y_cam_width * sub_y_cam_height > 2 * padding_length * padding_length) {
				shared_memory_flag = 0;
			}

			// fill the projected padding
			if (shared_memory_flag == 1) {
				sub_y_cam = &float_sub_y_ref[padding_length * padding_length];
				int p = threadIdx.x + threadIdx.y * shared_width;

				while (p < sub_y_cam_width * sub_y_cam_height) {
					int cam_x = min_cam_x + p % sub_y_cam_width;
					int cam_y = min_cam_y + p / sub_y_cam_width;
					if (cam_x < 0 || cam_y < 0 || cam_x >= width || cam_y >= height) {
						p += shared_width * shared_height;
						continue;
					}
					sub_y_cam[p] = y_cam[MI3(cam_x, cam_y, cam_n, width, height)];
					p += shared_width * shared_height;
				}
			}

			__syncthreads();

			// (ii) calculate cost against reference
			// Calculating cost in a window
			float cost = 0.0f;
			float cc = 0.0f;
			for (int k = -(half_window); k <= half_window; k++)
			{
				for (int l = -(half_window); l <= half_window; l++)
				{
					if (x + l < 0 || x + l >= width)
						continue;
					if (y + k < 0 || y + k >= height)
						continue;
					if (x_proj2 + l < 0 || x_proj2 + l >= width)
						continue;
					if (y_proj2 + k < 0 || y_proj2 + k >= height)
						continue;


					if (shared_memory_flag == 1) {
						if (x_proj2 - min_cam_x + l >= 0 && x_proj2 - min_cam_x + l < sub_y_cam_width && y_proj2 - min_cam_y + k >= 0 && y_proj2 - min_cam_y + k < sub_y_cam_height) {
							cost += fabsf(float_sub_y_ref[MI(padding_x + l, padding_y + k, padding_length)] - sub_y_cam[MI(x_proj2 - min_cam_x + l, y_proj2 - min_cam_y + k, sub_y_cam_width)]);
						}
					}
					else {
						cost += fabsf(float_sub_y_ref[MI(padding_x + l, padding_y + k, padding_length)] - y_cam[MI3(x_proj2 + l, y_proj2 + k, cam_n, width, height)]);
					}
					cc += 1.0f;
				}
			}
			cost /= cc;

			//  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
			// only the minimum cost for all the cameras is stored
			if (cam_n == 0 && zi == 0) {
				if (cost < 255.0) {
					best_depth = zi;
					best_cost = cost;
				}
				else {
					best_depth = 255;
					best_cost = 255.0;
				}
			}
			else if (best_cost > cost) {
				best_depth = zi;
				best_cost = cost;
			}
			else if (best_cost == cost && best_depth > zi) {
				best_depth = zi;
			}

			__syncthreads();
		}
	}
	depth[MI(x, y, width)] = (uint8_t) best_depth;
}

//__global__ void compute_reduced_uint8_t_all_cost_no_fill_better_pad_less_global_smart_full_shared_full_float_2D(float* cost_cube, uint8_t* depth, const float* y_ref, const float* y_cam)
//{
//	int x = blockIdx.x * blockDim.x + threadIdx.x;
//	int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//	__shared__ int width;
//	__shared__ int height;
//	__shared__ float znear;
//	__shared__ float zfar;
//	__shared__ float ZPlanes;
//	__shared__ int half_window;
//	__shared__ float K[9];
//	__shared__ float R[9];
//	__shared__ float t[3];
//	__shared__ float inv_K[9];
//	__shared__ float inv_R[9];
//	__shared__ float inv_t[3];
//	__shared__ int cam_x_proj[4];
//	__shared__ int cam_y_proj[4];
//	__shared__ float* sub_y_cam;
//	__shared__ int shared_width;
//	__shared__ int shared_height;
//
//	extern __shared__ float float_sub_y_ref[];
//
//	if (x >= *const_width || y >= *const_height)
//		return;
//
//	float best_cost = 255.0;
//	float best_depth = 0.0;
//
//	// Fill shared memory
//	if (threadIdx.x == 0) {
//		if (threadIdx.y == 0) width = *const_width;
//		else if (threadIdx.y == 1) height = *const_height;
//		else if (threadIdx.y == 2) znear = *const_znear;
//		else if (threadIdx.y == 3) zfar = *const_zfar;
//		else if (threadIdx.y == 4) ZPlanes = *const_ZPlanes;
//		else if (threadIdx.y == 5) half_window = *const_half_window;
//	}
//	else if (threadIdx.x == 1 && threadIdx.y < 9) K[threadIdx.y] = const_K[threadIdx.y];
//	else if (threadIdx.x == 2 && threadIdx.y < 9) R[threadIdx.y] = const_R[threadIdx.y];
//	else if (threadIdx.x == 3 && threadIdx.y < 3) t[threadIdx.y] = const_t[threadIdx.y];
//	else if (threadIdx.x == 4 && threadIdx.y < 9) inv_K[threadIdx.y] = const_inv_K[threadIdx.y];
//	else if (threadIdx.x == 5 && threadIdx.y < 9) inv_R[threadIdx.y] = const_inv_R[threadIdx.y];
//	else if (threadIdx.x == 6 && threadIdx.y < 3) inv_t[threadIdx.y] = const_inv_t[threadIdx.y];
//
//	int padding_length = N_THREADS + 2 * half_window;
//	int padding_x = half_window + threadIdx.x;
//	int padding_y = half_window + threadIdx.y;
//
//	int shared_ref_width;
//	int shared_ref_height;
//
//	// Compute ref padding parameters
//	if ((blockIdx.x + 1) * N_THREADS > *const_width)
//		shared_ref_width = *const_width - blockIdx.x * N_THREADS;
//	else
//		shared_ref_width = N_THREADS;
//	if ((blockIdx.y + 1) * N_THREADS > *const_height)
//		shared_ref_height = *const_heightt - blockIdx.y * N_THREADS;
//	else
//		shared_ref_height = N_THREADS;
//
//	// Fill ref padding
//	int p = threadIdx.x + threadIdx.y * shared_ref_width;
//
//	while (p < padding_length * padding_length) {
//		int cam_x = MI(p % padding_length - half_window, blockIdx.x, N_THREADS);
//		int cam_y = MI(p / padding_length - half_window, blockIdx.y, N_THREADS);
//		if (cam_x < 0 || cam_y < 0 || cam_x >= width || cam_y >= height) {
//			p += shared_ref_width * shared_ref_height;
//			continue;
//		}
//		float_sub_y_ref[p] = y_ref[MI(cam_x, cam_y, width)];
//		p += shared_ref_width * shared_ref_height;
//	}
//
//	__syncthreads();
//
//	for (int cam_n = 0; cam_n < *const_cam_count; cam_n++) {
//
//		if (cam_n != 0) {
//			if (threadIdx.x == 1 && threadIdx.y < 9) K[threadIdx.y] = const_K[MI(threadIdx.y, cam_n, 9)];
//			else if (threadIdx.x == 2 && threadIdx.y < 9) R[threadIdx.y] = const_R[MI(threadIdx.y, cam_n, 9)];
//			else if (threadIdx.x == 3 && threadIdx.y < 3) t[threadIdx.y] = const_t[MI(threadIdx.y, cam_n, 3)];
//		}
//
//		__syncthreads();
//
//		for (int zi = 0; zi < 256; zi++) {
//
//			// Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
//			float z = znear * zfar / (znear + ((zi / ZPlanes) * (zfar - znear))); //need to be in the x and y loops?
//
//			// 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
//			float X_ref = (inv_K[0] * x + inv_K[1] * y + inv_K[2]) * z;
//			float Y_ref = (inv_K[3] * x + inv_K[4] * y + inv_K[5]) * z;
//			float Z_ref = (inv_K[6] * x + inv_K[7] * y + inv_K[8]) * z;
//
//			// 3D in ref camera coordinates to 3D world
//			float X = inv_R[0] * X_ref + inv_R[1] * Y_ref + inv_R[2] * Z_ref - inv_t[0];
//			float Y = inv_R[3] * X_ref + inv_R[4] * Y_ref + inv_R[5] * Z_ref - inv_t[1];
//			float Z = inv_R[6] * X_ref + inv_R[7] * Y_ref + inv_R[8] * Z_ref - inv_t[2];
//
//			// 3D world to projected camera 3D coordinates
//			float X_proj = const_R[MI(0, cam_n, 9)] * X + const_R[MI(1, cam_n, 9)] * Y + const_R[MI(2, cam_n, 9)] * Z - const_t[MI(0, cam_n, 3)];
//			float Y_proj = const_R[MI(3, cam_n, 9)] * X + const_R[MI(4, cam_n, 9)] * Y + const_R[MI(5, cam_n, 9)] * Z - const_t[MI(1, cam_n, 3)];
//			float Z_proj = const_R[MI(6, cam_n, 9)] * X + const_R[MI(7, cam_n, 9)] * Y + const_R[MI(8, cam_n, 9)] * Z - const_t[MI(2, cam_n, 3)];
//
//			// Projected camera 3D coordinates to projected camera 2D coordinates
//			float x_proj = (const_K[MI(0, cam_n, 9)] * X_proj / Z_proj + const_K[MI(1, cam_n, 9)] * Y_proj / Z_proj + const_K[MI(2, cam_n, 9)]);
//			float y_proj = (const_K[MI(3, cam_n, 9)] * X_proj / Z_proj + const_K[MI(4, cam_n, 9)] * Y_proj / Z_proj + const_K[MI(5, cam_n, 9)]);
//
//			int x_proj2 = x_proj < 0 || x_proj >= width ? 0 : (int)roundf(x_proj);
//			int y_proj2 = y_proj < 0 || y_proj >= height ? 0 : (int)roundf(y_proj);
//
//			// Find projected padding corners
//			if (threadIdx.x == 0) {
//				if (threadIdx.y == 0) {
//					cam_x_proj[0] = x_proj2;
//					cam_y_proj[0] = y_proj2;
//				}
//				else if (threadIdx.y == N_THREADS - 1 || y == height - 1) {
//					cam_x_proj[1] = x_proj2;
//					cam_y_proj[1] = y_proj2;
//				}
//			}
//			else if (threadIdx.x == N_THREADS - 1 || x == width - 1) {
//				if (threadIdx.y == 0) {
//					cam_x_proj[2] = x_proj2;
//					cam_y_proj[2] = y_proj2;
//				}
//				else if (threadIdx.y == N_THREADS - 1 || y == height - 1) {
//					cam_x_proj[3] = x_proj2;
//					cam_y_proj[3] = y_proj2;
//					shared_width = threadIdx.x;
//					shared_height = threadIdx.y;
//				}
//			}
//
//			__syncthreads();
//
//			// Compute projected padding parameters
//			int min_cam_x = umin(cam_x_proj[0], cam_x_proj[2]) - half_window;
//			int min_cam_y = umin(cam_y_proj[0], cam_y_proj[1]) - half_window;
//			int sub_y_cam_width = umax(cam_x_proj[1], cam_x_proj[3]) + half_window - min_cam_x + 1;
//			int sub_y_cam_height = umax(cam_y_proj[2], cam_y_proj[3]) + half_window - min_cam_y + 1;
//			int shared_memory_flag = 1;
//
//			if (sub_y_cam_width * sub_y_cam_height > 2 * padding_length * padding_length) {
//				shared_memory_flag = 0;
//			}
//
//			// fill the projected padding
//			if (shared_memory_flag == 1) {
//				sub_y_cam = &float_sub_y_ref[padding_length * padding_length];
//				int p = threadIdx.x + threadIdx.y * shared_width;
//
//				while (p < sub_y_cam_width * sub_y_cam_height) {
//					int cam_x = min_cam_x + p % sub_y_cam_width;
//					int cam_y = min_cam_y + p / sub_y_cam_width;
//					if (cam_x < 0 || cam_y < 0 || cam_x >= width || cam_y >= height) {
//						p += shared_width * shared_height;
//						continue;
//					}
//					sub_y_cam[p] = y_cam[MI3(cam_x, cam_y, cam_n, width, height)];
//					p += shared_width * shared_height;
//				}
//			}
//
//			__syncthreads();
//
//			// (ii) calculate cost against reference
//			// Calculating cost in a window
//			float cost = 0.0f;
//			float cc = 0.0f;
//			for (int k = -(half_window); k <= half_window; k++)
//			{
//				for (int l = -(half_window); l <= half_window; l++)
//				{
//					if (x + l < 0 || x + l >= width)
//						continue;
//					if (y + k < 0 || y + k >= height)
//						continue;
//					if (x_proj2 + l < 0 || x_proj2 + l >= width)
//						continue;
//					if (y_proj2 + k < 0 || y_proj2 + k >= height)
//						continue;
//
//
//					if (shared_memory_flag == 1) {
//						if (x_proj2 - min_cam_x + l >= 0 && x_proj2 - min_cam_x + l < sub_y_cam_width && y_proj2 - min_cam_y + k >= 0 && y_proj2 - min_cam_y + k < sub_y_cam_height) {
//							cost += fabsf(float_sub_y_ref[MI(padding_x + l, padding_y + k, padding_length)] - sub_y_cam[MI(x_proj2 - min_cam_x + l, y_proj2 - min_cam_y + k, sub_y_cam_width)]);
//						}
//					}
//					else {
//						cost += fabsf(float_sub_y_ref[MI(padding_x + l, padding_y + k, padding_length)] - y_cam[MI3(x_proj2 + l, y_proj2 + k, cam_n, width, height)]);
//					}
//					cc += 1.0f;
//				}
//			}
//			cost /= cc;
//
//			//  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
//			// only the minimum cost for all the cameras is stored
//			if (cam_n == 0 && zi == 0) {
//				if (cost < 255.0) {
//					best_depth = zi;
//					best_cost = cost;
//				}
//				else {
//					best_depth = 255;
//					best_cost = 255.0;
//				}
//			}
//			else if (best_cost > cost) {
//				best_depth = zi;
//				best_cost = cost;
//			}
//			else if (best_cost == cost && best_depth > zi) {
//				best_depth = zi;
//			}
//
//			__syncthreads();
//		}
//	}
//	depth[MI(x, y, width)] = (uint8_t)best_depth;
//}



float* frame2frame_matching_naive_baseline(cam &ref, cam &cam_1, cv::Mat &cost_cube_plane, int zi, int half_window)
{
	uint mat_length;
	cv::Mat mat;

	// Only one plane
	mat_length = cost_cube_plane.total() * cost_cube_plane.channels();
	uint im_length = mat_length * 3 / 2;

	// Pass matrices into array
	float* new_cost_cube = new float[mat_length];
	float* mat_arr_plane = cost_cube_plane.isContinuous() ? (float*)cost_cube_plane.data : (float*)cost_cube_plane.clone().data;
	memcpy((void*)new_cost_cube, (void*)mat_arr_plane, mat_length * sizeof(float));

	mat = ref.YUV[0];
	uint8_t* y_ref = new uint8_t[im_length];
	uint8_t* mat_arr = mat.isContinuous() ? (uint8_t*)mat.data : (uint8_t*)mat.clone().data;
	memcpy((void*)y_ref, (void*)mat_arr, im_length * sizeof(uint8_t));

	mat = cam_1.YUV[0];
	uint8_t* y_cam = new uint8_t[im_length];
	mat_arr = mat.isContinuous() ? (uint8_t*)mat.data : (uint8_t*)mat.clone().data;
	memcpy((void*)y_cam, (void*)mat_arr, im_length * sizeof(uint8_t));

	// define pointers
	double* K = &cam_1.p.K[0]; double* R = &cam_1.p.R[0]; double* t = &cam_1.p.t[0];
	double* inv_K = &ref.p.K_inv[0]; double* inv_R = &ref.p.R_inv[0]; double* inv_t = &ref.p.t_inv[0];

	int* dev_width; int* dev_height; int* dev_zi; int* dev_half_window; int* dev_zplanes;
	float* dev_znear; float* dev_zfar; float* dev_cost_cube;
	double* dev_K; double* dev_R; double* dev_t; double* dev_inv_K; double* dev_inv_R; double* dev_inv_t;
	uint8_t* dev_Y_ref; uint8_t* dev_Y_cam;

	CHK(cudaSetDevice(0));

	// Alloc memory on GPU
	CHK(cudaMalloc((void**)&dev_width, sizeof(int)));
	CHK(cudaMalloc((void**)&dev_height, sizeof(int)));
	CHK(cudaMalloc((void**)&dev_zi, sizeof(int)));
	CHK(cudaMalloc((void**)&dev_znear, sizeof(float)));
	CHK(cudaMalloc((void**)&dev_zfar, sizeof(float)));
	CHK(cudaMalloc((void**)&dev_zplanes, sizeof(int)));
	CHK(cudaMalloc((void**)&dev_half_window, sizeof(int)));
	CHK(cudaMalloc((void**)&dev_K, 9 * sizeof(double)));
	CHK(cudaMalloc((void**)&dev_R, 9 * sizeof(double)));
	CHK(cudaMalloc((void**)&dev_t, 3 * sizeof(double)));
	CHK(cudaMalloc((void**)&dev_inv_K, 9 * sizeof(double)));
	CHK(cudaMalloc((void**)&dev_inv_R, 9 * sizeof(double)));
	CHK(cudaMalloc((void**)&dev_inv_t, 3 * sizeof(double)));
	CHK(cudaMalloc((void**)&dev_Y_ref, im_length * sizeof(uint8_t)));
	CHK(cudaMalloc((void**)&dev_Y_cam, im_length * sizeof(uint8_t)));
	CHK(cudaMalloc((void**)&dev_cost_cube, mat_length * sizeof(float)));

	// Transfer data to GPU
	CHK(cudaMemcpy(dev_width, &ref.width, sizeof(int), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_height, &ref.height, sizeof(int), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_zi, &zi, sizeof(int), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_znear, &ZNear, sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_zfar, &ZFar, sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_zplanes, &ZPlanes, sizeof(int), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_half_window, &half_window, sizeof(int), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_K, K, 9 * sizeof(double), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_R, R, 9 * sizeof(double), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_t, t, 3 * sizeof(double), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_inv_K, inv_K, 9 * sizeof(double), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_inv_R, inv_R, 9 * sizeof(double), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_inv_t, inv_t, 3 * sizeof(double), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_Y_ref, y_ref, im_length * sizeof(uint8_t), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_Y_cam, y_cam, im_length * sizeof(uint8_t), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_cost_cube, new_cost_cube, mat_length * sizeof(float), cudaMemcpyHostToDevice));

	int N_threads = 1024;
	dim3 thread_size(N_threads);
	dim3 block_size((mat_length + N_threads - 1) / N_threads);

	compute_cost_naive_baseline << <block_size, thread_size >> > (dev_width, dev_height, dev_zi, dev_znear, dev_zfar, dev_zplanes, dev_half_window, dev_K, 
		dev_R, dev_t, dev_inv_K, dev_inv_R, dev_inv_t, dev_cost_cube, dev_Y_ref, dev_Y_cam);
	
	//CHK(cudaGetLastError());
	cudaGetLastError();
	
	CHK(cudaMemcpy(new_cost_cube, dev_cost_cube, mat_length * sizeof(float), cudaMemcpyDeviceToHost));

	free(y_ref);
	free(y_cam);

	float* result = new_cost_cube;

	CHK(cudaFree(dev_width));
	CHK(cudaFree(dev_height));
	CHK(cudaFree(dev_zi));
	CHK(cudaFree(dev_znear));
	CHK(cudaFree(dev_zfar));
	CHK(cudaFree(dev_zplanes));
	CHK(cudaFree(dev_half_window));
	CHK(cudaFree(dev_K));
	CHK(cudaFree(dev_R));
	CHK(cudaFree(dev_t));
	CHK(cudaFree(dev_inv_K));
	CHK(cudaFree(dev_inv_R));
	CHK(cudaFree(dev_inv_t));
	CHK(cudaFree(dev_Y_ref));
	CHK(cudaFree(dev_Y_cam));
	CHK(cudaFree(dev_cost_cube));

	CHK(cudaDeviceReset());
	return result;
}

float* frame2frame_matching_naive_float(cam& ref, cam& cam_1, cv::Mat& cost_cube_plane, int zi, int half_window)
{
	uint mat_length;
	cv::Mat mat;

	// Only one plane
	mat_length = cost_cube_plane.total() * cost_cube_plane.channels();
	uint im_length = mat_length * 3 / 2;

	// Pass matrices into array
	float* new_cost_cube = new float[mat_length];
	float* mat_arr_plane = cost_cube_plane.isContinuous() ? (float*)cost_cube_plane.data : (float*)cost_cube_plane.clone().data;
	memcpy((void*)new_cost_cube, (void*)mat_arr_plane, mat_length * sizeof(float));

	mat = ref.YUV[0];
	uint8_t* y_ref = new uint8_t[im_length];
	uint8_t* mat_arr = mat.isContinuous() ? (uint8_t*)mat.data : (uint8_t*)mat.clone().data;
	memcpy((void*)y_ref, (void*)mat_arr, im_length * sizeof(uint8_t));

	mat = cam_1.YUV[0];
	uint8_t* y_cam = new uint8_t[im_length];
	mat_arr = mat.isContinuous() ? (uint8_t*)mat.data : (uint8_t*)mat.clone().data;
	memcpy((void*)y_cam, (void*)mat_arr, im_length * sizeof(uint8_t));

	// define pointers
	float K[9];
	float R[9];
	float inv_K[9];
	float inv_R[9];
	for (int i=0; i<9; i++){
		K[i] = (float)cam_1.p.K[i];
		R[i] = (float)cam_1.p.R[i];
		inv_K[i] = (float)ref.p.K_inv[i];
		inv_R[i] = (float)ref.p.R_inv[i];
	}
	float t[3];
	float inv_t[3];
	for (int i = 0; i < 3; i++) {
		t[i] = (float)cam_1.p.t[i];
		inv_t[i] = (float)ref.p.t_inv[i];
	}
	float new_zi = (float) zi;
	float new_ZPlanes = (float) ZPlanes;

	int* dev_width; int* dev_height; int* dev_half_window;
	float* dev_znear; float* dev_zfar; float* dev_cost_cube; float* dev_zi; float* dev_zplanes;
	float* dev_K; float* dev_R; float* dev_t; float* dev_inv_K; float* dev_inv_R; float* dev_inv_t;
	uint8_t* dev_Y_ref; uint8_t* dev_Y_cam;

	CHK(cudaSetDevice(0));

	// Alloc memory on GPU
	CHK(cudaMalloc((void**)&dev_width, sizeof(int)));
	CHK(cudaMalloc((void**)&dev_height, sizeof(int)));
	CHK(cudaMalloc((void**)&dev_zi, sizeof(float)));
	CHK(cudaMalloc((void**)&dev_znear, sizeof(float)));
	CHK(cudaMalloc((void**)&dev_zfar, sizeof(float)));
	CHK(cudaMalloc((void**)&dev_zplanes, sizeof(float)));
	CHK(cudaMalloc((void**)&dev_half_window, sizeof(int)));
	CHK(cudaMalloc((void**)&dev_K, 9 * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_R, 9 * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_t, 3 * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_inv_K, 9 * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_inv_R, 9 * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_inv_t, 3 * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_Y_ref, im_length * sizeof(uint8_t)));
	CHK(cudaMalloc((void**)&dev_Y_cam, im_length * sizeof(uint8_t)));
	CHK(cudaMalloc((void**)&dev_cost_cube, mat_length * sizeof(float)));

	// Transfer data to GPU
	CHK(cudaMemcpy(dev_width, &ref.width, sizeof(int), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_height, &ref.height, sizeof(int), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_zi, &new_zi, sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_znear, &ZNear, sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_zfar, &ZFar, sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_zplanes, &new_ZPlanes, sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_half_window, &half_window, sizeof(int), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_K, K, 9 * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_R, R, 9 * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_t, t, 3 * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_inv_K, inv_K, 9 * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_inv_R, inv_R, 9 * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_inv_t, inv_t, 3 * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_Y_ref, y_ref, im_length * sizeof(uint8_t), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_Y_cam, y_cam, im_length * sizeof(uint8_t), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_cost_cube, new_cost_cube, mat_length * sizeof(float), cudaMemcpyHostToDevice));

	int N_threads = 1024;
	dim3 thread_size(N_threads);
	dim3 block_size((mat_length + N_threads - 1) / N_threads);

	
	compute_cost_naive_float << <block_size, thread_size >> > (dev_width, dev_height, dev_zi, dev_znear, dev_zfar, dev_zplanes, dev_half_window, dev_K,
		dev_R, dev_t, dev_inv_K, dev_inv_R, dev_inv_t, dev_cost_cube, dev_Y_ref, dev_Y_cam);

	cudaGetLastError();

	CHK(cudaMemcpy(new_cost_cube, dev_cost_cube, mat_length * sizeof(float), cudaMemcpyDeviceToHost));

	free(y_ref);
	free(y_cam);

	float* result = new_cost_cube;

	CHK(cudaFree(dev_width));
	CHK(cudaFree(dev_height));
	CHK(cudaFree(dev_zi));
	CHK(cudaFree(dev_znear));
	CHK(cudaFree(dev_zfar));
	CHK(cudaFree(dev_zplanes));
	CHK(cudaFree(dev_half_window));
	CHK(cudaFree(dev_K));
	CHK(cudaFree(dev_R));
	CHK(cudaFree(dev_t));
	CHK(cudaFree(dev_inv_K));
	CHK(cudaFree(dev_inv_R));
	CHK(cudaFree(dev_inv_t));
	CHK(cudaFree(dev_Y_ref));
	CHK(cudaFree(dev_Y_cam));
	CHK(cudaFree(dev_cost_cube));

	CHK(cudaDeviceReset());
	return result;
}

float* frame2frame_matching_naive_float_2D(cam& ref, cam& cam_1, cv::Mat& cost_cube_plane, int zi, int half_window)
{
	uint mat_length;
	cv::Mat mat;

	// Only one plane
	mat_length = cost_cube_plane.total() * cost_cube_plane.channels();
	uint im_length = mat_length * 3 / 2;

	// Pass matrices into array
	float* new_cost_cube = new float[mat_length];
	float* mat_arr_plane = cost_cube_plane.isContinuous() ? (float*)cost_cube_plane.data : (float*)cost_cube_plane.clone().data;
	memcpy((void*)new_cost_cube, (void*)mat_arr_plane, mat_length * sizeof(float));

	mat = ref.YUV[0];
	uint8_t* y_ref = new uint8_t[im_length];
	uint8_t* mat_arr = mat.isContinuous() ? (uint8_t*)mat.data : (uint8_t*)mat.clone().data;
	memcpy((void*)y_ref, (void*)mat_arr, im_length * sizeof(uint8_t));

	mat = cam_1.YUV[0];
	uint8_t* y_cam = new uint8_t[im_length];
	mat_arr = mat.isContinuous() ? (uint8_t*)mat.data : (uint8_t*)mat.clone().data;
	memcpy((void*)y_cam, (void*)mat_arr, im_length * sizeof(uint8_t));

	// define pointers
	float K[9];
	float R[9];
	float inv_K[9];
	float inv_R[9];
	for (int i = 0; i < 9; i++) {
		K[i] = (float)cam_1.p.K[i];
		R[i] = (float)cam_1.p.R[i];
		inv_K[i] = (float)ref.p.K_inv[i];
		inv_R[i] = (float)ref.p.R_inv[i];
	}
	float t[3];
	float inv_t[3];
	for (int i = 0; i < 3; i++) {
		t[i] = (float)cam_1.p.t[i];
		inv_t[i] = (float)ref.p.t_inv[i];
	}
	float new_zi = (float)zi;
	float new_ZPlanes = (float)ZPlanes;

	int* dev_width; int* dev_height; int* dev_half_window;
	float* dev_znear; float* dev_zfar; float* dev_cost_cube; float* dev_zi; float* dev_zplanes;
	float* dev_K; float* dev_R; float* dev_t; float* dev_inv_K; float* dev_inv_R; float* dev_inv_t;
	uint8_t* dev_Y_ref; uint8_t* dev_Y_cam;

	CHK(cudaSetDevice(0));

	// Alloc memory on GPU
	CHK(cudaMalloc((void**)&dev_width, sizeof(int)));
	CHK(cudaMalloc((void**)&dev_height, sizeof(int)));
	CHK(cudaMalloc((void**)&dev_zi, sizeof(float)));
	CHK(cudaMalloc((void**)&dev_znear, sizeof(float)));
	CHK(cudaMalloc((void**)&dev_zfar, sizeof(float)));
	CHK(cudaMalloc((void**)&dev_zplanes, sizeof(float)));
	CHK(cudaMalloc((void**)&dev_half_window, sizeof(int)));
	CHK(cudaMalloc((void**)&dev_K, 9 * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_R, 9 * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_t, 3 * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_inv_K, 9 * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_inv_R, 9 * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_inv_t, 3 * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_Y_ref, im_length * sizeof(uint8_t)));
	CHK(cudaMalloc((void**)&dev_Y_cam, im_length * sizeof(uint8_t)));
	CHK(cudaMalloc((void**)&dev_cost_cube, mat_length * sizeof(float)));

	// Transfer data to GPU
	CHK(cudaMemcpy(dev_width, &ref.width, sizeof(int), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_height, &ref.height, sizeof(int), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_zi, &new_zi, sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_znear, &ZNear, sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_zfar, &ZFar, sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_zplanes, &new_ZPlanes, sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_half_window, &half_window, sizeof(int), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_K, K, 9 * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_R, R, 9 * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_t, t, 3 * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_inv_K, inv_K, 9 * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_inv_R, inv_R, 9 * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_inv_t, inv_t, 3 * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_Y_ref, y_ref, im_length * sizeof(uint8_t), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_Y_cam, y_cam, im_length * sizeof(uint8_t), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_cost_cube, new_cost_cube, mat_length * sizeof(float), cudaMemcpyHostToDevice));

	dim3 thread_size(N_THREADS, N_THREADS);
	dim3 block_size(
		(ref.width + N_THREADS - 1) / N_THREADS,
		(ref.height + N_THREADS - 1) / N_THREADS,
		1);


	compute_cost_naive_float_2D << <block_size, thread_size >> > (dev_width, dev_height, dev_zi, dev_znear, dev_zfar, dev_zplanes, dev_half_window, dev_K,
		dev_R, dev_t, dev_inv_K, dev_inv_R, dev_inv_t, dev_cost_cube, dev_Y_ref, dev_Y_cam);

	cudaGetLastError();

	CHK(cudaMemcpy(new_cost_cube, dev_cost_cube, mat_length * sizeof(float), cudaMemcpyDeviceToHost));

	free(y_ref);
	free(y_cam);

	float* result = new_cost_cube;

	CHK(cudaFree(dev_width));
	CHK(cudaFree(dev_height));
	CHK(cudaFree(dev_zi));
	CHK(cudaFree(dev_znear));
	CHK(cudaFree(dev_zfar));
	CHK(cudaFree(dev_zplanes));
	CHK(cudaFree(dev_half_window));
	CHK(cudaFree(dev_K));
	CHK(cudaFree(dev_R));
	CHK(cudaFree(dev_t));
	CHK(cudaFree(dev_inv_K));
	CHK(cudaFree(dev_inv_R));
	CHK(cudaFree(dev_inv_t));
	CHK(cudaFree(dev_Y_ref));
	CHK(cudaFree(dev_Y_cam));
	CHK(cudaFree(dev_cost_cube));

	CHK(cudaDeviceReset());
	return result;
}

float* frame2frame_matching_partially_shared_float_2D(cam& ref, cam& cam_1, cv::Mat& cost_cube_plane, int zi, int half_window)
{
	uint mat_length;
	cv::Mat mat;

	// Only one plane
	mat_length = cost_cube_plane.total() * cost_cube_plane.channels();
	uint im_length = mat_length * 3 / 2;

	// Pass matrices into array
	float* new_cost_cube = new float[mat_length];
	float* mat_arr_plane = cost_cube_plane.isContinuous() ? (float*)cost_cube_plane.data : (float*)cost_cube_plane.clone().data;
	memcpy((void*)new_cost_cube, (void*)mat_arr_plane, mat_length * sizeof(float));

	mat = ref.YUV[0];
	uint8_t* y_ref = new uint8_t[im_length];
	uint8_t* mat_arr = mat.isContinuous() ? (uint8_t*)mat.data : (uint8_t*)mat.clone().data;
	memcpy((void*)y_ref, (void*)mat_arr, im_length * sizeof(uint8_t));

	mat = cam_1.YUV[0];
	uint8_t* y_cam = new uint8_t[im_length];
	mat_arr = mat.isContinuous() ? (uint8_t*)mat.data : (uint8_t*)mat.clone().data;
	memcpy((void*)y_cam, (void*)mat_arr, im_length * sizeof(uint8_t));

	// define pointers
	float K[9];
	float R[9];
	float inv_K[9];
	float inv_R[9];
	for (int i = 0; i < 9; i++) {
		K[i] = (float)cam_1.p.K[i];
		R[i] = (float)cam_1.p.R[i];
		inv_K[i] = (float)ref.p.K_inv[i];
		inv_R[i] = (float)ref.p.R_inv[i];
	}
	float t[3];
	float inv_t[3];
	for (int i = 0; i < 3; i++) {
		t[i] = (float)cam_1.p.t[i];
		inv_t[i] = (float)ref.p.t_inv[i];
	}
	float new_zi = (float)zi;
	float new_ZPlanes = (float)ZPlanes;

	int* dev_width; int* dev_height; int* dev_half_window;
	float* dev_znear; float* dev_zfar; float* dev_cost_cube; float* dev_zi; float* dev_zplanes;
	float* dev_K; float* dev_R; float* dev_t; float* dev_inv_K; float* dev_inv_R; float* dev_inv_t;
	uint8_t* dev_Y_ref; uint8_t* dev_Y_cam;

	CHK(cudaSetDevice(0));

	// Alloc memory on GPU
	CHK(cudaMalloc((void**)&dev_width, sizeof(int)));
	CHK(cudaMalloc((void**)&dev_height, sizeof(int)));
	CHK(cudaMalloc((void**)&dev_zi, sizeof(float)));
	CHK(cudaMalloc((void**)&dev_znear, sizeof(float)));
	CHK(cudaMalloc((void**)&dev_zfar, sizeof(float)));
	CHK(cudaMalloc((void**)&dev_zplanes, sizeof(float)));
	CHK(cudaMalloc((void**)&dev_half_window, sizeof(int)));
	CHK(cudaMalloc((void**)&dev_K, 9 * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_R, 9 * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_t, 3 * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_inv_K, 9 * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_inv_R, 9 * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_inv_t, 3 * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_Y_ref, im_length * sizeof(uint8_t)));
	CHK(cudaMalloc((void**)&dev_Y_cam, im_length * sizeof(uint8_t)));
	CHK(cudaMalloc((void**)&dev_cost_cube, mat_length * sizeof(float)));

	// Transfer data to GPU
	CHK(cudaMemcpy(dev_width, &ref.width, sizeof(int), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_height, &ref.height, sizeof(int), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_zi, &new_zi, sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_znear, &ZNear, sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_zfar, &ZFar, sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_zplanes, &new_ZPlanes, sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_half_window, &half_window, sizeof(int), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_K, K, 9 * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_R, R, 9 * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_t, t, 3 * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_inv_K, inv_K, 9 * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_inv_R, inv_R, 9 * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_inv_t, inv_t, 3 * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_Y_ref, y_ref, im_length * sizeof(uint8_t), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_Y_cam, y_cam, im_length * sizeof(uint8_t), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_cost_cube, new_cost_cube, mat_length * sizeof(float), cudaMemcpyHostToDevice));

	dim3 thread_size(N_THREADS, N_THREADS);
	dim3 block_size(
		(ref.width + N_THREADS - 1) / N_THREADS,
		(ref.height + N_THREADS - 1) / N_THREADS,
		1);
	int shared_memory_size = (N_THREADS + 2 * half_window);

	compute_cost_partially_shared_float_2D <<<block_size, thread_size, shared_memory_size * shared_memory_size * sizeof(uint8_t)>>> (dev_width, dev_height, dev_zi,
	dev_znear, dev_zfar, dev_zplanes, dev_half_window, dev_K, dev_R, dev_t, dev_inv_K, dev_inv_R, dev_inv_t, dev_cost_cube, dev_Y_ref, dev_Y_cam);

	cudaGetLastError();

	CHK(cudaMemcpy(new_cost_cube, dev_cost_cube, mat_length * sizeof(float), cudaMemcpyDeviceToHost));

	free(y_ref);
	free(y_cam);

	float* result = new_cost_cube;

	CHK(cudaFree(dev_width));
	CHK(cudaFree(dev_height));
	CHK(cudaFree(dev_zi));
	CHK(cudaFree(dev_znear));
	CHK(cudaFree(dev_zfar));
	CHK(cudaFree(dev_zplanes));
	CHK(cudaFree(dev_half_window));
	CHK(cudaFree(dev_K));
	CHK(cudaFree(dev_R));
	CHK(cudaFree(dev_t));
	CHK(cudaFree(dev_inv_K));
	CHK(cudaFree(dev_inv_R));
	CHK(cudaFree(dev_inv_t));
	CHK(cudaFree(dev_Y_ref));
	CHK(cudaFree(dev_Y_cam));
	CHK(cudaFree(dev_cost_cube));

	CHK(cudaDeviceReset());
	return result;
}

float* frame2frame_matching_shared_float_2D(cam& ref, cam& cam_1, cv::Mat& cost_cube_plane, int zi, int half_window)
{
	uint mat_length;
	cv::Mat mat;

	// Only one plane
	mat_length = cost_cube_plane.total() * cost_cube_plane.channels();
	uint im_length = mat_length * 3 / 2;

	// Pass matrices into array
	float* new_cost_cube = new float[mat_length];
	float* mat_arr_plane = cost_cube_plane.isContinuous() ? (float*)cost_cube_plane.data : (float*)cost_cube_plane.clone().data;
	memcpy((void*)new_cost_cube, (void*)mat_arr_plane, mat_length * sizeof(float));

	mat = ref.YUV[0];
	uint8_t* y_ref = new uint8_t[im_length];
	uint8_t* mat_arr = mat.isContinuous() ? (uint8_t*)mat.data : (uint8_t*)mat.clone().data;
	memcpy((void*)y_ref, (void*)mat_arr, im_length * sizeof(uint8_t));

	mat = cam_1.YUV[0];
	uint8_t* y_cam = new uint8_t[im_length];
	mat_arr = mat.isContinuous() ? (uint8_t*)mat.data : (uint8_t*)mat.clone().data;
	memcpy((void*)y_cam, (void*)mat_arr, im_length * sizeof(uint8_t));

	// define pointers
	float K[9];
	float R[9];
	float inv_K[9];
	float inv_R[9];
	for (int i = 0; i < 9; i++) {
		K[i] = (float)cam_1.p.K[i];
		R[i] = (float)cam_1.p.R[i];
		inv_K[i] = (float)ref.p.K_inv[i];
		inv_R[i] = (float)ref.p.R_inv[i];
	}
	float t[3];
	float inv_t[3];
	for (int i = 0; i < 3; i++) {
		t[i] = (float)cam_1.p.t[i];
		inv_t[i] = (float)ref.p.t_inv[i];
	}
	float new_zi = (float)zi;
	float new_ZPlanes = (float)ZPlanes;

	int* dev_width; int* dev_height; int* dev_half_window;
	float* dev_znear; float* dev_zfar; float* dev_cost_cube; float* dev_zi; float* dev_zplanes;
	float* dev_K; float* dev_R; float* dev_t; float* dev_inv_K; float* dev_inv_R; float* dev_inv_t;
	uint8_t* dev_Y_ref; uint8_t* dev_Y_cam;

	CHK(cudaSetDevice(0));

	// Alloc memory on GPU
	CHK(cudaMalloc((void**)&dev_width, sizeof(int)));
	CHK(cudaMalloc((void**)&dev_height, sizeof(int)));
	CHK(cudaMalloc((void**)&dev_zi, sizeof(float)));
	CHK(cudaMalloc((void**)&dev_znear, sizeof(float)));
	CHK(cudaMalloc((void**)&dev_zfar, sizeof(float)));
	CHK(cudaMalloc((void**)&dev_zplanes, sizeof(float)));
	CHK(cudaMalloc((void**)&dev_half_window, sizeof(int)));
	CHK(cudaMalloc((void**)&dev_K, 9 * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_R, 9 * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_t, 3 * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_inv_K, 9 * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_inv_R, 9 * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_inv_t, 3 * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_Y_ref, im_length * sizeof(uint8_t)));
	CHK(cudaMalloc((void**)&dev_Y_cam, im_length * sizeof(uint8_t)));
	CHK(cudaMalloc((void**)&dev_cost_cube, mat_length * sizeof(float)));

	// Transfer data to GPU
	CHK(cudaMemcpy(dev_width, &ref.width, sizeof(int), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_height, &ref.height, sizeof(int), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_zi, &new_zi, sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_znear, &ZNear, sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_zfar, &ZFar, sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_zplanes, &new_ZPlanes, sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_half_window, &half_window, sizeof(int), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_K, K, 9 * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_R, R, 9 * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_t, t, 3 * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_inv_K, inv_K, 9 * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_inv_R, inv_R, 9 * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_inv_t, inv_t, 3 * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_Y_ref, y_ref, im_length * sizeof(uint8_t), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_Y_cam, y_cam, im_length * sizeof(uint8_t), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_cost_cube, new_cost_cube, mat_length * sizeof(float), cudaMemcpyHostToDevice));

	dim3 thread_size(N_THREADS, N_THREADS);
	dim3 block_size(
		(ref.width + N_THREADS - 1) / N_THREADS,
		(ref.height + N_THREADS - 1) / N_THREADS,
		1);
	int shared_memory_size = (N_THREADS + 2 * half_window);

	compute_cost_shared_float_2D << <block_size, thread_size, 3 * shared_memory_size* shared_memory_size * sizeof(uint8_t) >> > (dev_width, dev_height, dev_zi,
		dev_znear, dev_zfar, dev_zplanes, dev_half_window, dev_K, dev_R, dev_t, dev_inv_K, dev_inv_R, dev_inv_t, dev_cost_cube, dev_Y_ref, dev_Y_cam);

	cudaGetLastError();

	CHK(cudaMemcpy(new_cost_cube, dev_cost_cube, mat_length * sizeof(float), cudaMemcpyDeviceToHost));

	free(y_ref);
	free(y_cam);

	float* result = new_cost_cube;

	CHK(cudaFree(dev_width));
	CHK(cudaFree(dev_height));
	CHK(cudaFree(dev_zi));
	CHK(cudaFree(dev_znear));
	CHK(cudaFree(dev_zfar));
	CHK(cudaFree(dev_zplanes));
	CHK(cudaFree(dev_half_window));
	CHK(cudaFree(dev_K));
	CHK(cudaFree(dev_R));
	CHK(cudaFree(dev_t));
	CHK(cudaFree(dev_inv_K));
	CHK(cudaFree(dev_inv_R));
	CHK(cudaFree(dev_inv_t));
	CHK(cudaFree(dev_Y_ref));
	CHK(cudaFree(dev_Y_cam));
	CHK(cudaFree(dev_cost_cube));

	CHK(cudaDeviceReset());
	return result;
}

float* frame2frame_matching_shared_full_float_2D(cam& ref, cam& cam_1, cv::Mat& cost_cube_plane, int zi, int half_window)
{
	uint mat_length;
	cv::Mat mat;

	// Only one plane
	mat_length = cost_cube_plane.total() * cost_cube_plane.channels();
	uint im_length = mat_length * 3 / 2;

	// Pass matrices into array
	float* new_cost_cube = new float[mat_length];
	float* mat_arr_plane = cost_cube_plane.isContinuous() ? (float*)cost_cube_plane.data : (float*)cost_cube_plane.clone().data;
	memcpy((void*)new_cost_cube, (void*)mat_arr_plane, mat_length * sizeof(float));

	ref.YUV[0].convertTo(mat, CV_32F);
	float* y_ref = new float[im_length];
	float* mat_arr = mat.isContinuous() ? (float*)mat.data : (float*)mat.clone().data;
	memcpy((void*)y_ref, (void*)mat_arr, im_length * sizeof(float));

	cam_1.YUV[0].convertTo(mat, CV_32F);
	float* y_cam = new float[im_length];
	mat_arr = mat.isContinuous() ? (float*)mat.data : (float*)mat.clone().data;
	memcpy((void*)y_cam, (void*)mat_arr, im_length * sizeof(float));

	// define pointers
	float K[9];
	float R[9];
	float inv_K[9];
	float inv_R[9];
	for (int i = 0; i < 9; i++) {
		K[i] = (float)cam_1.p.K[i];
		R[i] = (float)cam_1.p.R[i];
		inv_K[i] = (float)ref.p.K_inv[i];
		inv_R[i] = (float)ref.p.R_inv[i];
	}
	float t[3];
	float inv_t[3];
	for (int i = 0; i < 3; i++) {
		t[i] = (float)cam_1.p.t[i];
		inv_t[i] = (float)ref.p.t_inv[i];
	}
	float new_zi = (float)zi;
	float new_ZPlanes = (float)ZPlanes;
	float* dev_cost_cube;
	float* dev_Y_ref; float* dev_Y_cam;

	CHK(cudaSetDevice(0));

	// Alloc memory on GPU
	CHK(cudaMalloc((void**)&dev_Y_ref, im_length * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_Y_cam, im_length * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_cost_cube, mat_length * sizeof(float)));

	// Transfer data to GPU
	CHK(cudaMemcpyToSymbol(const_width, &ref.width, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_height, &ref.height, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_zi, &new_zi, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_znear, &ZNear, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_zfar, &ZFar, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_ZPlanes, &new_ZPlanes, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_half_window, &half_window, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_K, K, 9 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_R, R, 9 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_t, t, 3 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_inv_K, inv_K, 9 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_inv_R, inv_R, 9 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_inv_t, inv_t, 3 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_Y_ref, y_ref, im_length * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_Y_cam, y_cam, im_length * sizeof(float), cudaMemcpyHostToDevice));

	CHK(cudaMemcpy(dev_cost_cube, new_cost_cube, mat_length * sizeof(float), cudaMemcpyHostToDevice));

	dim3 thread_size(N_THREADS, N_THREADS);
	dim3 block_size(
		(ref.width + N_THREADS - 1) / N_THREADS,
		(ref.height + N_THREADS - 1) / N_THREADS,
		1);
	int shared_memory_size = (N_THREADS + 2 * half_window);

	compute_cost_shared_full_float_2D << <block_size, thread_size, 3 * shared_memory_size * shared_memory_size * sizeof(float) >> > (dev_cost_cube, dev_Y_ref, dev_Y_cam);

	cudaGetLastError();

	CHK(cudaMemcpy(new_cost_cube, dev_cost_cube, mat_length * sizeof(float), cudaMemcpyDeviceToHost));

	free(y_ref);
	free(y_cam);

	float* result = new_cost_cube;

	CHK(cudaFree(dev_Y_ref));
	CHK(cudaFree(dev_Y_cam));
	CHK(cudaFree(dev_cost_cube));

	CHK(cudaDeviceReset());
	return result;
}

float* frame2frame_matching_smart_naive_full_float_2D(cam& ref, cam& cam_1, std::vector<cv::Mat>& cost_cube, int half_window)
{
	uint mat_length;
	cv::Mat mat;

	// Only one plane
	mat_length = ref.height * ref.width;
	uint im_length = mat_length * 3 / 2;

	// Pass matrices into array
	float* new_cost_cube = new float[mat_length * ZPlanes];
	for (int i = 0; i < ZPlanes; i++)
	{
		mat = cost_cube[i];
		float* mat_arr = mat.isContinuous() ? (float*)mat.data : (float*)mat.clone().data;
		memcpy((void*)&(new_cost_cube[i * mat_length]), (void*)mat_arr, mat_length * sizeof(float));
	}

	ref.YUV[0].convertTo(mat, CV_32F);
	float* y_ref = new float[im_length];
	float* mat_arr = mat.isContinuous() ? (float*)mat.data : (float*)mat.clone().data;
	memcpy((void*)y_ref, (void*)mat_arr, im_length * sizeof(float));

	cam_1.YUV[0].convertTo(mat, CV_32F);
	float* y_cam = new float[im_length];
	mat_arr = mat.isContinuous() ? (float*)mat.data : (float*)mat.clone().data;
	memcpy((void*)y_cam, (void*)mat_arr, im_length * sizeof(float));

	// define pointers
	float K[9];
	float R[9];
	float inv_K[9];
	float inv_R[9];
	for (int i = 0; i < 9; i++) {
		K[i] = (float)cam_1.p.K[i];
		R[i] = (float)cam_1.p.R[i];
		inv_K[i] = (float)ref.p.K_inv[i];
		inv_R[i] = (float)ref.p.R_inv[i];
	}
	float t[3];
	float inv_t[3];
	for (int i = 0; i < 3; i++) {
		t[i] = (float)cam_1.p.t[i];
		inv_t[i] = (float)ref.p.t_inv[i];
	}
	float new_ZPlanes = (float)ZPlanes;

	float* dev_cost_cube;
	float* dev_Y_ref; float* dev_Y_cam;

	CHK(cudaSetDevice(0));

	// Alloc memory on GPU
	CHK(cudaMalloc((void**)&dev_Y_ref, im_length * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_Y_cam, im_length * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_cost_cube, mat_length * ZPlanes * sizeof(float)));

	// Transfer data to GPU
	CHK(cudaMemcpyToSymbol(const_width, &ref.width, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_height, &ref.height, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_znear, &ZNear, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_zfar, &ZFar, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_ZPlanes, &new_ZPlanes, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_half_window, &half_window, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_K, K, 9 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_R, R, 9 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_t, t, 3 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_inv_K, inv_K, 9 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_inv_R, inv_R, 9 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_inv_t, inv_t, 3 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_Y_ref, y_ref, im_length * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_Y_cam, y_cam, im_length * sizeof(float), cudaMemcpyHostToDevice));

	CHK(cudaMemcpy(dev_cost_cube, new_cost_cube, mat_length * ZPlanes * sizeof(float), cudaMemcpyHostToDevice));

	dim3 thread_size(N_THREADS, N_THREADS);
	dim3 block_size(
		(ref.width + N_THREADS - 1) / N_THREADS,
		(ref.height + N_THREADS - 1) / N_THREADS,
		1);
	int shared_memory_size = (N_THREADS + 2 * half_window);

	compute_cost_smart_naive_full_float_2D << <block_size, thread_size, 3 * shared_memory_size * shared_memory_size * sizeof(float) >> > (dev_cost_cube, dev_Y_ref, dev_Y_cam);

	cudaGetLastError();

	CHK(cudaMemcpy(new_cost_cube, dev_cost_cube, mat_length * ZPlanes * sizeof(float), cudaMemcpyDeviceToHost));

	free(y_ref);
	free(y_cam);

	float* result = new_cost_cube;

	CHK(cudaFree(dev_Y_ref));
	CHK(cudaFree(dev_Y_cam));
	CHK(cudaFree(dev_cost_cube));

	CHK(cudaDeviceReset());

	return result;
}

float* frame2frame_matching_smart_shared_full_float_2D(cam& ref, cam& cam_1, std::vector<cv::Mat>& cost_cube, int half_window)
{	
	uint mat_length;
	cv::Mat mat;
	
	// Only one plane
	mat_length = ref.height * ref.width;
	uint im_length = mat_length * 3 / 2;

	// Pass matrices into array
	float* new_cost_cube = new float[mat_length * ZPlanes];
	for (int i = 0; i < ZPlanes; i++)
	{
		mat = cost_cube[i];
		float* mat_arr = mat.isContinuous() ? (float*)mat.data : (float*)mat.clone().data;
		memcpy((void*)&(new_cost_cube[i * mat_length]), (void*)mat_arr, mat_length * sizeof(float));
	}

	ref.YUV[0].convertTo(mat, CV_32F);
	float* y_ref = new float[im_length];
	float* mat_arr = mat.isContinuous() ? (float*)mat.data : (float*)mat.clone().data;
	memcpy((void*)y_ref, (void*)mat_arr, im_length * sizeof(float));

	cam_1.YUV[0].convertTo(mat, CV_32F);
	float* y_cam = new float[im_length];
	mat_arr = mat.isContinuous() ? (float*)mat.data : (float*)mat.clone().data;
	memcpy((void*)y_cam, (void*)mat_arr, im_length * sizeof(float));

	// define pointers
	float K[9];
	float R[9];
	float inv_K[9];
	float inv_R[9];
	for (int i = 0; i < 9; i++) {
		K[i] = (float)cam_1.p.K[i];
		R[i] = (float)cam_1.p.R[i];
		inv_K[i] = (float)ref.p.K_inv[i];
		inv_R[i] = (float)ref.p.R_inv[i];
	}
	float t[3];
	float inv_t[3];
	for (int i = 0; i < 3; i++) {
		t[i] = (float)cam_1.p.t[i];
		inv_t[i] = (float)ref.p.t_inv[i];
	}
	float new_ZPlanes = (float)ZPlanes;

	float* dev_cost_cube;
	float* dev_Y_ref; float* dev_Y_cam;

	CHK(cudaSetDevice(0));

	// Alloc memory on GPU
	CHK(cudaMalloc((void**)&dev_Y_ref, im_length * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_Y_cam, im_length * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_cost_cube, mat_length * ZPlanes * sizeof(float)));

	// Transfer data to GPU
	CHK(cudaMemcpyToSymbol(const_width, &ref.width, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_height, &ref.height, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_znear, &ZNear, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_zfar, &ZFar, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_ZPlanes, &new_ZPlanes, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_half_window, &half_window, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_K, K, 9 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_R, R, 9 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_t, t, 3 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_inv_K, inv_K, 9 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_inv_R, inv_R, 9 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_inv_t, inv_t, 3 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_Y_ref, y_ref, im_length * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_Y_cam, y_cam, im_length * sizeof(float), cudaMemcpyHostToDevice));

	CHK(cudaMemcpy(dev_cost_cube, new_cost_cube, mat_length * ZPlanes * sizeof(float), cudaMemcpyHostToDevice));

	dim3 thread_size(N_THREADS, N_THREADS);
	dim3 block_size(
		(ref.width + N_THREADS - 1) / N_THREADS,
		(ref.height + N_THREADS - 1) / N_THREADS,
		1);
	int shared_memory_size = (N_THREADS + 2 * half_window);

	compute_cost_smart_shared_full_float_2D << <block_size, thread_size, 3 * shared_memory_size * shared_memory_size * sizeof(float) >> > (dev_cost_cube, dev_Y_ref, dev_Y_cam);

	cudaGetLastError();

	CHK(cudaMemcpy(new_cost_cube, dev_cost_cube, mat_length * ZPlanes * sizeof(float), cudaMemcpyDeviceToHost));

	free(y_ref);
	free(y_cam);

	float* result = new_cost_cube;

	CHK(cudaFree(dev_Y_ref));
	CHK(cudaFree(dev_Y_cam));
	CHK(cudaFree(dev_cost_cube));

	CHK(cudaDeviceReset());
	
	return result;
}

float* frame2frame_matching_smart_full_shared_full_float_2D(cam& ref, cam& cam_1, std::vector<cv::Mat>& cost_cube, int half_window)
{
	uint mat_length;
	cv::Mat mat;

	// Only one plane
	mat_length = ref.height * ref.width;
	uint im_length = mat_length * 3 / 2;

	// Pass matrices into array
	float* new_cost_cube = new float[mat_length * ZPlanes];
	for (int i = 0; i < ZPlanes; i++)
	{
		mat = cost_cube[i];
		float* mat_arr = mat.isContinuous() ? (float*)mat.data : (float*)mat.clone().data;
		memcpy((void*)&(new_cost_cube[i * mat_length]), (void*)mat_arr, mat_length * sizeof(float));
	}

	ref.YUV[0].convertTo(mat, CV_32F);
	float* y_ref = new float[im_length];
	float* mat_arr = mat.isContinuous() ? (float*)mat.data : (float*)mat.clone().data;
	memcpy((void*)y_ref, (void*)mat_arr, im_length * sizeof(float));

	cam_1.YUV[0].convertTo(mat, CV_32F);
	float* y_cam = new float[im_length];
	mat_arr = mat.isContinuous() ? (float*)mat.data : (float*)mat.clone().data;
	memcpy((void*)y_cam, (void*)mat_arr, im_length * sizeof(float));

	// define pointers
	float K[9];
	float R[9];
	float inv_K[9];
	float inv_R[9];
	for (int i = 0; i < 9; i++) {
		K[i] = (float)cam_1.p.K[i];
		R[i] = (float)cam_1.p.R[i];
		inv_K[i] = (float)ref.p.K_inv[i];
		inv_R[i] = (float)ref.p.R_inv[i];
	}
	float t[3];
	float inv_t[3];
	for (int i = 0; i < 3; i++) {
		t[i] = (float)cam_1.p.t[i];
		inv_t[i] = (float)ref.p.t_inv[i];
	}
	float new_ZPlanes = (float)ZPlanes;

	float* dev_cost_cube;
	float* dev_Y_ref; float* dev_Y_cam;

	CHK(cudaSetDevice(0));

	// Alloc memory on GPU
	CHK(cudaMalloc((void**)&dev_Y_ref, im_length * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_Y_cam, im_length * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_cost_cube, mat_length * ZPlanes * sizeof(float)));

	// Transfer data to GPU
	CHK(cudaMemcpyToSymbol(const_width, &ref.width, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_height, &ref.height, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_znear, &ZNear, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_zfar, &ZFar, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_ZPlanes, &new_ZPlanes, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_half_window, &half_window, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_K, K, 9 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_R, R, 9 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_t, t, 3 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_inv_K, inv_K, 9 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_inv_R, inv_R, 9 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_inv_t, inv_t, 3 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_Y_ref, y_ref, im_length * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_Y_cam, y_cam, im_length * sizeof(float), cudaMemcpyHostToDevice));

	CHK(cudaMemcpy(dev_cost_cube, new_cost_cube, mat_length * ZPlanes * sizeof(float), cudaMemcpyHostToDevice));

	dim3 thread_size(N_THREADS, N_THREADS);
	dim3 block_size(
		(ref.width + N_THREADS - 1) / N_THREADS,
		(ref.height + N_THREADS - 1) / N_THREADS,
		1);
	int shared_memory_size = (N_THREADS + 2 * half_window);

	compute_cost_smart_full_shared_full_float_2D << <block_size, thread_size, 3 * shared_memory_size * shared_memory_size * sizeof(float) >> > (dev_cost_cube, dev_Y_ref, dev_Y_cam);

	cudaGetLastError();

	CHK(cudaMemcpy(new_cost_cube, dev_cost_cube, mat_length * ZPlanes * sizeof(float), cudaMemcpyDeviceToHost));

	free(y_ref);
	free(y_cam);

	float* result = new_cost_cube;

	CHK(cudaFree(dev_Y_ref));
	CHK(cudaFree(dev_Y_cam));
	CHK(cudaFree(dev_cost_cube));

	CHK(cudaDeviceReset());

	return result;
}

float* frame2frame_matching_all_smart_full_shared_full_float_2D(cam& ref, std::vector<cam>& cam_vector, std::vector<cv::Mat>& cost_cube, int half_window)
{
	uint mat_length;
	cv::Mat mat;

	// Only one plane
	mat_length = ref.height * ref.width;
	uint im_length = mat_length * 3 / 2;
	const uint cam_count = cam_vector.size() - 1;

	// Pass matrices into array
	float* new_cost_cube = new float[mat_length * ZPlanes];
	for (int i = 0; i < ZPlanes; i++)
	{
		mat = cost_cube[i];
		float* mat_arr = mat.isContinuous() ? (float*)mat.data : (float*)mat.clone().data;
		memcpy((void*)&(new_cost_cube[i * mat_length]), (void*)mat_arr, mat_length * sizeof(float));
	}

	ref.YUV[0].convertTo(mat, CV_32F);
	float* y_ref = new float[im_length];
	float* mat_arr = mat.isContinuous() ? (float*)mat.data : (float*)mat.clone().data;
	memcpy((void*)y_ref, (void*)mat_arr, im_length * sizeof(float));

	float* y_cams = new float[im_length * cam_count];
	for (int i = 1; i <= cam_count; i++)
	{
		cam_vector.at(i).YUV[0].convertTo(mat, CV_32F);
		float* mat_arr = mat.isContinuous() ? (float*)mat.data : (float*)mat.clone().data;
		memcpy((void*)&(y_cams[(i-1) * mat_length]), (void*)mat_arr, mat_length * sizeof(float));
	}

	// define pointers
	float* K = new float[9 * cam_count];
	float* R = new float[9 * cam_count];
	float inv_K[9];
	float inv_R[9];
	float* t = new float[3 * cam_count];
	float inv_t[3];
	for (int i = 0; i < 9; i++) {
		for (int cam_n = 1; cam_n <= cam_count; cam_n++) {
			K[MI(i, cam_n-1, 9)] = (float)cam_vector.at(cam_n).p.K[i];
			R[MI(i, cam_n-1, 9)] = (float)cam_vector.at(cam_n).p.R[i];
		}
		inv_K[i] = (float)ref.p.K_inv[i];
		inv_R[i] = (float)ref.p.R_inv[i];
	}
	for (int i = 0; i < 3; i++) {
		for (int cam_n = 1; cam_n <= cam_count; cam_n++) 
			t[MI(i, cam_n-1, 3)] = (float)cam_vector.at(cam_n).p.t[i];
		inv_t[i] = (float)ref.p.t_inv[i];
	}
	float new_ZPlanes = (float)ZPlanes;

	float* dev_cost_cube;
	float* dev_Y_ref; float* dev_Y_cams;

	CHK(cudaSetDevice(0));

	// Alloc memory on GPU
	CHK(cudaMalloc((void**)&dev_Y_ref, im_length * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_Y_cams, im_length * cam_count * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_cost_cube, mat_length * ZPlanes * sizeof(float)));

	// Transfer data to GPU
	CHK(cudaMemcpyToSymbol(const_width, &ref.width, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_height, &ref.height, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_znear, &ZNear, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_zfar, &ZFar, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_ZPlanes, &new_ZPlanes, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_half_window, &half_window, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_K, K, 9 * cam_count * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_R, R, 9 * cam_count * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_t, t, 3 * cam_count * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_inv_K, inv_K, 9 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_inv_R, inv_R, 9 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_inv_t, inv_t, 3 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_cam_count, &cam_count, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_Y_ref, y_ref, im_length * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_Y_cams, y_cams, im_length * cam_count * sizeof(float), cudaMemcpyHostToDevice));

	CHK(cudaMemcpy(dev_cost_cube, new_cost_cube, mat_length * ZPlanes * sizeof(float), cudaMemcpyHostToDevice));

	dim3 thread_size(N_THREADS, N_THREADS);
	dim3 block_size(
		(ref.width + N_THREADS - 1) / N_THREADS,
		(ref.height + N_THREADS - 1) / N_THREADS,
		1);
	int shared_memory_size = (N_THREADS + 2 * half_window);

	compute_all_cost_smart_full_shared_full_float_2D << <block_size, thread_size, 3 * shared_memory_size * shared_memory_size * sizeof(float) >> > (dev_cost_cube, dev_Y_ref, dev_Y_cams);

	cudaGetLastError();

	CHK(cudaMemcpy(new_cost_cube, dev_cost_cube, mat_length * ZPlanes * sizeof(float), cudaMemcpyDeviceToHost));

	free(y_ref);
	free(y_cams);
	free(K);
	free(R);
	free(t);

	float* result = new_cost_cube;

	CHK(cudaFree(dev_Y_ref));
	CHK(cudaFree(dev_Y_cams));
	CHK(cudaFree(dev_cost_cube));

	CHK(cudaDeviceReset());

	return result;
}

float* frame2frame_matching_all_no_fill_smart_full_shared_full_float_2D(cam& ref, std::vector<cam>& cam_vector, std::vector<cv::Mat>& cost_cube, int half_window)
{
	uint mat_length;
	cv::Mat mat;

	// Only one plane
	mat_length = ref.height * ref.width;
	uint im_length = mat_length * 3 / 2;
	const uint cam_count = cam_vector.size() - 1;

	float* new_cost_cube = new float[mat_length * ZPlanes];

	// Pass matrices into array
	ref.YUV[0].convertTo(mat, CV_32F);
	float* y_ref = new float[im_length];
	float* mat_arr = mat.isContinuous() ? (float*)mat.data : (float*)mat.clone().data;
	memcpy((void*)y_ref, (void*)mat_arr, im_length * sizeof(float));

	float* y_cams = new float[im_length * cam_count];
	for (int i = 1; i <= cam_count; i++)
	{
		cam_vector.at(i).YUV[0].convertTo(mat, CV_32F);
		float* mat_arr = mat.isContinuous() ? (float*)mat.data : (float*)mat.clone().data;
		memcpy((void*)&(y_cams[(i - 1) * mat_length]), (void*)mat_arr, mat_length * sizeof(float));
	}


	// define pointers
	float* K = new float[9 * cam_count];
	float* R = new float[9 * cam_count];
	float inv_K[9];
	float inv_R[9];
	float* t = new float[3 * cam_count];
	float inv_t[3];
	for (int i = 0; i < 9; i++) {
		for (int cam_n = 1; cam_n <= cam_count; cam_n++) {
			K[MI(i, cam_n - 1, 9)] = (float)cam_vector.at(cam_n).p.K[i];
			R[MI(i, cam_n - 1, 9)] = (float)cam_vector.at(cam_n).p.R[i];
		}
		inv_K[i] = (float)ref.p.K_inv[i];
		inv_R[i] = (float)ref.p.R_inv[i];
	}
	for (int i = 0; i < 3; i++) {
		for (int cam_n = 1; cam_n <= cam_count; cam_n++)
			t[MI(i, cam_n - 1, 3)] = (float)cam_vector.at(cam_n).p.t[i];
		inv_t[i] = (float)ref.p.t_inv[i];
	}
	float new_ZPlanes = (float)ZPlanes;

	float* dev_cost_cube;
	float* dev_Y_ref; float* dev_Y_cams;

	CHK(cudaSetDevice(0));

	// Alloc memory on GPU
	CHK(cudaMalloc((void**)&dev_Y_ref, im_length * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_Y_cams, im_length * cam_count * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_cost_cube, mat_length * ZPlanes * sizeof(float)));

	// Transfer data to GPU
	CHK(cudaMemcpyToSymbol(const_width, &ref.width, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_height, &ref.height, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_znear, &ZNear, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_zfar, &ZFar, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_ZPlanes, &new_ZPlanes, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_half_window, &half_window, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_K, K, 9 * cam_count * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_R, R, 9 * cam_count * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_t, t, 3 * cam_count * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_inv_K, inv_K, 9 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_inv_R, inv_R, 9 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_inv_t, inv_t, 3 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_cam_count, &cam_count, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_Y_ref, y_ref, im_length * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_Y_cams, y_cams, im_length * cam_count * sizeof(float), cudaMemcpyHostToDevice));

	dim3 thread_size(N_THREADS, N_THREADS);
	dim3 block_size(
		(ref.width + N_THREADS - 1) / N_THREADS,
		(ref.height + N_THREADS - 1) / N_THREADS,
		1);
	int shared_memory_size = (N_THREADS + 2 * half_window);

	compute_all_cost_no_fill_smart_full_shared_full_float_2D << <block_size, thread_size, 3 * shared_memory_size * shared_memory_size * sizeof(float) >> > (dev_cost_cube, dev_Y_ref, dev_Y_cams);

	cudaGetLastError();

	CHK(cudaMemcpy(new_cost_cube, dev_cost_cube, mat_length * ZPlanes * sizeof(float), cudaMemcpyDeviceToHost));

	free(y_ref);
	free(y_cams);
	free(K);
	free(R);
	free(t);

	float* result = new_cost_cube;

	CHK(cudaFree(dev_Y_ref));
	CHK(cudaFree(dev_Y_cams));
	CHK(cudaFree(dev_cost_cube));

	CHK(cudaDeviceReset());

	return result;
}

float* frame2frame_matching_all_no_fill_better_pad_smart_full_shared_full_float_2D(cam& ref, std::vector<cam>& cam_vector, std::vector<cv::Mat>& cost_cube, int half_window)
{
	uint mat_length;
	cv::Mat mat;

	// Only one plane
	mat_length = ref.height * ref.width;
	uint im_length = mat_length * 3 / 2;
	const uint cam_count = cam_vector.size() - 1;

	float* new_cost_cube = new float[mat_length * ZPlanes];

	// Pass matrices into array
	ref.YUV[0].convertTo(mat, CV_32F);
	float* y_ref = new float[im_length];
	float* mat_arr = mat.isContinuous() ? (float*)mat.data : (float*)mat.clone().data;
	memcpy((void*)y_ref, (void*)mat_arr, im_length * sizeof(float));

	float* y_cams = new float[im_length * cam_count];
	for (int i = 1; i <= cam_count; i++)
	{
		cam_vector.at(i).YUV[0].convertTo(mat, CV_32F);
		float* mat_arr = mat.isContinuous() ? (float*)mat.data : (float*)mat.clone().data;
		memcpy((void*)&(y_cams[(i - 1) * mat_length]), (void*)mat_arr, mat_length * sizeof(float));
	}

	// define pointers
	float* K = new float[9 * cam_count];
	float* R = new float[9 * cam_count];
	float inv_K[9];
	float inv_R[9];
	float* t = new float[3 * cam_count];
	float inv_t[3];
	for (int i = 0; i < 9; i++) {
		for (int cam_n = 1; cam_n <= cam_count; cam_n++) {
			K[MI(i, cam_n - 1, 9)] = (float)cam_vector.at(cam_n).p.K[i];
			R[MI(i, cam_n - 1, 9)] = (float)cam_vector.at(cam_n).p.R[i];
		}
		inv_K[i] = (float)ref.p.K_inv[i];
		inv_R[i] = (float)ref.p.R_inv[i];
	}
	for (int i = 0; i < 3; i++) {
		for (int cam_n = 1; cam_n <= cam_count; cam_n++)
			t[MI(i, cam_n - 1, 3)] = (float)cam_vector.at(cam_n).p.t[i];
		inv_t[i] = (float)ref.p.t_inv[i];
	}
	float new_ZPlanes = (float)ZPlanes;

	float* dev_cost_cube;
	float* dev_Y_ref; float* dev_Y_cams;

	CHK(cudaSetDevice(0));

	// Alloc memory on GPU
	CHK(cudaMalloc((void**)&dev_Y_ref, im_length * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_Y_cams, im_length * cam_count * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_cost_cube, mat_length * ZPlanes * sizeof(float)));

	// Transfer data to GPU
	CHK(cudaMemcpyToSymbol(const_width, &ref.width, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_height, &ref.height, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_znear, &ZNear, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_zfar, &ZFar, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_ZPlanes, &new_ZPlanes, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_half_window, &half_window, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_K, K, 9 * cam_count * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_R, R, 9 * cam_count * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_t, t, 3 * cam_count * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_inv_K, inv_K, 9 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_inv_R, inv_R, 9 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_inv_t, inv_t, 3 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_cam_count, &cam_count, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_Y_ref, y_ref, im_length * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_Y_cams, y_cams, im_length * cam_count * sizeof(float), cudaMemcpyHostToDevice));

	dim3 thread_size(N_THREADS, N_THREADS);
	dim3 block_size(
		(ref.width + N_THREADS - 1) / N_THREADS,
		(ref.height + N_THREADS - 1) / N_THREADS,
		1);
	int shared_memory_size = (N_THREADS + 2 * half_window);

	compute_all_cost_no_fill_better_pad_smart_full_shared_full_float_2D << <block_size, thread_size, 3 * shared_memory_size * shared_memory_size * sizeof(float) >> > (dev_cost_cube, dev_Y_ref, dev_Y_cams);

	cudaGetLastError();

	CHK(cudaMemcpy(new_cost_cube, dev_cost_cube, mat_length * ZPlanes * sizeof(float), cudaMemcpyDeviceToHost));

	free(y_ref);
	free(y_cams);
	free(K);
	free(R);
	free(t);

	float* result = new_cost_cube;

	CHK(cudaFree(dev_Y_ref));
	CHK(cudaFree(dev_Y_cams));
	CHK(cudaFree(dev_cost_cube));

	CHK(cudaDeviceReset());

	return result;
}

float* frame2frame_matching_reduced_float_all_no_fill_better_pad_smart_full_shared_full_float_2D(cam& ref, std::vector<cam>& cam_vector, int half_window)
{
	uint mat_length;
	cv::Mat mat;

	// Only one plane
	mat_length = ref.height * ref.width;
	uint im_length = mat_length * 3 / 2;
	const uint cam_count = cam_vector.size() - 1;

	float* new_cost_cube = new float[mat_length * 2];

	// Pass matrices into array
	ref.YUV[0].convertTo(mat, CV_32F);
	float* y_ref = new float[im_length];
	float* mat_arr = mat.isContinuous() ? (float*)mat.data : (float*)mat.clone().data;
	memcpy((void*)y_ref, (void*)mat_arr, im_length * sizeof(float));

	float* y_cams = new float[im_length * cam_count];
	for (int i = 1; i <= cam_count; i++)
	{
		cam_vector.at(i).YUV[0].convertTo(mat, CV_32F);
		float* mat_arr = mat.isContinuous() ? (float*)mat.data : (float*)mat.clone().data;
		memcpy((void*)&(y_cams[(i - 1) * mat_length]), (void*)mat_arr, mat_length * sizeof(float));
	}


	// define pointers
	float* K = new float[9 * cam_count];
	float* R = new float[9 * cam_count];
	float inv_K[9];
	float inv_R[9];
	float* t = new float[3 * cam_count];
	float inv_t[3];
	for (int i = 0; i < 9; i++) {
		for (int cam_n = 1; cam_n <= cam_count; cam_n++) {
			K[MI(i, cam_n - 1, 9)] = (float)cam_vector.at(cam_n).p.K[i];
			R[MI(i, cam_n - 1, 9)] = (float)cam_vector.at(cam_n).p.R[i];
		}
		inv_K[i] = (float)ref.p.K_inv[i];
		inv_R[i] = (float)ref.p.R_inv[i];
	}
	for (int i = 0; i < 3; i++) {
		for (int cam_n = 1; cam_n <= cam_count; cam_n++)
			t[MI(i, cam_n - 1, 3)] = (float)cam_vector.at(cam_n).p.t[i];
		inv_t[i] = (float)ref.p.t_inv[i];
	}
	float new_ZPlanes = (float)ZPlanes;

	float* dev_cost_cube;
	float* dev_Y_ref; float* dev_Y_cams;

	CHK(cudaSetDevice(0));

	// Alloc memory on GPU
	CHK(cudaMalloc((void**)&dev_Y_ref, im_length * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_Y_cams, im_length * cam_count * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_cost_cube, mat_length * 2 * sizeof(float)));

	// Transfer data to GPU
	CHK(cudaMemcpyToSymbol(const_width, &ref.width, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_height, &ref.height, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_znear, &ZNear, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_zfar, &ZFar, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_ZPlanes, &new_ZPlanes, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_half_window, &half_window, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_K, K, 9 * cam_count * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_R, R, 9 * cam_count * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_t, t, 3 * cam_count * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_inv_K, inv_K, 9 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_inv_R, inv_R, 9 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_inv_t, inv_t, 3 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_cam_count, &cam_count, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_Y_ref, y_ref, im_length * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_Y_cams, y_cams, im_length * cam_count * sizeof(float), cudaMemcpyHostToDevice));

	dim3 thread_size(N_THREADS, N_THREADS);
	dim3 block_size(
		(ref.width + N_THREADS - 1) / N_THREADS,
		(ref.height + N_THREADS - 1) / N_THREADS,
		1);
	int shared_memory_size = (N_THREADS + 2 * half_window);

	compute_reduced_float_all_cost_no_fill_better_pad_smart_full_shared_full_float_2D << <block_size, thread_size, 3 * shared_memory_size * shared_memory_size * sizeof(float) >> > (dev_cost_cube, dev_Y_ref, dev_Y_cams);

	cudaGetLastError();

	CHK(cudaMemcpy(new_cost_cube, dev_cost_cube, mat_length * 2 * sizeof(float), cudaMemcpyDeviceToHost));

	free(y_ref);
	free(y_cams);
	free(K);
	free(R);
	free(t);

	float* result = new_cost_cube;

	CHK(cudaFree(dev_Y_ref));
	CHK(cudaFree(dev_Y_cams));
	CHK(cudaFree(dev_cost_cube));

	CHK(cudaDeviceReset());

	return result;
}

uint8_t* frame2frame_matching_reduced_uint8_t_all_no_fill_better_pad_smart_full_shared_full_float_2D(cam& ref, std::vector<cam>& cam_vector, int half_window)
{
	uint mat_length;
	cv::Mat mat;

	// Only one plane
	mat_length = ref.height * ref.width;
	uint im_length = mat_length * 3 / 2;
	const uint cam_count = cam_vector.size() - 1;

	uint8_t* depth = new uint8_t[mat_length];

	// Pass matrices into array
	ref.YUV[0].convertTo(mat, CV_32F);
	float* y_ref = new float[im_length];
	float* mat_arr = mat.isContinuous() ? (float*)mat.data : (float*)mat.clone().data;
	memcpy((void*)y_ref, (void*)mat_arr, im_length * sizeof(float));

	float* y_cams = new float[im_length * cam_count];
	for (int i = 1; i <= cam_count; i++)
	{
		cam_vector.at(i).YUV[0].convertTo(mat, CV_32F);
		float* mat_arr = mat.isContinuous() ? (float*)mat.data : (float*)mat.clone().data;
		memcpy((void*)&(y_cams[(i - 1) * mat_length]), (void*)mat_arr, mat_length * sizeof(float));
	}

	// define pointers
	float* K = new float[9 * cam_count];
	float* R = new float[9 * cam_count];
	float inv_K[9];
	float inv_R[9];
	float* t = new float[3 * cam_count];
	float inv_t[3];
	for (int i = 0; i < 9; i++) {
		for (int cam_n = 1; cam_n <= cam_count; cam_n++) {
			K[MI(i, cam_n - 1, 9)] = (float)cam_vector.at(cam_n).p.K[i];
			R[MI(i, cam_n - 1, 9)] = (float)cam_vector.at(cam_n).p.R[i];
		}
		inv_K[i] = (float)ref.p.K_inv[i];
		inv_R[i] = (float)ref.p.R_inv[i];
	}
	for (int i = 0; i < 3; i++) {
		for (int cam_n = 1; cam_n <= cam_count; cam_n++)
			t[MI(i, cam_n - 1, 3)] = (float)cam_vector.at(cam_n).p.t[i];
		inv_t[i] = (float)ref.p.t_inv[i];
	}
	float new_ZPlanes = (float)ZPlanes;

	uint8_t* dev_depth;
	float* dev_cost_cube;
	float* dev_Y_ref; float* dev_Y_cams;

	CHK(cudaSetDevice(0));

	// Alloc memory on GPU
	CHK(cudaMalloc((void**)&dev_Y_ref, im_length * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_Y_cams, im_length * cam_count * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_cost_cube, mat_length * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_depth, mat_length * sizeof(uint8_t)));

	// Transfer data to GPU
	CHK(cudaMemcpyToSymbol(const_width, &ref.width, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_height, &ref.height, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_znear, &ZNear, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_zfar, &ZFar, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_ZPlanes, &new_ZPlanes, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_half_window, &half_window, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_K, K, 9 * cam_count * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_R, R, 9 * cam_count * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_t, t, 3 * cam_count * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_inv_K, inv_K, 9 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_inv_R, inv_R, 9 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_inv_t, inv_t, 3 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_cam_count, &cam_count, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_Y_ref, y_ref, im_length * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_Y_cams, y_cams, im_length * cam_count * sizeof(float), cudaMemcpyHostToDevice));

	dim3 thread_size(N_THREADS, N_THREADS);
	dim3 block_size(
		(ref.width + N_THREADS - 1) / N_THREADS,
		(ref.height + N_THREADS - 1) / N_THREADS,
		1);
	int shared_memory_size = (N_THREADS + 2 * half_window);

	compute_reduced_uint8_t_all_cost_no_fill_better_pad_smart_full_shared_full_float_2D << <block_size, thread_size, 3 * shared_memory_size * shared_memory_size * sizeof(float) >> > (dev_cost_cube, dev_depth, dev_Y_ref, dev_Y_cams);

	cudaGetLastError();

	CHK(cudaMemcpy(depth, dev_depth, mat_length * sizeof(uint8_t), cudaMemcpyDeviceToHost));

	free(y_ref);
	free(y_cams);
	free(K);
	free(R);
	free(t);

	uint8_t* result = depth;

	CHK(cudaFree(dev_Y_ref));
	CHK(cudaFree(dev_Y_cams));
	CHK(cudaFree(dev_cost_cube));
	CHK(cudaFree(dev_depth));

	CHK(cudaDeviceReset());

	return result;
}

uint8_t* frame2frame_matching_reduced_uint8_t_all_no_fill_better_pad_less_global_smart_full_shared_full_float_2D(cam& ref, std::vector<cam>& cam_vector, int half_window)
{
	uint mat_length;
	cv::Mat mat;
	// Only one plane
	mat_length = ref.height * ref.width;
	uint im_length = mat_length * 3 / 2;
	const uint cam_count = cam_vector.size() - 1;

	uint8_t* depth = new uint8_t[mat_length];

	// Pass matrices into array
	ref.YUV[0].convertTo(mat, CV_32F);
	float* y_ref = new float[im_length];
	float* mat_arr = mat.isContinuous() ? (float*)mat.data : (float*)mat.clone().data;
	memcpy((void*)y_ref, (void*)mat_arr, im_length * sizeof(float));

	float* y_cams = new float[im_length * cam_count];
	for (int i = 1; i <= cam_count; i++)
	{
		cam_vector.at(i).YUV[0].convertTo(mat, CV_32F);
		float* mat_arr = mat.isContinuous() ? (float*)mat.data : (float*)mat.clone().data;
		memcpy((void*)&(y_cams[(i - 1) * mat_length]), (void*)mat_arr, mat_length * sizeof(float));
	}


	// define pointers
	float* K = new float[9 * cam_count];
	float* R = new float[9 * cam_count];
	float inv_K[9];
	float inv_R[9];
	float* t = new float[3 * cam_count];
	float inv_t[3];
	for (int i = 0; i < 9; i++) {
		for (int cam_n = 1; cam_n <= cam_count; cam_n++) {
			K[MI(i, cam_n - 1, 9)] = (float)cam_vector.at(cam_n).p.K[i];
			R[MI(i, cam_n - 1, 9)] = (float)cam_vector.at(cam_n).p.R[i];
		}
		inv_K[i] = (float)ref.p.K_inv[i];
		inv_R[i] = (float)ref.p.R_inv[i];
	}
	for (int i = 0; i < 3; i++) {
		for (int cam_n = 1; cam_n <= cam_count; cam_n++)
			t[MI(i, cam_n - 1, 3)] = (float)cam_vector.at(cam_n).p.t[i];
		inv_t[i] = (float)ref.p.t_inv[i];
	}
	float new_ZPlanes = (float)ZPlanes;

	uint8_t* dev_depth;
	float* dev_cost_cube;
	float* dev_Y_ref; float* dev_Y_cams;

	CHK(cudaSetDevice(0));

	// Alloc memory on GPU
	CHK(cudaMalloc((void**)&dev_Y_ref, im_length * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_Y_cams, im_length * cam_count * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_cost_cube, mat_length * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_depth, mat_length * sizeof(uint8_t)));

	// Transfer data to GPU
	CHK(cudaMemcpyToSymbol(const_width, &ref.width, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_height, &ref.height, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_znear, &ZNear, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_zfar, &ZFar, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_ZPlanes, &new_ZPlanes, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_half_window, &half_window, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_K, K, 9 * cam_count * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_R, R, 9 * cam_count * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_t, t, 3 * cam_count * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_inv_K, inv_K, 9 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_inv_R, inv_R, 9 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_inv_t, inv_t, 3 * sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpyToSymbol(const_cam_count, &cam_count, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_Y_ref, y_ref, im_length * sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_Y_cams, y_cams, im_length * cam_count * sizeof(float), cudaMemcpyHostToDevice));

	dim3 thread_size(N_THREADS, N_THREADS);
	dim3 block_size(
		(ref.width + N_THREADS - 1) / N_THREADS,
		(ref.height + N_THREADS - 1) / N_THREADS,
		1);
	int shared_memory_size = (N_THREADS + 2 * half_window);

	compute_reduced_uint8_t_all_cost_no_fill_better_pad_less_global_smart_full_shared_full_float_2D << <block_size, thread_size, 3 * shared_memory_size * shared_memory_size * sizeof(float) >> > (dev_cost_cube, dev_depth, dev_Y_ref, dev_Y_cams);

	CHK(cudaMemcpy(depth, dev_depth, mat_length * sizeof(uint8_t), cudaMemcpyDeviceToHost));



	cudaGetLastError();

	free(y_ref);
	free(y_cams);
	free(K);
	free(R);
	free(t);

	uint8_t* result = depth;

	CHK(cudaFree(dev_Y_ref));
	CHK(cudaFree(dev_Y_cams));
	CHK(cudaFree(dev_cost_cube));
	CHK(cudaFree(dev_depth));

	CHK(cudaDeviceReset());

	return result;
}