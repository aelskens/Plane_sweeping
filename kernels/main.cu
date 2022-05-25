#include "main.cuh"

#include <cstdio>
#include <iostream>

#define N_THREADS 32
#define MI(x, y, width) ((x) + (y) * (width))

#define CHK(code) \
do { \
    if ((code) != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s %s %i\n", \
                        cudaGetErrorString((code)), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

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

	dev_test_vecAdd << <1, N >> > (dev_a, dev_b, dev_c, N);

	cudaMemcpy(c, dev_c, N * sizeof(int),
		cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	printf("%s\n", cudaGetErrorString(cudaGetLastError()));

	for (int i = 0; i < N; ++i) {
		printf("%i + %i = %i\n", a[i], b[i], c[i]);
	}
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
	double z_proj = Z_proj;

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
	float z_proj = Z_proj;

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
	float z_proj = Z_proj;

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
	__shared__ int y_cam_pos_x0;
	__shared__ int y_cam_pos_y0;
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
	float z_proj = Z_proj;

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
	__shared__ int y_cam_pos_x0;
	__shared__ int y_cam_pos_y0;
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
	//float z_proj = Z_proj;

	int x_proj2 = x_proj < 0 ? 0 : (x_proj >= width ? width : (int)roundf(x_proj));
	int y_proj2 = y_proj < 0 ? 0 : (y_proj >= height ? height : (int)roundf(y_proj));

	if (threadIdx.x == 0) {
		if (threadIdx.y == 0) {
			cam_x_proj[0] = x_proj2;
			cam_y_proj[0] = y_proj2;
		}
		else if (threadIdx.y == N_THREADS-1) {
			cam_x_proj[1] = x_proj2;
			cam_y_proj[1] = y_proj2;
		}
	}
	else if (threadIdx.x == N_THREADS - 1) {
		if (threadIdx.y == 0) {
			cam_x_proj[2] = x_proj2;
			cam_y_proj[2] = y_proj2;
		}
		else if (threadIdx.y == N_THREADS - 1) {
			cam_x_proj[3] = x_proj2;
			cam_y_proj[3] = y_proj2;
		}
	}

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

	int min_cam_x = umin(cam_x_proj[0], cam_x_proj[2]) - half_window;
	int min_cam_y = umin(cam_y_proj[0], cam_y_proj[1]) - half_window;
	int sub_y_cam_width = umax(cam_x_proj[1], cam_x_proj[3]) + half_window - min_cam_x + 1;
	int sub_y_cam_height = umax(cam_y_proj[2], cam_y_proj[3]) + half_window - min_cam_y + 1;
	int shared_memory_flag = 1;

	if (sub_y_cam_width * sub_y_cam_height > 2 * padding_length * padding_length) {
		//if (threadIdx.x == 0 && threadIdx.y == 0) printf("I am %d %d and in flag 0\n", blockIdx.x, blockIdx.y);
		shared_memory_flag = 0;
	}

	/*if(blockIdx.x==55 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) 
		printf("min x=%d, min y=%d, width=%d, height=%d\n", min_cam_x, min_cam_y, sub_y_cam_width, sub_y_cam_height);*/
	/*if (blockIdx.x == 0 && blockIdx.y == 33 && threadIdx.x == 0 && threadIdx.y == 0)
		printf("cam_x_proj0 = %d, cam_x_proj1 = %d, cam_x_proj2 = %d, cam_x_proj3 = %d, min_cam_x = %d for sub_y_cam_width = %d\n", cam_x_proj[0], cam_x_proj[1], cam_x_proj[2], cam_x_proj[3], min_cam_x, sub_y_cam_width);
	if (blockIdx.x == 0 && blockIdx.y == 33 && threadIdx.x == 0 && threadIdx.y == 0)
		printf("cam_y_proj0 = %d, cam_y_proj1 = %d, cam_y_proj2 = %d, cam_y_proj3 = %d, min_cam_y = %d for sub_y_cam_width = %d\n", cam_y_proj[0], cam_y_proj[1], cam_y_proj[2], cam_y_proj[3], min_cam_y, sub_y_cam_height);*/

	if(shared_memory_flag == 1){
		sub_y_cam = &sub_y_ref[padding_length * padding_length];
		int p = threadIdx.x + threadIdx.y * N_THREADS;

		while(p < sub_y_cam_width * sub_y_cam_height){
			int cam_x = min_cam_x + p % sub_y_cam_width;
			int cam_y = min_cam_y + p / sub_y_cam_width;
			if (cam_x < 0 || cam_y < 0 || cam_x >= width || cam_y >= height) {
				p += N_THREADS * N_THREADS;
				continue;
			}
			sub_y_cam[p] = y_cam[MI(cam_x, cam_y, width)];
			//if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.y == 0 && blockIdx.x == 1) printf("I am block %d, %d x_proj2 = %d, min_cam_x = %d, x_proj2 = %d, min_cam_x = %d, sub_y_cam[p] = %d, y_cam_element = %d \n", blockIdx.x, blockIdx.y, x_proj2, min_cam_x, y_proj2, min_cam_y, sub_y_cam[p], y_cam[MI(x_proj2, y_proj2, width)]);
			p += N_THREADS * N_THREADS;
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
				//cost += fabsf((float)sub_y_ref[MI(padding_x + l, padding_y + k, padding_length)] - (float)sub_y_cam[MI(x_proj2 - min_cam_x + l, y_proj2 - min_cam_y + k, sub_y_cam_width)]);
				//cost += fabsf((float)sub_y_ref[MI(padding_x + l, padding_y + k, padding_length)]);
				if (threadIdx.x == 0 && threadIdx.y == 0){
					//printf("I am block %d, %d  sub_y_cam_element = %d, y_cam_element = %d \n", blockIdx.x, blockIdx.y, sub_y_cam[MI(x_proj2 - min_cam_x + l, y_proj2 - min_cam_y + k, sub_y_cam_width)], y_cam[MI(x_proj2 + l, y_proj2 + k, width)]);
					printf("I am block %d, %d x_proj2 = %d, min_cam_x = %d, x_proj2 = %d, min_cam_x = %d, l = %d, k =  %d, sub_y_cam_element = %d, y_cam_element = %d \n", blockIdx.x, blockIdx.y, x_proj2, min_cam_x, y_proj2, min_cam_y, l, k, sub_y_cam[MI(x_proj2 - min_cam_x + l, y_proj2 - min_cam_y + k, sub_y_cam_width)], y_cam[MI(x_proj2 + l, y_proj2 + k, width)]);
				}
			}
			else 
				cost += fabsf((float)sub_y_ref[MI(padding_x + l, padding_y + k, padding_length)] - (float)y_cam[MI(x_proj2 + l, y_proj2 + k, width)]);
			//cost += fabsf((float) sub_y_ref[MI(padding_x + l, padding_y + k, padding_length)]);
			
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

void test(cv::Mat const& Y) {
	cv::Mat* dev_Y;
	uchar* Y_arr = Y.isContinuous()? Y.data: Y.clone().data;
	
	cudaSetDevice(0);
	cudaMalloc((void**)&dev_Y, sizeof(int));
	cudaMemcpy(&dev_Y, Y_arr, 1920*1080*sizeof(float), cudaMemcpyHostToDevice);


	cv::namedWindow("Y", cv::WindowFlags::WINDOW_AUTOSIZE);
	cv::imshow("Y", Y);
	cv::waitKey(0);
}

//void frame2frame_matching(cam& ref, cam& cam_1, std::vector<cv::Mat> &cost_cube, int zi, int half_window)
float* frame2frame_matching_naive_baseline(cam &ref, cam &cam_1, cv::Mat &cost_cube_plane, int zi, int half_window)
{
	printf("Naive cost frame2frame_matching:\n");

	uint mat_length;
	cv::Mat mat;

	/*// Full cost cube
	mat_length = cost_cube[0].total() * cost_cube[0].channels();
	uint im_length = mat_length * 3 / 2;
	printf("size: %d\n",mat_length);

	float* new_cost_cube = new float[mat_length *ZPlanes];
	for (int i = 0; i < ZPlanes; i++)
	{
		mat = cost_cube[i];
		float* mat_arr = mat.isContinuous() ? (float*)mat.data : (float*)mat.clone().data;
		memcpy((void*) &(new_cost_cube[i* mat_length]),(void*) mat_arr, mat_length * sizeof(float));
	}*/

	// Only one plane
	mat_length = cost_cube_plane.total() * cost_cube_plane.channels();
	uint im_length = mat_length * 3 / 2;

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

	double* K = &cam_1.p.K[0]; double* R = &cam_1.p.R[0]; double* t = &cam_1.p.t[0];
	double* inv_K = &ref.p.K_inv[0]; double* inv_R = &ref.p.R_inv[0]; double* inv_t = &ref.p.t_inv[0];

	int* dev_width; int* dev_height; int* dev_zi; int* dev_half_window; int* dev_zplanes;
	float* dev_znear; float* dev_zfar; float* dev_cost_cube;
	double* dev_K; double* dev_R; double* dev_t; double* dev_inv_K; double* dev_inv_R; double* dev_inv_t;
	uint8_t* dev_Y_ref; uint8_t* dev_Y_cam;

	CHK(cudaSetDevice(0));

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
	//CHK(cudaMalloc((void**)&dev_cost_cube, ZPlanes * mat_length * sizeof(float)));
	CHK(cudaMalloc((void**)&dev_cost_cube, mat_length * sizeof(float)));

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
	//CHK(cudaMemcpy(dev_cost_cube, new_cost_cube, ZPlanes * mat_length * sizeof(float), cudaMemcpyHostToDevice));

	int N_threads = 1024;
	dim3 thread_size(N_threads);
	dim3 block_size((mat_length + N_threads - 1) / N_threads);

	/*compute_cost_naive << <block_size, thread_size>> > (dev_width, dev_height, dev_zi, dev_znear, dev_zfar, dev_zplanes, dev_half_window, dev_cam_K,
		dev_cam_R, dev_cam_t, dev_ref_inv_K, dev_ref_inv_R, dev_ref_inv_t, dev_Y_ref, dev_Y_cam, dev_cost_cube);*/

	compute_cost_naive_baseline << <block_size, thread_size >> > (dev_width, dev_height, dev_zi, dev_znear, dev_zfar, dev_zplanes, dev_half_window, dev_K, 
		dev_R, dev_t, dev_inv_K, dev_inv_R, dev_inv_t, dev_cost_cube, dev_Y_ref, dev_Y_cam);
	
	//CHK(cudaGetLastError());
	cudaGetLastError();
	
	//CHK(cudaMemcpy(cost_cube_plane.data, dev_cost_cube, mat_length * sizeof(float), cudaMemcpyDeviceToHost));
	CHK(cudaMemcpy(new_cost_cube, dev_cost_cube, mat_length * sizeof(float), cudaMemcpyDeviceToHost));

	//memcpy((void*)result.data, (void*)&new_cost_cube[0], mat_length * sizeof(float));
	//cv::Mat result = cv::Mat(ref.width, ref.height, CV_32FC1, &new_cost_cube);
	float* result = new_cost_cube;

	CHK(cudaFree(dev_width));
	CHK(cudaDeviceReset());
	return result;
}

float* frame2frame_matching_naive_float(cam& ref, cam& cam_1, cv::Mat& cost_cube_plane, int zi, int half_window)
{
	printf("Naive cost frame2frame_matching:\n");

	uint mat_length;
	cv::Mat mat;


	// Only one plane
	mat_length = cost_cube_plane.total() * cost_cube_plane.channels();
	uint im_length = mat_length * 3 / 2;

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

	//float* K = &cam_1.p.K[0]; float* R = &cam_1.p.R[0]; float* t = &cam_1.p.t[0];
	//double* inv_K = &ref.p.K_inv[0]; double* inv_R = &ref.p.R_inv[0]; double* inv_t = &ref.p.t_inv[0];

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

	//CHK(cudaGetLastError());
	cudaGetLastError();

	CHK(cudaMemcpy(new_cost_cube, dev_cost_cube, mat_length * sizeof(float), cudaMemcpyDeviceToHost));

	float* result = new_cost_cube;

	CHK(cudaFree(dev_width));
	CHK(cudaDeviceReset());
	return result;
}

float* frame2frame_matching_naive_float_2D(cam& ref, cam& cam_1, cv::Mat& cost_cube_plane, int zi, int half_window)
{
	printf("Naive cost frame2frame_matching:\n");

	uint mat_length;
	cv::Mat mat;


	// Only one plane
	mat_length = cost_cube_plane.total() * cost_cube_plane.channels();
	uint im_length = mat_length * 3 / 2;

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

	//float* K = &cam_1.p.K[0]; float* R = &cam_1.p.R[0]; float* t = &cam_1.p.t[0];
	//double* inv_K = &ref.p.K_inv[0]; double* inv_R = &ref.p.R_inv[0]; double* inv_t = &ref.p.t_inv[0];

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

	//CHK(cudaGetLastError());
	cudaGetLastError();

	CHK(cudaMemcpy(new_cost_cube, dev_cost_cube, mat_length * sizeof(float), cudaMemcpyDeviceToHost));

	float* result = new_cost_cube;

	CHK(cudaFree(dev_width));
	CHK(cudaDeviceReset());
	return result;
}

float* frame2frame_matching_partially_shared_float_2D(cam& ref, cam& cam_1, cv::Mat& cost_cube_plane, int zi, int half_window)
{
	printf("Naive cost frame2frame_matching:\n");

	uint mat_length;
	cv::Mat mat;

	// Only one plane
	mat_length = cost_cube_plane.total() * cost_cube_plane.channels();
	uint im_length = mat_length * 3 / 2;

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

	//CHK(cudaGetLastError());
	cudaGetLastError();

	CHK(cudaMemcpy(new_cost_cube, dev_cost_cube, mat_length * sizeof(float), cudaMemcpyDeviceToHost));

	float* result = new_cost_cube;

	CHK(cudaFree(dev_width));
	CHK(cudaDeviceReset());
	return result;
}

float* frame2frame_matching_shared_float_2D(cam& ref, cam& cam_1, cv::Mat& cost_cube_plane, int zi, int half_window)
{
	printf("Naive cost frame2frame_matching:\n");

	uint mat_length;
	cv::Mat mat;

	// Only one plane
	mat_length = cost_cube_plane.total() * cost_cube_plane.channels();
	uint im_length = mat_length * 3 / 2;

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

	//CHK(cudaGetLastError());
	cudaGetLastError();

	CHK(cudaMemcpy(new_cost_cube, dev_cost_cube, mat_length * sizeof(float), cudaMemcpyDeviceToHost));

	float* result = new_cost_cube;

	CHK(cudaFree(dev_width));
	CHK(cudaDeviceReset());
	return result;
}