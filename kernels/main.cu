#include "main.cuh"

#include <cstdio>

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

__global__ void test_gpu(int* width, int* height, int* zi, float* znear, float* zfar, float* ZPlanes, int* half_window, double* K, double* R, double* t,
	double* inv_K, double* inv_R, double* inv_t, float* cost_cube, uint8_t* y_ref, uint8_t* y_cam)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	//int l = blockIdx.y * blockDim.y + threadIdx.y;

	cost_cube[k] = (float) k;
	//if(k%100000==0) printf("GPU %d ref:%u cam:%u cost_cube:%f\n", k, y_ref[k], y_cam[k], cost_cube[k]);
	//if(k<9) printf("GPU %d KRt %f %f %f\n", k, K[k], R[k], t[k]);

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
float* frame2frame_matching(cam &ref, cam &cam_1, cv::Mat &cost_cube_plane, int zi, int half_window)
{
	printf("Naive cost frame2frame_matching:\n");

	uint mat_length;
	cv::Mat mat; //, result;

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
	printf("size: %d\n", mat_length);

	float* new_cost_cube = new float[mat_length];
	float* mat_arr_plane = cost_cube_plane.isContinuous() ? (float*)cost_cube_plane.data : (float*)cost_cube_plane.clone().data;
	memcpy((void*)new_cost_cube, (void*)mat_arr_plane, mat_length * sizeof(float));

	mat = ref.YUV[0];
	uint8_t* y_ref = new uint8_t[im_length];
	uint8_t* mat_arr = mat.isContinuous() ? (uint8_t*)mat.data : (uint8_t*)mat.clone().data;
	printf("mat size th: %d\n", mat.total()* mat.channels());
	printf("im_length: %d\n", im_length);
	memcpy((void*)y_ref, (void*)mat_arr, im_length * sizeof(uint8_t));

	printf("last val = %f\n", y_ref[im_length-1]);

	mat = cam_1.YUV[0];
	uint8_t* y_cam = new uint8_t[im_length];
	mat_arr = mat.isContinuous() ? (uint8_t*)mat.data : (uint8_t*)mat.clone().data;
	memcpy((void*)y_cam, (void*)mat_arr, im_length * sizeof(uint8_t));

	double* K = &cam_1.p.K[0]; double* R = &cam_1.p.R[0]; double* t = &cam_1.p.t[0];
	double* inv_K = &ref.p.K[0]; double* inv_R = &ref.p.R[0]; double* inv_t = &ref.p.t[0];

	int* dev_width; int* dev_height; int* dev_zi; int* dev_half_window;
	float* dev_znear; float* dev_zfar; float* dev_zplanes; float* dev_cost_cube;
	double* dev_K; double* dev_R; double* dev_t; double* dev_inv_K; double* dev_inv_R; double* dev_inv_t;
	uint8_t* dev_Y_ref; uint8_t* dev_Y_cam;

	CHK(cudaSetDevice(0));

	CHK(cudaMalloc((void**)&dev_width, sizeof(int)));
	CHK(cudaMalloc((void**)&dev_height, sizeof(int)));
	CHK(cudaMalloc((void**)&dev_zi, sizeof(int)));
	CHK(cudaMalloc((void**)&dev_znear, sizeof(float)));
	CHK(cudaMalloc((void**)&dev_zfar, sizeof(float)));
	CHK(cudaMalloc((void**)&dev_zplanes, sizeof(float)));
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
	CHK(cudaMemcpy(dev_zplanes, &ZPlanes, sizeof(float), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_half_window, &half_window, sizeof(int), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_K, K, 9 * sizeof(double), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_R, R, 9 * sizeof(double), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_t, t, 3 * sizeof(double), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_inv_K, inv_K, 9 * sizeof(double), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_inv_R, inv_R, 9 * sizeof(double), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_inv_t, inv_t, 3 * sizeof(double), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_Y_ref, y_ref, im_length * sizeof(uint8_t), cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_Y_cam, y_cam, im_length * sizeof(uint8_t), cudaMemcpyHostToDevice));
	//CHK(cudaMemcpy(dev_cost_cube, new_cost_cube, ZPlanes * mat_length * sizeof(float), cudaMemcpyHostToDevice));

	int N_threads = 1024;
	dim3 thread_size(N_threads);
	dim3 block_size((mat_length + N_threads - 1) / N_threads);

	/*compute_cost_naive << <block_size, thread_size>> > (dev_width, dev_height, dev_zi, dev_znear, dev_zfar, dev_zplanes, dev_half_window, dev_cam_K,
		dev_cam_R, dev_cam_t, dev_ref_inv_K, dev_ref_inv_R, dev_ref_inv_t, dev_Y_ref, dev_Y_cam, dev_cost_cube);*/

	test_gpu << <block_size, thread_size >> > (dev_width, dev_height, dev_zi, dev_znear, dev_zfar, dev_zplanes, dev_half_window, dev_K, dev_R, dev_t,
		dev_inv_K, dev_inv_R, dev_inv_t, dev_cost_cube, dev_Y_ref, dev_Y_cam);
	
	//CHK(cudaGetLastError());
	cudaGetLastError();

	printf("Avant print CPU");

	for(int k=0; k<mat_length; k+=100000)  printf("CPU0 new_cost_cube %d ref:%u cam:%u cost_cube:%f\n", k, y_ref[k], y_cam[k], new_cost_cube[k]);
	//for (int k = 0; k < 9; k++) printf("CPU %d KRt %f %f %f\n", k, K[k], R[k], t[k]);
	
	printf("Thread size %d, block size %d\n", thread_size.x, block_size.x);

	//CHK(cudaMemcpy(cost_cube_plane.data, dev_cost_cube, mat_length * sizeof(float), cudaMemcpyDeviceToHost));
	CHK(cudaMemcpy(new_cost_cube, dev_cost_cube, mat_length * sizeof(float), cudaMemcpyDeviceToHost));
	//memcpy((void*)result.data, (void*)&new_cost_cube[0], mat_length * sizeof(float));
	//cv::Mat result = cv::Mat(ref.width, ref.height, CV_32FC1, &new_cost_cube);
	float* result = new_cost_cube;

	for (int k = 0; k < mat_length; k += 100000)  printf("CPU1 new_cost_cube %d ref:%u cam:%u cost_cube:%f\n", k, y_ref[k], y_cam[k], new_cost_cube[k]);
	CHK(cudaFree(dev_width));
	CHK(cudaDeviceReset());
	return result;
}