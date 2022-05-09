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

	dev_test_vecAdd << <1, N >> > (dev_a, dev_b, dev_c, N);

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


__global__ void compute_cost_naive(float* cost, float* cc, std::vector<cv::Mat> const& ref, std::vector<cv::Mat> const& cam, int* id_x, int* id_y, int N)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	int l = blockIdx.y * blockDim.y + threadIdx.y;

	if (k >= N || l >= N) return;

	
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

//std::vector<cv::Mat> frame2frame_matching(std::vector<cam> const &ref, std::vector<cam> const &cam, std::vector<cv::Mat> const &cost_cube, int zi, int half_window)
void frame2frame_matching(cam &ref, cam &cam_1, std::vector<cv::Mat> &cost_cube, int zi, int half_window)
{
	printf("Naive cost frame2frame_matching:\n");

	uint mat_length;

	std::vector<float*> new_cost_cube;
	for (auto& mat : cost_cube)
	{
		mat_length = mat.total()*mat.channels();
		float* mat_arr = mat.isContinuous() ? (float*)mat.data : (float*)mat.clone().data;
		new_cost_cube.push_back(mat_arr);
	}


	/*mat_length = cost_cube[0].total() * cost_cube[0].channels();
	float* new_cost_cube = new float[mat_length];
	for (int i=0; i<ZPlanes; i++)
	{
		cv::Mat mat = cost_cube[i];
		float* mat_arr = mat.isContinuous() ? (float*)mat.data : (float*)mat.clone().data;
		std::copy(mat_arr, mat_arr+mat_length-1, new_cost_cube);
	}*/

	int* dev_width; int* dev_height; int* dev_zi; int* dev_half_window;
	float* dev_znear; float* dev_zfar; float* dev_zplanes;
	std::vector<double>* dev_cam_K; std::vector<double>* dev_cam_R; std::vector<double>* dev_cam_t; std::vector<double>* dev_ref_inv_K; std::vector<double>* dev_ref_inv_R; std::vector<double>* dev_ref_inv_t;
	cv::Mat* dev_Y_cam; cv::Mat* dev_Y_ref;
	std::vector<float*>* dev_cost_cube;

	cudaSetDevice(0);

	// width, height, zi, Znear, ZFar, ZPlanes, K, R, t, inv_K, inv_R, inv_t, window, Y_cam, Y_ref, cost_cube

	cudaMalloc((void**)&dev_width, sizeof(int));
	cudaMalloc((void**)&dev_height, sizeof(int));
	cudaMalloc((void**)&dev_zi, sizeof(int));
	cudaMalloc((void**)&dev_znear, sizeof(float));
	cudaMalloc((void**)&dev_zfar, sizeof(float));
	cudaMalloc((void**)&dev_zplanes, sizeof(float));
	cudaMalloc((void**)&dev_half_window, sizeof(int));
	cudaMalloc((void**)&dev_cam_K, 9 * sizeof(double));
	cudaMalloc((void**)&dev_cam_R, 9 * sizeof(double));
	cudaMalloc((void**)&dev_cam_t, 3 * sizeof(double));
	cudaMalloc((void**)&dev_ref_inv_K, 9 * sizeof(double));
	cudaMalloc((void**)&dev_ref_inv_R, 9 * sizeof(double));
	cudaMalloc((void**)&dev_ref_inv_t, 3 * sizeof(double));
	cudaMalloc((void**)&dev_Y_ref, ref.YUV[0].cols * ref.YUV[0].rows * sizeof(float));
	cudaMalloc((void**)&dev_Y_cam, cam_1.YUV[0].cols * cam_1.YUV[0].rows * sizeof(float));
	cudaMalloc((void**)&dev_cost_cube, ZPlanes * mat_length * sizeof(float));

	cudaMemcpy(dev_width, &ref.width, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_height, &ref.height, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_zi, &zi, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_znear, &ZNear, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_zfar, &ZFar, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_zplanes, &ZPlanes, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_half_window, &half_window, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cam_K, &cam_1.p.K, 9 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cam_R, &cam_1.p.R, 9 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cam_t, &cam_1.p.t, 3 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ref_inv_K, &ref.p.K_inv, 9 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ref_inv_R, &ref.p.R_inv, 9 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ref_inv_t, &ref.p.t_inv, 3 * sizeof(double), cudaMemcpyHostToDevice);
	//cudaMemcpy(dev_Y_ref, &ref.YUV[0], ref.YUV[0].cols * ref.YUV[0].rows * sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(dev_Y_cam, &cam_1.YUV[0], mat_length * sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(dev_cost_cube, &new_cost_cube, ZPlanes * mat_length * sizeof(float), cudaMemcpyHostToDevice);

	int N_threads = 1024;
	dim3 thread_size(N_threads);
	//dim3 block_size(((ref.height * ref.width) + N_threads - 1) / N_threads);
	//std::cout << "thread size " << thread_size << ", block size " << block_size << std::endl;
	//printf("Thread size %d, block size %d\n", thread_size, block_size);
}