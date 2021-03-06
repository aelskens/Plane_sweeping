#include "../kernels/main.cuh"

#include <cstdio>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <string>
#include <time.h>
#include <chrono>

std::vector<cam> read_cams(std::string const& folder)
{
	// Init parameters
	std::vector<params<double>> cam_params_vector = get_cam_params();

	// Init cameras
	std::vector<cam> cam_array(cam_params_vector.size());
	for (int i = 0; i < cam_params_vector.size(); i++)
	{
		// Name
		std::string name = folder + "/v" + std::to_string(i) + ".png";

		// Read PNG file
		cv::Mat im_rgb = cv::imread(name);
		cv::Mat im_yuv;
		const int width = im_rgb.cols;
		const int height = im_rgb.rows;

		// Convert to YUV420
		cv::cvtColor(im_rgb, im_yuv, cv::COLOR_BGR2YUV_I420); 
		const int size = width * height * 1.5; // YUV 420

		std::vector<cv::Mat> YUV;
		cv::split(im_yuv, YUV);

		// Params
		cam_array.at(i) = cam(name, width, height, size, YUV, cam_params_vector.at(i));
	}

	return cam_array;
}

std::vector<cv::Mat> sweeping_plane(cam const ref, std::vector<cam> const& cam_vector, int window = 3)
{
	// Initialization to MAX value
	// std::vector<float> cost_cube(ref.width * ref.height * ZPlanes, 255.f);
	std::vector<cv::Mat> cost_cube(ZPlanes);
	for (int i = 0; i < cost_cube.size(); ++i)
	{
		cost_cube[i] = cv::Mat(ref.height, ref.width, CV_32FC1, 255.);
	}

	// For each camera in the setup (reference is skipped)
	for (auto& cam : cam_vector)
	{
		if (cam.name == ref.name)
			continue;

		std::cout << "Cam: " << cam.name << std::endl;
		// For each pixel and candidate: (i) calculate projection index, (ii) calculate cost against reference, (iii) store minimum cost
		for (int zi = 0; zi < ZPlanes; zi++)
		{
			std::cout << "Plane " << zi << std::endl;
			for (int y = 0; y < ref.height; y++)
			{
				for (int x = 0; x < ref.width; x++)
				{
					// (i) calculate projection index

					// Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
					double z = ZNear * ZFar / (ZNear + (((double)zi / (double)ZPlanes) * (ZFar - ZNear))); //need to be in the x and y loops?

					// 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
					double X_ref = (ref.p.K_inv[0] * x + ref.p.K_inv[1] * y + ref.p.K_inv[2]) * z;
					double Y_ref = (ref.p.K_inv[3] * x + ref.p.K_inv[4] * y + ref.p.K_inv[5]) * z;
					double Z_ref = (ref.p.K_inv[6] * x + ref.p.K_inv[7] * y + ref.p.K_inv[8]) * z;

					// 3D in ref camera coordinates to 3D world
					double X = ref.p.R_inv[0] * X_ref + ref.p.R_inv[1] * Y_ref + ref.p.R_inv[2] * Z_ref - ref.p.t_inv[0];
					double Y = ref.p.R_inv[3] * X_ref + ref.p.R_inv[4] * Y_ref + ref.p.R_inv[5] * Z_ref - ref.p.t_inv[1];
					double Z = ref.p.R_inv[6] * X_ref + ref.p.R_inv[7] * Y_ref + ref.p.R_inv[8] * Z_ref - ref.p.t_inv[2];

					// 3D world to projected camera 3D coordinates
					double X_proj = cam.p.R[0] * X + cam.p.R[1] * Y + cam.p.R[2] * Z - cam.p.t[0];
					double Y_proj = cam.p.R[3] * X + cam.p.R[4] * Y + cam.p.R[5] * Z - cam.p.t[1];
					double Z_proj = cam.p.R[6] * X + cam.p.R[7] * Y + cam.p.R[8] * Z - cam.p.t[2];

					// Projected camera 3D coordinates to projected camera 2D coordinates
					double x_proj = (cam.p.K[0] * X_proj / Z_proj + cam.p.K[1] * Y_proj / Z_proj + cam.p.K[2]);
					double y_proj = (cam.p.K[3] * X_proj / Z_proj + cam.p.K[4] * Y_proj / Z_proj + cam.p.K[5]);
					double z_proj = Z_proj;

					x_proj = x_proj < 0 || x_proj >= cam.width ? 0 : roundf(x_proj);
					y_proj = y_proj < 0 || y_proj >= cam.height ? 0 : roundf(y_proj);

					// (ii) calculate cost against reference
					// Calculating cost in a window
					float cost = 0.0f;
					float cc = 0.0f;
					for (int k = -window / 2; k <= window / 2; k++)
					{
						for (int l = -window / 2; l <= window / 2; l++)
						{
							if (x + l < 0 || x + l >= ref.width)
								continue;
							if (y + k < 0 || y + k >= ref.height)
								continue;
							if (x_proj + l < 0 || x_proj + l >= cam.width)
								continue;
							if (y_proj + k < 0 || y_proj + k >= cam.height)
								continue;

							// Y
							cost += fabs(ref.YUV[0].at<uint8_t>(y + k, x + l) - cam.YUV[0].at<uint8_t>((int)y_proj + k, (int)x_proj + l));
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
					cost_cube[zi].at<float>(y, x) = fminf(cost_cube[zi].at<float>(y, x), cost);
				}
			}
		}
	}

	return cost_cube;
}

std::vector<cv::Mat> sweeping_plane_linear(cam const ref, std::vector<cam> const& cam_vector, int window = 3)
{
	// Initialization to MAX value
	// std::vector<float> cost_cube(ref.width * ref.height * ZPlanes, 255.f);
	std::vector<cv::Mat> cost_cube(ZPlanes);
	for (int i = 0; i < cost_cube.size(); ++i)
	{
		cost_cube[i] = cv::Mat(ref.height, ref.width, CV_32FC1, 255.);
	}

	// For each camera in the setup (reference is skipped)
	for (auto& cam : cam_vector)
	{
		if (cam.name == ref.name)
			continue;

		std::cout << "Cam: " << cam.name << std::endl;
		// For each pixel and candidate: (i) calculate projection index, (ii) calculate cost against reference, (iii) store minimum cost
		for (int zi = 0; zi < ZPlanes; zi++)
		{
			std::cout << "Plane " << zi << std::endl;
			for (int p = 0; p < ref.height * ref.width; p++)
			{
				int x = p % ref.width;
				int y = p / ref.width;

				// Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
				double z = ZNear * ZFar / (ZNear + (((double)zi / (double)ZPlanes) * (ZFar - ZNear))); //need to be in the x and y loops?

				// 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
				double X_ref = (ref.p.K_inv[0] * x + ref.p.K_inv[1] * y + ref.p.K_inv[2]) * z;
				double Y_ref = (ref.p.K_inv[3] * x + ref.p.K_inv[4] * y + ref.p.K_inv[5]) * z;
				double Z_ref = (ref.p.K_inv[6] * x + ref.p.K_inv[7] * y + ref.p.K_inv[8]) * z;

				// 3D in ref camera coordinates to 3D world
				double X = ref.p.R_inv[0] * X_ref + ref.p.R_inv[1] * Y_ref + ref.p.R_inv[2] * Z_ref - ref.p.t_inv[0];
				double Y = ref.p.R_inv[3] * X_ref + ref.p.R_inv[4] * Y_ref + ref.p.R_inv[5] * Z_ref - ref.p.t_inv[1];
				double Z = ref.p.R_inv[6] * X_ref + ref.p.R_inv[7] * Y_ref + ref.p.R_inv[8] * Z_ref - ref.p.t_inv[2];

				// 3D world to projected camera 3D coordinates
				double X_proj = cam.p.R[0] * X + cam.p.R[1] * Y + cam.p.R[2] * Z - cam.p.t[0];
				double Y_proj = cam.p.R[3] * X + cam.p.R[4] * Y + cam.p.R[5] * Z - cam.p.t[1];
				double Z_proj = cam.p.R[6] * X + cam.p.R[7] * Y + cam.p.R[8] * Z - cam.p.t[2];

				// Projected camera 3D coordinates to projected camera 2D coordinates
				double x_proj = (cam.p.K[0] * X_proj / Z_proj + cam.p.K[1] * Y_proj / Z_proj + cam.p.K[2]);
				double y_proj = (cam.p.K[3] * X_proj / Z_proj + cam.p.K[4] * Y_proj / Z_proj + cam.p.K[5]);
				double z_proj = Z_proj;

				x_proj = x_proj < 0 || x_proj >= cam.width ? 0 : roundf(x_proj);
				y_proj = y_proj < 0 || y_proj >= cam.height ? 0 : roundf(y_proj);

				// (ii) calculate cost against reference
				// Calculating cost in a window
				float cost = 0.0f;
				float cc = 0.0f;
				for (int k = -window / 2; k <= window / 2; k++)
				{
					for (int l = -window / 2; l <= window / 2; l++)
					{
						if (x + l < 0 || x + l >= ref.width)
							continue;
						if (y + k < 0 || y + k >= ref.height)
							continue;
						if (x_proj + l < 0 || x_proj + l >= cam.width)
							continue;
						if (y_proj + k < 0 || y_proj + k >= cam.height)
							continue;

						// Y
						cost += fabs(ref.YUV[0].at<uint8_t>(y + k, x + l) - cam.YUV[0].at<uint8_t>((int)y_proj + k, (int)x_proj + l));
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
				cost_cube[zi].at<float>(y, x) = fminf(cost_cube[zi].at<float>(y, x), cost);
			}
		}
	}

	return cost_cube;
}

std::vector<cv::Mat> sweeping_plane_cost_plane_gpu(cam ref, std::vector<cam> & cam_vector, int window = 3, int mode = 11)
{
	// Initialization to MAX value
	std::vector<cv::Mat> cost_cube(ZPlanes);
	for (int i = 0; i < cost_cube.size(); ++i)
	{
		cost_cube[i] = cv::Mat(ref.height, ref.width, CV_32FC1, 255.);
	}

	// For each camera in the setup (reference is skipped)
	for (cam cam : cam_vector)
	{
		if (cam.name == ref.name || mode >= 9)
				continue;

		// For each pixel and candidate: (i) calculate projection index, (ii) calculate cost against reference, (iii) store minimum cost
		for (int zi = 0; zi < ZPlanes; zi++)
		{
			float* result;
			if(mode >= 6 && zi!=0) continue;

			switch (mode) {
				case 0: 
					result = frame2frame_matching_naive_baseline(ref, cam, cost_cube[zi], zi, window / 2);
					break;
				case 1: 
					result = frame2frame_matching_naive_float(ref, cam, cost_cube[zi], zi, window / 2);
					break;
				case 2: 
					result = frame2frame_matching_naive_float_2D(ref, cam, cost_cube[zi], zi, window / 2);
					break;
				case 3: 
					result = frame2frame_matching_partially_shared_float_2D(ref, cam, cost_cube[zi], zi, window / 2);
					break;
				case 4: 
					result = frame2frame_matching_shared_float_2D(ref, cam, cost_cube[zi], zi, window / 2);
					break;
				case 5:
					result = frame2frame_matching_shared_full_float_2D(ref, cam, cost_cube[zi], zi, window / 2);
					break;
				case 6:
					result = frame2frame_matching_smart_naive_full_float_2D(ref, cam, cost_cube, window / 2);
					break;
				case 7:
					result = frame2frame_matching_smart_shared_full_float_2D(ref, cam, cost_cube, window / 2);
					break;
				case 8:
					result = frame2frame_matching_smart_full_shared_full_float_2D(ref, cam, cost_cube, window / 2);
					break;
			}

			if (mode < 6){
				cv::Mat result_mat = cv::Mat(1080, 1920, CV_32FC1, result);
				cost_cube[zi] = result_mat;
			}
			else {
				for (int i = 0; i < 256; i++) {
					cv::Mat result_mat = cv::Mat(1080, 1920, CV_32FC1, &result[i * ref.height * ref.width]);
					cost_cube[i] = result_mat;
				}
			} 

		}
	}

	if (mode >= 9){
		float* result;

		switch (mode) {
			case 9:
				result = frame2frame_matching_all_smart_full_shared_full_float_2D(ref, cam_vector, cost_cube, window / 2);
				break;
			case 10:
				result = frame2frame_matching_all_no_fill_smart_full_shared_full_float_2D(ref, cam_vector, cost_cube, window / 2);
				break;
			case 11:
				result = frame2frame_matching_all_no_fill_better_pad_smart_full_shared_full_float_2D(ref, cam_vector, cost_cube, window / 2);
				break;
		}

		for (int i = 0; i < 256; i++) {
			cv::Mat result_mat = cv::Mat(1080, 1920, CV_32FC1, &result[i * ref.height * ref.width]);
			cost_cube[i] = result_mat;
		}
	
	}

	return cost_cube;
}

cv::Mat sweeping_plane_cost_plane_and_findmin_gpu(cam ref, std::vector<cam>& cam_vector, int window = 3, int mode = 3)
{
	float* float_result;
	uint8_t* uint8_t_result;
	cv::Mat tmp_depth, depth;

	switch (mode) {
		case 0:
			float_result = frame2frame_matching_reduced_float_all_no_fill_better_pad_smart_full_shared_full_float_2D(ref, cam_vector, window / 2);
			tmp_depth = cv::Mat(ref.height, ref.width, CV_32FC1, &float_result[0]);
			tmp_depth.convertTo(depth, CV_8U);
			break;
		case 1:
			uint8_t_result = frame2frame_matching_reduced_uint8_t_all_no_fill_better_pad_smart_full_shared_full_float_2D(ref, cam_vector, window / 2);
			depth = cv::Mat(ref.height, ref.width, CV_8U, &uint8_t_result[0]);
			break;
		case 2:
			uint8_t_result = frame2frame_matching_reduced_uint8_t_all_no_fill_better_pad_less_global_smart_full_shared_full_float_2D(ref, cam_vector, window / 2);
			depth = cv::Mat(ref.height, ref.width, CV_8U, &uint8_t_result[0]);
			break;
	}


	return depth;
}

cv::Mat find_min(std::vector<cv::Mat> const& cost_cube)
{
	const int zPlanes = cost_cube.size();
	const int height = cost_cube[0].size().height;
	const int width = cost_cube[0].size().width;

	cv::Mat ret(height, width, CV_32FC1, 255.);
	cv::Mat depth(height, width, CV_8U, 255);

	for (int zi = 0; zi < zPlanes; zi++)
	{
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				if (cost_cube[zi].at<float>(y, x) < ret.at<float>(y, x))
				{
					ret.at<float>(y, x) = cost_cube[zi].at<float>(y, x);
					depth.at<u_char>(y, x) = zi;
				}
			}
		}
	}

	return depth;
}


int main()
{
	// Read cams
	std::chrono::steady_clock::time_point begin_read = std::chrono::steady_clock::now();
	std::vector<cam> cam_vector = read_cams("data");
	std::chrono::steady_clock::time_point end_read = std::chrono::steady_clock::now();
	int delta_time_read = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end_read - begin_read).count();
	std::cout << "Elapsed time for read_cam " << delta_time_read << "[ms]" << std::endl;
	
	// Warm-up
	wrap_test_vectorAdd();



	std::vector<cv::Mat> cost_cube;

	

	/////////////////
	// CPU

	std::chrono::steady_clock::time_point begin_CPU = std::chrono::steady_clock::now();
	cost_cube = sweeping_plane(cam_vector.at(0), cam_vector, 5); 
	//std::vector<cv::Mat> cost_cube = sweeping_plane_linear(cam_vector.at(0), cam_vector, 5);
	cv::Mat depth = find_min(cost_cube);
	std::chrono::steady_clock::time_point end_CPU = std::chrono::steady_clock::now();
	int delta_time_CPU = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end_CPU - begin_CPU).count();
	std::cout << "Elapsed time CPU " << delta_time_CPU << "[ms]" << std::endl;
	std::string name = "../../screenshots/All_cams/CPU.png";
	cv::imwrite(name, depth);


	//////////
	// GPU cost_cube

	std::string versions[12] = {"Naive_baseline", "Naive_float", "Naive_float_2D", "Partially_shared_float", "Shared_float", "Shared_full_float", "Smart_naive_full_float", "Smart_shared_full_float", "Smart_full_shared_full_float", "Smart_all_cams_full_shared_full_float", "Smart_all_cams_no_fill_full_shared_full_float", "Smart_all_cams_no_fill_better_pad_full_shared_full_float" };

	for (int i=0; i<12; i++) {
		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		cost_cube = sweeping_plane_cost_plane_gpu(cam_vector.at(0), cam_vector, 5, i);
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		int delta_time = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
		std::cout << "Elapsed time " << versions[i] << " " << delta_time << "[ms]" << std::endl;

		begin = std::chrono::steady_clock::now();
		cv::Mat depth = find_min(cost_cube);
		end = std::chrono::steady_clock::now();
		delta_time = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
		std::cout << "Elapsed time find_min " << delta_time << "[ms]" << std::endl;

		std::string name = "../../screenshots/All_cams/" + versions[i] + ".png";
		cv::imwrite(name, depth); 
		/*cv::namedWindow("Depth", cv::WINDOW_NORMAL);
		cv::imshow("Depth", depth);
		cv::waitKey(0);*/
	}

	////////
	// GPU depth map

	std::string versions2[3] = { "Reduced_float_smart_all_cams_no_fill_better_pad_full_shared_full_float", "Reduced_uint8_t_smart_all_cams_no_fill_better_pad_full_shared_full_float", "Reduced_uint8_t_smart_all_cams_no_fill_better_pad_less_global_full_shared_full_float"};

	for (int i = 0; i < 3; i++) {
		//time_t start_time = time(0);
		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

		cv::Mat depth = sweeping_plane_cost_plane_and_findmin_gpu(cam_vector.at(0), cam_vector, 5, i);

		//time_t end_time = time(0);
		//int delta_time = (int) (end_time - start_time);
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		int delta_time = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
		//std::cout << "Elapsed time " << delta_time/60 << "m" << delta_time%60  << "s"<< std::endl;
		std::cout << "Elapsed time " << versions2[i] << " " << delta_time << "[ms]" << std::endl;
		std::string name = "../../screenshots/All_cams/" + versions2[i] + ".png";
		cv::imwrite(name, depth);
		/*cv::namedWindow("Depth", cv::WINDOW_NORMAL);
		cv::imshow("Depth", depth);
		cv::waitKey(0);*/
	}


	return 0;
}