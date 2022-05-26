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
		//printf("width = %d, height = %d\n", width, height);

		// Convert to YUV420
		cv::cvtColor(im_rgb, im_yuv, cv::COLOR_BGR2YUV_I420); // uint8_t; cv::COLOR_BGR2YUV is equivalent to R+G+B/3
		const int size = width * height * 1.5; // YUV 420

		std::vector<cv::Mat> YUV;
		cv::split(im_yuv, YUV);
		/*printf("imYUVwidth = %d, imYUVheight = %d\n", im_yuv.cols, im_yuv.rows);
		printf("Ywidth = %d, Yheight = %d\n", YUV[0].cols, YUV[0].rows);
		printf("Uwidth = %d, Uheight = %d\n", YUV[1].cols, YUV[1].rows);
		printf("Vwidth = %d, Vheight = %d\n", YUV[2].cols, YUV[2].rows);*/

		// Params
		cam_array.at(i) = cam(name, width, height, size, YUV, cam_params_vector.at(i));
	}

	return cam_array;

	// cv::Mat U(height / 2, width / 2, CV_8UC1, cam_array.at(0).image.data() + (int)(width * height * 1.25));
	// cv::namedWindow("im", cv::WINDOW_NORMAL);
	// cv::imshow("im", U);
	// cv::waitKey(0);
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

	// Visualize costs
	// for (int zi = 0; zi < ZPlanes; zi++)
	// {
	// 	std::cout << "plane " << zi << std::endl;
	// 	cv::namedWindow("Cost", cv::WINDOW_NORMAL);
	// 	cv::imshow("Cost", cost_cube.at(zi) / 255.f);
	// 	cv::waitKey(0);
	// }
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


				/*if (p % 500 == 0)
					printf("p = %d, x = %d, y = %d\n", p, x, y);*/
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

	// Visualize costs
	// for (int zi = 0; zi < ZPlanes; zi++)
	// {
	// 	std::cout << "plane " << zi << std::endl;
	// 	cv::namedWindow("Cost", cv::WINDOW_NORMAL);
	// 	cv::imshow("Cost", cost_cube.at(zi) / 255.f);
	// 	cv::waitKey(0);
	// }
	return cost_cube;
}

std::vector<cv::Mat> sweeping_plane_cost_plane_gpu(cam ref, std::vector<cam> & cam_vector, int window = 3, int mode = 4)
{
	// Initialization to MAX value
	// std::vector<float> cost_cube(ref.width * ref.height * ZPlanes, 255.f);
	std::vector<cv::Mat> cost_cube(ZPlanes);
	for (int i = 0; i < cost_cube.size(); ++i)
	{
		cost_cube[i] = cv::Mat(ref.height, ref.width, CV_32FC1, 255.);
	}

	// For each camera in the setup (reference is skipped)
	for (cam cam : cam_vector)
	{
		if (cam.name == ref.name)
			continue;

		std::cout << "Cam: " << cam.name << std::endl;
		// For each pixel and candidate: (i) calculate projection index, (ii) calculate cost against reference, (iii) store minimum cost
		for (int zi = 0; zi < ZPlanes; zi++)
		{
			std::cout << "Plane " << zi << std::endl;
			float* result;
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
			}
			cv::Mat result_mat = cv::Mat(1080, 1920, CV_32FC1, result);
			cost_cube[zi] = result_mat;
		}
	}

	// Visualize costs
	 /*for (int zi = 0; zi < ZPlanes; zi++)
	 {
	 	std::cout << "plane " << zi << std::endl;
	 	cv::namedWindow("Cost", cv::WINDOW_NORMAL);
	 	cv::imshow("Cost", cost_cube.at(zi) / 255.f);
	 	cv::waitKey(0);
	 }*/
	return cost_cube;
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

// Test to take YUV matrix from a cam
void test_YUV_mat(std::vector<cv::Mat> const &YUV)
{
	/*cv::namedWindow("Y", cv::WindowFlags::WINDOW_AUTOSIZE);
	cv::imshow("Y", YUV[0]);
	cv::namedWindow("U", cv::WindowFlags::WINDOW_AUTOSIZE);
	cv::imshow("U", YUV[1]);
	cv::namedWindow("V", cv::WindowFlags::WINDOW_AUTOSIZE);
	cv::imshow("V", YUV[2]);
	cv::waitKey(0);*/
	printf("rows %i\n", YUV[0].rows);
	printf("columns %i\n", YUV[0].cols);

	// type(element) in Y from YUV == double bad for GPU?
	/*int i = 0;
	cv::MatConstIterator_<double> it = YUV[0].begin<double>(), it_end = YUV[0].end<double>();
	for (; it != it_end; it++) {
		float r = *it;
		std::cout << r << "\n" << std::endl;
		std::cout << *it << "\n" << std::endl;
		std::cout << typeid(*it).name() << std::endl;
		break;
	}*/
}

int main()
{
	// Read cams
	std::vector<cam> cam_vector = read_cams("data");

	// Test call a CUDA function
	wrap_test_vectorAdd();


	//// Test passing parameters
	//std::vector<cv::Mat> cost_cube;
	//for (int i=0; i<256; i++) cost_cube.push_back(cv::Mat(1920, 1080, CV_32FC1));
	////test(cam_vector.at(0).YUV[0]);
	//float* result = frame2frame_matching(cam_vector.at(0), cam_vector.at(1), cost_cube.at(0), 0, 5);
	//for (int k = 0; k < 1080*1920; k += 100000)  printf("main.cpp result %d cost_cube:%f\n", k, result[k]);
	//cv::Mat result_mat = cv::Mat(1080, 1920, CV_32FC1, result);
	//cost_cube.at(0) = result_mat;
	//for (int k = 0; k < 1080 * 1920; k += 100000)  printf("main.cpp cost_cube %d cost_cube:%f\n", k, cost_cube.at(0).at<float>(k/1920, k%1920));

	
	/////////////////////////////////////////////////////////////////////////////
	//// Test to take YUV matrix from a cam
	////test_YUV_mat(cam_vector.at(0).YUV);

	
	////////////////////////////////////////////////////////////////////////////
	//// Sweeping algorithm for camera 0
	std::vector<cv::Mat> cost_cube;

	for (int i=2; i<5; i++) {
		time_t start_time = time(0);
		//std::vector<cv::Mat> cost_cube = sweeping_plane(cam_vector.at(0), cam_vector, 5); // data type inn each Mat == float
		//std::cout << typeid(cost_cube[0].at<float>(0, 0)).name() << std::endl; 
		//std::vector<cv::Mat> cost_cube = sweeping_plane_linear(cam_vector.at(0), cam_vector, 5);
		cost_cube = sweeping_plane_cost_plane_gpu(cam_vector.at(0), cam_vector, 5, i);
		time_t end_time = time(0);
		int delta_time = (int) (end_time - start_time);
		std::cout << "Elapsed time " << delta_time/60 << "m" << delta_time%60  << "s"<< std::endl;
	}

	//// Find min cost and generate depth map
	cv::Mat depth = find_min(cost_cube);
	/*time_t second_end_time = time(0);
	delta_time = (int)(second_end_time - end_time);
	std::cout << "Elapsed time for find_min " << delta_time / 60 << "m" << delta_time % 60 << "s" << std::endl;*/
	cv::namedWindow("Depth", cv::WINDOW_NORMAL);
	cv::imshow("Depth", depth);
	//cv::imwrite("../Depth_shared_maybe_fixed_1.png", depth);
	cv::waitKey(0);

	//printf("%f", depth.at<float>(0, 0));

	return 0;
}