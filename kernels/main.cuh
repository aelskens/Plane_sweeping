#pragma once
#include <cuda_runtime.h>
#include <cuda.h>

#include "../src/cam_params.hpp"
#include "../src/constants.hpp"

#include <vector>
#include <typeinfo>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

// This is the public interface of our cuda function, called directly in main.cpp
void wrap_test_vectorAdd();
//void test(cv::Mat const &Y);
float* frame2frame_matching_naive_baseline(cam& ref, cam& cam_1, cv::Mat& cost_cube_plane, int zi, int half_window);
float* frame2frame_matching_naive_float(cam& ref, cam& cam_1, cv::Mat& cost_cube_plane, int zi, int half_window);
float* frame2frame_matching_naive_float_2D(cam& ref, cam& cam_1, cv::Mat& cost_cube_plane, int zi, int half_window);
float* frame2frame_matching_partially_shared_float_2D(cam &ref, cam &cam_1, cv::Mat &cost_cube_plane, int zi, int half_window);
float* frame2frame_matching_shared_float_2D(cam &ref, cam &cam_1, cv::Mat &cost_cube_plane, int zi, int half_window);
//void compute_cost_naive(float* cost, float* cc, std::vector<cv::Mat> const& ref, std::vector<cv::Mat> const& cam, int* id_x, int* id_y, int N);