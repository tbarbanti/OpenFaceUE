///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt
//
//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace: an open source facial behavior analysis toolkit
//       Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency
//       in IEEE Winter Conference on Applications of Computer Vision, 2016  
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-speci?c normalisation for automatic Action Unit detection
//       Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
//       Constrained Local Neural Fields for robust facial landmark detection in the wild.
//       Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency. 
//       in IEEE Int. Conference on Computer Vision Workshops, 300 Faces in-the-Wild Challenge, 2013.    
//
///////////////////////////////////////////////////////////////////////////////

#include <Face_utils.h>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

// For FHOG visualisation
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>

using namespace std;

namespace FaceAnalysis
{

	// Pick only the more stable/rigid points under changes of expression
	void extract_rigid_points(cv::Mat_<float>& source_points, cv::Mat_<float>& destination_points)
	{
		if(source_points.rows == 68)
		{
			cv::Mat_<float> tmp_source = source_points.clone();
			source_points = cv::Mat_<float>();

			// Push back the rigid points (some face outline, eyes, and nose)
			source_points.push_back(tmp_source.row(1));
			source_points.push_back(tmp_source.row(2));
			source_points.push_back(tmp_source.row(3));
			source_points.push_back(tmp_source.row(4));
			source_points.push_back(tmp_source.row(12));
			source_points.push_back(tmp_source.row(13));
			source_points.push_back(tmp_source.row(14));
			source_points.push_back(tmp_source.row(15));
			source_points.push_back(tmp_source.row(27));
			source_points.push_back(tmp_source.row(28));
			source_points.push_back(tmp_source.row(29));
			source_points.push_back(tmp_source.row(31));
			source_points.push_back(tmp_source.row(32));
			source_points.push_back(tmp_source.row(33));
			source_points.push_back(tmp_source.row(34));
			source_points.push_back(tmp_source.row(35));
			source_points.push_back(tmp_source.row(36));
			source_points.push_back(tmp_source.row(39));
			source_points.push_back(tmp_source.row(40));
			source_points.push_back(tmp_source.row(41));
			source_points.push_back(tmp_source.row(42));
			source_points.push_back(tmp_source.row(45));
			source_points.push_back(tmp_source.row(46));
			source_points.push_back(tmp_source.row(47));

			cv::Mat_<float> tmp_dest = destination_points.clone();
			destination_points = cv::Mat_<float>();

			// Push back the rigid points
			destination_points.push_back(tmp_dest.row(1));
			destination_points.push_back(tmp_dest.row(2));
			destination_points.push_back(tmp_dest.row(3));
			destination_points.push_back(tmp_dest.row(4));
			destination_points.push_back(tmp_dest.row(12));
			destination_points.push_back(tmp_dest.row(13));
			destination_points.push_back(tmp_dest.row(14));
			destination_points.push_back(tmp_dest.row(15));
			destination_points.push_back(tmp_dest.row(27));
			destination_points.push_back(tmp_dest.row(28));
			destination_points.push_back(tmp_dest.row(29));
			destination_points.push_back(tmp_dest.row(31));
			destination_points.push_back(tmp_dest.row(32));
			destination_points.push_back(tmp_dest.row(33));
			destination_points.push_back(tmp_dest.row(34));
			destination_points.push_back(tmp_dest.row(35));
			destination_points.push_back(tmp_dest.row(36));
			destination_points.push_back(tmp_dest.row(39));
			destination_points.push_back(tmp_dest.row(40));
			destination_points.push_back(tmp_dest.row(41));
			destination_points.push_back(tmp_dest.row(42));
			destination_points.push_back(tmp_dest.row(45));
			destination_points.push_back(tmp_dest.row(46));
			destination_points.push_back(tmp_dest.row(47));
		}
	}

	// Aligning a face to a common reference frame
	void AlignFace(cv::Mat& aligned_face, const cv::Mat& frame, const cv::Mat_<float>& detected_landmarks, cv::Vec6d params_global, const PDM& pdm, bool rigid, float sim_scale, int out_width, int out_height)
	{
		// Will warp to scaled mean shape
		cv::Mat_<float> similarity_normalised_shape = pdm.mean_shape * sim_scale;
	
		// Discard the z component
		similarity_normalised_shape = similarity_normalised_shape(cv::Rect(0, 0, 1, 2*similarity_normalised_shape.rows/3)).clone();

		cv::Mat_<float> source_landmarks = detected_landmarks.reshape(1, 2).t();
		cv::Mat_<float> destination_landmarks = similarity_normalised_shape.reshape(1, 2).t();

		// Aligning only the more rigid points
		if(rigid)
		{
			extract_rigid_points(source_landmarks, destination_landmarks);
		}

		// TODO rem the doubles here
		cv::Matx22d scale_rot_matrix = AlignShapesWithScale(source_landmarks, destination_landmarks);
		cv::Matx23d warp_matrix;

		warp_matrix(0,0) = scale_rot_matrix(0,0);
		warp_matrix(0,1) = scale_rot_matrix(0,1);
		warp_matrix(1,0) = scale_rot_matrix(1,0);
		warp_matrix(1,1) = scale_rot_matrix(1,1);

		double tx = params_global[4];
		double ty = params_global[5];

		cv::Vec2d T(tx, ty);
		T = scale_rot_matrix * T;

		// Make sure centering is correct
		warp_matrix(0,2) = -T(0) + out_width/2;
		warp_matrix(1,2) = -T(1) + out_height/2;

		cv::warpAffine(frame, aligned_face, warp_matrix, cv::Size(out_width, out_height), cv::INTER_LINEAR);
	}

	// Aligning a face to a common reference frame
	void AlignFaceMask(cv::Mat& aligned_face, const cv::Mat& frame, const cv::Mat_<float>& detected_landmarks, cv::Vec6f params_global, const PDM& pdm, const cv::Mat_<int>& triangulation, bool rigid, float sim_scale, int out_width, int out_height)
	{
		// Will warp to scaled mean shape
		cv::Mat_<float> similarity_normalised_shape = pdm.mean_shape * sim_scale;
	
		// Discard the z component
		similarity_normalised_shape = similarity_normalised_shape(cv::Rect(0, 0, 1, 2*similarity_normalised_shape.rows/3)).clone();

		cv::Mat_<float> source_landmarks = detected_landmarks.reshape(1, 2).t();
		cv::Mat_<float> destination_landmarks = similarity_normalised_shape.reshape(1, 2).t();

		// Aligning only the more rigid points
		if(rigid)
		{
			extract_rigid_points(source_landmarks, destination_landmarks);
		}

		cv::Matx22f scale_rot_matrix = AlignShapesWithScale(source_landmarks, destination_landmarks);
		cv::Matx23f warp_matrix;

		warp_matrix(0,0) = scale_rot_matrix(0,0);
		warp_matrix(0,1) = scale_rot_matrix(0,1);
		warp_matrix(1,0) = scale_rot_matrix(1,0);
		warp_matrix(1,1) = scale_rot_matrix(1,1);

		float tx = params_global[4];
		float ty = params_global[5];

		cv::Vec2f T(tx, ty);
		T = scale_rot_matrix * T;

		// Make sure centering is correct
		warp_matrix(0,2) = -T(0) + out_width/2;
		warp_matrix(1,2) = -T(1) + out_height/2;

		cv::warpAffine(frame, aligned_face, warp_matrix, cv::Size(out_width, out_height), cv::INTER_LINEAR);

		// Move the destination landmarks there as well
		cv::Matx22f warp_matrix_2d(warp_matrix(0,0), warp_matrix(0,1), warp_matrix(1,0), warp_matrix(1,1));
		
		destination_landmarks = cv::Mat(detected_landmarks.reshape(1, 2).t()) * cv::Mat(warp_matrix_2d).t();

		destination_landmarks.col(0) = destination_landmarks.col(0) + warp_matrix(0,2);
		destination_landmarks.col(1) = destination_landmarks.col(1) + warp_matrix(1,2);
		
		// Move the eyebrows up to include more of upper face
		destination_landmarks.at<float>(0,1) -= (30/0.7)*sim_scale;
		destination_landmarks.at<float>(16,1) -= (30 / 0.7)*sim_scale;

		destination_landmarks.at<float>(17,1) -= (30 / 0.7)*sim_scale;
		destination_landmarks.at<float>(18,1) -= (30 / 0.7)*sim_scale;
		destination_landmarks.at<float>(19,1) -= (30 / 0.7)*sim_scale;
		destination_landmarks.at<float>(20,1) -= (30 / 0.7)*sim_scale;
		destination_landmarks.at<float>(21,1) -= (30 / 0.7)*sim_scale;
		destination_landmarks.at<float>(22,1) -= (30 / 0.7)*sim_scale;
		destination_landmarks.at<float>(23,1) -= (30 / 0.7)*sim_scale;
		destination_landmarks.at<float>(24,1) -= (30 / 0.7)*sim_scale;
		destination_landmarks.at<float>(25,1) -= (30 / 0.7)*sim_scale;
		destination_landmarks.at<float>(26,1) -= (30 / 0.7)*sim_scale;

		destination_landmarks = cv::Mat(destination_landmarks.t()).reshape(1, 1).t();

		FaceAnalysis::PAW paw(destination_landmarks, triangulation, 0, 0, aligned_face.cols-1, aligned_face.rows-1);
		
		// Mask each of the channels (a bit of a roundabout way, but OpenCV 3.1 in debug mode doesn't seem to be able to handle a more direct way using split and merge)
		vector<cv::Mat> aligned_face_channels(aligned_face.channels());
		
		for (int c = 0; c < aligned_face.channels(); ++c)
		{
			cv::extractChannel(aligned_face, aligned_face_channels[c], c);
		}

		for(size_t i = 0; i < aligned_face_channels.size(); ++i)
		{
			cv::multiply(aligned_face_channels[i], paw.pixel_mask, aligned_face_channels[i], 1.0, CV_8U);
		}

		if(aligned_face.channels() == 3)
		{
			cv::Mat planes[] = { aligned_face_channels[0], aligned_face_channels[1], aligned_face_channels[2] };
			cv::merge(planes, 3, aligned_face);
		}
		else
		{
			aligned_face = aligned_face_channels[0];
		}
	}


	void Visualise_FHOG(const cv::Mat_<double>& descriptor, int num_rows, int num_cols, cv::Mat& visualisation)
	{

		// First convert to dlib format
		dlib::array2d<dlib::matrix<float,31,1> > hog(num_rows, num_cols);
		
		cv::MatConstIterator_<double> descriptor_it = descriptor.begin();
		for(int y = 0; y < num_cols; ++y)
		{
			for(int x = 0; x < num_rows; ++x)
			{
				for(unsigned int o = 0; o < 31; ++o)
				{
					hog[y][x](o) = *descriptor_it++;
				}
			}
		}

		// Draw the FHOG to OpenCV format
		auto fhog_vis = dlib::draw_fhog(hog);
		visualisation = dlib::toMat(fhog_vis).clone();
	}

	// Create a row vector Felzenszwalb HOG descriptor from a given image
	void Extract_FHOG_descriptor(cv::Mat_<double>& descriptor, const cv::Mat& image, int& num_rows, int& num_cols, int cell_size)
	{
		
		dlib::array2d<dlib::matrix<float,31,1> > hog;
		if(image.channels() == 1)
		{
			dlib::cv_image<uchar> dlib_warped_img(image);
			dlib::extract_fhog_features(dlib_warped_img, hog, cell_size);
		}
		else
		{
			dlib::cv_image<dlib::bgr_pixel> dlib_warped_img(image);
			dlib::extract_fhog_features(dlib_warped_img, hog, cell_size);
		}

		// Convert to a usable format
		num_cols = hog.nc();
		num_rows = hog.nr();

		descriptor = cv::Mat_<double>(1, num_cols * num_rows * 31);
		cv::MatIterator_<double> descriptor_it = descriptor.begin();
		for(int y = 0; y < num_cols; ++y)
		{
			for(int x = 0; x < num_rows; ++x)
			{
				for(unsigned int o = 0; o < 31; ++o)
				{
					*descriptor_it++ = (double)hog[y][x](o);
				}
			}
		}
	}

	// Extract summary statistics (mean, stdev, min, max) from each dimension of a descriptor, each row is a descriptor
	void ExtractSummaryStatistics(const cv::Mat_<double>& descriptors, cv::Mat_<double>& sum_stats, bool use_mean, bool use_stdev, bool use_max_min)
	{
		// Using four summary statistics at the moment 
		// Means, stds, mins, maxs
		int num_stats = 0;

		if(use_mean)
			num_stats++;

		if(use_stdev)
			num_stats++;

		if(use_max_min)
			num_stats++;

		sum_stats = cv::Mat_<double>(1, descriptors.cols * num_stats, 0.0);
		for(int i = 0; i < descriptors.cols; ++i)
		{
			cv::Scalar mean, stdev;
			cv::meanStdDev(descriptors.col(i), mean, stdev);

			int add = 0;

			if(use_mean)
			{
				sum_stats.at<double>(0, i*num_stats + add) = mean[0];
				add++;
			}

			if(use_stdev)
			{
				sum_stats.at<double>(0, i*num_stats + add) = stdev[0];
				add++;
			}

			if(use_max_min)
			{
				double min, max;
				cv::minMaxIdx(descriptors.col(i), &min, &max);
				sum_stats.at<double>(0, i*num_stats + add) = max - min;
				add++;
			}
		}		
	}

	void AddDescriptor(cv::Mat_<double>& descriptors, cv::Mat_<double> new_descriptor, int curr_frame, int num_frames_to_keep)
	{
		if(descriptors.empty())
		{
			descriptors = cv::Mat_<double>(num_frames_to_keep, new_descriptor.cols, 0.0);
		}

		int row_to_change = curr_frame % num_frames_to_keep;

		new_descriptor.copyTo(descriptors.row(row_to_change));
	}	

	//===========================================================================
	// Point set and landmark manipulation functions
	//===========================================================================
	// Using Kabsch's algorithm for aligning shapes
	//This assumes that align_from and align_to are already mean normalised
	cv::Matx22f AlignShapesKabsch2D(const cv::Mat_<float>& align_from, const cv::Mat_<float>& align_to)
	{

		cv::SVD svd(align_from.t() * align_to);

		// make sure no reflection is there
		// corr ensures that we do only rotaitons and not reflections
		float d = cv::determinant(svd.vt.t() * svd.u.t());

		cv::Matx22f corr = cv::Matx22f::eye();
		if (d > 0)
		{
			corr(1, 1) = 1;
		}
		else
		{
			corr(1, 1) = -1;
		}

		cv::Matx22f R;
		cv::Mat(svd.vt.t()*cv::Mat(corr)*svd.u.t()).copyTo(R);

		return R;
	}

	//=============================================================================
	// Basically Kabsch's algorithm but also allows the collection of points to be different in scale from each other
	cv::Matx22f AlignShapesWithScale(cv::Mat_<float>& src, cv::Mat_<float> dst)
	{
		int n = src.rows;

		// First we mean normalise both src and dst
		float mean_src_x = cv::mean(src.col(0))[0];
		float mean_src_y = cv::mean(src.col(1))[0];

		float mean_dst_x = cv::mean(dst.col(0))[0];
		float mean_dst_y = cv::mean(dst.col(1))[0];

		cv::Mat_<float> src_mean_normed = src.clone();
		src_mean_normed.col(0) = src_mean_normed.col(0) - mean_src_x;
		src_mean_normed.col(1) = src_mean_normed.col(1) - mean_src_y;

		cv::Mat_<float> dst_mean_normed = dst.clone();
		dst_mean_normed.col(0) = dst_mean_normed.col(0) - mean_dst_x;
		dst_mean_normed.col(1) = dst_mean_normed.col(1) - mean_dst_y;

		// Find the scaling factor of each
		cv::Mat src_sq;
		cv::pow(src_mean_normed, 2, src_sq);

		cv::Mat dst_sq;
		cv::pow(dst_mean_normed, 2, dst_sq);

		float s_src = sqrt(cv::sum(src_sq)[0] / n);
		float s_dst = sqrt(cv::sum(dst_sq)[0] / n);

		src_mean_normed = src_mean_normed / s_src;
		dst_mean_normed = dst_mean_normed / s_dst;

		float s = s_dst / s_src;

		// Get the rotation
		cv::Matx22f R = AlignShapesKabsch2D(src_mean_normed, dst_mean_normed);

		cv::Matx22f	A;
		cv::Mat(s * R).copyTo(A);

		cv::Mat_<float> aligned = (cv::Mat(cv::Mat(A) * src.t())).t();
		cv::Mat_<float> offset = dst - aligned;

		float t_x = cv::mean(offset.col(0))[0];
		float t_y = cv::mean(offset.col(1))[0];

		return A;

	}


	//===========================================================================
	// Visualisation functions
	//===========================================================================
	void Project(cv::Mat_<float>& dest, const cv::Mat_<float>& mesh, float fx, float fy, float cx, float cy)
	{
		dest = cv::Mat_<float>(mesh.rows, 2, 0.0);

		int num_points = mesh.rows;

		float X, Y, Z;


		cv::Mat_<float>::const_iterator mData = mesh.begin();
		cv::Mat_<float>::iterator projected = dest.begin();

		for (int i = 0; i < num_points; i++)
		{
			// Get the points
			X = *(mData++);
			Y = *(mData++);
			Z = *(mData++);

			float x;
			float y;

			// if depth is 0 the projection is different
			if (Z != 0)
			{
				x = ((X * fx / Z) + cx);
				y = ((Y * fy / Z) + cy);
			}
			else
			{
				x = X;
				y = Y;
			}

			// Project and store in dest matrix
			(*projected++) = x;
			(*projected++) = y;
		}

	}

	//===========================================================================
	// Angle representation conversion helpers
	//===========================================================================

	// Using the XYZ convention R = Rx * Ry * Rz, left-handed positive sign
	cv::Matx33f Euler2RotationMatrix(const cv::Vec3f& eulerAngles)
	{
		cv::Matx33f rotation_matrix;

		float s1 = sin(eulerAngles[0]);
		float s2 = sin(eulerAngles[1]);
		float s3 = sin(eulerAngles[2]);

		float c1 = cos(eulerAngles[0]);
		float c2 = cos(eulerAngles[1]);
		float c3 = cos(eulerAngles[2]);

		rotation_matrix(0, 0) = c2 * c3;
		rotation_matrix(0, 1) = -c2 *s3;
		rotation_matrix(0, 2) = s2;
		rotation_matrix(1, 0) = c1 * s3 + c3 * s1 * s2;
		rotation_matrix(1, 1) = c1 * c3 - s1 * s2 * s3;
		rotation_matrix(1, 2) = -c2 * s1;
		rotation_matrix(2, 0) = s1 * s3 - c1 * c3 * s2;
		rotation_matrix(2, 1) = c3 * s1 + c1 * s2 * s3;
		rotation_matrix(2, 2) = c1 * c2;

		return rotation_matrix;
	}
	
	// Using the XYZ convention R = Rx * Ry * Rz, left-handed positive sign
	cv::Vec3f RotationMatrix2Euler(const cv::Matx33f& rotation_matrix)
	{
		float q0 = sqrt(1 + rotation_matrix(0, 0) + rotation_matrix(1, 1) + rotation_matrix(2, 2)) / 2.0f;
		float q1 = (rotation_matrix(2, 1) - rotation_matrix(1, 2)) / (4.0f*q0);
		float q2 = (rotation_matrix(0, 2) - rotation_matrix(2, 0)) / (4.0f*q0);
		float q3 = (rotation_matrix(1, 0) - rotation_matrix(0, 1)) / (4.0f*q0);

		float t1 = 2.0f * (q0*q2 + q1*q3);

		float yaw = asin(2.0 * (q0*q2 + q1*q3));
		float pitch = atan2(2.0 * (q0*q1 - q2*q3), q0*q0 - q1*q1 - q2*q2 + q3*q3);
		float roll = atan2(2.0 * (q0*q3 - q1*q2), q0*q0 + q1*q1 - q2*q2 - q3*q3);

		return cv::Vec3f(pitch, yaw, roll);
	}

	cv::Vec3f Euler2AxisAngle(const cv::Vec3f& euler)
	{
		cv::Matx33f rotMatrix = Euler2RotationMatrix(euler);
		cv::Vec3f axis_angle;
		cv::Rodrigues(rotMatrix, axis_angle);
		return axis_angle;
	}

	cv::Vec3f AxisAngle2Euler(const cv::Vec3f& axis_angle)
	{
		cv::Matx33f rotation_matrix;
		cv::Rodrigues(axis_angle, rotation_matrix);
		return RotationMatrix2Euler(rotation_matrix);
	}

	cv::Matx33f AxisAngle2RotationMatrix(const cv::Vec3f& axis_angle)
	{
		cv::Matx33f rotation_matrix;
		cv::Rodrigues(axis_angle, rotation_matrix);
		return rotation_matrix;
	}

	cv::Vec3f RotationMatrix2AxisAngle(const cv::Matx33f& rotation_matrix)
	{
		cv::Vec3f axis_angle;
		cv::Rodrigues(rotation_matrix, axis_angle);
		return axis_angle;
	}

	//============================================================================
	// Matrix reading functionality
	//============================================================================

	// Reading in a matrix from a stream
	void ReadMat(std::ifstream& stream, cv::Mat &output_mat)
	{
		// Read in the number of rows, columns and the data type
		int row, col, type;

		stream >> row >> col >> type;

		output_mat = cv::Mat(row, col, type);

		switch (output_mat.type())
		{
		case CV_64FC1:
		{
			cv::MatIterator_<double> begin_it = output_mat.begin<double>();
			cv::MatIterator_<double> end_it = output_mat.end<double>();

			while (begin_it != end_it)
			{
				stream >> *begin_it++;
			}
		}
		break;
		case CV_32FC1:
		{
			cv::MatIterator_<float> begin_it = output_mat.begin<float>();
			cv::MatIterator_<float> end_it = output_mat.end<float>();

			while (begin_it != end_it)
			{
				stream >> *begin_it++;
			}
		}
		break;
		case CV_32SC1:
		{
			cv::MatIterator_<int> begin_it = output_mat.begin<int>();
			cv::MatIterator_<int> end_it = output_mat.end<int>();
			while (begin_it != end_it)
			{
				stream >> *begin_it++;
			}
		}
		break;
		case CV_8UC1:
		{
			cv::MatIterator_<uchar> begin_it = output_mat.begin<uchar>();
			cv::MatIterator_<uchar> end_it = output_mat.end<uchar>();
			while (begin_it != end_it)
			{
				stream >> *begin_it++;
			}
		}
		break;
		default:
			printf("ERROR(%s,%d) : Unsupported Matrix type %d!\n", __FILE__, __LINE__, output_mat.type()); abort();


		}
	}

	void ReadMatBin(std::ifstream& stream, cv::Mat &output_mat)
	{
		// Read in the number of rows, columns and the data type
		int row, col, type;

		stream.read((char*)&row, 4);
		stream.read((char*)&col, 4);
		stream.read((char*)&type, 4);

		output_mat = cv::Mat(row, col, type);
		int size = output_mat.rows * output_mat.cols * output_mat.elemSize();
		stream.read((char *)output_mat.data, size);

	}

	// Skipping lines that start with # (together with empty lines)
	void SkipComments(std::ifstream& stream)
	{
		while (stream.peek() == '#' || stream.peek() == '\n' || stream.peek() == ' ' || stream.peek() == '\r')
		{
			std::string skipped;
			std::getline(stream, skipped);
		}
	}


}