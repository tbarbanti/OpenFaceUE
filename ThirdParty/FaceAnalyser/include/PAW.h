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

#ifndef __PAWA_h_
#define __PAWA_h_

// OpenCV includes
#include <opencv2/core/core.hpp>

namespace FaceAnalysis
{
  //===========================================================================
  /** 
      A Piece-wise Affine Warp
	  The ideas for this piece-wise affine triangular warping are taken from the
	  Active appearance models revisited by Iain Matthews and Simon Baker in IJCV 2004
	  This is used for both validation of landmark detection, and for avatar animation

	  The code is based on the CLM tracker by Jason Saragih et al.
  */	

class PAW{
public:    
	// Number of pixels after the warping to neutral shape
    int     number_of_pixels; 

	// Minimum x coordinate in destination
    float  min_x;

	// minimum y coordinate in destination
	float  min_y;

	// Destination points (landmarks to be warped to)
	cv::Mat_<float> destination_landmarks;

	// Destination points (landmarks to be warped from)
	cv::Mat_<float> source_landmarks;

	// Triangulation, each triangle is warped using an affine transform
	cv::Mat_<int> triangulation;

	// Triangle index, indicating which triangle each of destination pixels lies in
	cv::Mat_<int> triangle_id;

	// Indicating if the destination warped pixels is valid (lies within a face)
	cv::Mat_<uchar> pixel_mask;

	// A number of precomputed coefficients that are helpful for quick warping
	
	// affine coefficients for all triangles (see Matthews and Baker 2004)
	// 6 coefficients for each triangle (are computed from alpha and beta)
	// This is computed during each warp based on source landmarks
	cv::Mat_<float> coefficients;

	// matrix of (c,x,y) coeffs for alpha
	cv::Mat_<float> alpha;

	// matrix of (c,x,y) coeffs for alpha
	cv::Mat_<float> beta;

	// x-source of warped points
	cv::Mat_<float> map_x;

	// y-source of warped points
	cv::Mat_<float> map_y;

	// Default constructor
    PAW(){;}

	// Construct a warp from a destination shape and triangulation
	PAW(const cv::Mat_<float>& destination_shape, const cv::Mat_<int>& triangulation);

	// The final optional argument allows for optimisation if the triangle indices from previous frame are known (for tracking in video)
	PAW(const cv::Mat_<float>& destination_shape, const cv::Mat_<int>& triangulation, float in_min_x, float in_min_y, float in_max_x, float in_max_y);

	// Copy constructor
	PAW(const PAW& other);

	void Read(std::ifstream &s);

	// The actual warping
    void Warp(const cv::Mat& image_to_warp, cv::Mat& destination_image, const cv::Mat_<float>& landmarks_to_warp);
	
	// Compute coefficients needed for warping
    void CalcCoeff();

	// Perform the actual warping
    void WarpRegion(cv::Mat_<float>& map_x, cv::Mat_<float>& map_y);

    inline int NumberOfLandmarks() const {return destination_landmarks.rows/2;} ;
    inline int NumberOfTriangles() const {return triangulation.rows;} ;

	// The width and height of the warped image
    inline int constWidth() const {return pixel_mask.cols;}
    inline int Height() const {return pixel_mask.rows;}
    
private:

	// Helper functions for dealing with triangles
	static bool sameSide(float x0, float y0, float x1, float y1, float x2, float y2, float x3, float y3);
	static bool pointInTriangle(float x0, float y0, float x1, float y1, float x2, float y2, float x3, float y3);
	static int findTriangle(const cv::Point_<float>& point, const std::vector<std::vector<float>>& control_points, int guess = -1);

  };
  //===========================================================================
}
#endif
