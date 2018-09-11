# Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Goals
The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[distorted_chessboard]: ./doc/images/distorted_chessboard.png "Distorted Chessboard"
[undistorted_chessboard]: ./doc/images/undistorted_chessboard.png "Undistorted Chessboard"
[topview_images]: ./doc/images/topview_images.png "Top view images"

[channel_comparison1]: ./doc/images/channel_comparison1.png "Color channel comparison"
[channel_comparison2]: ./doc/images/channel_comparison2.png "Color channel comparison"

[gauss_derivative]: ./doc/images/gauss_derivative.png "Gaussian derivative curve"
[gauss]: ./doc/images/gauss.png "Gauss curve"
[spatial_kernel]: ./doc/images/spatial_kernel.png "Spatial kernel"
[merged_thresholded_response]: ./doc/images/merged_thresholded_response.png "Merged thresholded response"

[lane_histogram]: ./doc/images/lane_histogram.png "Lane Histogram"
[bounding_boxes_left]: ./doc/images/bounding_boxes_left.png "Bounding boxes yellow line"
[bounding_boxes_right]: ./doc/images/bounding_boxes_right.png "Bounding boxes white line"
[polynomials_search_left]: ./doc/images/polynomials_search_left.png "Search with polynomials yellow line"
[polynomials_search_right]: ./doc/images/polynomials_search_right.png "Search with polynomials white line"

[detected_lane]: ./doc/images/detected_lane.png "Detected lane"

## Camera Calibration

The code for this step is contained in a IPython notebook located in 
"./Camera_Calibration.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `all_object_points` is just a replicated array of coordinates, and `object_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `corners` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection to `all_image_points`.  

I then used the output `all_object_points` and `all_image_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

| Distorted Image with <br> found points| Undistorted Image |
|:---:|:---:|
| ![alt text][distorted_chessboard] | ![alt text][undistorted_chessboard] | 

The distortion coefficients and camera matrix are stored in a pickle file `camera_cal_data/camera_calibration_data.p` to use them later in lane detection pipline. 


## Pipeline

The code pipeline is contained in a separate IPython notebook that is located at `./Advanced_Lane_Detection.ipynb` It shows step by step what has been done to detect the current lane. Each explanation is followed by code and example images that show effects.   

The implementation is based on the technics taught in the online course ["Self Driving Car Engineer" from Udacity](https://udacity.com/course/self-driving-car-engineer-nanodegree--nd013) and on the paper ["Real time Detection of Lane Markers in Urban Streets" by Mohamed Aly](http://www.mohamedaly.info/research/lane-detection)

### 1. Image distortion correction and top-view

The driving lane is detected best in an undistorted top view image. Lines that are actually parallel, converge at the horizon in an images captured by a camera. The calculation of a top view gets rid of this effect, so that lane markings appear parallel. Further we may focus only on the road.

The previously stored distortion coefficients and camera matrix are used to correct the distortion. This works the same way as it was done above with the chessboard. The top view is calculated by mapping a straight lane to a rectangle. 
```
# top left, top right, bottom right, bottom left corners of
# a straight lane
corners = [[571,464], [713,464], [1032, 661], [288,661]]
margin_horz = 240
```
The rectangle is calculated by
```
# top left, top right, bottom right, bottom left corners of
dest_rect= np.float32([
    [margin_horz,             0],
    [img_width - margin_horz, 0],
    [img_width - margin_horz, img_height],
    [margin_horz,             img_height]])
``` 

The correction and warping is done in code cell 3 in the IPython file. This shows the result of several images.  

![alt text][topview_images]

#### Evaluation of color channel

We need to know which color channels are useful to extract lane dividers. Lane dividers have three color properties, hue (the color), saturation and lightness, to contrast with the road. The saturation of a yellow line is usually high, whereas we may count on brightness for white lines. In order to take advantage of this we may transform the image to the HLS color space.

![alt text][channel_comparison1]
![alt text][channel_comparison2]

We can see that the saturation channel and hue channel is suitable to extract yellow lines, whereas the lightness channel seems to extract white markers.

### 2. Marker extraction
Different to the approach from Udacity, I use gaussian spatial filters to detect vertical lines, as described in Mohamed Aly's paper on the lighness channel and on the saturation channel. The response of this filter is then thresholded.

Two functions are used to build the filter. A derivative of gaussian curve in horizontal directions and a gaussian curve for vertical directions.

|Derivative of gaussian curve in horizontal directions|Gaussian curve in Vertical directions|Kernel|
|:---:|:---:|:---:|
| ![alt text][gauss_derivative] | ![alt text][gauss] | ![alt text][spatial_kernel] |
 
The response of the kernel is high when it is moved over vertical lines. This is done on the saturation channel and on the lightness channel of the top view image. Masks that are created using the top view are applied on the response image to remove responses on dark lines. And on lines that are not yellow. So we retain two images. One for yellow lines and one for white lines. As we are interested on high responses we threshold and normalize the response images. See Code cells 6, 7, 8, 9  and 10

This is the merged thresholded response 

![alt text][merged_thresholded_response]

The lane markers stand out now clearly. Both images are merged here only to visualize the lane markers in one image. But that is not useful for further processing, as the yellow line on the left might be merged with some not yellow detections. The algorithm proceeds with the yellow lane marker and white lane marker separately.   
 
### 2. Find and Filter Line Candidates

The candidates for lines are found using a smoothed histogram, that is generated for the bottom third of image. The local maxima in the histogram are used as a location hypothesis for line extraction. The algorithm does not limit the count of lines. It tries to find as many as it could, provided that the peaks in the histogram exceed a certain threshold. See Code Cell 13,14

![alt text][lane_histogram]

Sliding windows are drawn around the candidates and moved up along a probable path to the top of the images. If a window doesn't contain enough points, the area is not taken into account for further processing. The algorithm stops if there are too many empty windows or the top of the image is reached. See Cell 16

| Bounding boxes <br> yellow line| Bounding boxes <br> white line |
|:---:|:---:|
| ![alt text][bounding_boxes_left] | ![alt text][bounding_boxes_right] | 

Second order polynomials are fitted based on the found areas. These serve as input for fitting with polynomials as this is a less expensive operation. Only if not enough lines are found by searching in areas spanned by polynomials, the algorithm falls back to the sliding window search. See Code Cell 17

| Search with polynomials <br> yellow line| Search with polynomials <br> white line |
|:---:|:---:|
| ![alt text][polynomials_search_left] | ![alt text][polynomials_search_right] | 
 
 ### 3. Filter Polynomials
 
 The previous step may detect more then two polynomials, but we need only two to find the current lane.
 
 The space between lane markings that confine a lane must be at least as broad as a car or truck. And also not much larger. The lane markings are parallel to each other. So their polynomials f(x) = Ax^2 + Bx should be similar. Some sample points are generated along these polynomials and the delta of them is summarized. This gives a measure how different those functions are. Further the lane markings should be defined by a fair amount of pixels, whereas false detections may not be defined by that many. These properties can be used to filter false markings and select the best ones. See Code Cell 20

The length of the line may tell how sure we are that this is a lane marking, as lane markings should lead from the bottom to the top of an image. This measure is used to update previous detection.

### 4. Calculate Curvature and Lane Position

The curvature radius R for a polynomial f(x) = Ax^2 + Bx is calculated by:
```
R = ((1 + (2 * A * y + B) ** 2) ** (3/2)) / np.abs(2 * A)
```

As this is the curvature in pixels we should convert the inputs to real world space. We have to messure how long and wide the warped lane section are. See code sell 18. 
```
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30 / length_of_lane_in_pixel # meters per pixel in y dimension
xm_per_pix = 3.7 / size_of_lane_in_pixel  # meters per pixel in x dimension
```

The position in the lane is calculated by substracting the image center from lane center. The lane center can be calculated by getting the middle of a point on the left and a point on the right polynomial. See code cell 19

### 5. Draw the lane on a camera image

Draw the a polygon along the two polynoms and and warp the image back to the original space using the inverse pespective matrix. Add this image to the original camera image. See code cell 21

![alt text][detected_lane]

### 6. Next round 

Continue with the next image and use the previous detection to support the next detection. 
* Sample 70% percent of the detected points and add them to a image generated by marker extraction.
* Update old lines with a length factor (length of detected lane marker / image height)

The whole pipeline is implemented in code cell 22

## Result

Here are the final videos

* [Project Video](https://github.com/monsieurmona/CarND-Advanced-Lane-Lines/blob/master/output_video/project_video_output.mp4)
* [Challenge video](https://github.com/monsieurmona/CarND-Advanced-Lane-Lines/blob/master/output_video/challenge_video_output.mp4)

## Discussion

This pipeline will likely fail at crossings where we have lot more lines. It might not do well at night. 

I would think a 3D image would help to improve the detection a lot. We could classify the ground and detect only on this area, where all obstacles were removed.

Further the image brightness and contrast should be adjusted to the lightning conditions.    




