Lane Detection and Obstacle Avoidance System

Overview

This project implements a real-time lane detection and obstacle avoidance system using computer vision techniques. It processes video input to detect road lanes, identify obstacles (e.g., orange or blue objects), and provide driving decisions (e.g., "Go Straight," "Turn Left," "Turn Right") using a state machine. The system is built with Python, leveraging OpenCV for image processing, NumPy for numerical computations, and a custom LaneStateMachine class for stable decision-making.

Key features:





Detects yellow lane lines using LAB color space and Hough Transform.



Identifies obstacles using HSV color-based segmentation.



Shades the lane region and calculates a center line for navigation.



Uses a state machine to ensure stable driving decisions based on lane and obstacle detection.



Processes video input at 960x540 resolution with a 5-frame stability threshold for decisions.

Requirements





Python 3.8+



OpenCV (opencv-python) 4.5.0+



NumPy 1.21.0+



A video file (e.g., vid1.mp4) or webcam for input.

Installation





Clone the Repository:

git clone <repository-url>
cd <repository-folder>



Set Up a Virtual Environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate



Install Dependencies:

pip install opencv-python numpy



Prepare Video Input:





Place your video file (e.g., vid1.mp4) in the DIP Project Videos (1) folder or update the videoPath in main.py to point to your video file.

Usage





Run the Script:

python main.py



Input:





The script processes the video specified in videoPath (default: ./DIP Project Videos (1)/vid1.mp4).



Alternatively, modify main.py to use a webcam by setting cap = cv2.VideoCapture(0).



Output:





Windows:





Lane and Obstacle Detection: Displays the video with detected lanes (yellow overlay), center line (green), obstacles (red rectangles), and driving decisions (text).



Combined: Shows the yellow lane mask.



Edges: Displays the Canny edge detection output.



Obstacle Detection Mask: Shows the orange/blue obstacle mask.



Decisions: Text on the output frame indicates actions (e.g., "Go Straight," "Turn RIGHT - Obstacle on LEFT") in colors:





Green: Safe to proceed.



Red: Turn or stop due to obstacles.



Yellow: No lane detected.



Controls:





Press q to exit the video processing loop.

Project Structure





main.py: Main script containing the lane detection, obstacle identification, and state machine logic.



DIP Project Videos (1)/vid1.mp4: Sample video file for testing (replace with your own video).

How It Works





Lane Detection:





Crops the video frame (10% margin on sides, 30% from top) to focus on the road.



Converts to LAB color space to detect yellow lanes using a color mask.



Applies morphological operations and Canny edge detection to identify lane edges.



Uses Hough Transform to detect lane lines, classified as left or right based on slope and angle.



Obstacle Detection:





Converts the frame to HSV color space to detect orange and blue objects (potential obstacles).



Identifies contours with area > 500 pixels and draws bounding boxes.



Decision Making:





A LaneStateMachine ensures stable decisions by requiring 5 consistent frames before changing state.



Decisions include:





"No Lane Detected" if no lanes are found.



"Go Straight" if lanes are detected without obstacles.



"Turn LEFT/RIGHT - Obstacle on RIGHT/LEFT" if an obstacle overlaps the lane region.



Visualization:





Shades the lane region in yellow and draws a green center line.



Overlays obstacles with red rectangles and displays decisions as text.

Example Output

For a frame with detected lanes and an obstacle:





Lanes are shaded yellow, with a green center line.



An obstacle (e.g., orange cone) is marked with a red rectangle.



Text like "Turn RIGHT - Obstacle on LEFT" appears in red if an obstacle is detected in the lane.

Limitations





Optimized for yellow lanes; may require tuning for white or other lane colors.



Obstacle detection is limited to orange and blue objects; adjust HSV ranges for other colors.



Assumes clear video input; performance may degrade in low-light or noisy conditions.

Future Improvements





Add support for white lane detection by expanding the color mask.



Implement machine learning (e.g., CNNs) for more robust obstacle detection.



Optimize for real-time performance on lower-end hardware.



Add logging for decision history and performance metrics.


