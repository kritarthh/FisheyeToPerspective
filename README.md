# Requirements
+ opencv
+ hdf5

# Usage
+ Place the calibration file in the `files/` directory with name `calib.txt`
+ Place the folders or files that you want to convert inside the `files/input/` folder
+ The output will be generated in the `files/output/` folder in the same folder structure as the input folder structure
+ The execulable takes 7 arguments
  + arg1: the location of directory containing the `files/` directory
  + arg2: horizontal left view angle
  + arg3: horizontal right view angle
  + arg4: horizontal view angles count
  + arg5: vertial down view angle
  + arg6: vertial up view angle
  + arg7: horizontal view angles count
+ Example Usage:
  + ./fishToPersp -30 +30 11 0 0 1
