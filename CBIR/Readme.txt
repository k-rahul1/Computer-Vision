Content-based Image Retrieval

////Author////
Rahul Kumar
Northeastern University

//////Time travel day//////
Note: 1 time travel day is used



Operating System - Ubuntu 20.04.5 LTS
IDE - VS Code

Following folders are available in CBIR code file:
Note: To execute the whole program, first run the feature generation program and then run CBIR program

1. feature_generation_prog
---contents: main.cpp, feature_generator.cpp, feature_generator.h, filter.cpp, filter.h, csv_util.cpp, csv_util.h, Makefile
Instructions to run the program:
1. Type 'make' in the terminal to build the file.
2. Type './feature' to run the executable //also pass the path of directory of image database as argument
3. Input the option to apply any feature method

It will generate a csv file with image names and corresponding feature vectors.

2. CBIR_prog
---contents: CBIR.cpp, distanceMetric.cpp, distanceMetric.h, feature_generator.cpp, feature_generator.h, filter.cpp, filter.h, csv_util.cpp, csv_util.h, Makefile
Instructions to run the program:
1. Type 'make' in the terminal to build the file.
2. Type './feature' to run the executable
3. Input the target image name in the terminal  //keep the image in the working directory
4. Input the option to apply any feature method //keep the corresponding csv file in the working directory
5. Input the number of matches required

The program will return the image names of N best matches

Note: To implement the custom matching method, download the dataset from the below link and use this dataset to generate the csv file:
https://drive.google.com/drive/folders/1416kdgWgVzrJdirZaoS2c122Zc4zR-8_?usp=share_link

This link contains the images of sunset along with images from the Olympus dataset.
