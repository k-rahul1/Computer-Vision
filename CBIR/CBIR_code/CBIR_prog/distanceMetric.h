//Created by Rahul Kumar

//Function declation for different distance metric methods

//Implementing Euclidean distance as distance metric
float blDistanceMetric(std::vector<float> &target, std::vector<float> &feature);

//Implementing Histogram Intersection as distance metric
float histDistanceMetric(std::vector<float> &target, std::vector<float> &feature);

//Implementing histogram intersection as distance metric
float MultihistDistanceMetric(std::vector<float> &target1, std::vector<float> &target2, std::vector<float> &feature1, std::vector<float> &feature2 );

//Implementing Euclidean distance as distance metric
float colorTextureDistanceMetric(std::vector<float> &target, std::vector<float> &feature);

//Implementing weighted Euclidean distance as distance metric
float customDistanceMetric(std::vector<float> &target,std::vector<float> &feature);

// Implementing Manhattan Distance as distance metric
float gaborDistanceMetric(std::vector<float> &target,std::vector<float> &feature);

//Implementing Manhattan distance for Color Histogram and calculating standard deviation for Spatial variance
float spatialDistanceMetric(std::vector<float> &target,std::vector<float> &feature);