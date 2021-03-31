

#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/xfeatures2d.hpp>
#include "opencv2/core.hpp"
#include <fstream>



#include <iostream>
using namespace cv;
using namespace std;


int load(vector<Mat> &images,vector <int> &labels ) {

vector<String> car;
glob("/home/ash/opencv_test/object_dataset/car/*.jpg", car, false);


Mat img;
size_t count_car = car.size(); //number of png files in images folder
for (size_t i=0; i<count_car; i++)
{	
	img = imread(car[i],0);
	resize(img,img, Size(256,256));


	images.push_back(img);
	labels.push_back(0);

}

vector<String> cat;
glob("/home/ash/opencv_test/object_dataset/cat/*.jpg", cat, false);
size_t count_cat = cat.size(); //number of png files in images folder
for (size_t i=0; i<count_cat; i++)
{	
	img = imread(cat[i],0);
	resize(img,img, Size(256,256));


	images.push_back(img);
	labels.push_back(1);

}



vector<String> flower;
glob("/home/ash/opencv_test/object_dataset/flower/*.jpg", flower, false);
size_t count_flower = flower.size(); //number of png files in images folder
for (size_t i=0; i<count_flower; i++)
{	
	img = imread(flower[i],0);
	resize(img,img, Size(256,256));
	

	images.push_back(img);
	labels.push_back(2);

}

return 0;
}

int feature_extract(vector <Mat>images,vector <int>labels, Mat &featuresUnclustered)
{
int total;
for(size_t i =0; i<images.size(); i++)
{

Ptr<xfeatures2d::SURF> siftptr;
Mat descriptors;
siftptr = xfeatures2d::SURF::create(100,4,3,true, true);


vector<KeyPoint> keypoints;

siftptr->detectAndCompute(images[i],noArray(),keypoints,descriptors);
Mat output_image_1;
drawKeypoints (images[i], keypoints, output_image_1);


featuresUnclustered.push_back(descriptors);

}


return 0;
}


int kmeans(Mat desc){
//int i;
/*
Mat labels;
  Mat img(500, 500, CV_8UC3);
Mat centers;
    Scalar colorTab[] =
    {
        Scalar(0, 0, 255),
        Scalar(0,255,0),
        Scalar(255,100,100),
        Scalar(255,0,255),
        Scalar(0,255,255)
    };
	cout<<"hI";
// double compactness = kmeans(desc, 3, labels,
            TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
               3, KMEANS_PP_CENTERS, centers);
//cout<<centers;
        img = Scalar::all(0);



/*        for( i = 0; i < 300; i++ )
        {
            int clusterIdx = labels.at<int>(i);
            Point ipt = desc.at<Point2f>(i);
            circle( img, ipt, 2, colorTab[clusterIdx], FILLED, LINE_AA );
        }
        for (i = 0; i < (int)centers.size(); ++i)
        {
            Point2f c = centers[0][i];
            circle( img, c, 40, colorTab[i], 1, LINE_AA );
        }
        cout << "Compactness: " << compactness << endl;
        imshow("clusters", img);
	waitKey(0);
*/



    cv::Mat projectedPointsImage = cv::Mat(512, 512, CV_8UC3, cv::Scalar::all(255));


    cv::Mat labels; cv::Mat centers;
    int k =3; // searched clusters in k-means

    cv::kmeans(desc, k, labels, cv::TermCriteria(TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0), 10, cv::KMEANS_PP_CENTERS, centers);

	ofstream outfile;

        outfile.open("clusters.txt");
        if(outfile.is_open()){
            for(int i=0; i<3; i++){
                for(int j=0; j<128; j++){
                    cout<<centers.at<float>(i,j)<<" ";
                    outfile<<centers.at<float>(i,j)<<"\n ";
                }
                cout<<endl;
                outfile<<endl;
            }
            outfile.close();
return 0;
}
int main(){
vector<Mat> images;
Mat features;
vector <int> labels;
load(images,labels);
feature_extract(images,labels,features);
cout<<features.dims;

kmeans(features);
}
