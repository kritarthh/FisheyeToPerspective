#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <dirent.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

bool replace(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}

int listdir(string path, std::vector<string>& files)
{
    struct dirent *entry;
    DIR *dp;

    dp = opendir(path.c_str());
    if (dp == NULL) {
        // perror("opendir: Path does not exist or could not be read.");
        return -1;
    }

    // create output folder if doesnot exist
    replace(path, "/output/", "/output_cropped/");
    system(("mkdir -p " + path).c_str());
    replace(path, "/output_cropped/", "/output/");

    while ((entry = readdir(dp))) {
        string name = string(entry->d_name);
        if (name == "." || name == "..") {
            continue;
        }
        if (name.size() > 4 && name.substr(name.size() - 4) == ".jpg") {
            cout << "Image found: " << name << endl;
            files.push_back(path + "/" + name);
            // fishToPersp(path + "/" + name, fs, o);
        } else {
            listdir(path + "/" + name, files);
        }
    }
    closedir(dp);
    return 0;
}

int main(int argc, char const *argv[])
{
    std::vector<string> files;

    string dirPath = string(realpath(argv[1], NULL)) + "/files/output/";

    listdir(dirPath, files);

    #pragma omp parallel for
    for (int i = 0; i < files.size(); ++i) {
        string path = files[i];
        replace(path, "/output/", "/output_cropped/");
        if (FILE *file = fopen(path.c_str(), "r")) {
            fclose(file);
            cout << i << " Skipping: " << path.substr(path.find_last_of("/") + 1) << endl;
            continue;
        }
        replace(path, "/output_cropped/", "/output/");
        cout << i << " Cropping and Resizing: " << path.substr(path.find_last_of("/") + 1) << endl;
        Mat image = imread(path);
        replace(path, "/output/", "/output_cropped/");
        Mat outImage(Size(320, 160), CV_8UC3);
        cv::Rect roi;
        roi.x = 0;
        roi.y = image.rows / 2;
        roi.width = image.cols;
        roi.height = image.rows / 2;

        /* Crop the original image to the defined ROI */
        Mat crop = image(roi);
        resize(crop, outImage, Size(320, 160));
        // cv::imshow("crop", outImage);
        // cv::waitKey(0);

        cv::imwrite(path, outImage);
    }
    return 0;
}