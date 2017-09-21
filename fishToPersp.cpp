#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;
#define PI 3.1415926536

#define CMV_MAX_BUF 1024
#define MAX_POL_LENGTH 64

struct ocam_model
{
    double pol[MAX_POL_LENGTH];    // the polynomial coefficients: pol[0] + x"pol[1] + x^2*pol[2] + ... + x^(N-1)*pol[N-1]
    int    length_pol;             // length of polynomial
    double invpol[MAX_POL_LENGTH]; // the coefficients of the inverse polynomial
    int    length_invpol;          // length of inverse polynomial
    double xc;                     // row coordinate of the center
    double yc;                     // column coordinate of the center
    double c;                      // affine parameter
    double d;                      // affine parameter
    double e;                      // affine parameter
    int    width;                  // image width
    int    height;                 // image height
};

int get_ocam_model(struct ocam_model *myocam_model, char *filename)
{
    double *pol           = myocam_model->pol;
    double *invpol        = myocam_model->invpol; 
    double *xc            = &(myocam_model->xc);
    double *yc            = &(myocam_model->yc); 
    double *c             = &(myocam_model->c);
    double *d             = &(myocam_model->d);
    double *e             = &(myocam_model->e);
    int    *width         = &(myocam_model->width);
    int    *height        = &(myocam_model->height);
    int    *length_pol    = &(myocam_model->length_pol);
    int    *length_invpol = &(myocam_model->length_invpol);
    FILE   *f;
    char    buf[CMV_MAX_BUF];
    int     i;

    //Open file
    if(!(f=fopen(filename,"r"))){
        printf("File %s cannot be opened\n", filename);                
        return -1;
    }

    //Read polynomial coefficients
    fgets(buf,CMV_MAX_BUF,f);
    fscanf(f,"\n");
    fscanf(f,"%d", length_pol);
    for (i = 0; i < *length_pol; i++){
        fscanf(f," %lf",&pol[i]);
    }

    //Read inverse polynomial coefficients
    fscanf(f,"\n");
    fgets(buf,CMV_MAX_BUF,f);
    fscanf(f,"\n");
    fscanf(f,"%d", length_invpol);
    for (i = 0; i < *length_invpol; i++){
        fscanf(f," %lf",&invpol[i]);
    }

    //Read center coordinates
    fscanf(f,"\n");
    fgets(buf,CMV_MAX_BUF,f);
    fscanf(f,"\n");
    fscanf(f,"%lf %lf\n", xc, yc);

    //Read affine coefficients
    fgets(buf,CMV_MAX_BUF,f);
    fscanf(f,"\n");
    fscanf(f,"%lf %lf %lf\n", c,d,e);

    //Read image size
    fgets(buf,CMV_MAX_BUF,f);
    fscanf(f,"\n");
    fscanf(f,"%d %d", height, width);

    fclose(f);
    return 0;
}

Point2f getInputPoint(int x, int y,float width, float height, Point2f &center, float angle_x, float angle_y)
{
    Point2f pfish;
    float theta, phi, r;
    Point3f psph;
    Point3f psph2;
    Point3f psph3;
    
    float FOV =(float)PI/180 * 161.375;
    float FOV2 = (float)PI/180 * 129.1;

    theta = PI * (x / width - 0.5);
    phi = PI * (y / height - 0.5);
    psph.x = cos(phi) * sin(theta);
    psph.y = cos(phi) * cos(theta);
    psph.z = sin(phi);
    psph2.x = psph.x * cos(angle_x) + psph.y * sin(angle_x);
    psph2.y = psph.y * cos(angle_x) - psph.x * sin(angle_x);
    psph2.z = psph.z;
    psph3.z = psph2.z * cos(angle_y) - psph2.y * sin(angle_y);
    psph3.y = psph2.y * cos(angle_y) + psph2.z * sin(angle_y);
    psph3.x = psph2.x;
    theta = atan2(psph3.z,psph3.x);
    phi = atan2(sqrt(psph3.x*psph3.x+psph3.z*psph3.z),psph3.y);
    r = width * phi / FOV;
    pfish.x = center.x + r * cos(theta);
    pfish.y = center.y + r * sin(theta);
    return pfish;
}

Point2f revGetInputPoint(int x, int y, float width, float height, Point2f &center)
{
    Point2f ptpan;
    float theta,phi;
    Point3f psph;
    Point3f psph2;
    
    float FOV =(float)PI/180 * 161.375;
    float FOV2 = (float)PI/180 * 129.1;

    theta = atan2(y - center.y, x - center.x);
    phi = ((float)(x - center.x) / cos(theta)) * FOV / width;

    if (cos(theta) < 0.000001 && cos(theta) > -0.000001) {
        phi = (y - center.y) * FOV / width;
    }

    psph.y = 1/sqrt(1 + tan(phi) * tan(phi));
    psph.x = cos(theta) > 0 ?
                sqrt((1 - psph.y * psph.y)/(1 + tan(theta) * tan(theta))) :
                -sqrt((1 - psph.y * psph.y)/(1 + tan(theta) * tan(theta)));

    if (psph.x < 0.000001 && psph.x > -0.000001) {
        psph.z = tan(phi) / sqrt(1 + tan(phi) * tan(phi));
    } else {
        psph.z = psph.x * tan(theta);
    }

    theta = atan2(psph.x, psph.y);
    phi = asin(psph.z);

    ptpan.x = width * (theta/PI + 0.5);
    ptpan.y = height * (phi/PI + 0.5);

    return ptpan;
}

void world2cam(double point2D[2], double point3D[3], struct ocam_model *myocam_model)
{
    double *invpol     = myocam_model->invpol; 
    double xc          = (myocam_model->xc);
    double yc          = (myocam_model->yc); 
    double c           = (myocam_model->c);
    double d           = (myocam_model->d);
    double e           = (myocam_model->e);
    int    width       = (myocam_model->width);
    int    height      = (myocam_model->height);
    int length_invpol  = (myocam_model->length_invpol);
    double norm        = sqrt(point3D[0]*point3D[0] + point3D[1]*point3D[1]);
    double theta       = atan(point3D[2]/norm);
    double t, t_i;
    double rho, x, y;
    double invnorm;
    int i;

    if (norm != 0) {
        invnorm = 1/norm;
        t  = theta;
        rho = invpol[0];
        t_i = 1;

        for (i = 1; i < length_invpol; i++)
        {
            t_i *= t;
            rho += t_i*invpol[i];
        }

        x = point3D[0]*invnorm*rho;
        y = point3D[1]*invnorm*rho;

        point2D[0] = x*c + y*d + xc;
        point2D[1] = x*e + y   + yc;
    } else {
        point2D[0] = xc;
        point2D[1] = yc;
    }
}

void create_perspecive_undistortion_LUT( CvMat *mapx, CvMat *mapy, struct ocam_model *ocam_model, float sf)
{
    int i, j;
    int width = mapx->cols; //New width
    int height = mapx->rows;//New height     
    float *data_mapx = mapx->data.fl;
    float *data_mapy = mapy->data.fl;
    float Nxc = height/2.0;
    float Nyc = width/2.0;
    float Nz  = -width/sf;
    double M[3];
    double m[2];

    for (i=0; i<height; i++)
        for (j=0; j<width; j++) {   
            M[0] = (i - Nxc);
            M[1] = (j - Nyc);
            M[2] = Nz;
            world2cam(m, M, ocam_model);
            *( data_mapx + i*width+j ) = (float) m[1];
            *( data_mapy + i*width+j ) = (float) m[0];
        }
}

int main(int argc, char **argv)
{
    if (argc < 5) {
        cout << "Usage: ./fishToPersp <calibration_file> <input_image> <horizontal_viewangle> <vertical_viewangle>\n";
        return 0;
    }

    struct ocam_model o, o_cata;
    get_ocam_model(&o, argv[1]);

    Mat orignalImage = imread(argv[2]);

    if (orignalImage.empty()) {
        cout<<"Empty image\n";
        return 0;
    }

    float theta = strtof(argv[3],0) * PI / 180;
    float phi = strtof(argv[4],0) * PI / 180;

    Mat outImage(orignalImage.rows,orignalImage.cols,CV_8UC3);

    Point2f center;
    center.x = orignalImage.cols >> 1;
    center.y = orignalImage.rows >> 1;

    for(int i = 0; i < outImage.cols; i++)
    {
        for(int j = 0; j < outImage.rows; j++)
        {
            Point2f inP =  getInputPoint(i,j,orignalImage.cols,orignalImage.rows, center, theta, phi);
            Point inP2((int)inP.x,(int)inP.y);
            if(inP2.x >= orignalImage.cols || inP2.y >= orignalImage.rows)
                continue;
            if(inP2.x < 0 || inP2.y < 0)
                continue;
            Vec3b color = orignalImage.at<Vec3b>(inP2);
            outImage.at<Vec3b>(Point(i,j)) = color;
        }
    }

    orignalImage = outImage.clone();

    for(int i = 0; i < outImage.cols; i++)
    {
        for(int j = 0; j < outImage.rows; j++)
        {
            Point2f inP =  revGetInputPoint(i,j,orignalImage.cols,orignalImage.rows, center);
            Point inP2((int)inP.x,(int)inP.y);
            if(inP2.x >= orignalImage.cols || inP2.y >= orignalImage.rows)
                continue;
            if(inP2.x < 0 || inP2.y < 0)
                continue;
            Vec3b color = orignalImage.at<Vec3b>(inP2);
            outImage.at<Vec3b>(Point(i,j)) = color;
        }
    }

    double point3D[3];
    double point2D[2];

    IplImage *src       = new IplImage(outImage);
    IplImage *dst_persp = cvCreateImage(cvGetSize(src), 8, 3);
    CvMat* mapx_persp   = cvCreateMat(src->height, src->width, CV_32FC1);
    CvMat* mapy_persp   = cvCreateMat(src->height, src->width, CV_32FC1);

    float sf = 4.0;
    create_perspecive_undistortion_LUT( mapx_persp, mapy_persp, &o, sf );
    cvRemap(src, dst_persp, mapx_persp, mapy_persp, CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS, cvScalarAll(0));

    // crop image
    int offset_x = (src->width - src->height) >> 1;

    cvSetImageROI(dst_persp, cvRect(offset_x, 0, src->height, src->height));

    IplImage *tmp = cvCreateImage(cvGetSize(dst_persp),
                                  dst_persp->depth,
                                  dst_persp->nChannels);

    cvCopy(dst_persp, tmp, NULL);
    cvResetImageROI(dst_persp);

    dst_persp = cvCloneImage(tmp);

    cvSaveImage((string(argv[3]) + "_deg_x_" + string(argv[4]) + "_deg_y.jpg").c_str(), dst_persp);

    return 0;
}