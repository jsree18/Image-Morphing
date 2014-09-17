#include "mex.h"
#include <iostream>
#include <string>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\objdetect\objdetect.hpp>

using namespace cv;
using namespace std;

int** extractCharacteristicPoints(Rect face,Mat img);
Rect DetectFace(Mat img);
bool extractMouthPoints(Mat mouthImg, Point& llips, Point& rlips, Point& ulips, Point& dlips);
bool extractNosePoints(Mat noseImg,Point&,Point&);
bool extractEyesPoints(Mat eyeRegionImg,bool,Point leb[4], Point reb[4], Point& cleye, Point& creye,Point lefcps [5], Point refcps [5]);
void project(const Mat& mat, Mat* hp, Mat* vp = NULL);
bool findBoundingPoints(const vector<Point>& vp,Point* leftmost, Point* rightmost, Point* umost, Point* dmost);
bool extractEyeCorners (Mat eyeImg, Point efcps [5], Point ceye);

CascadeClassifier face_cascade;
string face_cascade_name = "haarcascade_frontalface_alt.xml";

int** getFeatures(string file1){
    if( !face_cascade.load( face_cascade_name ) ){
		cout<<"cascade error"<<endl;
	}
    
    int** points;
    Mat frame = imread(file1,1),img_gray;
    cvtColor( frame, img_gray, CV_BGR2GRAY );
	equalizeHist( img_gray, img_gray );
	GaussianBlur( img_gray, img_gray, Size(3, 3), 2, 2);
	
	Rect face = DetectFace( img_gray );
	points = extractCharacteristicPoints(face,img_gray);
    
    return points;
}

	Rect DetectFace(Mat img_gray){
	
	vector<Rect> faces;
	Rect g(0,0,0,0);
	face_cascade.detectMultiScale( img_gray, faces,1.1, 4, 0|CV_HAAR_FIND_BIGGEST_OBJECT|CV_HAAR_SCALE_IMAGE,Size(30, 30) );
	Mat f=img_gray;
	if(faces.size()>0)
	{	rectangle(f,Point(faces[0].x,faces[0].y),Point(faces[0].x+faces[0].width, faces[0].y+faces[0].height),Scalar(255, 255, 255));
	//imshow("testface",f);
	return faces[0];
	}
	else 
	return g;
	}
	
	int** extractCharacteristicPoints(Rect faceR,Mat img_gray){
	
	//resize(img_gray,img_gray,Size(ro,200));
	Mat face;
	if(faceR.area()==0){
	face=img_gray;
	faceR.x = 0;
	faceR.y = 0;
	faceR.width = img_gray.cols;
	faceR.height = img_gray.rows;
	}
	else
	face=img_gray(faceR);
	
	int ro1 = (face.rows*140)/img_gray.cols;
	int co1 = (face.cols*140)/img_gray.cols;
	Mat mouthImg, noseImg, eyeRegionImg ;
    Rect mr, er, nr;
	//imshow("1",face);
	//resize(face,face,Size(ro1,co1));
    
	mr.x = faceR.x + (faceR.width*0.2);
	mr.y = faceR.y + (faceR.height*0.5);
	mr.width = faceR.width*0.6;
	mr.height = faceR.height*0.5;

    cout<<"holyshit1"<<endl;
	mouthImg = img_gray(mr);
	cout<<"holyshit2"<<endl;
	Point llips,rlips,ulips,dlips;
	extractMouthPoints(mouthImg,llips,rlips,ulips,dlips);
	llips += mr.tl();
    rlips += mr.tl();
    ulips += mr.tl();
    dlips += mr.tl();


	nr.x = faceR.x + (faceR.width*0.3) ;
    nr.y = faceR.y + (faceR.height*0.5);
    nr.width = faceR.width * 0.4 ;
	nr.height =  min((ulips.y + mr.y) - nr.y + 1, int(faceR.height * 0.20));

	noseImg = img_gray(nr);
	Point lnstrl,rnstrl;
	extractNosePoints(noseImg, lnstrl,rnstrl);
	rnstrl += nr.tl();
    lnstrl += nr.tl();

	er.x = faceR.x + (faceR.width*0.1) ;
    er.y = faceR.y + (faceR.height*0.1);
    er.width = faceR.width * 0.8 ;
    er.height =  min((nr.y - er.y + 1), int(faceR.height * 0.5));

	eyeRegionImg = img_gray(er);
	Point leb[4],reb[4],lefcps[5],refcps [5];
	Point &cleye=lefcps[4];
	Point &creye=refcps[4];
	extractEyesPoints(eyeRegionImg,true,leb,reb,cleye,creye,lefcps,refcps);

	for (int i = 0; i< 5; i++){
		if(i<4){
			leb[i] += er.tl();
			reb[i] += er.tl();
		}
		lefcps[i] += er.tl();
		refcps[i] += er.tl();
    }

	int** oo;
	oo=new int*[6];
	for(int i = 0; i < 6; ++i)
    oo[i] = new int[2];

	oo[0][0] = llips.x;
	oo[0][1] = llips.y;
	oo[1][0] = rlips.x;
	oo[1][1] = rlips.y;
	//oo[2][0] = ulips.x;
	//oo[2][1] = ulips.y;
	//oo[3][0] = dlips.x;
	//oo[3][1] = dlips.y;
	oo[2][0] = lnstrl.x;
	oo[2][1] = lnstrl.y;
	oo[3][0] = rnstrl.x;
	oo[3][1] = rnstrl.y;
	oo[4][0] = lefcps[4].x;
	oo[4][1] = lefcps[4].y;
	oo[5][0] = refcps[4].x;
	oo[5][1] = refcps[4].y;
	/*Mat op=img_gray.clone();
	for(int i=0;i<8;i++){
	cout<<"("<<oo[i][0]<<","<<oo[0][1]<<")"<<endl;
	circle(op, Point(oo[i][0],oo[i][1]), 1,CV_RGB(255,255,255),2, 8, 0);
	}
	imshow("op",op);
	*/
	return oo;
}

bool extractMouthPoints(Mat mouthImg, Point& llips, Point& rlips, Point& ulips, Point& dlips){
	int** points=NULL;
	Mat mouthImgTh, mouthSobel, kernel(1,5,CV_8UC1,1);
	Sobel(mouthImg, mouthSobel, CV_8UC1, 0, 2, 3);
	threshold(mouthSobel, mouthImgTh, 255,255, THRESH_BINARY | THRESH_OTSU);

	Mat mouthAdapTh;
    adaptiveThreshold(mouthSobel,mouthAdapTh,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY,((mouthSobel.size().width) % 2 == 1? (mouthSobel.size().width) : (mouthSobel.size().width) + 1 ),-15);

    vector<vector<Point> > contours;
    Rect bigBox;
    Mat imContours;
	mouthAdapTh.copyTo(imContours);
	findContours(imContours,contours,CV_RETR_LIST,CV_CHAIN_APPROX_NONE);

	int cind;
    for( int  j = 0 ; j < contours.size(); j++)
    {
        Mat matcnt (contours[j]);
        Rect bbox = boundingRect (matcnt);

        if (bbox.width > bigBox.width) {
            bigBox = bbox;
            cind = j;
        }
    }

	if (contours.size()>0)
		findBoundingPoints(contours.at(cind),&llips,&rlips,&ulips, &dlips);
	else
		return false;

	Mat mouthEroded;
    erode(mouthAdapTh, mouthEroded, kernel);
	Mat lip2 = mouthEroded(bigBox);
    Mat vPrj (lip2.size().height, 1, CV_32FC1);
    Point p1,p2;
    double pv1,pv2;

	project(lip2, NULL, &vPrj);

	minMaxLoc(vPrj,&pv2, &pv1,&p2, &p1);

    ulips.y = p1.y + bigBox.tl().y;

    ulips.x = int((rlips.x + llips.x)/2);

    Mat lipmap;
    Mat mouthComp = Mat(mouthImg.size(),mouthImg.type(), CV_RGB(255,255,255)) - mouthImg  ;
    addWeighted( mouthSobel, 0.5, mouthComp, 0.5, 0, lipmap);
    Mat mouthAdapTh3;
    adaptiveThreshold(lipmap, mouthAdapTh3,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY,((mouthSobel.size().width) % 2 == 1? (mouthSobel.size().width) : (mouthSobel.size().width) + 1 ),-10);


    mouthAdapTh3.copyTo(imContours);
    //mouthImgTh.copyTo(imContours);
    findContours(imContours,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);

    //Sorting by area and darkness. TODO: improve heuristics to sort by (e.g: use area or width/mean value (max area and darkest)*relative down position)
    const int CAREAS_SZ = 500;
    int cAreasIdx [CAREAS_SZ];
    int cAreas [CAREAS_SZ];

	int j;
    for( j = 0 ; j < contours.size() && j < CAREAS_SZ; j++)
    {
        int aux1, aux2, auxIdx1, auxIdx2;
        Mat matcnt (contours[j]);
        Rect bbox = boundingRect(matcnt);

        double costf = bbox.width; //sum(lipmap(bbox)) [0] ;// (sum(eyeNeg(halfsideBox))[0]))  + ((bbox.y + (bbox.height/2)) / halfImContours.size().height ;

        int i = 0;
        while (i < j && costf <= cAreas[i]) i++;

        aux1 = costf;
        auxIdx1 = j;
        while (i <= j)
        {
            aux2 = cAreas[i];
            cAreas[i] = aux1;
            aux1 = aux2;

            auxIdx2 = cAreasIdx[i];
            cAreasIdx[i] = auxIdx1;
            auxIdx1 = auxIdx2;

            i++;
        }
    }

    if (contours.size() >= 1) {
        Mat matcnt (contours.at(cAreasIdx[0]));
        bigBox = boundingRect(matcnt);
        findBoundingPoints(contours.at(cAreasIdx[0]),NULL, NULL, NULL, &dlips);
        //Adjust de upper and lower lip 'x' coordinates to the middle. Otherwise they will be moving with noise.
        ulips.x = int((rlips.x + llips.x)/2);
        dlips.x = int((rlips.x + llips.x)/2);

    }

	return true;
}

bool extractNosePoints(Mat noseImg, Point& lnstrl, Point& rnstrl)
{
    Mat noseSobel;

    //Sobel filter. Horizontal, 2nd order, single channel 255 bit-depth, 3x3 kernel.
    Sobel(noseImg, noseSobel, CV_8UC1, 0, 2 , 3);

    Rect nBox;
    Mat vPrj (noseImg.size().height, 1, CV_32FC1); //One coloumn
    Mat hPrj (1, noseImg.size().width, CV_32FC1); //One coloumn
    Point p1,p2;
    double pv1,pv2;

    project(noseSobel,&hPrj, &vPrj);


    float vTh, hTh;
    minMaxLoc(vPrj,&pv2, &pv1,&p2, &p1);
    vTh = pv1/2;
    minMaxLoc(hPrj,&pv2, &pv1,&p2, &p1);
    hTh = pv1/2;

    int r = 0,cl;

    for (r=0;r < vPrj.size().height; r++)
        if (vPrj.at<float>(r,0) > vTh) break;
    nBox.y = r;
    for (r = vPrj.size().height-1; r >= nBox.y; r--)
        if (vPrj.at<float>(r,0) > vTh) break;
    nBox.height = r - nBox.y + 1;

    for (cl=0;cl < hPrj.size().width; cl++)
        if (hPrj.at<float>(0,cl) > hTh) break;
    nBox.x = cl;
    for (cl = hPrj.size().width -1; cl >= nBox.x; cl--)
        if (hPrj.at<float>(0,cl) > hTh) break;
    nBox.width = cl - nBox.x + 1;

//DEBUG
//    rectangle(noseImg,nBox.tl(),nBox.br(),CV_RGB (255,0,0),1);

    Mat noseStrImg;
    Rect halfNoseBox;


    halfNoseBox = nBox;
    if (halfNoseBox.width !=0 && halfNoseBox.height !=0){

        //Left half
        halfNoseBox.width = std::max(0.0, std::ceil(halfNoseBox.width * 0.5)); 
        noseImg(halfNoseBox).copyTo(noseStrImg);
        minMaxLoc(noseStrImg,&pv2, &pv1,&p2, &p1);
        p2.x = p2.x + halfNoseBox.x;
        p2.y = p2.y + halfNoseBox.y;

        lnstrl = p2;

        //Right half
        halfNoseBox.x += std::max(0, halfNoseBox.width-1); 
        noseImg(halfNoseBox).copyTo(noseStrImg);
        minMaxLoc(noseStrImg,&pv2, &pv1,&p2, &p1);
        p2.x = p2.x + halfNoseBox.x;
        p2.y = p2.y + halfNoseBox.y;

        rnstrl = p2;
    }


    return true;
}

bool extractEyesPoints(Mat eyeRegionImg, bool doBluring, Point leb[4], Point reb[4], Point& cleye, Point& creye, Point lefcps [5], Point refcps [5])
{
    // Alias for eyebrows'points (left, right, up, down).
    Point&  lleb = leb[0];
    Point&  rleb = leb[1];
    Point&  uleb = leb[2];
    Point&  dleb = leb[3];
    Point&  lreb = reb[0];
    Point&  rreb = reb[1];
    Point&  ureb = reb[2];
    Point&  dreb = reb[3];


    //We will get the four widest areas of the contours extracted from the Otsu Thresholded image of the Sobel transform of the gaussian blured version (to get rid of foredhead wrinkles, hair, etc) of the original image.
    vector<vector<Point> > contours;
    Rect bigBox1, bigBox2;
    Mat imContours;

    Mat blur, blurSobel, blurSTh;

    eyeRegionImg.copyTo(blur);
    if (doBluring) GaussianBlur( eyeRegionImg , blur, Size(5, 5), 2, 2);

    //Sobel filter. Horizontal, 2nd order, single channel 255 bit-depth, 3x3 kernel.
    Sobel(blur, blurSobel,CV_8UC1, 0, 2 , 3);

    //Thresholding so only hard edges stay.Automatic threshold estimated with Otsu algorithm. Both numbers 255 are ignored (dummy).
    threshold(blurSobel, blurSTh, 255,255, THRESH_BINARY | THRESH_OTSU);
    
    blurSTh.copyTo(imContours);


    //Dividing eye region in left and right halfs. This made the algorithm more robust to face rotation.
    Rect halfsideBox;
    Mat eyeNeg =  Mat(eyeRegionImg.size(),eyeRegionImg.type(), CV_RGB(255,255,255)) - eyeRegionImg  ;
    for (int iteration = 0; iteration < 2; iteration++)
    {

        //Left half
        if ( iteration == 0 )
        {
            halfsideBox.x = 0;
            halfsideBox.y = 0;
            halfsideBox.width = imContours.size().width * 0.5;
            halfsideBox.height = imContours.size().height;
        }
        else
        {
            halfsideBox.x = std::floor(imContours.size().width * 0.5);
            halfsideBox.y = 0;
            halfsideBox.width = imContours.size().width * 0.5 - 1;
            halfsideBox.height = imContours.size().height;

        }

        Mat halfImContours = imContours(halfsideBox);

        findContours(halfImContours,contours,CV_RETR_LIST,CV_CHAIN_APPROX_NONE);

        //Sorting by area and darkness.
        const int CAREAS_SZ = 500;
        int cAreasIdx [CAREAS_SZ];
        int cAreas [CAREAS_SZ];
        int j;
        for( j = 0 ; j < contours.size() && j < CAREAS_SZ; j++)
        {
            int aux1, aux2, auxIdx1, auxIdx2;
            Mat matcnt (contours[j]);
            Rect bbox = boundingRect(matcnt);

            //Cost funtion to maximize: max(area * darkest)
            double costf = sum(eyeNeg(halfsideBox)(bbox)) [0] ; //Other things tryed: (sum(eyeNeg(halfsideBox))[0]))  + ((bbox.y + (bbox.height/2)) / halfImContours.size().height ; //bbox.area();

            //cout << "Area " << j << ": " << area << endl;

            //Sorting
            int i = 0;
            while (i < j && costf <= cAreas[i]) i++;

            aux1 = costf;
            auxIdx1 = j;
            while (i <= j)
            {
                aux2 = cAreas[i];
                cAreas[i] = aux1;
                aux1 = aux2;

                auxIdx2 = cAreasIdx[i];
                cAreasIdx[i] = auxIdx1;
                auxIdx1 = auxIdx2;

                i++;
            }
        }


        if (contours.size() >= 2) {

            //Position 0 will have the eyebrow and 1 the eye.
            if (boundingRect(contours[cAreasIdx[1]]).y < boundingRect(contours[cAreasIdx[0]]).y)
            {
                int aux = cAreasIdx[0];
                cAreasIdx[0] = cAreasIdx[1];
                cAreasIdx[1] = aux;
            }

            if (iteration == 0)
            {
                findBoundingPoints(contours.at(cAreasIdx[0]),&lleb,&rleb,&uleb, &dleb);
                lleb += halfsideBox.tl();
                rleb += halfsideBox.tl();
                uleb += halfsideBox.tl();
                dleb += halfsideBox.tl();
            }
            else
            {
                findBoundingPoints(contours.at(cAreasIdx[0]),&lreb,&rreb,&ureb, &dreb);
                lreb += halfsideBox.tl();
                rreb += halfsideBox.tl();
                ureb += halfsideBox.tl();
                dreb += halfsideBox.tl();
            }

            //Extracting Eye Center. Considered the darkest point in eye contour.
            Rect eyeBox= boundingRect(Mat(contours[cAreasIdx[1]]));
            Point p1,p2;
            double pv1,pv2;

            minMaxLoc(blur(eyeBox),&pv2, &pv1,&p2, &p1);
            p2.x = p2.x + eyeBox.x;
            p2.y = p2.y + eyeBox.y;

            if (iteration == 0 )
                cleye = p2 + halfsideBox.tl(); // Center of Left Eye
            else
                creye = p2 + halfsideBox.tl(); // Center of Right Eye

            //DEBUG
            for (int i = 0; i < 2 && i < contours.size(); i++ )
            {
                Point laux ;
                Point raux ;
                Point uaux ;
                Point daux ;

                findBoundingPoints(contours[cAreasIdx[i]],&laux,&raux,&uaux, &daux);

                Rect br;
                br.x = laux.x;
                br.y = uaux.y;
                br.width = raux.x - br.x + 1;
                br.height = daux.y - br.y + 1;

                //DEBUG
                Scalar colorR;

                colorR = i >= 1? CV_RGB (120,120,120) : CV_RGB (255,255,255);

                rectangle(eyeNeg, br.tl() + halfsideBox.tl() ,br.br() + halfsideBox.tl(),colorR,1);

            }


            //DEBUG
            //findBoundingPoints(contours.at(cAreasIdx[1]),&lefcps[0],&lefcps[1],&lefcps[2], &lefcps[3]);
            //lefcps[0] += halfsideBox.tl();
            //lefcps[1] += halfsideBox.tl();
            //lefcps[2] += halfsideBox.tl();
            //lefcps[3] += halfsideBox.tl();
            //if (iteration == 0){
            //    debuglefcps = lefcps[3];
            //}
            //line(eyeRegionImg,Point(iteration * eyeRegionImg.size().width/2, lefcps[2].y), Point(eyeRegionImg.size().width/(2-iteration) -1, lefcps[2].y),  CV_RGB (255,255,255), 1);
            //circle (eyeRegionImg, lefcps[2], 0, CV_RGB (255,255,255),-1);
            //line(eyeRegionImg,Point(0, lefcps[3].y), Point(eyeRegionImg.size().width/2 -1, lefcps[3].y),  CV_RGB (255,255,255), 1);
            //circle (eyeRegionImg, lefcps[3], 0, CV_RGB (255,255,255),-1);



        }else
            return false; //Some eyebrow or eye was not found

        //DEBUG
        //cv::imshow( "NegEyes", eyeNeg );
        //cvMoveWindow("NegEyes", 1400, 300 );

    }

    //Finding right eye corners
    Rect ler; //Left Eye Region
    ler.x = 0;
    ler.y = std::max (0, std::min(eyeRegionImg.size().height-1, dleb.y + 1)); //Not including eyebrow point.
    ler.width = std::max(0, int(std::ceil(eyeRegionImg.size().width / 2.0)));
    ler.height = std::max(0, eyeRegionImg.rows - ler.y);

    //Left Eye facial Characteristic Points: left and right corners, up and down eyelids, and center  respectively from 0 to 4.
    if (cleye.y < ler.y) cleye.y = ler.y; //To be sure that eyecenter is in left eye region (ler) and below eyebrow.
    extractEyeCorners (eyeRegionImg(ler), lefcps, cleye - ler.tl());



     //Finding right eye corners
    Rect rer; //Left Eye Region
    rer.x = std::max(0, int(std::ceil(eyeRegionImg.size().width / 2.0)) - 1);
    rer.y = std::max (0, std::min(eyeRegionImg.size().height-1, dreb.y + 1)); //Not including eyebrow point.
    rer.width = std::max(0, eyeRegionImg.size().width - rer.x);
    rer.height = std::max(0, eyeRegionImg.rows - rer.y);

     //Right Eye facial Characteristic Points: left and right corners, up and down eyelids, and center  respectively from 0 to 4.
    if (creye.y < rer.y) creye.y = rer.y; //To be sure that eyecenter is in left eye region (ler) and below eyebrow.
    extractEyeCorners (eyeRegionImg(rer), refcps, creye - rer.tl());


    //Relocating points
    int i=5;
    while (0 < i--)
    {
        lefcps [i] += ler.tl();
        refcps [i] += rer.tl();
    }



    return true; //OK
}

bool extractEyeCorners (Mat eyeImg, Point efcps [5], Point ceye)
{
    Mat eyesSobel, eyesTh, eyesSTh;
    Mat eyeBlur;

    //equalizeHist( eyeImg, eyeImg );

    GaussianBlur( eyeImg , eyeBlur, Size(3,3), 2, 2 );

    //Sobel filter. Horizontal, 2nd order, single channel 255 bit-depth, 3x3 kernel.
    Sobel(eyeBlur, eyesSobel,CV_8UC1, 0, 2 , 3, 1, 0, BORDER_REPLICATE);

    //Thresholding so only hard edges stay.Automatic threshold estimated with Otsu algorithm. Both numbers 255 are ignored (dummy).
    //threshold(eyesSobel, eyesSTh, 255,255, THRESH_BINARY | THRESH_OTSU);
    adaptiveThreshold(eyesSobel, eyesSTh,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY,3,-2);


    //The extraction will be based on:
    //  - Sorting contours by area*darkness
    //  - Considering only the 4 first of those.
    //  - The first of them will be the upper eyelid contour.
    //  - The lowest located of them will be the lower eyelid.
    //  - Upper and lower eyelids contours could be the same one.
    //  - The bounding points of the union of these two contours are the 4 characteristic points of the eye.
    //  - The eye center is not extracted here. It is keeped the value from previous algorithms.
    vector<vector<Point> > contours;
    Mat imContours;

    eyesSTh.copyTo(imContours);
    findContours(imContours,contours,CV_RETR_LIST,CV_CHAIN_APPROX_NONE);

    //Sorting by area and darkness.
    const int CAREAS_SZ = 500;
    int cAreasIdx [CAREAS_SZ];
    int cAreas [CAREAS_SZ];
    int j;
    for( j = 0 ; j < contours.size() && j < CAREAS_SZ; j++)
    {
        int aux1, aux2, auxIdx1, auxIdx2;
        Mat matcnt (contours[j]);
        Rect bbox = boundingRect(matcnt);

        //Cost funtion to maximize: max(area * darkest)
        double costf = sum(eyesSTh(bbox)) [0] ; //Other things tryed: (sum(eyeNeg(halfsideBox))[0]))  + ((bbox.y + (bbox.height/2)) / halfImContours.size().height ; //bbox.area();

        //cout << "Area " << j << ": " << area << endl;

        //Sorting
        int i = 0;
        while (i < j && costf <= cAreas[i]) i++;

        aux1 = costf;
        auxIdx1 = j;
        while (i <= j)
        {
            aux2 = cAreas[i];
            cAreas[i] = aux1;
            aux1 = aux2;

            auxIdx2 = cAreasIdx[i];
            cAreasIdx[i] = auxIdx1;
            auxIdx1 = auxIdx2;

            i++;
        }
    }

    //We will get consider only the 4 first contours. Then the lowest located of those, will be the low eyelid.
    //The upper eyelid will be the first in sort.
    int uIdx = -1 , lIdx = -1;
    if (contours.size() > 0)
    {
        uIdx = cAreasIdx[0]; //Upper eyelid will be the widest and darkest

        //Low eyelid will be the lowest located among the 4 wider.
        int maxy = 0;
        for( j = 0 ; j < contours.size() && j < 4; j++)
        {
            Mat matcnt (contours[cAreasIdx[j]]);
            Rect bbox = boundingRect(matcnt);

            if (bbox.br().y > maxy )
            {
                maxy = bbox.br().y;
                lIdx = cAreasIdx[j];
            }
        }
    }



    Point laux, raux, uaux, daux;

    if (uIdx >=0 && lIdx >= 0)
    {
        findBoundingPoints(contours[uIdx],&efcps[0], &efcps[1], &efcps[2], &efcps[3]);
        findBoundingPoints(contours[lIdx],&laux, &raux, &uaux, &daux);

        //Getting leftmost and rightmost
        efcps[0] = (laux.x < efcps[0].x) ? laux : efcps[0];
        efcps[1] = (raux.x > efcps[1].x) ? raux : efcps[1];

        //Getting uppermost and downmost
        efcps[2] = (uaux.y < efcps[2].y) ? uaux : efcps[2];
        efcps[3] = (daux.y > efcps[3].y) ? daux : efcps[3];

        //Centering the upper and lower points
        efcps[2].x = (efcps[1].x + efcps[0].x) / 2;
        efcps[3].x = efcps[2].x;

        //Centero of the eye will be the same. Maybe improved in the future.
        efcps[4] = ceye;



    }else
        return false; //cerr << "Exception: NO EYE CORNERS DETECTED !! " << endl;

    return 0;
}

void project(const Mat& mat, Mat* hpp, Mat* vpp)
{
    Mat &hp = (*hpp), &vp = (*vpp); //To make easy use of vpp and hpp
    int r,cl; //row and coloumn indexes.


    if (&vp != NULL)
        for (r=0;r < vp.size().height; r++)
            vp.at<float>(r,0) = sum(mat.row(r))[0];

    if (&hp != NULL)
        for (cl=0;cl < hp.size().width; cl++)
            hp.at<float>(0,cl) = sum(mat.col(cl))[0];
}

bool findBoundingPoints(const vector<Point>& vp,Point* leftmost, Point* rightmost, Point* umost, Point* dmost)
{

    int maxxi = 0; // max and min 'x' index
    int minxi = 0;
    int maxyi = 0; // max and min 'y' index
    int minyi = 0;

    for( int  j = 0 ; j < vp.size(); j++)
    {
        const Point& p = vp.at(j);

        if (p.x < vp.at(minxi).x)
            minxi = j;

        if (p.x > vp.at(maxxi).x)
            maxxi = j;

        if (p.y < vp.at(minyi).y)
            minyi = j;

        if (p.y > vp.at(maxyi).y)
            maxyi = j;


    }

    if (leftmost != NULL) *leftmost = vp.at(minxi);
    if (rightmost!= NULL) *rightmost= vp.at(maxxi);
    if (umost != NULL) *umost = vp.at(minyi);
    if (dmost != NULL) *dmost = vp.at(maxyi);

    return true;

}

void mexFunction( int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[] )
{
    double *returnPtr;
    string str = mxArrayToString(prhs[0]);
    int** op = getFeatures(str);
    plhs[0] = mxCreateNumericMatrix(6,2, mxDOUBLE_CLASS, mxREAL);
    returnPtr = mxGetPr(plhs[0]);
    for(mwSize i=0;i<6;i++){
        for(mwSize j=0;j<2;j++)
            returnPtr[(i*2)+j] = op[i][j];
    }
}