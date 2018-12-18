#include "pch.h"
#include "../train_bovw/common_code.hpp"
#include <tclap/CmdLine.h>
#include <opencv2/ml.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define IMG_WIDTH 300

using namespace std;
using namespace cv;
using namespace ml;
using namespace TCLAP;

/**
 * \brief Read the trained classifier from a file.
 * \param[in] classifierFile is the path to the file.
 * \return the trained classifier.
 */
Ptr<StatModel> ReadClassifier(string& classifierFile)
{
	// read the classifier
	Ptr<StatModel> trainedClassifier;

	const int classifierType = GetClassifierValue();

	switch (classifierType)
	{
		case 1:
		{
			const Ptr<KNearest> kNNclassifier = KNearest::load<KNearest>(classifierFile);
			const int k = GetKnnValue();

			kNNclassifier->setDefaultK(k);
			trainedClassifier = kNNclassifier;

			break;
		}
		case 2:
		{
			const Ptr<SVM> svm = Algorithm::load<SVM>(classifierFile);
			trainedClassifier = svm;

			break;
		}
		case 3:
		{
			const Ptr<RTrees> randomForest = Algorithm::load<RTrees>(classifierFile);
			trainedClassifier = randomForest;

			break;
		}
		case 4:
		{
			const Ptr<Boost> boosting = Algorithm::load<Boost>(classifierFile);
			trainedClassifier = boosting;

			break;
		}
		default:
		{
			cerr << "ERROR: Unknown classifier type!" << endl;

			exit(-1);
		}
	}

	return trainedClassifier;
}

/**
 * \brief Get the input images source from the user.
 * \return true if the web camera should be used.
 */
bool UseCamera()
{
	int useCamera = 0;

	clog << "Select input type (1 = Images, 2 = web camera): ";
	cin >> useCamera;

	return useCamera == 2;
}

int main(const int argc, char* argv[])
{
	CmdLine cmd("Test a BoVW model", ' ', "0.0");

	ValueArg<string> img("", "img", "Path to images folder.", false, "./images/image_001.jpg", "string");
	cmd.add(img);
	ValueArg<string> classifier("", "classifier", "Path to the classifier.", false, "../classifier.yml", "string");
	cmd.add(classifier);
	ValueArg<string> dict("", "dict", "Path to the dictionary.", false, "../dictionary.yml", "string");
	cmd.add(dict);
	ValueArg<string> config_file("", "config_file", "Path to the configuration file.", false, "../data/05_ObjectCategories_conf.txt", "string");
	cmd.add(config_file);

	cmd.parse(argc, argv);

	vector<string> categories;
	vector<int> numberOfSamplesPerCategory;
	LoadDatasetInformation(config_file.getValue(), categories, numberOfSamplesPerCategory);

	int i = 0;

	const auto descriptor = GetDescriptorValue();
	const Ptr<StatModel> trainedClassifier = ReadClassifier(classifier.getValue());
	const string fileNamePrefix = "./images/image_0";
	const bool useCamera = UseCamera();

	VideoCapture capture;
	if (useCamera && !capture.open(0))
	{
		cerr << "ERROR: Cannot open the web camera!";

		exit(-1);
	}

	for (;;)
	{
		string imageName;
		Mat image;

		if (useCamera)
		{
			capture >> image;

			if (image.empty())
			{
				return 0;
			}
		}
		else
		{
			i++;
			if (i < 10)
			{
				imageName = fileNamePrefix + "0" + to_string(i) + ".jpg";
			}
			else
			{
				imageName = fileNamePrefix + to_string(i) + ".jpg";
			}

			image = imread(imageName, IMREAD_GRAYSCALE);
		}

		resize(image, image, Size(IMG_WIDTH, round(IMG_WIDTH * image.rows / image.cols)));

		vector<int> siftScales{ 9, 13 }; // 5 , 9

		Mat descriptors;
		switch (descriptor)
		{
			case 1:
			{
				descriptors = ExtractSparseSIFTDescriptors(image);
				break;
			}
			case 2:
			{
				descriptors = ExtractSURFDescriptors(image);
				break;
			}
			case 3:
			{
				descriptors = ExtractDenseSIFTDescriptors(image, siftScales);
				break;
			}
			case 4:
			{
				descriptors = ExtractPHOWDescriptors(image, siftScales);
				break;
			}
			default:
			{
				cerr << "ERROR: Unknown descriptor type!";

				exit(-1);
			}
		}

		// read the dictionary
		FileStorage dictionaryFile;
		int keywords;

		dictionaryFile.open(dict.getValue(), FileStorage::READ);
		dictionaryFile["keywords"] >> keywords;

		const Ptr<KNearest> dictionary = Algorithm::read<KNearest>(dictionaryFile.root());

		dictionaryFile.release();

		// calculate the BoVW
		const Mat bovw = ComputeBovw(dictionary, keywords, descriptors);
		   
		// predict the labels
		Mat predictedLabels;
		trainedClassifier->predict(bovw, predictedLabels);

		const string category = categories[predictedLabels.at<float>(0, 0)];

		namedWindow("Image");

		Mat imageToShow;

		if (useCamera)
		{
			imageToShow = image;
		}
		else
		{
			imageToShow = imread(imageName);
		}

		resize(imageToShow, imageToShow, Size(IMG_WIDTH, round(IMG_WIDTH * image.rows / image.cols)));

		putText(imageToShow, category, cvPoint(30, 30), FONT_HERSHEY_PLAIN, 1, cvScalar(0, 0, 255), 1, CV_AA);

		imshow("Image", imageToShow);

		if (useCamera)
		{
			if (waitKey(100) == 27)
			{
				break;
			}
		}
		else
		{
			waitKey(0);
		}

		if (!useCamera && i == 10)
		{
			break;
		}

	}

	destroyAllWindows();
	
	if (useCamera)
	{
		capture.release();
	}
}
