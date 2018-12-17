#include "pch.h"
#include <tclap/CmdLine.h>
#include <opencv2/ml.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "../train_bovw/common_code.hpp"
#include <opencv2/highgui.hpp>

#define IMG_WIDTH 300

using namespace std;
using namespace cv;
using namespace ml;
using namespace TCLAP;

/**
 * \brief Read the trained classifier from a file.
 * \param classifierFile is the path to the file.
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
			clog << "ERROR: Unknown classifier type!" << endl;

			exit(-1);
		}
	}

	return trainedClassifier;
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
	
	const auto descriptor = GetDescriptorValue();

	const Ptr<StatModel> trainedClassifier = ReadClassifier(classifier.getValue());
	
	const string fileNamePrefix = "./images/image_0";

	for (int i = 1; i <= 10; i++)
	{
		string imageName;
		if (i < 10)
		{
			imageName = fileNamePrefix + "0" + to_string(i) + ".jpg";
		}
		else
		{
			imageName = fileNamePrefix + to_string(i) + ".jpg";
		}


		Mat image = imread(imageName, IMREAD_GRAYSCALE);
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
				clog << "Unknown descriptor type.";
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

		namedWindow(category);
		imshow(category, imread(imageName));

		if (waitKey(0) == 27) break;

		destroyAllWindows();
	}
}
