/*! @file common_code.cpp
	@brief Useful for building a Bag of Visual Words model
	@authors Fundamentos de Sistemas Inteligentes en Vision
*/

#include "pch.h"
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "common_code.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;
using namespace ml;
using namespace xfeatures2d;

void Basename(const string& path, string& directoryName, string& fileName, string& extension)
{
	directoryName = "";
	fileName = path;
	extension = "";

	auto position = path.rfind('/');

	if (position != string::npos)
	{
		directoryName = path.substr(0, position);
		fileName = path.substr(position + 1);
	}

	position = fileName.rfind('.');

	if (position != string::npos)
	{
		extension = fileName.substr(position + 1);
		fileName = fileName.substr(0, position);
	}
}

string ComputeSampleFilename(const string& baseName, const string& category, const int sampleIndex)
{
	ostringstream filename;

	filename << baseName << "/101_ObjectCategories/" << category << "/image_" << setfill('0') << setw(4) << sampleIndex << ".jpg";
	
	return filename.str();
}

int LoadDatasetInformation(const string& fileName, vector<string>& categories, vector<int>& samplesPerCategory)
{
	int returnCode = 0;
	ifstream inputFile(fileName);

	if (!inputFile)
	{
		returnCode = 1;
	}
	else
	{
		while (inputFile && returnCode == 0)
		{
			string categoryName;
			int numberOfSamples;

			inputFile >> categoryName >> numberOfSamples;
			
			if (!inputFile)
			{
				if (!inputFile.eof())
				{
					returnCode = 2;
				}
			}
			else
			{
				categories.push_back(categoryName);
				samplesPerCategory.push_back(numberOfSamples);
			}
		}
	}

	return returnCode;
}

void RandomSampling(int total, int numberOfTrainingSamples, int numberOfTestingSamples, vector<int>& trainingSamples, vector<int>& testingSamples)
{
	assert(numberOfTrainingSamples < total);

	trainingSamples.resize(0);
	testingSamples.resize(0);
	
	vector<bool> sampled(total, false);
	
	while (numberOfTrainingSamples > 0)
	{
		const auto s = int(double(total) * rand() / (RAND_MAX + 1.0));
		int i = 0;

		while (sampled[i] && unsigned(i) < sampled.size())
		{
			++i; // advance to the first unsampled
		}
		
		int c = 0;
		
		while (c < s) // count s unsampled
		{
			while (sampled[++i]); // advance to next unsampled
			++c;
		}

		assert(!sampled[i]);
		
		trainingSamples.push_back(i + 1);
		
		sampled[i] = true;
		
		--total;
		--numberOfTrainingSamples;
	}

	if (numberOfTestingSamples >= total)
	{
		for (size_t i = 0; i < sampled.size(); ++i)
		{
			if (!sampled[i])
			{
				testingSamples.push_back(i + 1);
			}
		}
	}
	else
	{
		while (numberOfTestingSamples > 0)
		{
			const auto s = int(double(total) * rand() / (RAND_MAX + 1.0));
			int i = 0;

			while (sampled[i] && unsigned(i) < sampled.size())
			{
				++i; // the first unsampled
			}

			int c = 0;
			
			while (c < s) // count s unsampled
			{
				while (sampled[++i]); // advance to next unsampled
				++c;
			}
			
			testingSamples.push_back(i + 1);
			
			sampled[i] = true;
			
			--total;
			--numberOfTestingSamples;
		}
	}
}

void CreateTrainingAndTestingDatasets(vector<int>& samplesPerCategory, const int numberOfTrainingSamples, const int numberOfTestingSamples,
	vector<vector<int>>& trainingSamples, vector<vector<int>>& testingSamples)
{
	trainingSamples.resize(0);
	testingSamples.resize(0);

	for (int i : samplesPerCategory)
	{
		vector<int> trainingSample;
		vector<int> testingSample;

		RandomSampling(i, numberOfTrainingSamples, numberOfTestingSamples, trainingSample, testingSample);
		
		trainingSamples.push_back(trainingSample);
		testingSamples.push_back(testingSample);
	}
}

Mat ComputeConfusionMatrix(const int numberOfCategories, const Mat& trueLabels, const Mat& predictedLabels)
{
	CV_Assert(trueLabels.rows == predictedLabels.rows);
	CV_Assert(trueLabels.type() == CV_32FC1);
	CV_Assert(predictedLabels.type() == CV_32FC1);

	Mat confusionMatrix = Mat::zeros(numberOfCategories, numberOfCategories, CV_32F);

	for (int i = 0; i < trueLabels.rows; ++i)
	{
		confusionMatrix.at<float>(trueLabels.at<float>(i), predictedLabels.at<float>(i)) += 1.0;
	}

	return confusionMatrix;
}

void ComputeRecognitionRate(const Mat& confusionMatrix, double& mean, double& deviation)
{
	CV_Assert(confusionMatrix.rows == confusionMatrix.cols && confusionMatrix.rows > 1);
	CV_Assert(confusionMatrix.depth() == CV_32F);

	mean = 0.0;
	deviation = 0.0;

	for (int c = 0; c < confusionMatrix.rows; ++c)
	{
		const double class_Rate = confusionMatrix.at<float>(c, c) / sum(confusionMatrix.row(c))[0];
		mean += class_Rate;
		deviation += class_Rate * class_Rate;
	}

	mean /= double(confusionMatrix.rows);
	deviation = sqrt(deviation / double(confusionMatrix.rows) - mean * mean);
}


Mat ExtractSparseSIFTDescriptors(const Mat& image, const int numberOfDescriptors)
{
	Ptr<SIFT> sparseSift = SIFT::create(numberOfDescriptors);
	vector<KeyPoint> keypoints;
	Mat descriptors;
	
	sparseSift->detectAndCompute(image, noArray(), keypoints, descriptors);
	
	return descriptors;
}

Mat ExtractDenseSIFTDescriptors(const Mat& image, const vector<int>& siftScales)
{
	Ptr<SIFT> denseSift = SIFT::create();
	vector<KeyPoint> keypoints;
	Mat descriptors;

	for (int siftScale : siftScales)
	{
		for (int j = siftScale; j < image.rows - siftScale; j += siftScale)
		{
			for (int k = siftScale; k < image.cols - siftScale; k += siftScale)
			{
				keypoints.emplace_back(KeyPoint(float(k), float(j), float(siftScale * 2)));
			}
		}
	}

	denseSift->detectAndCompute(image, noArray(), keypoints, descriptors, true);

	return descriptors;
}

Mat ExtractSURFDescriptors(const Mat& image)
{
	Ptr<SURF> surf = SURF::create();
	vector<KeyPoint> keyPoints;
	Mat descriptors;
	
	surf->detectAndCompute(image, noArray(), keyPoints, descriptors);
	
	return descriptors;
}

Mat ExtractPHOWDescriptors(const Mat& image, vector<int>& siftScales)
{
	int rows = image.rows;
	if (rows % 2 != 0)
	{
		rows--;
	}
	int columns = image.cols;
	if (columns % 2 != 0)
	{
		columns++;
	}

	// extract the dense SIFT descriptor from the image
	Mat descriptors = ExtractDenseSIFTDescriptors(image, siftScales);
	
	// divide image into quarters and extract the dense SIFT descriptor from each sub-image
	const int rowRange = rows / 2;
	const int columnRange = columns / 2;

	for (int i = 0; i < rows; i += rowRange)
	{
		for (int j = 0; j < columns; j += columnRange)
		{
			Rect extractedRectangle(j, i, columnRange, rowRange);
			Mat subImage = image(extractedRectangle);

			vconcat(ExtractDenseSIFTDescriptors(subImage, siftScales), descriptors, descriptors);
		}
	}

	return descriptors;
}

Mat ComputeBovw(const Ptr<KNearest>& dictionary, const int dictionarySize, Mat& imageDescriptors, const bool normalize)
{
	Mat bovw = Mat::zeros(1, dictionarySize, CV_32F);
	Mat visualWords;

	CV_Assert(imageDescriptors.type() == CV_32F);
	
	dictionary->findNearest(imageDescriptors, 1, visualWords);
	
	CV_Assert(visualWords.depth() == CV_32F);
	
	for (int i = 0; i < imageDescriptors.rows; ++i)
	{
		bovw.at<float>(visualWords.at<float>(i))++;
	}

	if (normalize)
	{
		bovw /= float(imageDescriptors.rows);
	}

	return bovw;
}

int GetDescriptorValue()
{
	int descriptor = 1;
	bool ok = false;

	while (!ok)
	{
		clog << "Select the descriptor type (1 = SIFT, 2 = SURF, 3 = Dense SIFT, 4 = PHOW): ";
		cin >> descriptor;

		if (descriptor > 0 && descriptor < 5)
		{
			ok = true;
		}
	}

	return descriptor;
}

int GetKnnValue()
{
	int kNN = 1;

	clog << "Enter 'k' value for kNN classifier: ";
	cin >> kNN;

	return kNN;
}

int GetClassifierValue()
{
	int classifier = 1;
	bool ok = false;

	while (!ok)
	{
		clog << "Select the classifier type (1 = kNN, 2 = SVM, 3 = Random Forest, 4 = Boosting): ";
		cin >> classifier;

		if (classifier > 0 && classifier < 5)
		{
			ok = true;
		}
	}

	return classifier;
}

double GetSvmMarginValue()
{
	double margin = 1;

	clog << "Enter SVM classifier margin: ";
	cin >> margin;

	return margin;
}

SVM::KernelTypes GetSvmKernelType()
{
	int kernelType = 1;
	bool ok = false;

	while (!ok)
	{
		clog << "Select the SVM classifier kernel type (1 = Linear, 2 = Radial, 3 = Polynomial): ";
		cin >> kernelType;

		if (kernelType > 0 && kernelType < 4)
		{
			ok = true;
		}
	}

	switch (kernelType)
	{
		case 1:  return  SVM::LINEAR;
		case 2:  return  SVM::RBF;
		case 3:  return  SVM::POLY;
		default: return  SVM::LINEAR;
	}
}

int GetRandomForestMaxDepth()
{
	int maxDepth = 1;

	clog << "Enter Random Forest maximum depth: ";
	cin >> maxDepth;

	return maxDepth;
}

int GetRandomForestMinimumSamplesCount()
{
	int minSamples = 1;

	clog << "Enter Random Forest minimum samples count: ";
	cin >> minSamples;

	return minSamples;
}

int GetNumberOfTrees()
{
	int trees = 1;

	clog << "Enter the number of trees: ";
	cin >> trees;

	return trees;
}
