/*! @file train_bovw.cpp
	@brief Train a Bag of Visual Words model
	@authors Fundamentos de Sistemas Inteligentes en Vision
*/

#include "pch.h"
#include "common_code.hpp"
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <tclap/CmdLine.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define IMG_WIDTH 300

using namespace std;
using namespace cv;
using namespace ml;
using namespace TCLAP;


/**
 * @brief Writes the final statistics to the console.
 * @param[in] recognitionRates are recognition rates - mean and deviation.
 */
void WriteFinalStatistics(const vector<float> recognitionRates)
{
	clog << "################################ FINAL STATISTICS ################################" << endl;

	double mean = 0.0;
	double deviation = 0.0;

	for (float v : recognitionRates)
	{
		mean += v;
		deviation += v * v;
	}

	mean /= double(recognitionRates.size());
	deviation = deviation / double(recognitionRates.size()) - mean * mean;
	deviation = sqrt(deviation);

	clog << "Recognition Rate mean " << mean * 100.0 << "% deviation " << deviation * 100.0 << endl;
}

/**
 * @brief Saves the best models to files.
 * @param[in] keywords is the number of keywords.
 * @param[in] bestDictionary is the best dictionary.
 * @param[in] bestClassifier is the best classifier.
 */
void SaveBestModels(const int keywords, Ptr<KNearest>& bestDictionary, Ptr<StatModel>& bestClassifier)
{
	FileStorage dictFile;

	dictFile.open("../dictionary.yml", FileStorage::WRITE);
	dictFile << "keywords" << keywords;
	
	bestDictionary->write(dictFile);
	
	dictFile.release();
	
	bestClassifier->save("../classifier.yml");
}

/**
 * @brief Loads the data set and writes on the command line.
 * @param[in] categories are the data set categories.
 * @param[in] samplesPerCategory is the number of samples per category.
 * @param[in] basename is the path to the configuration file.
 * @param[in] configFile is the data set configuration file.
 */
void LoadAndWriteDataSet(vector<string>& categories, vector<int>& samplesPerCategory, const string& basename, const string& configFile)
{
	const string datasetDescriptionFile = basename + "/" + configFile;

	int returnCode;
	if ((returnCode = LoadDatasetInformation(datasetDescriptionFile, categories, samplesPerCategory)) != 0)
	{
		cerr << "Error: could not load dataset information from '" << datasetDescriptionFile << "' (" << returnCode << ")." << endl;
		exit(-1);
	}

	cout << "Found " << categories.size() << " categories: " << endl;

	if (categories.size() < 2)
	{
		cerr << "Error: at least two categories are needed." << endl;
		exit(-1);
	}

	for (const auto& category : categories)
	{
		cout << "\t" << category << endl;
	}
}

/**
 * @brief Show the classifier's normalized confusion matrix.
 * @param confusionMatrix is the confusion matrix of the best classifier.
 */
void ShowConfusionMatrix(const Mat& confusionMatrix)
{
	clog << "The best classifier's confusion matrix:" << endl;
	clog << setprecision(2) << fixed;

	for (int i = 0; i < confusionMatrix.rows; i++)
	{
		int sum = 0;
		for (int j = 0; j < confusionMatrix.cols; j++)
		{
			sum += confusionMatrix.at<float>(Point(i, j));
		}

		for (int j = 0; j < confusionMatrix.cols; j++)
		{
			clog << "| " << setw(6) << confusionMatrix.at<float>(Point(i, j)) / sum * 100.0 << "% ";
		}
		clog << "|" << endl;
	}
}

int main(const int argc, char* argv[])
{
	CmdLine cmd("Train and test a BoVW model", ' ', "0.0");

	ValueArg<string> basenameArg("", "Basename", "Basename for the dataset.", false, "../data", "pathname");
	cmd.add(basenameArg);
	ValueArg<string> configFile("", "config_file", "configuration file for the dataset.", false, "05_ObjectCategories_conf.txt", "pathname");
	cmd.add(configFile);
	ValueArg<int> n_runsArg("", "n_runs", "Number of trials train/set to compute the recognition rate. Default 10.", false, 10, "int");
	cmd.add(n_runsArg);
	ValueArg<int> dict_runs("", "dict_runs", "Number of trials to select the best dictionary. Default 5.", false, 5, "int");
	cmd.add(dict_runs);
	ValueArg<int> ndesc("", "ndesc", "Number of descriptors per image. Value 0 means extract all. Default 0.", false, 0, "int");
	cmd.add(ndesc);
	ValueArg<int> keywords("", "keywords", "Number of keywords generated. Default 100.", false, 100, "int");
	cmd.add(keywords);
	ValueArg<int> ntrain("", "ntrain", "Number of samples per class used to train. Default 15.", false, 15, "int");
	cmd.add(ntrain);
	ValueArg<int> ntest("", "ntest", "Number of samples per class used to test. Default 50.", false, 50, "int");
	cmd.add(ntest);

	cmd.parse(argc, argv);

	vector<string> categories;
	vector<int> samples_per_cat;

	LoadAndWriteDataSet(categories, samples_per_cat, basenameArg.getValue(), configFile.getValue());

	vector<float> recognitionRates(n_runsArg.getValue(), 0.0);
	vector<int> siftScales{ 9, 13 }; // 5 , 9

	Ptr<KNearest> bestDictionary;
	Ptr<StatModel> bestClassifier;
	Mat bestClassifierConfusionMatrix;
	double bestRecognitionRate = 0.0;

	const int descriptorType = GetDescriptorValue();
	const int kNN = GetKnnValue();

	for (int trial = 0; trial < n_runsArg.getValue(); trial++)
	{
		clog << "########## TRIAL " << trial + 1 << " ##########" << endl;

		vector<vector<int>> train_samples;
		vector<vector<int>> test_samples;

		CreateTrainingAndTestingDatasets(samples_per_cat, ntrain.getValue(), ntest.getValue(), train_samples, test_samples);

		clog << "Training ..." << endl;
		clog << "\tCreating dictionary ... " << endl;
		clog << "\t\tComputing descriptors..." << endl;

		Mat trainingDescriptors;
		vector<int> numberOfDescriptorsPerSample;

		numberOfDescriptorsPerSample.resize(0);

		for (size_t c = 0; c < train_samples.size(); ++c)
		{
			clog << "  " << setfill(' ') << setw(3) << (c * 100) / train_samples.size() << " %   \015";

			for (size_t s = 0; s < train_samples[c].size(); ++s)
			{
				string filename = ComputeSampleFilename(basenameArg.getValue(), categories[c], train_samples[c][s]);
				Mat img = imread(filename, IMREAD_GRAYSCALE);

				if (img.empty())
				{
					cerr << "Error: could not read image '" << filename << "'." << endl;
					exit(-1);
				}
				
				// fix size
				resize(img, img, Size(IMG_WIDTH, round(IMG_WIDTH * img.rows / img.cols)));

				Mat descriptors;

				switch (descriptorType)
				{
					case 1:
					{
						descriptors = ExtractSparseSIFTDescriptors(img, ndesc.getValue());
						break;
					}
					case 2:
					{
						descriptors = ExtractSURFDescriptors(img);
						break;
					}
					case 3:
					{
						descriptors = ExtractDenseSIFTDescriptors(img, siftScales);
						break;
					}
					case 4:
					{
						descriptors = ExtractPHOWDescriptors(img, siftScales);
						break;
					}
					default:
					{
						clog << "Unknown descriptor type.";
						exit(-1);
					}
				}

				if (trainingDescriptors.empty())
				{
					trainingDescriptors = descriptors;
				}
				else
				{
					Mat dst;
					vconcat(trainingDescriptors, descriptors, dst);
					trainingDescriptors = dst;
				}

				numberOfDescriptorsPerSample.push_back(descriptors.rows); // we could really have less of wished descriptors
			}

		}
		
		clog << endl;
		
		CV_Assert(numberOfDescriptorsPerSample.size() == (categories.size()*ntrain.getValue()));
		
		clog << "\t\tDescriptors size = " << trainingDescriptors.rows*trainingDescriptors.cols * sizeof(float) / (1024.0 *1024.0) << " MiB." << endl;
		clog << "\tGenerating " << keywords.getValue() << " keywords ..." << endl;
		
		Mat keyws;
		Mat labels;
		
		const double compactness = kmeans(trainingDescriptors, keywords.getValue(), labels,
			TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 10, 1.0),
			dict_runs.getValue(),
			KMEANS_PP_CENTERS, //cv::KMEANS_RANDOM_CENTERS,
			keyws);
		
		CV_Assert(keywords.getValue() == keyws.rows);
		
		// free not needed memory
		labels.release();

		clog << "\tGenerating the dictionary ... " << endl;

		Ptr<KNearest> dictionary = KNearest::create();
		dictionary->setAlgorithmType(KNearest::BRUTE_FORCE);
		dictionary->setIsClassifier(true);

		Mat indexes(keyws.rows, 1, CV_32S);
		for (int i = 0; i < keyws.rows; ++i)
		{
			indexes.at<int>(i) = i;
		}
		
		dictionary->train(keyws, ROW_SAMPLE, indexes);
		
		clog << "\tDictionary compactness " << compactness << endl;

		clog << "\tTrain classifier ... " << endl;

		// compute the corresponding BoVW for each train image
		clog << "\t\tGenerating the a BoVW descriptor per train image." << endl;

		int row_start = 0;
		Mat train_bovw;
		vector<float> train_labels_v;
		train_labels_v.resize(0);

		for (size_t c = 0, i = 0; c < train_samples.size(); ++c)
		{
			for (size_t s = 0; s < train_samples[c].size(); ++s, ++i)
			{
				Mat descriptors = trainingDescriptors.rowRange(row_start, row_start + numberOfDescriptorsPerSample[i]);
				row_start += numberOfDescriptorsPerSample[i];
				
				Mat bovw = ComputeBovw(dictionary, keyws.rows, descriptors);
				train_labels_v.push_back(c);

				if (train_bovw.empty())
				{
					train_bovw = bovw;
				}
				else
				{
					Mat destination;
					vconcat(train_bovw, bovw, destination);
					train_bovw = destination;
				}
			}
		}

		// free not needed memory
		trainingDescriptors.release();

		// create the classifier
		// train a kNN classifier using the training BoVWs like patterns.
		Ptr<KNearest> knnClassifier = KNearest::create();
		knnClassifier->setAlgorithmType(KNearest::BRUTE_FORCE);
		knnClassifier->setDefaultK(kNN);
		knnClassifier->setIsClassifier(true);
		Ptr<StatModel> classifier = knnClassifier;

		Mat train_labels(train_labels_v);
		classifier->train(train_bovw, ROW_SAMPLE, train_labels);

		// free not needed memory
		train_bovw.release();
		train_labels_v.resize(0);

		clog << "Testing .... " << endl;

		// load test images, generate descriptors and quantize getting a BoVW for each image
		// classify and compute errors

		// compute the corresponding BoVW for each test image
		clog << "\tCompute image descriptors for test images..." << endl;
		Mat testingBovw;
		vector<float> trueLabels;
		trueLabels.resize(0);

		for (size_t c = 0; c < test_samples.size(); ++c)
		{
			clog << "  " << setfill(' ') << setw(3) << (c * 100) / train_samples.size() << " %   \015";
			
			for (size_t s = 0; s < test_samples[c].size(); ++s)
			{
				string filename = ComputeSampleFilename(basenameArg.getValue(), categories[c], test_samples[c][s]);
				Mat img = imread(filename, IMREAD_GRAYSCALE);
			
				if (img.empty())
					cerr << "Error: could not read image '" << filename << "'." << endl;
				else
				{
					// fix size
					resize(img, img, Size(IMG_WIDTH, round(IMG_WIDTH*img.rows / img.cols)));

					Mat descriptors;

					switch (descriptorType)
					{
						case 1:
						{
							descriptors = ExtractSparseSIFTDescriptors(img, ndesc.getValue());
							break;
						}
						case 2:
						{
							descriptors = ExtractSURFDescriptors(img);
							break;
						}
						case 3:
						{
							descriptors = ExtractDenseSIFTDescriptors(img, siftScales);
							break;
						}
						case 4:
						{
							descriptors = ExtractPHOWDescriptors(img, siftScales);
							break;
						}
						default:
						{
							clog << "Unknown descriptor type.";
							exit(-1);
						}
					}

					Mat bovw = ComputeBovw(dictionary, keyws.rows, descriptors);
					
					if (testingBovw.empty())
					{
						testingBovw = bovw;
					}
					else
					{
						Mat dst;
						vconcat(testingBovw, bovw, dst);
						testingBovw = dst;
					}

					trueLabels.push_back(c);
				}
			}
		}

		clog << endl;
		clog << "\tThere are " << testingBovw.rows << " test images." << endl;

		// classify the test samples
		clog << "\tClassifying test images." << endl;
		Mat predictedLabels;

		classifier->predict(testingBovw, predictedLabels);

		CV_Assert(predictedLabels.depth() == CV_32F);
		CV_Assert(predictedLabels.rows == testingBovw.rows);
		CV_Assert(unsigned(predictedLabels.rows) == trueLabels.size());

		// compute the classifier's confusion matrix
		clog << "\tComputing confusion matrix." << endl;
		Mat confusionMatrix = ComputeConfusionMatrix(categories.size(), Mat(trueLabels), predictedLabels);

		CV_Assert(int(sum(confusionMatrix)[0]) == testingBovw.rows);
		
		double recognitionRateMean, recognitionRateDeviation;
		ComputeRecognitionRate(confusionMatrix, recognitionRateMean, recognitionRateDeviation);
		
		cerr << "Recognition rate mean = " << recognitionRateMean * 100 << "% deviation " << recognitionRateDeviation * 100 << endl;
		
		recognitionRates[trial] = recognitionRateMean;

		if (trial == 0 || recognitionRateMean > bestRecognitionRate)
		{
			bestDictionary = dictionary;
			bestClassifier = classifier;
			bestRecognitionRate = recognitionRateMean;
			bestClassifierConfusionMatrix = confusionMatrix;
		}
	}

	SaveBestModels(keywords.getValue(), bestDictionary, bestClassifier);

	ShowConfusionMatrix(bestClassifierConfusionMatrix);

	WriteFinalStatistics(recognitionRates);
	
	return 0;
}
