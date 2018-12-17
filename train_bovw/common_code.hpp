/*! @file common_code.hpp
	@brief Useful for building a Bag of Visual Words model
	@authors Fundamentos de Sistemas Inteligentes en Vision
*/

#ifndef __COMMON_CODE_HPP__
#define __COMMON_CODE_HPP__

#include <string>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;
using namespace ml;

/**
 * @brief Get the parts of the file name from the absolute path.
 * @param[in] path is the path to the file.
 * @param[out] directoryName is the name of the directory.
 * @param[out] fileName is the name of the file.
 * @param[out] extension is the file extension.
 */
void Basename(const string& path, string& directoryName, string& fileName, string& extension);

/**
 * @brief Generate the corresponding filename of a sample in the dataset.
 * @param[in] baseName is the base name.
 * @param[in] category is the name of the category.
 * @param[in] sampleIndex is the sample index.
 * @return The sample file name with its path.
 */
string ComputeSampleFilename(const string& baseName, const string& category, int sampleIndex);

/**
 * @brief Load the dataset description. Expected is one row per category with category name and the number of samples.
 * @arg[in] fileName is the pathname of the file.
 * @arg[out] categories is a vector with the names of the categories.
 * @arg[out] samplesPerCategory is a vector with the number of samples per each category.
 * @return returnCode: 0 -> success, 1 -> could not open file, 2 -> wrong file format.
 */
int LoadDatasetInformation(const string& fileName, vector<string>& categories, vector<int>& samplesPerCategory);

/**
 * @brief Create a dataset to train and test.
 * @arg[in] samplesPerCategory is the total samples per category.
 * @arg[in] numberOfTrainingSamples is the number of training samples per category.
 * @arg[in] numberOfTestingSamples is the number of testing samples per category.
 * @arg[out] trainingSamples is the sample's index per category that will be used to train.
 * @arg[out] testingSamples is the sample's index per category that will be used to test.
 */
void CreateTrainingAndTestingDatasets(vector<int>& samplesPerCategory, int numberOfTrainingSamples, int numberOfTestingSamples,
	vector<vector<int>>& trainingSamples, vector<vector<int>>& testingSamples);

/**
 * @brief Compute the recognition rate from a confusion matrix.
 * @param[in] confusionMatrix is the CxC confusion matrix.
 * @param[out] mean is the recognition rate mean on classes C.
 * @param[out] deviation is the recognition rate deviation on classes C.
 */
void ComputeRecognitionRate(const Mat& confusionMatrix, double& mean, double& deviation);

/**
  * @brief Compute the confusion matrix.
  * @param[in] numberOfCategories is the number of different categories.
  * @param[in] trueLabels is a vector with the true labels.
  * @param[in] predictedLabels is a vector with the predicted labels.
  * @return the confusion matrix.
  */
Mat ComputeConfusionMatrix(int numberOfCategories, const Mat& trueLabels, const Mat& predictedLabels);

/**
 * @brief Extract sparse SIFT descriptors of a given image.
 * @param[in] image is the input image.
 * @param[in] numberOfDescriptors is the maximum number of descriptors to be returned.
 * @return the set of SIFT descriptors.
 */
Mat ExtractSparseSIFTDescriptors(const Mat& image, int numberOfDescriptors = 0);

/**
 * @brief Extract dense SIFT descriptors of a given image.
 * @param[in] image is the input image.
 * @param[in] siftScales are the scales for the dense SIFT extraction.
 * @return the set of dense SIFT descriptors.
 */
Mat ExtractDenseSIFTDescriptors(const Mat& image, const vector<int>& siftScales);

/**
 * @brief Extract SURF descriptors of a given image.
 * @param[in] image is the input image.
 * @return the set of SURF descriptors.
 */
Mat ExtractSURFDescriptors(const Mat& image);

/**
 * \brief Extract PHOW descriptors of a given image.
 * \param[in] image is the input image.
 * \param[in] siftScales are the scales for the dense SIFT extraction. 
 * \return the set of PHOW descriptors.
 */
Mat ExtractPHOWDescriptors(const Mat& image, vector<int>& siftScales);

/**
 * @brief Compute the Bag of Visual Words descriptor.
 * @param[in] dictionary is the visual dictionary.
 * @param[in] dictionarySize is number of visual words.
 * @param[in] imageDescriptors is the set of keypoint descriptors used to represent the image.
 * @param[in] normalize is a flag that indicates whether the BoVW descriptor should be normalized.
 * @return the BoVW descriptor.
 */
Mat ComputeBovw(const Ptr<KNearest>& dictionary, int dictionarySize, Mat& imageDescriptors, bool normalize = true);

/**
 * @brief Get the descriptor type from the user.
 * @return the descriptor type (1 = SIFT, 2 = SURF, 3 = Dense SIFT, 4 = PHOW)
 */
int GetDescriptorValue();

/**
 * @brief Get the 'k' value for kNN classifier from the user.
 * @return the 'k' value for the kNN classifier.
 */
int GetKnnValue();

/**
 * @brief Get the classifier type from the user.
 * @return the classifier type (1 = kNN, 2 = SVM, 3 = Random Forest, 4 = Boosting)
 */
int GetClassifierValue();

/**
 * @brief Get the SVM margin value from the user.
 * @return the SVM margin value.
 */
double GetSvmMarginValue();

/**
 * @brief Get the SVM classifier kernel type from the user.
 * @return the type of the SVM kernel.
 */
SVM::KernelTypes GetSvmKernelType();

/**
 * @brief Get the maximum depth of the Random Forest from the user.
 * @return the Random forest maximum depth.
 */
int GetRandomForestMaxDepth();

/**
 * @brief Get the random forest minimal samples count.
 * @return the minimum samples to the random forest.
 */
int GetRandomForestMinimumSamplesCount();

/**
 * @brief Get the number of trees for the random forest from the user.
 * @return the number of trees.
 */
int GetNumberOfTrees();

#endif //__COMMON_CODE_HPP__
