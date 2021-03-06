/**
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	main_func.cpp
 * @date	04 December 2019
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is Classification Example with one FC Layer
 *              The base model for feature extractor is mobilenet v2 with
 * 1280*7*7 feature size. It read the Classification.ini in res directory and
 * run according to the configureation.
 *
 */
#include <climits>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include <stdlib.h>
#include <time.h>

#include "layers.h"
#include "neuralnet.h"
#include "tensor.h"

#define TRAINING true

/**
 * @brief     Data size for each category
 */
const unsigned int total_train_data_size = 100;

const unsigned int total_val_data_size = 10;

const unsigned int total_test_data_size = 100;

const unsigned int buffer_size = 100;

/**
 * @brief     Number of category : Three
 */
const unsigned int total_label_size = 10;

/**
 * @brief     Max Epoch
 */
const unsigned int iteration = 3000;

const unsigned int mini_batch = 32;

const unsigned int feature_size = 62720;

using namespace std;

/**
 * @brief     location of resources ( ../../res/ )
 */
string data_path;

bool duplicate[total_label_size * total_train_data_size];
bool valduplicate[total_label_size * total_val_data_size];

/**
 * @brief     step function
 * @param[in] x value to be distinguished
 * @retval 0.0 or 1.0
 */
float stepFunction(float x) {
  if (x > 0.9) {
    return 1.0;
  }

  if (x < 0.1) {
    return 0.0;
  }

  return x;
}

/**
 * @brief     Generate Random integer value between min to max
 * @param[in] min : minimum value
 * @param[in] max : maximum value
 * @retval    min < random value < max
 */
static int rangeRandom(int min, int max) {
  int n = max - min + 1;
  int remainder = RAND_MAX % n;
  int x;
  do {
    x = rand();
  } while (x >= RAND_MAX - remainder);
  return min + x % n;
}

/**
 * @brief     load data at specific position of file
 * @param[in] F  ifstream (input file)
 * @param[out] outVec
 * @param[out] outLabel
 * @param[in] id th data to get
 * @retval true/false false : end of data
 */
bool getData(std::ifstream &F, std::vector<float> &outVec,
             std::vector<float> &outLabel, int id) {
  uint64_t position = (feature_size + total_label_size) * id * sizeof(float);
  if ((position + (feature_size + total_label_size) * sizeof(float)) >
      ULLONG_MAX)
    return false;
  F.seekg(position, std::ios::beg);
  for (unsigned int i = 0; i < feature_size; i++)
    F.read((char *)&outVec[i], sizeof(float));
  for (unsigned int i = 0; i < total_label_size; i++)
    F.read((char *)&outLabel[i], sizeof(float));

  return true;
}

/**
 * @brief      get data which size is mini batch for train
 * @param[out] outVec
 * @param[out] outLabel
 * @param[out] status for error handling
 * @retval true/false
 */
bool getMiniBatch_train(std::vector<std::vector<std::vector<float>>> &outVec,
                        std::vector<std::vector<std::vector<float>>> &outLabel,
                        int &status) {
  std::vector<int> memI;
  std::vector<int> memJ;
  unsigned int count = 0;
  int data_size = total_train_data_size;

  std::string filename = "trainingSet.dat";
  std::ifstream F(filename, std::ios::in | std::ios::binary);

  for (unsigned int i = 0; i < total_label_size * data_size; i++) {
    if (!duplicate[i])
      count++;
  }

  if (count < mini_batch) {
    for (unsigned int i = 0; i < total_label_size * data_size; ++i) {
      duplicate[i] = false;
    }
    return false;
  }

  count = 0;
  while (count < mini_batch) {
    int nomI = rangeRandom(0, total_label_size * data_size - 1);
    if (!duplicate[nomI]) {
      memI.push_back(nomI);
      duplicate[nomI] = true;
      count++;
    }
  }

  for (unsigned int i = 0; i < count; i++) {
    std::vector<std::vector<float>> out;
    std::vector<std::vector<float>> outL;
    std::vector<float> o;
    std::vector<float> l;

    o.resize(feature_size);
    l.resize(total_label_size);

    getData(F, o, l, memI[i]);

    out.push_back(o);
    outL.push_back(l);

    outVec.push_back(out);
    outLabel.push_back(outL);
  }

  F.close();
  return true;
}

/**
 * @brief      get data which size is mini batch for validation
 * @param[out] outVec
 * @param[out] outLabel
 * @param[out] status for error handling
 * @retval true/false false : end of data
 */
bool getMiniBatch_val(std::vector<std::vector<std::vector<float>>> &outVec,
                      std::vector<std::vector<std::vector<float>>> &outLabel,
                      int &status) {
  std::vector<int> memI;
  std::vector<int> memJ;
  unsigned int count = 0;
  int data_size = total_val_data_size;

  std::string filename = "trainingSet.dat";
  std::ifstream F(filename, std::ios::in | std::ios::binary);

  for (unsigned int i = 0; i < total_label_size * data_size; i++) {
    if (!valduplicate[i])
      count++;
  }

  if (count < mini_batch) {
    for (unsigned int i = 0; i < total_label_size * data_size; ++i) {
      valduplicate[i] = false;
    }
    return false;
  }

  count = 0;
  while (count < mini_batch) {
    int nomI = rangeRandom(0, total_label_size * data_size - 1);
    if (!valduplicate[nomI]) {
      memI.push_back(nomI);
      valduplicate[nomI] = true;
      count++;
    }
  }

  for (unsigned int i = 0; i < count; i++) {
    std::vector<std::vector<float>> out;
    std::vector<std::vector<float>> outL;
    std::vector<float> o;
    std::vector<float> l;

    o.resize(feature_size);
    l.resize(total_label_size);

    getData(F, o, l, memI[i]);

    out.push_back(o);
    outL.push_back(l);

    outVec.push_back(out);
    outLabel.push_back(outL);
  }

  F.close();
  return true;
}

/**
 * @brief     create NN
 *            Get Feature from tflite & run foword & back propatation
 * @param[in]  arg 1 : configuration file path
 * @param[in]  arg 2 : resource path
 */
int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "./nntrainer_classification_func Config.ini\n";
    exit(0);
  }
  const vector<string> args(argv + 1, argv + argc);
  std::string config = args[0];
  data_path = args[1];

  srand(time(NULL));
  std::string ini_file = data_path + "ini.bin";
  std::vector<std::vector<float>> inputVector, outputVector;
  std::vector<std::vector<float>> inputValVector, outputValVector;
  std::vector<std::vector<float>> inputTestVector, outputTestVector;

  // This is to check duplication of data
  for (unsigned int i = 0; i < total_label_size * total_train_data_size; ++i) {
    duplicate[i] = false;
  }

  for (unsigned int i = 0; i < total_label_size * total_val_data_size; ++i) {
    valduplicate[i] = false;
  }

  /**
   * @brief     Neural Network Create & Initialization
   */
  nntrainer::NeuralNetwork NN;
  NN.setConfig(config);
  NN.init();
  NN.readModel();

  /**
   * @brief     Neural Network Train & validation
   */
  NN.train(getMiniBatch_train, getMiniBatch_val, nullptr);

  /**
   * @brief     Finalize NN
   */
  NN.finalize();
  return 0;
}
