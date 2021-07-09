// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file        unittest_tizen_capi_dataset.cpp
 * @date        14 July 2020
 * @brief       Unit test utility for dataset.
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>

#include <nntrainer.h>
#include <nntrainer_internal.h>
#include <nntrainer_test_util.h>

static const std::string getTestResPath(const std::string &filename) {
  return getResPath(filename, {"test"});
}

/**
 * @brief Neural Network Dataset Create / Destroy Test (negative test)
 */
TEST(nntrainer_capi_dataset, create_destroy_01_n) {
  ml_train_dataset_h dataset = nullptr;
  int status;
  status = ml_train_dataset_create_with_file(&dataset, NULL, NULL, NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_dataset_destroy(dataset);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Dataset Create / Destroy Test (negative test)
 */
TEST(nntrainer_capi_dataset, create_destroy_02_n) {
  ml_train_dataset_h dataset = nullptr;
  int status;
  status =
    ml_train_dataset_create_with_file(&dataset, "nofile.txt", NULL, NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_dataset_destroy(dataset);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Dataset Create / Destroy Test (negative test)
 */
TEST(nntrainer_capi_dataset, create_destroy_03_n) {
  ml_train_dataset_h dataset = nullptr;
  int status;
  status = ml_train_dataset_create_with_generator(&dataset, NULL, NULL, NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_dataset_destroy(dataset);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Dataset Create / Destroy Test (negative test)
 */
TEST(nntrainer_capi_dataset, create_destroy_04_n) {
  int status;
  status = ml_train_dataset_create_with_file(NULL, NULL, NULL, NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_dataset_create_with_generator(NULL, NULL, NULL, NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_dataset_destroy(NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Dataset Create / Destroy Test (positive test )
 */
TEST(nntrainer_capi_dataset, create_destroy_05_p) {
  ml_train_dataset_h dataset;
  int status;
  status = ml_train_dataset_create_with_file(
    &dataset, getTestResPath("trainingSet.dat").c_str(),
    getTestResPath("valSet.dat").c_str(),
    getTestResPath("testSet.dat").c_str());
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_dataset_destroy(dataset);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_dataset_create_with_file(
    &dataset, getTestResPath("trainingSet.dat").c_str(), NULL, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_dataset_destroy(dataset);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_dataset_destroy(dataset);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Dataset Create / Destroy Test (positive test )
 */
TEST(nntrainer_capi_dataset, create_destroy_06_p) {
  ml_train_dataset_h dataset;
  int status;

  status = ml_train_dataset_create_with_generator(&dataset, getBatch_train,
                                                  NULL, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_dataset_destroy(dataset);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_dataset_create_with_generator(&dataset, getBatch_train,
                                                  getBatch_val, getBatch_val);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_dataset_destroy(dataset);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Dataset set Property Test (positive test )
 */
TEST(nntrainer_capi_dataset, set_dataset_property_01_p) {
  ml_train_dataset_h dataset;
  int status;

  status = ml_train_dataset_create_with_generator(&dataset, getBatch_train,
                                                  NULL, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_dataset_set_property(dataset, "buffer_size=10", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_dataset_destroy(dataset);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Dataset set Property Test (positive test )
 */
TEST(nntrainer_capi_dataset, set_dataset_property_02_p) {
  ml_train_dataset_h dataset;
  int status;

  status = ml_train_dataset_create_with_file(
    &dataset, getTestResPath("trainingSet.dat").c_str(), NULL, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_dataset_set_property(dataset, "buffer_size=100", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_dataset_destroy(dataset);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Dataset set Property Test (negative test )
 */
TEST(nntrainer_capi_dataset, set_dataset_property_03_n) {
  ml_train_dataset_h dataset;
  int status;

  status = ml_train_dataset_create_with_generator(&dataset, getBatch_train,
                                                  NULL, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_dataset_set_property(dataset, "user_data=10", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_dataset_set_property(dataset, "user_data", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_dataset_destroy(dataset);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Dataset set Property Test (positive test )
 */
TEST(nntrainer_capi_dataset, set_dataset_property_04_p) {
  ml_train_dataset_h dataset;
  int status = ML_ERROR_NONE;

  status = ml_train_dataset_create_with_generator(&dataset, getBatch_train,
                                                  NULL, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status =
    ml_train_dataset_set_property(dataset, "user_data", (void *)&status, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_dataset_destroy(dataset);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Dataset set Property Test (positive test)
 */
TEST(nntrainer_capi_dataset, set_dataset_property_for_usage_01_p) {
  ml_train_dataset_h dataset;
  int status = ML_ERROR_NONE;

  status = ml_train_dataset_create_with_generator(
    &dataset, getBatch_train, getBatch_train, getBatch_train);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_dataset_set_property_for_usage(
    dataset, ML_TRAIN_DATASET_DATA_USAGE_TRAIN, "buffer_size=1", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_dataset_set_property_for_usage(
    dataset, ML_TRAIN_DATASET_DATA_USAGE_VALID, "buffer_size=1", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_dataset_set_property_for_usage(
    dataset, ML_TRAIN_DATASET_DATA_USAGE_TEST, "buffer_size=1", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_dataset_destroy(dataset);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Dataset set Property Test (negative test)
 */
TEST(nntrainer_capi_dataset,
     set_dataset_property_for_usage_does_not_exist_valid_n) {
  ml_train_dataset_h dataset;
  int status = ML_ERROR_NONE;

  status = ml_train_dataset_create_with_generator(&dataset, getBatch_train,
                                                  nullptr, getBatch_train);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_dataset_set_property_for_usage(
    dataset, ML_TRAIN_DATASET_DATA_USAGE_VALID, "buffer_size=1", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_dataset_destroy(dataset);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Dataset set Property Test (negative test)
 */
TEST(nntrainer_capi_dataset,
     set_dataset_property_for_usage_does_not_exist_test_n) {
  ml_train_dataset_h dataset;
  int status = ML_ERROR_NONE;

  status = ml_train_dataset_create_with_generator(&dataset, getBatch_train,
                                                  nullptr, nullptr);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_dataset_set_property_for_usage(
    dataset, ML_TRAIN_DATASET_DATA_USAGE_TEST, "buffer_size=1", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_dataset_destroy(dataset);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Dataset Create / Destroy Test (negative test )
 */
TEST(nntrainer_capi_dataset, set_dataset_01_n) {
  ml_train_model_h model;
  int status;

  status = ml_train_model_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_set_dataset(model, NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Dataset Create / Destroy Test (positive test )
 */
TEST(nntrainer_capi_dataset, set_dataset_02_p) {
  ml_train_model_h model;
  ml_train_dataset_h dataset;
  int status;

  status = ml_train_model_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_dataset_create_with_file(
    &dataset, getTestResPath("trainingSet.dat").c_str(), NULL, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_set_dataset(model, dataset);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Dataset Create / Destroy Test (positive test )
 */
TEST(nntrainer_capi_dataset, set_dataset_03_p) {
  ml_train_model_h model;
  ml_train_dataset_h dataset1, dataset2;
  int status;

  status = ml_train_model_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_dataset_create_with_file(
    &dataset1, getTestResPath("trainingSet.dat").c_str(), NULL, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_dataset_create_with_file(
    &dataset2, getTestResPath("valSet.dat").c_str(), NULL, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_set_dataset(model, dataset1);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_dataset_destroy(dataset1);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_model_set_dataset(model, dataset2);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_dataset_destroy(dataset2);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_dataset_destroy(dataset1);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error duing IniGoogleTest" << std::endl;
    return 0;
  }

  /** ignore tizen feature check while running the testcases */
  set_feature_state(SUPPORTED);

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error duing RUN_ALL_TSETS()" << std::endl;
  }

  /** reset tizen feature check state */
  set_feature_state(NOT_CHECKED_YET);

  return result;
}
