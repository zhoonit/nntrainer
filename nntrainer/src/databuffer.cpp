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
 * @file	databuffer.cpp
 * @date	04 December 2019
 * @brief	This is buffer object to handle big data
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "databuffer.h"
#include "nntrainer_error.h"
#include <assert.h>
#include <climits>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <iomanip>
#include <mutex>
#include <nntrainer_log.h>
#include <sstream>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <thread>

#define NN_EXCEPTION_NOTI(val)                             \
  do {                                                     \
    switch (type) {                                        \
    case BUF_TRAIN: {                                      \
      std::lock_guard<std::mutex> lgtrain(readyTrainData); \
      trainReadyFlag = val;                                \
      cv_train.notify_all();                               \
    } break;                                               \
    case BUF_VAL: {                                        \
      std::lock_guard<std::mutex> lgval(readyValData);     \
      valReadyFlag = val;                                  \
      cv_val.notify_all();                                 \
    } break;                                               \
    case BUF_TEST: {                                       \
      std::lock_guard<std::mutex> lgtest(readyTestData);   \
      testReadyFlag = val;                                 \
      cv_test.notify_all();                                \
    } break;                                               \
    default:                                               \
      break;                                               \
    }                                                      \
  } while (0)

static std::exception_ptr globalExceptionPtr = nullptr;

namespace nntrainer {

std::mutex data_lock;

std::mutex readyTrainData;
std::mutex readyValData;
std::mutex readyTestData;

std::condition_variable cv_train;
std::condition_variable cv_val;
std::condition_variable cv_test;

DataStatus trainReadyFlag;
DataStatus valReadyFlag;
DataStatus testReadyFlag;

static int rangeRandom(int min, int max) {
  int n = max - min + 1;
  int remainder = RAND_MAX % n;
  int x;
  do {
    x = rand();
  } while (x >= RAND_MAX - remainder);
  return min + x % n;
}

static long getFileSize(std::string file_name) {
  std::ifstream file_stream(file_name.c_str(), std::ios::in | std::ios::binary);
  if (file_stream.good()) {
    file_stream.seekg(0, std::ios::end);
    return file_stream.tellg();
  } else {
    return 0;
  }
}

int DataBuffer::run(BufferType type) {
  int status = ML_ERROR_NONE;
  switch (type) {
  case BUF_TRAIN:
    if (trainReadyFlag == DATA_ERROR)
      return ML_ERROR_INVALID_PARAMETER;

    if (validation[DATA_TRAIN]) {
      this->train_thread =
        std::thread(&DataBuffer::updateData, this, type, std::ref(status));
      if (globalExceptionPtr) {
        try {
          std::rethrow_exception(globalExceptionPtr);
        } catch (const std::exception &ex) {
          std::cout << ex.what() << "\n";
          return ML_ERROR_INVALID_PARAMETER;
        }
      }
    }
    break;
  case BUF_VAL:
    if (valReadyFlag == DATA_ERROR)
      return ML_ERROR_INVALID_PARAMETER;
    if (validation[DATA_VAL]) {
      this->val_thread =
        std::thread(&DataBuffer::updateData, this, type, std::ref(status));
      if (globalExceptionPtr) {
        try {
          std::rethrow_exception(globalExceptionPtr);
        } catch (const std::exception &ex) {
          std::cout << ex.what() << "\n";
          return ML_ERROR_INVALID_PARAMETER;
        }
      }
    }
    break;
  case BUF_TEST:
    if (testReadyFlag == DATA_ERROR)
      return ML_ERROR_INVALID_PARAMETER;

    if (validation[DATA_TEST]) {
      this->test_thread =
        std::thread(&DataBuffer::updateData, this, type, std::ref(status));
      if (globalExceptionPtr) {
        try {
          std::rethrow_exception(globalExceptionPtr);
        } catch (const std::exception &ex) {
          std::cout << ex.what() << "\n";
          return ML_ERROR_INVALID_PARAMETER;
        }
      }
    }
    break;
  default:
    ml_loge("Error: Not Supported Data Type");
    status = ML_ERROR_INVALID_PARAMETER;
    break;
  }

  return status;
}

int DataBuffer::clear(BufferType type) {
  int status = ML_ERROR_NONE;
  switch (type) {
  case BUF_TRAIN: {
    train_running = false;
    this->train_data.clear();
    this->train_data_label.clear();
    this->cur_train_bufsize = 0;
    this->rest_train = max_train;
    trainReadyFlag = DATA_NOT_READY;
    this->train_running = true;
    if (validation[DATA_TRAIN] && true == train_thread.joinable())
      train_thread.join();
  } break;
  case BUF_VAL: {
    val_running = false;
    this->val_data.clear();
    this->val_data_label.clear();
    this->cur_val_bufsize = 0;
    this->rest_val = max_val;
    valReadyFlag = DATA_NOT_READY;
    this->val_running = true;
    if (validation[DATA_VAL] && true == val_thread.joinable())
      val_thread.join();

  } break;
  case BUF_TEST: {
    test_running = false;
    this->test_data.clear();
    this->test_data_label.clear();
    this->cur_test_bufsize = 0;
    this->rest_test = max_test;
    testReadyFlag = DATA_NOT_READY;
    this->test_running = true;
    if (validation[DATA_TEST] && true == test_thread.joinable())
      test_thread.join();
  } break;
  default:
    ml_loge("Error: Not Supported Data Type");
    status = ML_ERROR_INVALID_PARAMETER;
    break;
  }
  return status;
}

DataStatus DataBuffer::getStatus(BufferType type) {
  DataStatus ret = DATA_READY;
  if (globalExceptionPtr)
    ret = DATA_ERROR;

  switch (type) {
  case BUF_TRAIN:
    if ((train_data.size() < mini_batch) && trainReadyFlag)
      ret = DATA_NOT_READY;
    break;
  case BUF_VAL:
    if ((val_data.size() < mini_batch) && valReadyFlag)
      ret = DATA_NOT_READY;
    break;
  case BUF_TEST:
    if ((test_data.size() < mini_batch) && testReadyFlag)
      ret = DATA_NOT_READY;
    break;
  default:
    ml_loge("Error: Not Supported Data Type");
    ret = DATA_ERROR;
    break;
  }
  return ret;
}

bool DataBuffer::getDataFromBuffer(BufferType type, vec_3d &outVec,
                                   vec_3d &outLabel) {
  int nomI;
  unsigned int J, i, j, k;
  unsigned int width = input_size;
  unsigned int height = 1;

  switch (type) {
  case BUF_TRAIN: {
    std::vector<int> list;

    if (getStatus(BUF_TRAIN) != DATA_READY)
      return false;

    {
      std::unique_lock<std::mutex> ultest(readyTrainData);
      cv_train.wait(ultest, []() -> int { return trainReadyFlag; });
    }

    if (trainReadyFlag == DATA_ERROR) {
      return false;
    }

    data_lock.lock();
    for (k = 0; k < mini_batch; ++k) {
      nomI = rangeRandom(0, train_data.size() - 1);
      std::vector<std::vector<float>> v_height;
      for (j = 0; j < height; ++j) {
        J = j * width;
        std::vector<float> v_width;
        for (i = 0; i < width; ++i) {
          v_width.push_back(train_data[nomI][J + i]);
        }
        v_height.push_back(v_width);
      }

      list.push_back(nomI);
      outVec.push_back(v_height);
      outLabel.push_back({train_data_label[nomI]});
    }
    for (i = 0; i < mini_batch; ++i) {
      train_data.erase(train_data.begin() + list[i]);
      train_data_label.erase(train_data_label.begin() + list[i]);
      cur_train_bufsize--;
    }
  } break;
  case BUF_VAL: {
    std::vector<int> list;
    if (getStatus(BUF_VAL) != DATA_READY)
      return false;

    {
      std::unique_lock<std::mutex> ulval(readyValData);
      cv_val.wait(ulval, []() -> bool { return valReadyFlag; });
    }

    data_lock.lock();
    for (k = 0; k < mini_batch; ++k) {
      nomI = rangeRandom(0, val_data.size() - 1);
      std::vector<std::vector<float>> v_height;
      for (j = 0; j < height; ++j) {
        J = j * width;
        std::vector<float> v_width;
        for (i = 0; i < width; ++i) {
          v_width.push_back(val_data[nomI][J + i]);
        }
        v_height.push_back(v_width);
      }

      list.push_back(nomI);
      outVec.push_back(v_height);
      outLabel.push_back({val_data_label[nomI]});
    }
    for (i = 0; i < mini_batch; ++i) {
      val_data.erase(val_data.begin() + list[i]);
      val_data_label.erase(val_data_label.begin() + list[i]);
      cur_val_bufsize--;
    }
  } break;
  case BUF_TEST: {
    std::vector<int> list;
    if (getStatus(BUF_TEST) != DATA_READY)
      return false;

    {
      std::unique_lock<std::mutex> ultest(readyTestData);
      cv_test.wait(ultest, []() -> bool { return testReadyFlag; });
    }

    data_lock.lock();
    for (k = 0; k < mini_batch; ++k) {
      nomI = rangeRandom(0, test_data.size() - 1);
      std::vector<std::vector<float>> v_height;
      for (j = 0; j < height; ++j) {
        J = j * width;
        std::vector<float> v_width;
        for (i = 0; i < width; ++i) {
          v_width.push_back(test_data[nomI][J + i]);
        }
        v_height.push_back(v_width);
      }

      list.push_back(nomI);
      outVec.push_back(v_height);
      outLabel.push_back({test_data_label[nomI]});
    }
    for (i = 0; i < mini_batch; ++i) {
      test_data.erase(test_data.begin() + list[i]);
      test_data_label.erase(test_data_label.begin() + list[i]);
      cur_test_bufsize--;
    }
  } break;
  default:
    ml_loge("Error: Not Supported Data Type");
    return false;
    break;
  }
  data_lock.unlock();

  return true;
}

int DataBuffer::setClassNum(unsigned int num) {
  int status = ML_ERROR_NONE;
  if (num <= 0) {
    ml_loge("Error: number of class should be bigger than 0");
    SET_VALIDATION(false);
    return ML_ERROR_INVALID_PARAMETER;
  }
  if (class_num != 0 && class_num != num) {
    ml_loge("Error: number of class should be same with number of label label");
    SET_VALIDATION(false);
    return ML_ERROR_INVALID_PARAMETER;
  }
  class_num = num;
  return status;
}

int DataBuffer::setBufSize(unsigned int size) {
  int status = ML_ERROR_NONE;
  if (size < mini_batch) {
    ml_loge("Error: buffer size must be greater than batch size");
    SET_VALIDATION(false);
    return ML_ERROR_INVALID_PARAMETER;
  }
  bufsize = size;
  return status;
}

int DataBuffer::setMiniBatch(unsigned int size) {
  int status = ML_ERROR_NONE;
  if (size == 0) {
    ml_loge("Error: batch size must be greater than 0");
    SET_VALIDATION(false);
    return ML_ERROR_INVALID_PARAMETER;
  }
  mini_batch = size;
  return status;
}

int DataBuffer::setFeatureSize(unsigned int size) {
  int status = ML_ERROR_NONE;
  if (size == 0) {
    ml_loge("Error: batch size must be greater than 0");
    SET_VALIDATION(false);
    return ML_ERROR_INVALID_PARAMETER;
  }

  input_size = size;
  return status;
}

void DataBuffer::displayProgress(const int count, BufferType type, float loss) {
  int barWidth = 20;
  float max_size = max_train;
  switch (type) {
  case BUF_TRAIN:
    max_size = max_train;
    break;
  case BUF_VAL:
    max_size = max_val;
    break;
  case BUF_TEST:
    max_size = max_test;
    break;
  default:
    ml_loge("Error: Not Supported Data Type");
    break;
  }
  std::stringstream ssInt;
  ssInt << count * mini_batch;

  std::string str = ssInt.str();
  int len = str.length();

  if (max_size == 0) {
    int pad_left = (barWidth - len) / 2;
    int pad_right = barWidth - pad_left - len;
    std::string out_str =
      std::string(pad_left, ' ') + str + std::string(pad_right, ' ');
    std::cout << " [ ";
    std::cout << out_str;
    std::cout << " ] "
              << " ( Training Loss: " << loss << " )\r";
  } else {
    float progress;
    if (mini_batch > max_size)
      progress = 1.0;
    else
      progress = (((float)(count * mini_batch)) / max_size);

    int pos = barWidth * progress;
    std::cout << " [ ";
    for (int l = 0; l < barWidth; ++l) {
      if (l <= pos)
        std::cout << "=";
      else
        std::cout << " ";
    }
    std::cout << " ] " << int(progress * 100.0) << "% ( Training Loss: " << loss
              << " )\r";
  }

  std::cout.flush();
}

int DataBufferFromDataFile::init() {

  int status = ML_ERROR_NONE;

  if (!class_num) {
    ml_loge("Error: number of class must be set");
    SET_VALIDATION(false);
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (!this->input_size) {
    ml_loge("Error: featuer size must be set");
    SET_VALIDATION(false);
    return ML_ERROR_INVALID_PARAMETER;
  }

  this->cur_train_bufsize = 0;
  this->cur_val_bufsize = 0;
  this->cur_test_bufsize = 0;

  if (mini_batch == 0) {
    ml_loge("Error: mini batch size must be greater than 0");
    SET_VALIDATION(false);
    return ML_ERROR_INVALID_PARAMETER;
  }

  this->rest_train = max_train;
  this->rest_val = max_val;
  this->rest_test = max_test;

  this->train_running = true;
  this->val_running = true;
  this->test_running = true;

  trainReadyFlag = DATA_NOT_READY;
  valReadyFlag = DATA_NOT_READY;
  testReadyFlag = DATA_NOT_READY;
  return status;
}

void DataBufferFromDataFile::updateData(BufferType type, int &status) {
  unsigned int max_size = 0;
  unsigned int buf_size = 0;
  unsigned int *rest_size = NULL;
  unsigned int *cur_size = NULL;
  bool *running = NULL;
  std::vector<std::vector<float>> *data = NULL;
  std::vector<std::vector<float>> *datalabel = NULL;
  std::ifstream file;
  switch (type) {
  case BUF_TRAIN: {
    max_size = max_train;
    buf_size = bufsize;
    rest_size = &rest_train;
    cur_size = &cur_train_bufsize;
    running = &train_running;
    data = &train_data;
    datalabel = &train_data_label;
    std::ifstream train_stream(train_name, std::ios::in | std::ios::binary);
    file.swap(train_stream);
  } break;
  case BUF_VAL: {
    max_size = max_val;
    buf_size = bufsize;
    rest_size = &rest_val;
    cur_size = &cur_val_bufsize;
    running = &val_running;
    data = &val_data;
    datalabel = &val_data_label;
    std::ifstream val_stream(val_name, std::ios::in | std::ios::binary);
    file.swap(val_stream);
  } break;
  case BUF_TEST: {
    max_size = max_test;
    buf_size = bufsize;
    rest_size = &rest_test;
    cur_size = &cur_test_bufsize;
    running = &test_running;
    data = &test_data;
    datalabel = &test_data_label;
    std::ifstream test_stream(test_name, std::ios::in | std::ios::binary);
    file.swap(test_stream);
  } break;
  default:
    try {
      throw std::runtime_error("Error: Not Supported Data Type");
    } catch (...) {
      globalExceptionPtr = std::current_exception();
      NN_EXCEPTION_NOTI(DATA_ERROR);
      return;
    }
    break;
  }

  unsigned int I;
  std::vector<unsigned int> mark;
  mark.resize(max_size);
  file.clear();
  file.seekg(0, std::ios_base::end);
  uint64_t file_length = file.tellg();

  for (unsigned int i = 0; i < max_size; ++i) {
    mark[i] = i;
  }

  while ((*running) && mark.size() != 0) {
    if (buf_size - (*cur_size) > 0 && (*rest_size) > 0) {
      std::vector<float> vec;
      std::vector<float> veclabel;

      unsigned int id = rangeRandom(0, mark.size() - 1);
      I = mark[id];

      try {
        if (I > max_size) {
          ml_loge("Error: Test case id cannot exceed maximum number of test");
          status = ML_ERROR_INVALID_PARAMETER;
          throw std::runtime_error(
            "Error: Test case id cannot exceed maximum number of test");
        }
      } catch (...) {
        globalExceptionPtr = std::current_exception();
        NN_EXCEPTION_NOTI(DATA_ERROR);
        return;
      }

      mark.erase(mark.begin() + id);
      uint64_t position = (I * input_size + I * class_num) * sizeof(float);
      try {
        if (position > file_length || position > ULLONG_MAX) {
          ml_loge("Error: Cannot exceed max file size");
          status = ML_ERROR_INVALID_PARAMETER;
          throw std::runtime_error("Error: Cannot exceed max file size");
        }
      } catch (...) {
        globalExceptionPtr = std::current_exception();
        NN_EXCEPTION_NOTI(DATA_ERROR);
        return;
      }

      file.seekg(position, std::ios::beg);

      for (unsigned int j = 0; j < input_size; ++j) {
        float d;
        file.read((char *)&d, sizeof(float));
        vec.push_back(d);
      }

      for (unsigned int j = 0; j < class_num; ++j) {
        float d;
        file.read((char *)&d, sizeof(float));
        veclabel.push_back(d);
      }

      data_lock.lock();
      data->push_back(vec);
      datalabel->push_back(veclabel);
      (*rest_size)--;
      (*cur_size)++;
      data_lock.unlock();
    }

    if (buf_size == (*cur_size)) {
      NN_EXCEPTION_NOTI(DATA_READY);
    }
  }
  file.close();
}

int DataBufferFromDataFile::setDataFile(std::string path, DataType type) {
  int status = ML_ERROR_NONE;
  std::ifstream data_file(path.c_str());

  switch (type) {
  case DATA_TRAIN: {
    if (!data_file.good()) {
      ml_loge(
        "Error: Cannot open data file, Datafile is necessary for training");
      validation[type] = false;
      return ML_ERROR_INVALID_PARAMETER;
    }
    train_name = path;
  } break;
  case DATA_VAL: {
    if (!data_file.good()) {
      ml_logw("Warning: Cannot open validation data file. Cannot validate "
              "training result");
      validation[type] = false;
      break;
    }
    val_name = path;
  } break;
  case DATA_TEST: {
    if (!data_file.good()) {
      ml_logw(
        "Warning: Cannot open test data file. Cannot test training result");
      validation[type] = false;
      break;
    }
    test_name = path;
  } break;
  case DATA_LABEL: {
    std::string data;
    if (!data_file.good()) {
      ml_loge("Error: Cannot open label file");
      SET_VALIDATION(false);
      return ML_ERROR_INVALID_PARAMETER;
    }
    while (data_file >> data) {
      labels.push_back(data);
    }
    if (class_num != 0 && class_num != labels.size()) {
      ml_loge("Error: number of label should be same with number class number");
      SET_VALIDATION(false);
      return ML_ERROR_INVALID_PARAMETER;
    }
    class_num = labels.size();
  } break;
  case DATA_UNKNOWN:
  default:
    ml_loge("Error: Not Supported Data Type");
    SET_VALIDATION(false);
    return ML_ERROR_INVALID_PARAMETER;
    break;
  }
  return status;
}

int DataBufferFromDataFile::setFeatureSize(unsigned int size) {
  int status = ML_ERROR_NONE;
  long file_size = 0;

  status = DataBuffer::setFeatureSize(size);
  if (status != ML_ERROR_NONE)
    return status;

  if (validation[DATA_TRAIN]) {
    file_size = getFileSize(train_name);
    max_train = static_cast<unsigned int>(
      file_size / (class_num * sizeof(int) + input_size * sizeof(float)));
    if (max_train < mini_batch) {
      ml_logw(
        "Warning: number of training data is smaller than mini batch size");
    }
  } else {
    max_train = 0;
  }

  if (validation[DATA_VAL]) {
    file_size = getFileSize(val_name);
    max_val = static_cast<unsigned int>(
      file_size / (class_num * sizeof(int) + input_size * sizeof(float)));
    if (max_val < mini_batch) {
      ml_logw("Warning: number of val data is smaller than mini batch size");
    }
  } else {
    max_val = 0;
  }

  if (validation[DATA_TEST]) {
    file_size = getFileSize(test_name);
    max_test = static_cast<unsigned int>(
      file_size / (class_num * sizeof(int) + input_size * sizeof(float)));
    if (max_test < mini_batch) {
      ml_logw("Warning: number of test data is smaller than mini batch size");
    }
  } else {
    max_test = 0;
  }

  return status;
}

int DataBufferFromCallback::init() {
  int status = ML_ERROR_NONE;

  if (!class_num) {
    ml_loge("Error: number of class must be set");
    SET_VALIDATION(false);
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (!this->input_size) {
    ml_loge("Error: featuer size must be set");
    SET_VALIDATION(false);
    return ML_ERROR_INVALID_PARAMETER;
  }

  this->cur_train_bufsize = 0;
  this->cur_val_bufsize = 0;
  this->cur_test_bufsize = 0;

  this->max_train = 0;
  this->max_val = 0;
  this->max_test = 0;

  if (mini_batch == 0) {
    ml_loge("Error: mini batch size must be greater than 0");
    SET_VALIDATION(false);
    return ML_ERROR_INVALID_PARAMETER;
  }

  this->train_running = true;
  this->val_running = true;
  this->test_running = true;

  trainReadyFlag = DATA_NOT_READY;
  valReadyFlag = DATA_NOT_READY;
  testReadyFlag = DATA_NOT_READY;

  return status;
}

int DataBufferFromCallback::setFunc(
  BufferType type, std::function<bool(vec_3d &, vec_3d &, int &)> func) {

  int status = ML_ERROR_NONE;
  switch (type) {
  case BUF_TRAIN:
    callback_train = func;
    if (func == NULL)
      validation[0] = false;
    break;
  case BUF_VAL:
    callback_val = func;
    if (func == NULL)
      validation[1] = false;
    break;
  case BUF_TEST:
    callback_test = func;
    if (func == NULL)
      validation[2] = false;
    break;
  default:
    status = ML_ERROR_INVALID_PARAMETER;
    break;
  }

  return status;
}

void DataBufferFromCallback::updateData(BufferType type, int &status) {
  status = ML_ERROR_NONE;

  unsigned int buf_size = 0;
  unsigned int *cur_size = NULL;
  bool *running = NULL;
  std::vector<std::vector<float>> *data = NULL;
  std::vector<std::vector<float>> *datalabel = NULL;
  std::function<bool(vec_3d &, vec_3d &, int &)> callback;

  switch (type) {
  case BUF_TRAIN: {
    buf_size = bufsize;
    cur_size = &cur_train_bufsize;
    running = &train_running;
    data = &train_data;
    datalabel = &train_data_label;
    callback = callback_train;
  } break;
  case BUF_VAL: {
    buf_size = bufsize;
    cur_size = &cur_val_bufsize;
    running = &val_running;
    data = &val_data;
    datalabel = &val_data_label;
    callback = callback_val;
  } break;
  case BUF_TEST: {
    buf_size = bufsize;
    cur_size = &cur_test_bufsize;
    running = &test_running;
    data = &test_data;
    datalabel = &test_data_label;
    callback = callback_test;
  } break;
  default:
    break;
  }

  while ((*running)) {
    if (buf_size - (*cur_size) > 0) {
      vec_3d vec;
      vec_3d veclabel;

      bool endflag = callback(vec, veclabel, status);
      if (!endflag)
        break;

      if (vec.size() != veclabel.size()) {
        status = ML_ERROR_INVALID_PARAMETER;
      }

      for (unsigned int i = 0; i < vec.size(); ++i) {
        std::vector<float> v;
        std::vector<float> vl;
        for (unsigned int j = 0; j < vec[i].size(); ++j) {
          for (unsigned int k = 0; k < vec[i][j].size(); ++k) {
            v.push_back(vec[i][j][k]);
          }
        }
        for (unsigned int j = 0; j < veclabel[i].size(); ++j) {
          for (unsigned int k = 0; k < veclabel[i][j].size(); ++k) {
            vl.push_back(veclabel[i][j][k]);
          }
        }

        data_lock.lock();
        data->push_back(v);
        datalabel->push_back(vl);
        (*cur_size)++;
        data_lock.unlock();
      }
    }

    if (buf_size == (*cur_size)) {
      switch (type) {
      case BUF_TRAIN: {
        std::lock_guard<std::mutex> lgtrain(readyTrainData);
        trainReadyFlag = DATA_READY;
        cv_train.notify_all();
      } break;
      case BUF_VAL: {
        std::lock_guard<std::mutex> lgval(readyValData);
        valReadyFlag = DATA_READY;
        cv_val.notify_all();
      } break;
      case BUF_TEST: {
        std::lock_guard<std::mutex> lgtest(readyTestData);
        testReadyFlag = DATA_READY;
        cv_test.notify_all();
      } break;
      default:
        break;
      }
    }
  }
}

} /* namespace nntrainer */
