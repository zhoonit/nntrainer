// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   func_data_producer.cpp
 * @date   12 July 2021
 * @brief  This file contains various data producers from a callback
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <func_data_producer.h>

#include <nntrainer_error.h>

namespace nntrainer {

FuncDataProducer::FuncDataProducer(datagen_cb datagen_cb, void *user_data_) :
  cb(datagen_cb),
  user_data(user_data_) {}

FuncDataProducer::~FuncDataProducer() {}

const std::string FuncDataProducer::getType() const {
  return FuncDataProducer::type;
}

void FuncDataProducer::setProperty(const std::vector<std::string> &properties) {
  NNTR_THROW_IF(!properties.empty(), std::invalid_argument)
    << "properties is not empty, size: " << properties.size();
}

DataProducer::Generator
FuncDataProducer::finalize(const std::vector<TensorDim> &input_dims,
                           const std::vector<TensorDim> &label_dims) {
  return [cb = this->cb, ud = this->user_data, input_dims,
          label_dims]() -> DataProducer::Iteration {
    std::vector<Tensor> inputs;
    inputs.reserve(input_dims.size());
    float **input_data = new float *[input_dims.size()];

    for (unsigned int i = 0; i < input_dims.size(); ++i) {
      inputs.emplace_back(input_dims[i]);
      *(input_data + i) = inputs.back().getData();
    }

    std::vector<Tensor> labels;
    labels.reserve(label_dims.size());
    float **label_data = new float *[label_dims.size()];

    for (unsigned int i = 0; i < label_dims.size(); ++i) {
      labels.emplace_back(label_dims[i]);
      *(label_data + i) = labels.back().getData();
    }

    bool last = false;
    int status = cb(input_data, label_data, &last, ud);
    NNTR_THROW_IF(status != ML_ERROR_NONE, std::invalid_argument)
      << "[DataProducer] Callback returned error: " << status << '\n';

    if (last) {
      return {true, {}, {}};
    } else {
      return {false, inputs, labels};
    }
  };
}
} // namespace nntrainer
