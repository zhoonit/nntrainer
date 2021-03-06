/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
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
 * @file	util_func.h
 * @date	08 April 2020
 * @brief	This is collection of math functions.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __UTIL_FUNC_H__
#define __UTIL_FUNC_H__
#ifdef __cplusplus

#include "tensor.h"

namespace nntrainer {

/**
 * @brief     derivative softmax function for Tensor Type
 * @param[in] x Tensor
 * @retVal    Tensor
 */
Tensor softmaxPrime(Tensor x);

/**
 * @brief       Calculate softmax for Tensor Type
 * @param[in] t Tensor
 * @retval      Tensor
 */
Tensor softmax(Tensor t);

/**
 * @brief     random function
 * @param[in] x float
 */
float random(float x);

/**
 * @brief     sqrt function for float type
 * @param[in] x float
 */
float sqrtFloat(float x);

/**
 * @brief     log function for float type
 * @param[in] x float
 */
float logFloat(float x);

/**
 * @brief     sigmoid activation function
 * @param[in] x input
 */
float sigmoid(float x);

/**
 * @brief     derivative sigmoid function
 * @param[in] x input
 */
float sigmoidePrime(float x);

/**
 * @brief     tanh function for float type
 * @param[in] x input
 */
float tanhFloat(float x);

/**
 * @brief     derivative tanh function
 * @param[in] x input
 */
float tanhPrime(float x);

/**
 * @brief     relu activation function
 * @param[in] x input
 */
float relu(float x);

/**
 * @brief     derivative relu function
 * @param[in] x input
 */
float reluPrime(float x);

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __UTIL_FUNC_H__ */
