// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file ini_interpreter.cpp
 * @date 02 April 2021
 * @brief NNTrainer Ini Interpreter (partly moved from model_loader.c)
 * @see	https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <ini_interpreter.h>

#include <vector>

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <util_func.h>

static constexpr const char *FUNC_TAG = "[IniInterpreter] ";

static constexpr const char *UNKNOWN_STR = "Unknown";
static constexpr const char *NONE_STR = "NONE";
static constexpr const char *MODEL_STR = "model";
static constexpr const char *DATASET_STR = "dataset";
static constexpr const char *OPTIMIZER_STR = "optimizer";

namespace nntrainer {

namespace {

/** @todo:
 *  1. deprecate tag dispatching along with #1072
 *  2. deprecate getMergeableGraph (extendGraph should accept graph itself)
 */
class PlainLayer {};    /**< Plain Layer tag */
class BackboneLayer {}; /**< Backbone Layer tag */

/**
 * @brief convert section to list of string based properties
 *
 * @param ini ini handler
 * @param sec_name section name
 * @return std::vector<std::string> list of properties
 */
std::vector<std::string> section2properties(dictionary *ini,
                                            const std::string &sec_name) {
  int num_entries = iniparser_getsecnkeys(ini, sec_name.c_str());
  NNTR_THROW_IF(num_entries < 1, std::invalid_argument)
    << "there are no entries in the layer section: " << sec_name;
  ml_logd("number of entries for %s: %d", sec_name.c_str(), num_entries);

  std::unique_ptr<const char *[]> key_refs(new const char *[num_entries]);
  NNTR_THROW_IF(iniparser_getseckeys(ini, sec_name.c_str(), key_refs.get()) ==
                  nullptr,
                std::invalid_argument)
    << "failed to fetch keys for the section: " << sec_name;

  std::vector<std::string> properties;
  properties.reserve(num_entries - 1);

  for (int i = 0; i < num_entries; ++i) {
    std::string key(key_refs[i]);
    std::string prop_key = key.substr(key.find(":") + 1);

    if (istrequal(prop_key, "type") || istrequal(prop_key, "backbone")) {
      continue;
    }

    std::string value = iniparser_getstring(ini, key_refs[i], UNKNOWN_STR);
    NNTR_THROW_IF(value == UNKNOWN_STR || value.empty(), std::invalid_argument)
      << "parsing property failed key: " << key << " value: " << value;

    ml_logd("parsed properties: %s=%s", prop_key.c_str(), value.c_str());
    properties.push_back(prop_key + "=" + value);
  }

  return properties;
}

/**
 * @brief convert section to a layer object (later it might be layer node)
 *
 * @tparam T tag
 * @param ini dictionary * ini
 * @param sec_name section name
 * @param ac app context to search for the layer
 * @return std::shared_ptr<Layer> return layer object
 */
template <typename T>
std::shared_ptr<Layer> section2layer(dictionary *ini,
                                     const std::string &sec_name,
                                     const AppContext &ac) {
  throw std::invalid_argument("supported only with a tag for now");
}

template <>
std::shared_ptr<Layer> section2layer<PlainLayer>(dictionary *ini,
                                                 const std::string &sec_name,
                                                 const AppContext &ac) {

  const std::string &layer_type =
    iniparser_getstring(ini, (sec_name + ":Type").c_str(), UNKNOWN_STR);
  NNTR_THROW_IF(layer_type == UNKNOWN_STR, std::invalid_argument)
    << "section name is invalid, section name: " << sec_name;

  auto properties = section2properties(ini, sec_name);
  std::shared_ptr<ml::train::Layer> layer_ =
    ac.createObject<ml::train::Layer>(layer_type, properties);

  auto layer = std::static_pointer_cast<Layer>(layer_);
  layer->setName(sec_name);

  return layer;
}

template <>
std::shared_ptr<Layer> section2layer<BackboneLayer>(dictionary *ini,
                                                    const std::string &sec_name,
                                                    const AppContext &ac) {}

static std::shared_ptr<GraphRepresentation>
section2graph(dictionary *ini, const std::string &sec_name) {
  /// NYI!
  return nullptr;
}

bool graphSupported(const std::string &backbone_name) {
  /// NYI!
  return true;
}

std::vector<std::shared_ptr<Layer>>
getMergeableGraph(std::shared_ptr<const GraphRepresentation>) {
  return {};
};

} // namespace

void IniGraphInterpreter::serialize(
  std::shared_ptr<const GraphRepresentation> representation,
  const std::string &out) {}

std::shared_ptr<GraphRepresentation>
IniGraphInterpreter::deserialize(const std::string &in) {

  NNTR_THROW_IF(in.empty(), std::invalid_argument)
    << FUNC_TAG << "given in file is empty";

  NNTR_THROW_IF(!isFileExist(in), std::invalid_argument)
    << FUNC_TAG << "given ini file does not exist";

  dictionary *ini = iniparser_load(in.c_str());
  NNTR_THROW_IF(!ini, std::runtime_error) << "loading ini failed";

  auto freedict = [ini] { iniparser_freedict(ini); };

  /** Get number of sections in the file */
  int num_ini_sec = iniparser_getnsec(ini);
  NNTR_THROW_IF_CLEANUP(num_ini_sec < 0, std::invalid_argument, freedict)
    << FUNC_TAG << "invalid number of sections.";

  try {
    std::shared_ptr<GraphRepresentation> graph =
      std::make_shared<GraphRepresentation>();

    for (int idx = 0; idx < num_ini_sec; ++idx) {
      auto sec_name_ = iniparser_getsecname(ini, idx);
      NNTR_THROW_IF_CLEANUP(!sec_name_, std::runtime_error, )
        << "parsing a section name returned error, filename: " << in
        << "idx: " << idx;

      ml_logd("probing section_name: %s", sec_name_);
      std::string sec_name(sec_name_);

      if (istrequal(sec_name, MODEL_STR) || istrequal(sec_name, DATASET_STR) ||
          istrequal(sec_name, OPTIMIZER_STR)) {
        /// dedicated sections so skip
        continue;
      }
      /** Parse all the layers defined as sections in order */
      std::shared_ptr<Layer> layer;

      /**
       * If this section is a backbone, load backbone section from this
       * @note The order of backbones in the ini file defines the order on the
       * backbones in the model graph
       */
      const char *backbone_path =
        iniparser_getstring(ini, (sec_name + ":Backbone").c_str(), UNKNOWN_STR);

      const std::string &backbone = pathResolver(backbone_path);
      if (graphSupported(backbone)) {
        auto g = section2graph(ini, sec_name);

        /// @todo: deprecate this. We should extend graph from a graph
        auto tmp = getMergeableGraph(g);
        graph->extendGraph(tmp, sec_name);
        continue;
      }

      if (backbone_path == UNKNOWN_STR) {
        layer = section2layer<PlainLayer>(ini, sec_name, app_context);
      } else {
        /// @todo deprecate this as well with #1072
        layer = section2layer<BackboneLayer>(ini, sec_name, app_context);
      }

      graph->addLayer(layer);
    }
  } catch (...) {
    /** clean up and rethrow */
    freedict();
    throw;
  }

  freedict();
}

} // namespace nntrainer