// Copyright (c) 2026 AutoVision contributors. Apache-2.0.
// Header-only decode contract matching autovision.export.postprocess.
//
// Inputs are the raw ONNX outputs: logits (B, Q, C) and normalized cxcywh
// boxes (B, Q, 4). The header implements sigmoid + top-k + original-frame
// scaling, plus the class-offset NMS used by the PicoDet family. It is a
// reference for device code, not a stable ABI.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace autovision {

enum class Family { Detr, Picodet };

struct ImageSize {
  int width = 0;
  int height = 0;
};

struct Box {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
};

struct Detection {
  int label = 0;
  float score = 0.0f;
  Box box;
};

struct DecodeConfig {
  int num_classes = 0;
  int top_k = 300;
  float score_threshold = 0.0f;
  Family family = Family::Detr;
  float nms_iou = 0.6f;
  bool class_aware = true;
  int max_detections = 0;  // 0 = no cap
};

inline float sigmoid(float value) {
  const float clipped = std::min(80.0f, std::max(-80.0f, value));
  return 1.0f / (1.0f + std::exp(-clipped));
}

inline float iou(const Box& a, const Box& b) {
  const float x1 = std::max(a.x1, b.x1);
  const float y1 = std::max(a.y1, b.y1);
  const float x2 = std::min(a.x2, b.x2);
  const float y2 = std::min(a.y2, b.y2);
  const float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
  const float area_a =
      std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
  const float area_b =
      std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
  const float union_area = area_a + area_b - inter;
  return union_area > 0.0f ? inter / union_area : 0.0f;
}

// Matches torchvision.ops.nms with a class offset: labels are shifted by
// label * offset before IoU. Class-aware NMS passes offset > max(image),
// class-agnostic NMS passes offset=0. This mirrors the NumPy reference
// (`_class_offset_nms`), including stable descending-score order.
inline void class_offset_nms(std::vector<Detection>& detections,
                             float offset, float iou_threshold,
                             int max_detections) {
  if (detections.empty()) {
    return;
  }
  std::stable_sort(detections.begin(), detections.end(),
                   [](const Detection& a, const Detection& b) {
                     return a.score > b.score;
                   });
  std::vector<bool> removed(detections.size(), false);
  std::vector<Detection> kept;
  for (std::size_t i = 0; i < detections.size(); ++i) {
    if (removed[i]) {
      continue;
    }
    kept.push_back(detections[i]);
    if (max_detections > 0 &&
        static_cast<int>(kept.size()) >= max_detections) {
      break;
    }
    const float shift_i =
        static_cast<float>(detections[i].label) * offset;
    Box shifted_i = detections[i].box;
    shifted_i.x1 += shift_i;
    shifted_i.x2 += shift_i;
    for (std::size_t j = i + 1; j < detections.size(); ++j) {
      if (removed[j]) {
        continue;
      }
      const float shift_j =
          static_cast<float>(detections[j].label) * offset;
      Box shifted_j = detections[j].box;
      shifted_j.x1 += shift_j;
      shifted_j.x2 += shift_j;
      if (iou(shifted_i, shifted_j) > iou_threshold) {
        removed[j] = true;
      }
    }
  }
  detections.swap(kept);
}

// Decode one batch. `logits` and `boxes` are row-major; `sizes` are the
// original frame sizes used for the single rescale.
inline std::vector<std::vector<Detection>> decode(
    const float* logits, const float* boxes, int batch, int queries,
    const std::vector<ImageSize>& sizes, const DecodeConfig& config) {
  std::vector<std::vector<Detection>> output;
  if (config.num_classes <= 0 || batch <= 0 || queries <= 0) {
    return output;
  }
  output.resize(static_cast<std::size_t>(batch));
  for (int image = 0; image < batch; ++image) {
    struct Candidate {
      float score;
      int flat_index;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<std::size_t>(queries) * config.num_classes);
    const float* image_logits =
        logits + static_cast<std::size_t>(image) * queries * config.num_classes;
    for (int query = 0; query < queries; ++query) {
      for (int class_id = 0; class_id < config.num_classes; ++class_id) {
        const int flat = query * config.num_classes + class_id;
        candidates.push_back({sigmoid(image_logits[flat]), flat});
      }
    }
    const int keep = std::min(config.top_k,
                              static_cast<int>(candidates.size()));
    if (keep <= 0) {
      continue;
    }
    // Stable descending order, matching np.argsort(-scores, kind="stable")
    // in the NumPy reference: ties keep the lower flat (query, class) index.
    std::stable_sort(
        candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b) {
          return a.score > b.score;
        });
    const ImageSize size =
        image < static_cast<int>(sizes.size()) ? sizes[image] : ImageSize{};
    std::vector<Detection> detections;
    const float* image_boxes =
        boxes + static_cast<std::size_t>(image) * queries * 4;
    for (int index = 0; index < keep; ++index) {
      const Candidate& candidate = candidates[index];
      if (candidate.score <= config.score_threshold) {
        continue;
      }
      const int label = candidate.flat_index % config.num_classes;
      const int query = candidate.flat_index / config.num_classes;
      const float cx = image_boxes[query * 4 + 0];
      const float cy = image_boxes[query * 4 + 1];
      const float width = image_boxes[query * 4 + 2];
      const float height = image_boxes[query * 4 + 3];
      Detection detection;
      detection.label = label;
      detection.score = candidate.score;
      detection.box = {
          (cx - width / 2.0f) * static_cast<float>(size.width),
          (cy - height / 2.0f) * static_cast<float>(size.height),
          (cx + width / 2.0f) * static_cast<float>(size.width),
          (cy + height / 2.0f) * static_cast<float>(size.height),
      };
      detections.push_back(detection);
    }
    if (config.family == Family::Picodet && !detections.empty()) {
      const float offset =
          config.class_aware
              ? static_cast<float>(std::max(size.width, size.height)) + 1.0f
              : 0.0f;
      class_offset_nms(detections, offset, config.nms_iou,
                       config.max_detections);
    }
    output[static_cast<std::size_t>(image)] = std::move(detections);
  }
  return output;
}

}  // namespace autovision
