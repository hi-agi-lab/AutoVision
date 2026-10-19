// Copyright (c) 2026 AutoVision contributors. Apache-2.0.
// Host-side postprocess test: no ONNX Runtime / OpenCV required.
//
//   g++ -std=c++17 -O2 -Iexamples/cpp/src
//       examples/cpp/src/postprocess_test.cpp -o /tmp/postprocess_test
//   /tmp/postprocess_test
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "postprocess.hpp"

namespace {

bool close(float a, float b, float tolerance = 1e-4f) {
  return std::fabs(a - b) <= tolerance;
}

void test_detr_decode() {
  // 1 image, 3 queries, 2 classes. q0 -> class 0, q1 -> class 1.
  const float logits[] = {10.0f, -10.0f, -10.0f, 9.0f, -10.0f, -10.0f};
  const float boxes[] = {
      0.50f, 0.50f, 0.20f, 0.20f,
      0.25f, 0.25f, 0.10f, 0.10f,
      0.70f, 0.70f, 0.20f, 0.20f,
  };
  autovision::DecodeConfig config;
  config.num_classes = 2;
  config.top_k = 3;
  config.score_threshold = 0.5f;
  config.family = autovision::Family::Detr;

  auto output = autovision::decode(
      logits, boxes, 1, 3, {{100, 80}}, config);
  assert(output.size() == 1u);
  assert(output[0].size() == 2u);
  assert(output[0][0].label == 0);
  assert(output[0][1].label == 1);
  assert(close(output[0][0].box.x1, 40.0f));
  assert(close(output[0][0].box.y1, 32.0f));
  assert(close(output[0][0].box.x2, 60.0f));
  assert(close(output[0][0].box.y2, 48.0f));
  assert(close(output[0][1].box.x1, 20.0f));
  assert(close(output[0][1].box.y1, 16.0f));
  assert(close(output[0][1].box.x2, 30.0f));
  assert(close(output[0][1].box.y2, 24.0f));
}

void test_picodet_nms() {
  // Two identical high-score class-0 boxes: class-offset NMS keeps one.
  const float logits[] = {10.0f, -10.0f, 10.0f, -10.0f};
  const float boxes[] = {
      0.50f, 0.50f, 0.20f, 0.20f,
      0.50f, 0.50f, 0.20f, 0.20f,
  };
  autovision::DecodeConfig config;
  config.num_classes = 2;
  config.top_k = 2;
  config.score_threshold = 0.5f;
  config.family = autovision::Family::Picodet;
  config.nms_iou = 0.5f;
  config.class_aware = true;

  auto output = autovision::decode(
      logits, boxes, 1, 2, {{100, 80}}, config);
  assert(output.size() == 1u);
  assert(output[0].size() == 1u);
  assert(output[0][0].label == 0);
}

void test_picodet_class_aware_keeps_cross_class() {
  // Identical boxes at different classes must survive class-aware NMS:
  // the class offset shifts one label beyond the image extent.
  const float logits[] = {10.0f, -10.0f, -10.0f, 10.0f};
  const float boxes[] = {
      0.50f, 0.50f, 0.20f, 0.20f,
      0.50f, 0.50f, 0.20f, 0.20f,
  };
  autovision::DecodeConfig config;
  config.num_classes = 2;
  config.top_k = 2;
  config.score_threshold = 0.5f;
  config.family = autovision::Family::Picodet;
  config.nms_iou = 0.5f;
  config.class_aware = true;

  auto output = autovision::decode(
      logits, boxes, 1, 2, {{100, 80}}, config);
  assert(output.size() == 1u);
  assert(output[0].size() == 2u);
  assert(output[0][0].label == 0);
  assert(output[0][1].label == 1);
}

void test_picodet_class_agnostic_suppresses_cross_class() {
  // With class_aware=false the offset is zero, so identical boxes of
  // different classes collapse to one detection.
  const float logits[] = {10.0f, -10.0f, -10.0f, 10.0f};
  const float boxes[] = {
      0.50f, 0.50f, 0.20f, 0.20f,
      0.50f, 0.50f, 0.20f, 0.20f,
  };
  autovision::DecodeConfig config;
  config.num_classes = 2;
  config.top_k = 2;
  config.score_threshold = 0.5f;
  config.family = autovision::Family::Picodet;
  config.nms_iou = 0.5f;
  config.class_aware = false;

  auto output = autovision::decode(
      logits, boxes, 1, 2, {{100, 80}}, config);
  assert(output.size() == 1u);
  assert(output[0].size() == 1u);
  assert(output[0][0].label == 0);
}

void test_picodet_max_detections_cap() {
  const float logits[] = {10.0f, 9.0f, 8.0f};
  const float boxes[] = {
      0.20f, 0.20f, 0.10f, 0.10f,
      0.50f, 0.50f, 0.10f, 0.10f,
      0.80f, 0.80f, 0.10f, 0.10f,
  };
  autovision::DecodeConfig config;
  config.num_classes = 1;
  config.top_k = 3;
  config.score_threshold = 0.0f;
  config.family = autovision::Family::Picodet;
  config.nms_iou = 0.5f;
  config.max_detections = 2;

  auto output = autovision::decode(
      logits, boxes, 1, 3, {{100, 100}}, config);
  assert(output.size() == 1u);
  assert(output[0].size() == 2u);
}

void test_top_k_truncates() {
  const float logits[] = {10.0f, 9.0f, 8.0f, 7.0f};
  const float boxes[] = {
      0.50f, 0.50f, 0.20f, 0.20f,
      0.25f, 0.25f, 0.10f, 0.10f,
      0.75f, 0.75f, 0.10f, 0.10f,
      0.10f, 0.10f, 0.10f, 0.10f,
  };
  autovision::DecodeConfig config;
  config.num_classes = 1;
  config.top_k = 2;
  config.score_threshold = 0.0f;

  auto output = autovision::decode(
      logits, boxes, 1, 4, {{100, 100}}, config);
  assert(output.size() == 1u);
  assert(output[0].size() == 2u);
  assert(close(output[0][0].box.x1, 40.0f));
  assert(close(output[0][1].box.x1, 20.0f));
}

void test_stable_ties_keep_flat_order() {
  // Equal scores: stable descending sort keeps (query 0, class 0) before
  // (query 0, class 1), matching NumPy's stable argsort.
  const float logits[] = {0.0f, 0.0f};
  const float boxes[] = {0.50f, 0.50f, 0.20f, 0.20f};
  autovision::DecodeConfig config;
  config.num_classes = 2;
  config.top_k = 2;
  config.score_threshold = 0.0f;

  auto output = autovision::decode(
      logits, boxes, 1, 1, {{100, 80}}, config);
  assert(output.size() == 1u);
  assert(output[0].size() == 2u);
  assert(output[0][0].label == 0);
  assert(output[0][1].label == 1);
}

void test_batch_uses_each_original_size() {
  const float logits[] = {10.0f, 10.0f};
  const float boxes[] = {
      0.50f, 0.50f, 0.20f, 0.20f,
      0.50f, 0.50f, 0.20f, 0.20f,
  };
  autovision::DecodeConfig config;
  config.num_classes = 1;
  config.top_k = 1;
  config.score_threshold = 0.5f;

  auto output = autovision::decode(
      logits, boxes, 2, 1, {{100, 80}, {200, 40}}, config);
  assert(output.size() == 2u);
  assert(close(output[0][0].box.x1, 40.0f));
  assert(close(output[0][0].box.x2, 60.0f));
  assert(close(output[0][0].box.y1, 32.0f));
  assert(close(output[0][0].box.y2, 48.0f));
  assert(close(output[1][0].box.x1, 80.0f));
  assert(close(output[1][0].box.x2, 120.0f));
  assert(close(output[1][0].box.y1, 16.0f));
  assert(close(output[1][0].box.y2, 24.0f));
}

void test_threshold_is_strict() {
  // sigmoid(0) == 0.5; the reference keeps only scores strictly above the
  // threshold, so a 0.5 cutoff yields no detections.
  const float logits[] = {0.0f};
  const float boxes[] = {0.50f, 0.50f, 0.20f, 0.20f};
  autovision::DecodeConfig config;
  config.num_classes = 1;
  config.top_k = 1;
  config.score_threshold = 0.5f;

  auto output = autovision::decode(
      logits, boxes, 1, 1, {{100, 80}}, config);
  assert(output.size() == 1u);
  assert(output[0].empty());
}

void test_empty_threshold() {
  const float logits[] = {-10.0f, -10.0f};
  const float boxes[] = {0.5f, 0.5f, 0.2f, 0.2f};
  autovision::DecodeConfig config;
  config.num_classes = 2;
  config.top_k = 2;
  config.score_threshold = 0.5f;
  auto output = autovision::decode(
      logits, boxes, 1, 1, {{100, 80}}, config);
  assert(output.size() == 1u);
  assert(output[0].empty());
}

}  // namespace

int main() {
  test_detr_decode();
  test_picodet_nms();
  test_picodet_class_aware_keeps_cross_class();
  test_picodet_class_agnostic_suppresses_cross_class();
  test_picodet_max_detections_cap();
  test_top_k_truncates();
  test_stable_ties_keep_flat_order();
  test_batch_uses_each_original_size();
  test_empty_threshold();
  test_threshold_is_strict();
  std::printf("postprocess ok\n");
  return 0;
}
