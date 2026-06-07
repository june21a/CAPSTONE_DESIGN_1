"""Real-time constant-velocity collision prediction for detected vehicles."""

from __future__ import annotations

import math

import numpy as np


VEHICLE_CLASS_IDS = {0, 4}


def _normalize_angle(angle):
  return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _rotation(yaw):
  return np.array([[math.cos(yaw), -math.sin(yaw)],
                   [math.sin(yaw), math.cos(yaw)]], dtype=np.float64)


def bounding_box_corners(position, yaw, extent):
  """Return corners of an oriented box; extent is half-length and half-width."""
  local_corners = np.array([[extent[0], extent[1]],
                            [extent[0], -extent[1]],
                            [-extent[0], -extent[1]],
                            [-extent[0], extent[1]]], dtype=np.float64)
  return local_corners @ _rotation(yaw).T + position


def bounding_boxes_overlap(first, second):
  """Check two oriented rectangles using the separating axis theorem."""
  for corners in (first, second):
    edges = np.roll(corners, -1, axis=0) - corners
    for edge in edges[:2]:
      axis = np.array([-edge[1], edge[0]], dtype=np.float64)
      norm = np.linalg.norm(axis)
      if norm <= 1e-9:
        continue
      axis /= norm
      first_projection = first @ axis
      second_projection = second @ axis
      if (first_projection.max() < second_projection.min()
          or second_projection.max() < first_projection.min()):
        return False
  return True


class RealTimeCollisionPredictor:
  """Track detections and test constant-velocity box overlap over a horizon."""

  def __init__(self,
               fps=20.0,
               update_interval=0.1,
               prediction_horizon=3.0,
               prediction_step=0.1,
               ego_extent=(2.4508416652679443, 1.0641621351242065),
               confidence_threshold=0.5,
               max_match_distance=4.0,
               velocity_smoothing=0.6):
    self.fps = float(fps)
    self.update_interval = float(update_interval)
    self.prediction_horizon = float(prediction_horizon)
    self.prediction_step = float(prediction_step)
    self.ego_extent = np.asarray(ego_extent, dtype=np.float64)
    self.confidence_threshold = float(confidence_threshold)
    self.max_match_distance = float(max_match_distance)
    self.velocity_smoothing = float(velocity_smoothing)
    self.update_every_steps = max(1, int(round(self.update_interval * self.fps)))
    self.previous_step = None
    self.previous_ego_pose = None
    self.previous_detections = []
    self.next_track_id = 1
    self.latest_result = self._empty_result()

  def _empty_result(self):
    return {
        'status': 'SAFE',
        'collision_risk': False,
        'time_to_collision_s': None,
        'prediction_horizon_s': self.prediction_horizon,
        'prediction_step_s': self.prediction_step,
        'vehicles': [],
    }

  def should_update(self, step):
    return step % self.update_every_steps == 0

  def _prediction_times(self):
    count = int(math.floor(self.prediction_horizon / self.prediction_step))
    times = [round(index * self.prediction_step, 10) for index in range(count + 1)]
    if not math.isclose(times[-1], self.prediction_horizon):
      times.append(self.prediction_horizon)
    return times

  @staticmethod
  def _status_from_ttc(ttc):
    if ttc is None:
      return 'SAFE'
    if ttc <= 1.0:
      return 'IMMINENT'
    if ttc <= 2.0:
      return 'COLLISION RISK'
    return 'CAUTION'

  def _transform_previous_to_current(self, current_ego_pose):
    if self.previous_ego_pose is None:
      return []
    prev_x, prev_y, prev_yaw = self.previous_ego_pose
    curr_x, curr_y, curr_yaw = current_ego_pose
    position_delta = np.array([curr_x - prev_x, curr_y - prev_y], dtype=np.float64)
    position_delta = _rotation(curr_yaw).T @ position_delta
    yaw_delta = _normalize_angle(curr_yaw - prev_yaw)
    frame_rotation = _rotation(yaw_delta).T

    transformed = []
    for detection in self.previous_detections:
      item = detection.copy()
      item['position'] = frame_rotation @ (detection['position'] - position_delta)
      item['yaw'] = _normalize_angle(detection['yaw'] - yaw_delta)
      item['relative_velocity'] = frame_rotation @ detection['relative_velocity']
      transformed.append(item)
    return transformed

  def _prepare_detections(self, bounding_boxes):
    detections = []
    if bounding_boxes is None:
      return detections
    for raw_box in bounding_boxes:
      box = np.asarray(raw_box, dtype=np.float64)
      if box.size < 9:
        continue
      class_id = int(round(box[7]))
      confidence = float(box[8])
      if class_id not in VEHICLE_CLASS_IDS or confidence < self.confidence_threshold:
        continue
      speed = max(0.0, float(box[5]))
      yaw = float(box[4])
      detections.append({
          'position': box[:2].copy(),
          'extent': box[2:4].copy(),
          'yaw': yaw,
          'detected_speed': speed,
          'absolute_velocity': np.array([speed * math.cos(yaw), speed * math.sin(yaw)], dtype=np.float64),
          'relative_velocity': None,
          'track_id': None,
          'class_id': class_id,
          'confidence': confidence,
      })
    return detections

  def _match(self, previous, current):
    candidates = []
    for prev_idx, prev in enumerate(previous):
      for curr_idx, curr in enumerate(current):
        if prev['class_id'] != curr['class_id']:
          continue
        distance = float(np.linalg.norm(curr['position'] - prev['position']))
        if distance <= self.max_match_distance:
          candidates.append((distance, prev_idx, curr_idx))

    matches = []
    used_previous = set()
    used_current = set()
    for _, prev_idx, curr_idx in sorted(candidates):
      if prev_idx in used_previous or curr_idx in used_current:
        continue
      used_previous.add(prev_idx)
      used_current.add(curr_idx)
      matches.append((prev_idx, curr_idx))
    return matches

  def update(self, step, ego_speed, ego_pose, bounding_boxes):
    current = self._prepare_detections(bounding_boxes)
    ego_velocity = np.array([float(ego_speed), 0.0], dtype=np.float64)
    previous = self._transform_previous_to_current(ego_pose)

    if self.previous_step is not None:
      dt = (step - self.previous_step) / self.fps
      if dt > 0.0:
        for prev_idx, curr_idx in self._match(previous, current):
          prev = previous[prev_idx]
          curr = current[curr_idx]
          relative_velocity = (curr['position'] - prev['position']) / dt
          relative_velocity = (self.velocity_smoothing * prev['relative_velocity']
                               + (1.0 - self.velocity_smoothing) * relative_velocity)
          curr['relative_velocity'] = relative_velocity
          curr['absolute_velocity'] = relative_velocity + ego_velocity
          curr['track_id'] = prev['track_id']

    for detection in current:
      if detection['track_id'] is None:
        detection['track_id'] = self.next_track_id
        self.next_track_id += 1
      if detection['relative_velocity'] is None:
        detection['relative_velocity'] = detection['absolute_velocity'] - ego_velocity

    ego_position = np.zeros(2, dtype=np.float64)
    vehicles = []
    collision_ttcs = []
    for detection in current:
      timeline = []
      first_collision = None
      for time_s in self._prediction_times():
        ego_future = ego_position + ego_velocity * time_s
        vehicle_future = detection['position'] + detection['absolute_velocity'] * time_s
        ego_box = bounding_box_corners(ego_future, 0.0, self.ego_extent)
        vehicle_box = bounding_box_corners(vehicle_future, detection['yaw'], detection['extent'])
        overlap = bounding_boxes_overlap(ego_box, vehicle_box)
        if overlap and first_collision is None:
          first_collision = time_s
        timeline.append({
            'time_s': time_s,
            'relative_position_m': {
                'x': float(vehicle_future[0] - ego_future[0]),
                'y': float(vehicle_future[1] - ego_future[1]),
            },
            'bounding_box_overlap': overlap,
        })

      if first_collision is not None:
        collision_ttcs.append(first_collision)
      vehicles.append({
          'track_id': detection['track_id'],
          'position_m': {'x': float(detection['position'][0]), 'y': float(detection['position'][1])},
          'yaw_rad': detection['yaw'],
          'speed_mps': float(np.linalg.norm(detection['absolute_velocity'])),
          'extent_m': {'x': float(detection['extent'][0]), 'y': float(detection['extent'][1])},
          'collision_risk': first_collision is not None,
          'time_to_collision_s': first_collision,
          'timeline': timeline,
      })

    minimum_ttc = min(collision_ttcs, default=None)
    self.latest_result = {
        'status': self._status_from_ttc(minimum_ttc),
        'collision_risk': minimum_ttc is not None,
        'time_to_collision_s': minimum_ttc,
        'prediction_horizon_s': self.prediction_horizon,
        'prediction_step_s': self.prediction_step,
        'vehicles': vehicles,
    }
    self.previous_step = step
    self.previous_ego_pose = tuple(float(value) for value in ego_pose)
    self.previous_detections = current
    return self.latest_result
