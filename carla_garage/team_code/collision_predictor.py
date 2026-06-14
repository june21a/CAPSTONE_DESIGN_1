"""Real-time waypoint/CTRV collision prediction for detected vehicles."""

from __future__ import annotations

from collections import deque
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
  """Predict ego on waypoints and surrounding vehicles with CTRV."""

  def __init__(self,
               fps=20.0,
               update_interval=0.1,
               prediction_horizon=3.0,
               prediction_step=0.1,
               waypoint_interval=0.25,
               ego_extent=(2.4508416652679443, 1.0641621351242065),
               confidence_threshold=0.5,
               max_match_distance=4.0,
               velocity_smoothing=0.6,
               yaw_rate_smoothing=0.6,
               yaw_rate_deadband_deg=3.0,
               max_yaw_rate_deg=45.0,
               warning_confirmation_count=2,
               safe_release_seconds=0.5,
               longitudinal_margin=0.5,
               lateral_margin=0.2):
    self.fps = float(fps)
    self.update_interval = float(update_interval)
    self.prediction_horizon = float(prediction_horizon)
    self.prediction_step = float(prediction_step)
    self.waypoint_interval = float(waypoint_interval)
    self.ego_extent = np.asarray(ego_extent, dtype=np.float64)
    self.safety_margin = np.array([longitudinal_margin, lateral_margin], dtype=np.float64)
    self.confidence_threshold = float(confidence_threshold)
    self.max_match_distance = float(max_match_distance)
    self.velocity_smoothing = float(velocity_smoothing)
    self.yaw_rate_smoothing = float(yaw_rate_smoothing)
    self.yaw_rate_deadband = math.radians(yaw_rate_deadband_deg)
    self.max_yaw_rate = math.radians(max_yaw_rate_deg)
    self.warning_confirmation_count = max(1, int(warning_confirmation_count))
    self.safe_release_updates = max(1, int(round(safe_release_seconds / self.update_interval)))
    self.update_every_steps = max(1, int(round(self.update_interval * self.fps)))

    self.previous_step = None
    self.previous_ego_pose = None
    self.previous_detections = []
    self.next_track_id = 1

    self.pending_warning_key = None
    self.pending_warning_count = 0
    self.pending_ttcs = []
    self.active_warning = None
    self.active_ttc_history = deque(maxlen=5)
    self.safe_update_count = 0
    self.latest_result = self._empty_result()

  def _empty_result(self):
    return {
        'status': 'SAFE',
        'collision_risk': False,
        'time_to_collision_s': None,
        'direction': None,
        'track_id': None,
        'prediction_horizon_s': self.prediction_horizon,
        'prediction_step_s': self.prediction_step,
        'ego_prediction_model': 'pred_wp',
        'surrounding_prediction_model': 'CTRV',
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

  @staticmethod
  def _direction_from_position(position):
    x_pos, y_pos = float(position[0]), float(position[1])
    if x_pos >= abs(y_pos):
      return 'FRONT'
    if -x_pos >= abs(y_pos):
      return 'REAR'
    return 'RIGHT' if y_pos > 0.0 else 'LEFT'

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
      item['absolute_velocity'] = frame_rotation @ detection['absolute_velocity']
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
          'yaw_rate': 0.0,
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

  def _ego_trajectory(self, ego_speed, predicted_waypoints):
    times = self._prediction_times()
    waypoints = None
    if predicted_waypoints is not None:
      waypoints = np.asarray(predicted_waypoints, dtype=np.float64).reshape(-1, 2)
      waypoints = waypoints[np.all(np.isfinite(waypoints), axis=1)]

    if waypoints is None or len(waypoints) == 0:
      positions = [np.array([ego_speed * time_s, 0.0], dtype=np.float64) for time_s in times]
    else:
      path_points = np.vstack((np.zeros((1, 2), dtype=np.float64), waypoints))
      path_times = np.arange(len(path_points), dtype=np.float64) * self.waypoint_interval
      positions = []
      for time_s in times:
        if time_s <= path_times[-1]:
          positions.append(np.array([
              np.interp(time_s, path_times, path_points[:, 0]),
              np.interp(time_s, path_times, path_points[:, 1]),
          ]))
        elif len(path_points) > 1:
          last_velocity = (path_points[-1] - path_points[-2]) / self.waypoint_interval
          positions.append(path_points[-1] + last_velocity * (time_s - path_times[-1]))
        else:
          positions.append(path_points[-1].copy())

    yaws = []
    previous_yaw = 0.0
    for index, position in enumerate(positions):
      if index + 1 < len(positions):
        direction = positions[index + 1] - position
      elif index > 0:
        direction = position - positions[index - 1]
      else:
        direction = np.array([1.0, 0.0])
      if np.linalg.norm(direction) > 1e-4:
        previous_yaw = math.atan2(direction[1], direction[0])
      yaws.append(previous_yaw)
    return times, positions, yaws

  def _vehicle_state_at(self, detection, time_s):
    yaw_rate = detection['yaw_rate']
    speed = float(np.linalg.norm(detection['absolute_velocity']))
    yaw = detection['yaw']
    if abs(yaw_rate) < self.yaw_rate_deadband or speed < 1e-3:
      position = detection['position'] + detection['absolute_velocity'] * time_s
      return position, yaw

    future_yaw = _normalize_angle(yaw + yaw_rate * time_s)
    radius_factor = speed / yaw_rate
    position = detection['position'] + np.array([
        radius_factor * (math.sin(future_yaw) - math.sin(yaw)),
        radius_factor * (-math.cos(future_yaw) + math.cos(yaw)),
    ])
    return position, future_yaw

  def _display_candidate(self, vehicles):
    candidates = []
    for vehicle in vehicles:
      ttc = vehicle['time_to_collision_s']
      if ttc is None:
        continue
      direction = vehicle['direction']
      display_eligible = direction != 'REAR' or ttc <= 1.0
      vehicle['display_eligible'] = display_eligible
      if display_eligible:
        candidates.append(vehicle)
    return min(candidates, key=lambda item: item['time_to_collision_s'], default=None)

  def _stabilize_warning(self, candidate):
    candidate_key = None if candidate is None else (candidate['track_id'], candidate['direction'])
    if (candidate_key is not None and self.active_warning is not None
        and candidate_key == self.active_warning['key']):
      self.safe_update_count = 0
      self.active_ttc_history.append(candidate['time_to_collision_s'])
      median_ttc = float(np.median(self.active_ttc_history))
      self.active_warning.update({
          'status': self._status_from_ttc(median_ttc),
          'time_to_collision_s': median_ttc,
          'direction': candidate['direction'],
          'track_id': candidate['track_id'],
      })
      return self.active_warning

    if candidate_key is not None:
      if candidate_key == self.pending_warning_key:
        self.pending_warning_count += 1
        self.pending_ttcs.append(candidate['time_to_collision_s'])
      else:
        self.pending_warning_key = candidate_key
        self.pending_warning_count = 1
        self.pending_ttcs = [candidate['time_to_collision_s']]

      if self.pending_warning_count >= self.warning_confirmation_count:
        median_ttc = float(np.median(self.pending_ttcs[-5:]))
        self.active_ttc_history.clear()
        self.active_ttc_history.extend(self.pending_ttcs[-5:])
        self.active_warning = {
            'key': candidate_key,
            'status': self._status_from_ttc(median_ttc),
            'time_to_collision_s': median_ttc,
            'direction': candidate['direction'],
            'track_id': candidate['track_id'],
        }
        self.safe_update_count = 0
        return self.active_warning
    else:
      self.pending_warning_key = None
      self.pending_warning_count = 0
      self.pending_ttcs = []

    if self.active_warning is not None:
      self.safe_update_count += 1
      if self.safe_update_count < self.safe_release_updates:
        return self.active_warning

    self.active_warning = None
    self.active_ttc_history.clear()
    self.safe_update_count = 0
    return None

  def update(self, step, ego_speed, ego_pose, bounding_boxes, predicted_waypoints=None):
    current = self._prepare_detections(bounding_boxes)
    ego_velocity = np.array([float(ego_speed), 0.0], dtype=np.float64)
    previous = self._transform_previous_to_current(ego_pose)

    if self.previous_step is not None:
      dt = (step - self.previous_step) / self.fps
      if dt > 0.0:
        for prev_idx, curr_idx in self._match(previous, current):
          prev = previous[prev_idx]
          curr = current[curr_idx]
          measured_velocity = (curr['position'] - prev['position']) / dt
          curr['absolute_velocity'] = (self.velocity_smoothing * prev['absolute_velocity']
                                       + (1.0 - self.velocity_smoothing) * measured_velocity)
          curr['relative_velocity'] = curr['absolute_velocity'] - ego_velocity
          measured_yaw_rate = _normalize_angle(curr['yaw'] - prev['yaw']) / dt
          measured_yaw_rate = np.clip(measured_yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)
          curr['yaw_rate'] = (self.yaw_rate_smoothing * prev['yaw_rate']
                              + (1.0 - self.yaw_rate_smoothing) * measured_yaw_rate)
          curr['track_id'] = prev['track_id']

    for detection in current:
      if detection['track_id'] is None:
        detection['track_id'] = self.next_track_id
        self.next_track_id += 1
      if detection['relative_velocity'] is None:
        detection['relative_velocity'] = detection['absolute_velocity'] - ego_velocity

    times, ego_positions, ego_yaws = self._ego_trajectory(float(ego_speed), predicted_waypoints)
    ego_extent = self.ego_extent + self.safety_margin
    vehicles = []
    for detection in current:
      timeline = []
      first_collision = None
      collision_relative_position = None
      vehicle_extent = detection['extent'] + self.safety_margin
      for time_s, ego_position, ego_yaw in zip(times, ego_positions, ego_yaws):
        vehicle_position, vehicle_yaw = self._vehicle_state_at(detection, time_s)
        ego_box = bounding_box_corners(ego_position, ego_yaw, ego_extent)
        vehicle_box = bounding_box_corners(vehicle_position, vehicle_yaw, vehicle_extent)
        overlap = bounding_boxes_overlap(ego_box, vehicle_box)
        relative_position = vehicle_position - ego_position
        if overlap and first_collision is None:
          first_collision = time_s
          collision_relative_position = relative_position.copy()
        timeline.append({
            'time_s': time_s,
            'ego_position_m': {'x': float(ego_position[0]), 'y': float(ego_position[1])},
            'vehicle_position_m': {'x': float(vehicle_position[0]), 'y': float(vehicle_position[1])},
            'vehicle_yaw_rad': float(vehicle_yaw),
            'relative_position_m': {'x': float(relative_position[0]), 'y': float(relative_position[1])},
            'bounding_box_overlap': overlap,
        })

      direction_position = detection['position'] if collision_relative_position is None else collision_relative_position
      vehicles.append({
          'track_id': detection['track_id'],
          'position_m': {'x': float(detection['position'][0]), 'y': float(detection['position'][1])},
          'yaw_rad': float(detection['yaw']),
          'yaw_rate_deg_s': float(math.degrees(detection['yaw_rate'])),
          'speed_mps': float(np.linalg.norm(detection['absolute_velocity'])),
          'extent_m': {'x': float(detection['extent'][0]), 'y': float(detection['extent'][1])},
          'direction': self._direction_from_position(direction_position),
          'collision_risk': first_collision is not None,
          'time_to_collision_s': first_collision,
          'display_eligible': False,
          'timeline': timeline,
      })

    raw_candidate = self._display_candidate(vehicles)
    stable_warning = self._stabilize_warning(raw_candidate)
    if stable_warning is None:
      status = 'SAFE'
      minimum_ttc = None
      direction = None
      track_id = None
    else:
      status = stable_warning['status']
      minimum_ttc = stable_warning['time_to_collision_s']
      direction = stable_warning['direction']
      track_id = stable_warning['track_id']

    self.latest_result = {
        'status': status,
        'collision_risk': stable_warning is not None,
        'time_to_collision_s': minimum_ttc,
        'direction': direction,
        'track_id': track_id,
        'raw_time_to_collision_s': None if raw_candidate is None else raw_candidate['time_to_collision_s'],
        'prediction_horizon_s': self.prediction_horizon,
        'prediction_step_s': self.prediction_step,
        'ego_prediction_model': 'pred_wp',
        'surrounding_prediction_model': 'CTRV',
        'vehicles': vehicles,
    }
    self.previous_step = step
    self.previous_ego_pose = tuple(float(value) for value in ego_pose)
    self.previous_detections = current
    return self.latest_result
