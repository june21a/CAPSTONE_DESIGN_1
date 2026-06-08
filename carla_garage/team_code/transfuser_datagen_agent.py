"""
TransFuser inference agent with local datagen/debug recording enabled.

This keeps the driving stack from SensorAgent and writes the DataAgent training
dataset layout with GT labels collected from privileged simulator state.
"""

import os
import gzip
import math
import random
import re
import shutil

import cv2
import carla
import laspy
import numpy as np
import ujson

import transfuser_utils as t_u
from birds_eye_view.chauffeurnet import ObsManager
from birds_eye_view.run_stop_sign import RunStopSign
from nav_planner import extrapolate_waypoint_route
from sensor_agent import SensorAgent

DEFAULT_EGO_EXTENT = np.array([2.45, 1.06], dtype=np.float32)
DEFAULT_SIM_FPS = 20.0
DEFAULT_MAX_ACCELERATION = 1.89
DEFAULT_MAX_DECELERATION = 4.82
DEFAULT_COLLISION_REGION_RADIUS = float(np.linalg.norm(DEFAULT_EGO_EXTENT))
STRAIGHT_WAYPOINT_MAX_LATERAL_SPREAD = 0.5
STRAIGHT_WAYPOINT_MAX_HEADING_CHANGE_DEG = 5.0
COLLISION_POINT_SAME_LANE_LATERAL_MARGIN = float(DEFAULT_EGO_EXTENT[1]) + 0.5
MAX_CHECKED_FRAMES_BEFORE_EVENT = 10
COLLISION_EVENT_MIN_DISTANCE = 8.0
TRAJECTORY_VIS_MAX_SAFE_FRAMES = 5
TRAJECTORY_VIS_PIXELS_PER_METER = 5.0
TRAJECTORY_VIS_MIN_X = -32.0
TRAJECTORY_VIS_MAX_X = 32.0
TRAJECTORY_VIS_MIN_Y = -32.0
TRAJECTORY_VIS_MAX_Y = 32.0
VISUALIZED_CLASSES = {'ego_car', 'car', 'walker', 'static', 'traffic_light', 'stop_sign'}
TRAJECTORY_CLASS_COLORS = {
    'ego_car': (0, 190, 255),
    'car': (245, 145, 35),
    'walker': (40, 190, 95),
    'static': (125, 125, 125),
    'traffic_light': (215, 45, 45),
    'stop_sign': (190, 65, 180),
}
DEFAULT_TRAJECTORY_COLOR = (80, 80, 80)


def get_entry_point():
  return 'TransfuserDataAgent'


class TransfuserDataAgent(SensorAgent):
  """
  Runs the trained TransFuser model while collecting sensor/model/GT outputs.
  """

  def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
    if os.environ.get('DATAGEN', '0') == '1':
      os.environ.setdefault('COLLECT_SENSOR_DATA', '0')
      os.environ.setdefault('ATTENTION_VIS', '0')
      os.environ.setdefault('VISION_TASK_VIS', '0')

    super().setup(path_to_conf_file, route_index=route_index, traffic_manager=traffic_manager)
    self.datagen = os.environ.get('DATAGEN', '0') == '1'
    self.augmentation_translation = 0.0
    self.augmentation_rotation = 0.0
    self.gt_bev_manager = None
    self.gt_bev_manager_augmented = None
    self.gt_stop_sign_criteria = None
    self.augmented_vehicle_dummy = None
    self.generate_plan_safety_labels = os.environ.get('GENERATE_PLAN_SAFETY_LABELS', '1') == '1'
    self.plan_safety_label_frames = {}
    self.plan_safety_case_label = os.environ.get('SIM_CASE_LABEL', 'transfuser')
    self.plan_safety_unsafe_label = int(os.environ.get('PLAN_SAFETY_UNSAFE_LABEL', 0))
    self.plan_safety_safe_label = int(os.environ.get('PLAN_SAFETY_SAFE_LABEL', 1))
    self.delete_route_folder_without_collision = os.environ.get('DELETE_ROUTE_FOLDER_WITHOUT_COLLISION', '0') == '1'
    if self.save_path is not None and self.datagen:
      for folder in (
          'lidar',
          'rgb',
          'semantics',
          'semantics_augmented',
          'depth',
          'depth_augmented',
          'rgb_augmented',
          'bev_semantics',
          'bev_semantics_augmented',
          'boxes',
          'measurements',
      ):
        (self.save_path / folder).mkdir(parents=True, exist_ok=True)

  def _init(self):
    super()._init()
    if self.save_path is None or not self.datagen or self._ego_vehicle is None:
      return

    obs_config = {
        'width_in_pixels': self.config.lidar_resolution_width,
        'pixels_ev_to_bottom': self.config.lidar_resolution_height / 2.0,
        'pixels_per_meter': self.config.pixels_per_meter_collection,
        'history_idx': [-1],
        'scale_bbox': True,
        'scale_mask_col': 1.0,
        'map_folder': 'maps_2ppm_cv',
    }
    self.gt_stop_sign_criteria = RunStopSign(self._ego_vehicle.get_world())
    self.gt_bev_manager = ObsManager(obs_config, self.config)
    self.gt_bev_manager.attach_ego_vehicle(self._ego_vehicle, criteria_stop=self.gt_stop_sign_criteria)

    self.gt_bev_manager_augmented = ObsManager(obs_config, self.config)
    bb_copy = carla.BoundingBox(self._ego_vehicle.bounding_box.location, self._ego_vehicle.bounding_box.extent)
    transform_copy = carla.Transform(self._ego_vehicle.get_transform().location, self._ego_vehicle.get_transform().rotation)
    self.augmented_vehicle_dummy = t_u.CarlaActorDummy(self._ego_vehicle.get_world(), bb_copy, transform_copy,
                                                       self._ego_vehicle.id)
    self.gt_bev_manager_augmented.attach_ego_vehicle(self.augmented_vehicle_dummy,
                                                     criteria_stop=self.gt_stop_sign_criteria)

  def sensors(self):
    sensors = super().sensors()
    if self.save_path is None or not self.datagen:
      return sensors

    if self.config.augment:
      self.augmentation_translation = np.random.uniform(low=self.config.camera_translation_augmentation_min,
                                                        high=self.config.camera_translation_augmentation_max)
      self.augmentation_rotation = np.random.uniform(low=self.config.camera_rotation_augmentation_min,
                                                     high=self.config.camera_rotation_augmentation_max)

    sensors += [{
        'type': 'sensor.camera.rgb',
        'x': self.config.camera_pos[0],
        'y': self.config.camera_pos[1],
        'z': self.config.camera_pos[2],
        'roll': self.config.camera_rot_0[0],
        'pitch': self.config.camera_rot_0[1],
        'yaw': self.config.camera_rot_0[2],
        'width': self.config.camera_width,
        'height': self.config.camera_height,
        'fov': self.config.camera_fov,
        'id': 'rgb'
    }, {
        'type': 'sensor.camera.rgb',
        'x': self.config.camera_pos[0],
        'y': self.config.camera_pos[1] + self.augmentation_translation,
        'z': self.config.camera_pos[2],
        'roll': self.config.camera_rot_0[0],
        'pitch': self.config.camera_rot_0[1],
        'yaw': self.config.camera_rot_0[2] + self.augmentation_rotation,
        'width': self.config.camera_width,
        'height': self.config.camera_height,
        'fov': self.config.camera_fov,
        'id': 'rgb_augmented'
    }, {
        'type': 'sensor.camera.semantic_segmentation',
        'x': self.config.camera_pos[0],
        'y': self.config.camera_pos[1],
        'z': self.config.camera_pos[2],
        'roll': self.config.camera_rot_0[0],
        'pitch': self.config.camera_rot_0[1],
        'yaw': self.config.camera_rot_0[2],
        'width': self.config.camera_width,
        'height': self.config.camera_height,
        'fov': self.config.camera_fov,
        'id': 'semantics'
    }, {
        'type': 'sensor.camera.semantic_segmentation',
        'x': self.config.camera_pos[0],
        'y': self.config.camera_pos[1] + self.augmentation_translation,
        'z': self.config.camera_pos[2],
        'roll': self.config.camera_rot_0[0],
        'pitch': self.config.camera_rot_0[1],
        'yaw': self.config.camera_rot_0[2] + self.augmentation_rotation,
        'width': self.config.camera_width,
        'height': self.config.camera_height,
        'fov': self.config.camera_fov,
        'id': 'semantics_augmented'
    }, {
        'type': 'sensor.camera.depth',
        'x': self.config.camera_pos[0],
        'y': self.config.camera_pos[1],
        'z': self.config.camera_pos[2],
        'roll': self.config.camera_rot_0[0],
        'pitch': self.config.camera_rot_0[1],
        'yaw': self.config.camera_rot_0[2],
        'width': self.config.camera_width,
        'height': self.config.camera_height,
        'fov': self.config.camera_fov,
        'id': 'depth'
    }, {
        'type': 'sensor.camera.depth',
        'x': self.config.camera_pos[0],
        'y': self.config.camera_pos[1] + self.augmentation_translation,
        'z': self.config.camera_pos[2],
        'roll': self.config.camera_rot_0[0],
        'pitch': self.config.camera_rot_0[1],
        'yaw': self.config.camera_rot_0[2] + self.augmentation_rotation,
        'width': self.config.camera_width,
        'height': self.config.camera_height,
        'fov': self.config.camera_fov,
        'id': 'depth_augmented'
    }]
    return sensors

  def _on_plan_safety_candidate(self, tick_data, waypoints, target_speed, control, pred_waypoints=None):
    if self.save_path is None or not self.datagen:
      return
    if self.step % self.config.data_save_freq != 0:
      return

    frame = self.step // self.config.data_save_freq
    plan_safety_route = self._waypoints_to_list(waypoints)
    pred_waypoints_list = self._waypoints_to_list(pred_waypoints)
    route, route_original, next_command = self._route_measurement_from_planner(tick_data, plan_safety_route)
    target_point = self._tensor_to_list(tick_data.get('target_point'), default=[0.0, 0.0])
    target_point_next = self._tensor_to_list(tick_data.get('target_point_next'), default=target_point)
    command = self._command_from_tick(tick_data)
    speed = float(tick_data['speed'].detach().cpu().item())
    target_speed = speed if target_speed is None else float(target_speed)
    ego_matrix = None
    ego_location = [0.0, 0.0, 0.0]
    theta = 0.0
    if self._ego_vehicle is not None:
      ego_transform = self._ego_vehicle.get_transform()
      ego_matrix = ego_transform.get_matrix()
      ego_location = [ego_transform.location.x, ego_transform.location.y, ego_transform.location.z]
      theta = np.deg2rad(ego_transform.rotation.yaw)

    measurement = {
        'frame': frame,
        'step': int(self.step),
        'source': 'carla_transfuser',
        'route': route,
        'route_original': route_original,
        'pred_waypoints': pred_waypoints_list,
        'changed_route': False,
        'pos_global': ego_location,
        'theta': float(theta),
        'speed': speed,
        'target_speed': target_speed,
        'expert_target_speed': target_speed,
        'sim_disturbed_target_speed': target_speed,
        'speed_limit': 0.0,
        'target_point': target_point,
        'target_point_next': target_point_next,
        'command': command,
        'next_command': next_command,
        'aim_wp': target_point,
        'speed_reduced_by_obj_type': None,
        'speed_reduced_by_obj_id': None,
        'speed_reduced_by_obj_distance': None,
        'steer': float(getattr(control, 'steer', 0.0)),
        'throttle': float(getattr(control, 'throttle', 0.0)),
        'brake': bool(getattr(control, 'brake', 0.0) > 0.05),
        'control_brake': bool(getattr(control, 'brake', 0.0) > 0.05),
        'junction': self._ego_vehicle_in_junction(),
        'vehicle_hazard': False,
        'vehicle_affecting_id': None,
        'light_hazard': False,
        'walker_hazard': False,
        'walker_affecting_id': None,
        'stop_sign_hazard': False,
        'stop_sign_close': False,
        'walker_close': False,
        'walker_close_id': None,
        'angle': float(math.atan2(target_point[1], target_point[0])) if len(target_point) >= 2 else 0.0,
        'augmentation_translation': float(self.augmentation_translation),
        'augmentation_rotation': float(self.augmentation_rotation),
        'ego_matrix': ego_matrix,
        'sim_case_label': os.environ.get('SIM_CASE_LABEL', self.plan_safety_case_label),
        'sim_disturbed': os.environ.get('SIM_FAILURE_DISTURB', '0') == '1',
        'sim_disturbance_reason': os.environ.get('SIM_FAILURE_REASON', None),
    }

    with gzip.open(self.save_path / 'measurements' / f'{frame:04}.json.gz', 'wt', encoding='utf-8') as outfile:
      ujson.dump(measurement, outfile, indent=4)
    self._save_training_sensors(frame, tick_data)
    if self.generate_plan_safety_labels:
      label_measurement = dict(measurement)
      label_measurement['route'] = plan_safety_route
      self._add_plan_safety_label_candidate(frame, label_measurement)

  @staticmethod
  def _tensor_to_list(value, default=None):
    if value is None:
      return list(default) if default is not None else []
    if hasattr(value, 'detach'):
      value = value.detach().cpu().numpy()
    value = np.asarray(value).squeeze()
    if value.ndim == 0:
      return [float(value)]
    return value.astype(float).tolist()

  @staticmethod
  def _command_from_one_hot(command_one_hot):
    command = np.asarray(command_one_hot).squeeze()
    if command.size == 0:
      return 4
    return int(np.argmax(command)) + 1

  def _command_from_tick(self, tick_data):
    command = tick_data.get('command')
    if command is not None:
      if hasattr(command, 'detach'):
        command = command.detach().cpu().numpy()
      return self._command_from_one_hot(command)
    if hasattr(self, 'commands') and len(self.commands) > 0:
      return int(self.commands[-1])
    return 4

  def _route_measurement_from_planner(self, tick_data, fallback_route=None):
    fallback_route = fallback_route or []
    waypoint_planner = getattr(self, '_waypoint_planner', None)
    if self._ego_vehicle is not None and waypoint_planner is not None:
      ego_location = self._ego_vehicle.get_transform().location
      gps = tick_data.get('gps')
      compass = tick_data.get('compass')
      if gps is not None:
        gps = np.asarray(gps, dtype=np.float64)
        gps = np.append(gps[:2], ego_location.z)
        waypoint_route = waypoint_planner.run_step(gps)
        if len(waypoint_route) >= 2:
          waypoint_nodes = list(extrapolate_waypoint_route(waypoint_route, self.config.num_route_points_saved))
          if compass is None:
            compass = np.deg2rad(self._ego_vehicle.get_transform().rotation.yaw)
          route = [
              t_u.inverse_conversion_2d(node[0][:2], gps[:2], float(compass)).astype(float).tolist()
              for node in waypoint_nodes[:self.config.num_route_points_saved]
          ]
          if len(waypoint_nodes) > 1:
            next_command = int(waypoint_nodes[1][1].value)
          else:
            next_command = int(waypoint_nodes[0][1].value)
          return route, route.copy(), next_command

    return fallback_route, fallback_route.copy(), self._command_from_tick(tick_data)

  def _ego_vehicle_in_junction(self):
    world_map = getattr(self, 'world_map', None)
    if self._ego_vehicle is None or world_map is None:
      return False
    try:
      waypoint = world_map.get_waypoint(self._ego_vehicle.get_location(), lane_type=carla.LaneType.Any)
      return bool(waypoint is not None and waypoint.is_junction)
    except (RuntimeError, AttributeError):
      return False

  def _save_training_sensors(self, frame, tick_data):
    input_data = getattr(self, '_latest_raw_input_data', None)
    if input_data is None:
      return

    rgb = input_data['rgb'][1][:, :, :3]
    rgb_augmented = input_data['rgb_augmented'][1][:, :, :3]
    semantics = input_data['semantics'][1][:, :, 2]
    semantics_augmented = input_data['semantics_augmented'][1][:, :, 2]

    depth = input_data['depth'][1][:, :, :3]
    depth = (t_u.convert_depth(depth) * 255.0 + 0.5).astype(np.uint8)
    depth_augmented = input_data['depth_augmented'][1][:, :, :3]
    depth_augmented = (t_u.convert_depth(depth_augmented) * 255.0 + 0.5).astype(np.uint8)

    self._update_augmented_vehicle_dummy()
    if self.gt_bev_manager is None or self.gt_bev_manager_augmented is None:
      return

    if self.gt_stop_sign_criteria is not None:
      self.gt_stop_sign_criteria.tick(self._ego_vehicle)

    bev_semantics = self.gt_bev_manager.get_observation(close_traffic_lights=None)['bev_semantic_classes']
    bev_semantics_augmented = self.gt_bev_manager_augmented.get_observation(
        close_traffic_lights=None)['bev_semantic_classes']

    lidar = None
    if self.config.backbone not in ('aim') and len(self.lidar_buffer) > 0:
      lidar = np.asarray(self.lidar_buffer[-1], dtype=np.float32)
    elif 'lidar' in tick_data:
      lidar = np.asarray(tick_data['lidar'], dtype=np.float32)

    cv2.imwrite(str(self.save_path / 'rgb' / f'{frame:04}.jpg'), rgb)
    cv2.imwrite(str(self.save_path / 'rgb_augmented' / f'{frame:04}.jpg'), rgb_augmented)
    cv2.imwrite(str(self.save_path / 'semantics' / f'{frame:04}.png'), semantics)
    cv2.imwrite(str(self.save_path / 'semantics_augmented' / f'{frame:04}.png'), semantics_augmented)
    cv2.imwrite(str(self.save_path / 'depth' / f'{frame:04}.png'), depth)
    cv2.imwrite(str(self.save_path / 'depth_augmented' / f'{frame:04}.png'), depth_augmented)
    cv2.imwrite(str(self.save_path / 'bev_semantics' / f'{frame:04}.png'), bev_semantics)
    cv2.imwrite(str(self.save_path / 'bev_semantics_augmented' / f'{frame:04}.png'), bev_semantics_augmented)

    boxes = self._get_training_bounding_boxes(lidar)
    with gzip.open(self.save_path / 'boxes' / f'{frame:04}.json.gz', 'wt', encoding='utf-8') as outfile:
      ujson.dump(boxes, outfile, indent=4)

    if lidar is not None and len(lidar) > 0:
      header = laspy.LasHeader(point_format=self.config.point_format)
      header.offsets = np.min(lidar, axis=0)
      header.scales = np.array([self.config.point_precision, self.config.point_precision, self.config.point_precision])
      with laspy.open(self.save_path / 'lidar' / f'{frame:04}.laz', mode='w', header=header) as writer:
        point_record = laspy.ScaleAwarePointRecord.zeros(lidar.shape[0], header=header)
        point_record.x = lidar[:, 0]
        point_record.y = lidar[:, 1]
        point_record.z = lidar[:, 2]
        writer.write_points(point_record)

  def _update_augmented_vehicle_dummy(self):
    if self.augmented_vehicle_dummy is None or self._ego_vehicle is None:
      return
    bb_copy = carla.BoundingBox(self._ego_vehicle.bounding_box.location, self._ego_vehicle.bounding_box.extent)
    transform_copy = carla.Transform(self._ego_vehicle.get_transform().location, self._ego_vehicle.get_transform().rotation)
    augmented_loc = transform_copy.transform(carla.Location(0.0, self.augmentation_translation, 0.0))
    transform_copy.location = augmented_loc
    transform_copy.rotation.yaw = transform_copy.rotation.yaw + self.augmentation_rotation
    self.augmented_vehicle_dummy.bounding_box = bb_copy
    self.augmented_vehicle_dummy.transform = transform_copy

  def _get_training_bounding_boxes(self, lidar=None):
    if self._ego_vehicle is None or self._world is None:
      return []

    ego_transform = self._ego_vehicle.get_transform()
    ego_velocity = self._ego_vehicle.get_velocity()
    ego_control = self._ego_vehicle.get_control()
    ego_matrix = np.array(ego_transform.get_matrix())
    ego_yaw = np.deg2rad(ego_transform.rotation.yaw)
    ego_extent = self._ego_vehicle.bounding_box.extent

    boxes = [{
        'class': 'ego_car',
        'extent': [ego_extent.x, ego_extent.y, ego_extent.z],
        'position': [0.0, 0.0, 0.0],
        'yaw': 0.0,
        'num_points': -1,
        'distance': -1,
        'speed': self._get_forward_speed(ego_transform, ego_velocity),
        'brake': ego_control.brake,
        'id': int(self._ego_vehicle.id),
        'matrix': ego_transform.get_matrix(),
    }]

    actors = self._world.get_actors()
    for vehicle in actors.filter('*vehicle*'):
      if vehicle.id == self._ego_vehicle.id:
        continue
      if vehicle.get_location().distance(self._ego_vehicle.get_location()) > self.config.bb_save_radius:
        continue
      boxes.append(self._actor_box(vehicle, 'car', ego_matrix, ego_yaw, lidar))

    for walker in actors.filter('*walker*'):
      if walker.get_location().distance(self._ego_vehicle.get_location()) > self.config.bb_save_radius:
        continue
      boxes.append(self._actor_box(walker, 'walker', ego_matrix, ego_yaw, lidar))

    for static in actors.filter('*static*'):
      if static.get_location().distance(self._ego_vehicle.get_location()) > self.config.bb_save_radius:
        continue
      box = self._actor_box(static, 'static', ego_matrix, ego_yaw, lidar)
      box['type_id'] = static.type_id
      box['mesh_path'] = static.attributes['mesh_path'] if 'mesh_path' in static.attributes else None
      boxes.append(box)

    for traffic_light in actors.filter('*traffic_light*'):
      if not hasattr(traffic_light, 'bounding_box'):
        continue
      if traffic_light.get_location().distance(self._ego_vehicle.get_location()) > self.config.bb_save_radius:
        continue
      box = self._actor_box(traffic_light, 'traffic_light', ego_matrix, ego_yaw, lidar)
      box['state'] = str(traffic_light.state)
      box['affects_ego'] = False
      boxes.append(box)

    for stop_sign in actors.filter('*traffic.stop*'):
      if not hasattr(stop_sign, 'bounding_box'):
        continue
      if stop_sign.get_location().distance(self._ego_vehicle.get_location()) > self.config.bb_save_radius:
        continue
      box = self._actor_box(stop_sign, 'stop_sign', ego_matrix, ego_yaw, lidar)
      box['affects_ego'] = False
      boxes.append(box)

    return boxes

  def _actor_box(self, actor, class_name, ego_matrix, ego_yaw, lidar=None):
    transform = actor.get_transform()
    try:
      velocity = actor.get_velocity()
    except RuntimeError:
      velocity = carla.Vector3D()
    extent = actor.bounding_box.extent
    extent_list = [extent.x, extent.y, extent.z]
    yaw = t_u.normalize_angle(np.deg2rad(transform.rotation.yaw) - ego_yaw)
    relative_pos = t_u.get_relative_transform(ego_matrix, np.array(transform.get_matrix()))
    num_points = self._get_points_in_bbox(relative_pos, yaw, extent_list, lidar) if lidar is not None else -1

    result = {
        'class': class_name,
        'extent': extent_list,
        'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
        'yaw': yaw,
        'num_points': int(num_points),
        'distance': float(np.linalg.norm(relative_pos)),
        'speed': self._get_forward_speed(transform, velocity),
        'id': int(actor.id),
        'matrix': transform.get_matrix(),
    }
    if hasattr(actor, 'get_control') and class_name == 'car':
      control = actor.get_control()
      result.update({
          'brake': control.brake,
          'steer': control.steer,
          'throttle': control.throttle,
          'role_name': actor.attributes.get('role_name', ''),
          'type_id': actor.type_id,
      })
    return result

  @staticmethod
  def _get_forward_speed(transform, velocity):
    vel_np = np.array([velocity.x, velocity.y, velocity.z])
    pitch = np.deg2rad(transform.rotation.pitch)
    yaw = np.deg2rad(transform.rotation.yaw)
    orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    return float(np.dot(vel_np, orientation))

  @staticmethod
  def _get_points_in_bbox(vehicle_pos, vehicle_yaw, extent, lidar):
    rotation_matrix = np.array([[np.cos(vehicle_yaw), -np.sin(vehicle_yaw), 0.0],
                                [np.sin(vehicle_yaw), np.cos(vehicle_yaw), 0.0], [0.0, 0.0, 1.0]])
    vehicle_lidar = (rotation_matrix.T @ (lidar - vehicle_pos).T).T
    x, y, z = extent[0], extent[1], extent[2]
    return ((vehicle_lidar[:, 0] < x) & (vehicle_lidar[:, 0] > -x) & (vehicle_lidar[:, 1] < y) &
            (vehicle_lidar[:, 1] > -y) & (vehicle_lidar[:, 2] < z) & (vehicle_lidar[:, 2] > -z)).sum()

  def tick(self, input_data):
    self._latest_raw_input_data = input_data
    return super().tick(input_data)

  @staticmethod
  def _waypoints_to_list(waypoints):
    if waypoints is None:
      return []
    waypoints = np.asarray(waypoints, dtype=np.float32)
    if waypoints.ndim == 3:
      waypoints = waypoints[0]
    return [[float(point[0]), float(point[1])] for point in waypoints if len(point) >= 2]

  def _plan_waypoints_from_measurement(self, measurement):
    pred_waypoints = measurement.get('pred_waypoints')
    if isinstance(pred_waypoints, list) and pred_waypoints:
      return pred_waypoints[:self.config.pred_len], 'pred_waypoints'
    return measurement.get('route', [])[:self.config.pred_len], 'route'

  def _add_plan_safety_label_candidate(self, frame, measurement):
    self.plan_safety_case_label = measurement.get('sim_case_label', self.plan_safety_case_label)
    waypoints, plan_waypoint_source = self._plan_waypoints_from_measurement(measurement)
    self.plan_safety_label_frames[f'{frame:04}'] = [{
        'variant': measurement.get('sim_case_label', 'transfuser'),
        'plan_waypoint_source': plan_waypoint_source,
        'waypoints': waypoints,
        'target_speed': round(float(measurement.get('target_speed', 0.0)), 4),
        'expert_target_speed': round(float(measurement.get('expert_target_speed', 0.0)), 4),
        'will_collide': self.plan_safety_safe_label,
        'sim_disturbed': bool(measurement.get('sim_disturbed', False)),
        'sim_disturbance_reason': measurement.get('sim_disturbance_reason'),
    }]

  def _results_have_collision(self, results):
    if results is None:
      return False
    if isinstance(results, dict):
      infractions = results.get('infractions', {})
    else:
      infractions = getattr(results, 'infractions', {})
    collision_keys = ('collisions_layout', 'collisions_pedestrian', 'collisions_vehicle')
    return any(bool(infractions.get(key)) for key in collision_keys)

  @staticmethod
  def _collision_locations_from_results(results):
    if results is None:
      return []
    if isinstance(results, dict):
      infractions = results.get('infractions', {})
    else:
      infractions = getattr(results, 'infractions', {})

    collision_keys = ('collisions_layout', 'collisions_pedestrian', 'collisions_vehicle')
    location_pattern = re.compile(
        r'at \(x=([-+0-9.eE]+),\s*y=([-+0-9.eE]+),\s*z=([-+0-9.eE]+)\)')
    locations = []
    for key in collision_keys:
      for message in infractions.get(key, []) or []:
        match = location_pattern.search(str(message))
        if match:
          locations.append(
              np.array([float(match.group(1)), float(match.group(2)), float(match.group(3))], dtype=np.float64))
    return locations

  @staticmethod
  def _distinct_collision_locations(locations):
    distinct_locations = []
    for location in locations:
      if not any(
          float(np.linalg.norm(location[:2] - existing[:2])) < COLLISION_EVENT_MIN_DISTANCE
          for existing in distinct_locations):
        distinct_locations.append(location)
    return distinct_locations

  def _nearest_saved_measurement_frame_to_location(self, location):
    if self.save_path is None or location is None:
      return None
    measurements_dir = self.save_path / 'measurements'
    if not measurements_dir.is_dir():
      return None

    best_frame = None
    best_distance = None
    target = np.asarray(location[:2], dtype=np.float64)
    for measurement_path in sorted(measurements_dir.glob('*.json.gz')):
      try:
        with gzip.open(measurement_path, 'rt', encoding='utf-8') as infile:
          measurement = ujson.load(infile)
        pos_global = measurement.get('pos_global')
        frame = measurement.get('frame')
        frame = int(frame) if frame is not None else int(measurement_path.name.split('.', 1)[0])
        if pos_global is None or len(pos_global) < 2:
          continue
        distance = float(np.linalg.norm(np.asarray(pos_global[:2], dtype=np.float64) - target))
      except (OSError, EOFError, TypeError, ValueError):
        continue
      if best_distance is None or distance < best_distance:
        best_frame = frame
        best_distance = distance
    if best_frame is None:
      return None
    return best_frame, best_distance

  def _resolve_collision_events(self, results):
    collision_events = []
    locations = self._distinct_collision_locations(self._collision_locations_from_results(results))
    for location in locations:
      nearest = self._nearest_saved_measurement_frame_to_location(location)
      if nearest is None:
        continue
      frame, distance = nearest
      measurement = self._load_saved_measurement(frame)
      if measurement is None:
        continue
      collision_events.append({
          'frame': int(frame),
          'location': [round(float(value), 4) for value in location.tolist()],
          'location_distance': round(float(distance), 4),
          'measurement': measurement,
          'source': 'nearest_saved_measurement_to_collision_location',
      })
    collision_events.sort(key=lambda event: event['frame'])
    return collision_events

  def _resolve_collision_data_frame(self, results):
    collision_events = self._resolve_collision_events(results)
    if collision_events:
      return collision_events[0]['frame'], 'nearest_saved_measurement_to_collision_location'

    return None, 'unresolved'

  def _load_saved_measurement(self, frame):
    if self.save_path is None or frame is None:
      return None
    measurement_path = self.save_path / 'measurements' / f'{int(frame):04}.json.gz'
    if not measurement_path.is_file():
      return None
    try:
      with gzip.open(measurement_path, 'rt', encoding='utf-8') as infile:
        return ujson.load(infile)
    except (OSError, EOFError, ValueError):
      return None

  def _resolve_collision_region_radius(self, collision_data_frame):
    del collision_data_frame
    if self._ego_vehicle is not None:
      extent = self._ego_vehicle.bounding_box.extent
      ego_extent = [float(extent.x), float(extent.y)]
      return float(np.linalg.norm(np.asarray(ego_extent, dtype=np.float64))), 'ego_vehicle_bounding_box', ego_extent
    return DEFAULT_COLLISION_REGION_RADIUS, 'default_ego_extent', DEFAULT_EGO_EXTENT.tolist()

  @staticmethod
  def _speed_mps(measurement, key, default=0.0):
    try:
      return max(0.0, float(measurement.get(key, default) or default))
    except (TypeError, ValueError):
      return default

  @staticmethod
  def _collision_point_in_ego(measurement, collision_measurement):
    ego_matrix_raw = measurement.get('ego_matrix')
    collision_matrix_raw = collision_measurement.get('ego_matrix')
    if ego_matrix_raw is None or collision_matrix_raw is None:
      return None

    try:
      ego_matrix = np.array(ego_matrix_raw, dtype=np.float64)
      collision_matrix = np.array(collision_matrix_raw, dtype=np.float64)
      relative_matrix = np.linalg.inv(ego_matrix) @ collision_matrix
    except (TypeError, ValueError, np.linalg.LinAlgError):
      return None

    return relative_matrix[:2, 3].astype(np.float64)

  @staticmethod
  def _path_points_from_waypoints(waypoints):
    points = [np.zeros(2, dtype=np.float64)]
    for waypoint in waypoints:
      if not isinstance(waypoint, (list, tuple)) or len(waypoint) < 2:
        continue
      try:
        points.append(np.array([float(waypoint[0]), float(waypoint[1])], dtype=np.float64))
      except (TypeError, ValueError):
        continue
    return np.asarray(points, dtype=np.float64)

  def _path_distance_to_region(self, points, center, radius):
    if len(points) == 0:
      return None

    cumulative_distance = 0.0
    best_distance_along_path = None
    for index in range(len(points) - 1):
      start = points[index]
      end = points[index + 1]
      segment = end - start
      segment_length = float(np.linalg.norm(segment))
      if segment_length < 1e-6:
        distance_to_center = float(np.linalg.norm(center - start))
        closest_distance_along_path = cumulative_distance
      else:
        projection = float(np.clip(np.dot(center - start, segment) / (segment_length**2), 0.0, 1.0))
        closest = start + projection * segment
        distance_to_center = float(np.linalg.norm(center - closest))
        closest_distance_along_path = cumulative_distance + projection * segment_length

      if distance_to_center <= radius:
        if best_distance_along_path is None or closest_distance_along_path < best_distance_along_path:
          best_distance_along_path = closest_distance_along_path
      cumulative_distance += segment_length

    if len(points) == 1 and float(np.linalg.norm(center - points[0])) <= radius:
      return 0.0
    return best_distance_along_path

  def _straight_waypoint_rollout(self, waypoints):
    points = self._path_points_from_waypoints(waypoints)
    if len(points) < 3:
      return False

    path_points = points[1:]
    lateral_spread = float(path_points[:, 1].max() - path_points[:, 1].min())
    if lateral_spread > STRAIGHT_WAYPOINT_MAX_LATERAL_SPREAD:
      return False

    segment_vectors = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    valid_segments = segment_vectors[segment_lengths > 1e-3]
    if len(valid_segments) < 2:
      return True

    headings = np.unwrap(np.arctan2(valid_segments[:, 1], valid_segments[:, 0]))
    heading_change = math.degrees(float(headings.max() - headings.min()))
    return heading_change <= STRAIGHT_WAYPOINT_MAX_HEADING_CHANGE_DEG

  def _collision_point_same_lane_with_waypoints(self, waypoints, collision_point):
    points = self._path_points_from_waypoints(waypoints)
    if len(points) < 2:
      return False

    waypoint_lateral_offsets = points[1:, 1]
    collision_lateral_offset = float(collision_point[1])
    return bool(np.any(np.abs(waypoint_lateral_offsets - collision_lateral_offset) <=
                       COLLISION_POINT_SAME_LANE_LATERAL_MARGIN))

  @staticmethod
  def _velocity_rollout_distance(current_speed, target_speed, horizon_seconds, rollout_dt=0.05):
    speed = max(0.0, float(current_speed))
    target_speed = max(0.0, float(target_speed))
    remaining_time = max(0.0, float(horizon_seconds))
    distance = 0.0
    dt = max(1e-3, float(rollout_dt))

    while remaining_time > 1e-9:
      step_dt = min(dt, remaining_time)
      delta_speed = target_speed - speed
      if abs(delta_speed) < 1e-9:
        next_speed = speed
      elif delta_speed > 0.0:
        next_speed = min(target_speed, speed + DEFAULT_MAX_ACCELERATION * step_dt)
      else:
        next_speed = max(target_speed, speed - DEFAULT_MAX_DECELERATION * step_dt)
      distance += 0.5 * (speed + next_speed) * step_dt
      speed = next_speed
      remaining_time -= step_dt

    return distance

  def _evaluate_collision_reachability(self, measurement, collision_measurement, frame, collision_data_frame,
                                       collision_region_radius):
    current_speed = self._speed_mps(measurement, 'speed')
    target_speed = self._speed_mps(measurement, 'target_speed')
    waypoints, plan_waypoint_source = self._plan_waypoints_from_measurement(measurement)
    straight_rollout = self._straight_waypoint_rollout(waypoints)
    detail = {
        'current_speed': round(current_speed, 4),
        'target_speed': round(target_speed, 4),
        'plan_waypoint_source': plan_waypoint_source,
        'straight_waypoint_rollout': straight_rollout,
        'collision_point_same_lane_with_straight_waypoints': False,
        'collision_case_excluded_straight': False,
        'collision_case_outside_checked_window': False,
        'collision_case_unsafe': False,
        'collision_distance': None,
        'time_to_collision': None,
        'intersects_future_collision_region': False,
        'reachable_before_collision': False,
        'distance_to_collision_region_along_rollout': None,
        'reachable_distance_before_collision': None,
    }

    if collision_data_frame is None or collision_measurement is None or frame >= collision_data_frame:
      return False, detail

    if collision_data_frame - frame > MAX_CHECKED_FRAMES_BEFORE_EVENT:
      detail['collision_case_outside_checked_window'] = True
      return False, detail

    collision_point = self._collision_point_in_ego(measurement, collision_measurement)
    if collision_point is None:
      return False, detail

    detail['collision_point_ego'] = [round(float(collision_point[0]), 4), round(float(collision_point[1]), 4)]
    same_lane_collision_point = self._collision_point_same_lane_with_waypoints(waypoints, collision_point)
    detail['collision_point_same_lane_with_straight_waypoints'] = same_lane_collision_point
    if straight_rollout and not same_lane_collision_point:
      detail['collision_case_excluded_straight'] = True
      return False, detail

    collision_distance = float(np.linalg.norm(collision_point))
    time_to_collision = ((collision_data_frame - frame) * max(1, int(self.config.data_save_freq)) /
                         max(DEFAULT_SIM_FPS, 1e-6))
    detail['collision_distance'] = round(collision_distance, 4)
    detail['time_to_collision'] = round(time_to_collision, 4)

    points = self._path_points_from_waypoints(waypoints)
    distance_to_region = self._path_distance_to_region(points, collision_point, max(0.0, collision_region_radius))
    if distance_to_region is None:
      return False, detail

    reachable_distance = self._velocity_rollout_distance(current_speed, target_speed, time_to_collision)
    detail['intersects_future_collision_region'] = True
    detail['distance_to_collision_region_along_rollout'] = round(distance_to_region, 4)
    detail['reachable_distance_before_collision'] = round(reachable_distance, 4)
    detail['reachable_before_collision'] = reachable_distance >= distance_to_region
    detail['collision_case_unsafe'] = bool(detail['reachable_before_collision'])
    return bool(detail['reachable_before_collision']), detail

  def _evaluate_collision_events_reachability(self, measurement, frame, collision_events, collision_region_radius):
    event_details = []
    first_detail = None
    for event_index, event in enumerate(collision_events):
      event_frame = event['frame']
      if frame >= event_frame:
        continue
      unsafe, detail = self._evaluate_collision_reachability(
          measurement, event['measurement'], frame, event_frame, collision_region_radius)
      detail['collision_event_index'] = event_index
      detail['collision_data_frame'] = event_frame
      detail['collision_location'] = event['location']
      detail['collision_location_distance'] = event['location_distance']
      event_details.append(detail)
      if unsafe:
        detail['collision_case_unsafe'] = True
        return True, detail
      if first_detail is None:
        first_detail = detail

    if first_detail is None:
      current_speed = self._speed_mps(measurement, 'speed')
      target_speed = self._speed_mps(measurement, 'target_speed')
      return False, {
          'current_speed': round(current_speed, 4),
          'target_speed': round(target_speed, 4),
          'collision_case_unsafe': False,
          'collision_events_checked': 0,
      }

    first_detail['collision_events_checked'] = len(event_details)
    return False, first_detail

  @staticmethod
  def _normalize_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

  @staticmethod
  def _yaw_from_matrix(matrix):
    return math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))

  @staticmethod
  def _trajectory_image_point(x, y):
    col = int(round((y - TRAJECTORY_VIS_MIN_Y) * TRAJECTORY_VIS_PIXELS_PER_METER))
    row = int(round((TRAJECTORY_VIS_MAX_X - x) * TRAJECTORY_VIS_PIXELS_PER_METER))
    return col, row

  def _actor_pose_in_origin(self, origin_matrix, actor):
    actor_matrix_raw = actor.get('matrix')
    if actor_matrix_raw is None:
      return None

    try:
      actor_matrix = np.array(actor_matrix_raw, dtype=np.float64)
      relative_matrix = np.linalg.inv(origin_matrix) @ actor_matrix
    except (TypeError, ValueError, np.linalg.LinAlgError):
      return None

    yaw = self._normalize_angle(self._yaw_from_matrix(actor_matrix) - self._yaw_from_matrix(origin_matrix))
    return float(relative_matrix[0, 3]), float(relative_matrix[1, 3]), yaw

  @staticmethod
  def _actor_track_id(actor, fallback_index):
    actor_id = actor.get('id')
    if actor_id is None:
      return f"{actor.get('class', 'actor')}:{fallback_index}"
    return f"{actor.get('class', 'actor')}:{actor_id}"

  def _load_actor_tracks_from_dataset(self, frame, horizon, origin_matrix):
    tracks = {}
    for offset in range(horizon + 1):
      boxes_path = self.save_path / 'boxes' / f'{frame + offset:04}.json.gz'
      if not boxes_path.is_file():
        break
      try:
        with gzip.open(boxes_path, 'rt', encoding='utf-8') as infile:
          actors = ujson.load(infile)
      except (OSError, EOFError, ValueError):
        continue

      for fallback_index, actor in enumerate(actors):
        actor_class = str(actor.get('class', 'actor'))
        if actor_class not in VISUALIZED_CLASSES:
          continue
        pose = self._actor_pose_in_origin(origin_matrix, actor)
        if pose is None:
          continue
        track_id = self._actor_track_id(actor, fallback_index)
        if track_id not in tracks:
          tracks[track_id] = {
              'class': actor_class,
              'extent': actor.get('extent', DEFAULT_EGO_EXTENT.tolist()),
              'poses': [],
          }
        tracks[track_id]['poses'].append(pose)
        tracks[track_id]['extent'] = actor.get('extent', tracks[track_id]['extent'])
    return tracks

  def _draw_oriented_box(self, draw, pose, extent, color, width=2):
    x, y, yaw = pose
    extent_x = float(extent[0]) if len(extent) > 0 else float(DEFAULT_EGO_EXTENT[0])
    extent_y = float(extent[1]) if len(extent) > 1 else float(DEFAULT_EGO_EXTENT[1])
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    corners = []
    for local_x, local_y in ((extent_x, extent_y), (extent_x, -extent_y), (-extent_x, -extent_y),
                             (-extent_x, extent_y)):
      world_x = x + cos_yaw * local_x - sin_yaw * local_y
      world_y = y + sin_yaw * local_x + cos_yaw * local_y
      corners.append(self._trajectory_image_point(world_x, world_y))
    draw.line(corners + [corners[0]], fill=color, width=width)
    nose = self._trajectory_image_point(x + cos_yaw * extent_x, y + sin_yaw * extent_x)
    center = self._trajectory_image_point(x, y)
    draw.line((center, nose), fill=color, width=max(1, width))

  def _draw_candidate_waypoints(self, draw, candidate):
    image_points = []
    for waypoint in candidate.get('waypoints', []):
      if not isinstance(waypoint, (list, tuple)) or len(waypoint) < 2:
        continue
      try:
        image_points.append(self._trajectory_image_point(float(waypoint[0]), float(waypoint[1])))
      except (TypeError, ValueError):
        continue
    if len(image_points) > 1:
      draw.line(image_points, fill=(35, 80, 225), width=3)
    for index, point in enumerate(image_points):
      radius = 5 if index == 0 else 3
      fill = (255, 235, 60) if index == 0 else (35, 80, 225)
      outline = (20, 20, 20) if index == 0 else (255, 255, 255)
      draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=fill,
                   outline=outline, width=2 if index == 0 else 1)

  def _append_rgb_panel(self, trajectory_image, frame_key):
    for suffix in ('.jpg', '.png', '.jpeg'):
      rgb_path = self.save_path / 'rgb' / f'{int(frame_key):04}{suffix}'
      if rgb_path.is_file():
        break
    else:
      return trajectory_image

    rgb_image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb_image is None:
      return trajectory_image
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    target_height = trajectory_image.height
    target_width = max(1, int(round(rgb_image.shape[1] * target_height / max(rgb_image.shape[0], 1))))
    rgb_image = cv2.resize(rgb_image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    from PIL import Image, ImageDraw
    rgb_panel = Image.fromarray(rgb_image)
    combined = Image.new('RGB', (target_width + trajectory_image.width, target_height), (248, 248, 248))
    combined.paste(rgb_panel, (0, 0))
    combined.paste(trajectory_image, (target_width, 0))
    draw = ImageDraw.Draw(combined)
    draw.rectangle((0, 0, target_width - 1, target_height - 1), outline=(0, 0, 0), width=2)
    draw.line(((target_width, 0), (target_width, target_height)), fill=(0, 0, 0), width=2)
    draw.text((8, target_height - 18), f'RGB {frame_key}', fill=(255, 255, 255))
    return combined

  def _render_trajectory_visualization(self, frame_key, candidates, output_dir):
    try:
      from PIL import Image, ImageDraw
    except ImportError:
      print('Skipping plan-safety trajectory visualization because Pillow is not installed.')
      return None

    measurement = self._load_saved_measurement(int(frame_key))
    if measurement is None or measurement.get('ego_matrix') is None:
      return None
    try:
      origin_matrix = np.array(measurement['ego_matrix'], dtype=np.float64)
    except (TypeError, ValueError):
      return None

    tracks = self._load_actor_tracks_from_dataset(int(frame_key), MAX_CHECKED_FRAMES_BEFORE_EVENT, origin_matrix)
    if not tracks:
      return None

    width = max(1, int(round((TRAJECTORY_VIS_MAX_Y - TRAJECTORY_VIS_MIN_Y) * TRAJECTORY_VIS_PIXELS_PER_METER)))
    height = max(1, int(round((TRAJECTORY_VIS_MAX_X - TRAJECTORY_VIS_MIN_X) * TRAJECTORY_VIS_PIXELS_PER_METER)))
    image = Image.new('RGB', (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(image)

    zero_x = self._trajectory_image_point(0.0, TRAJECTORY_VIS_MIN_Y)[1]
    zero_y = self._trajectory_image_point(TRAJECTORY_VIS_MIN_X, 0.0)[0]
    draw.line(((0, zero_x), (width, zero_x)), fill=(220, 220, 220), width=1)
    draw.line(((zero_y, 0), (zero_y, height)), fill=(220, 220, 220), width=1)

    for track in tracks.values():
      actor_class = track['class']
      color = TRAJECTORY_CLASS_COLORS.get(actor_class, DEFAULT_TRAJECTORY_COLOR)
      points = [self._trajectory_image_point(pose[0], pose[1]) for pose in track['poses']]
      if len(points) > 1:
        draw.line(points, fill=color, width=4 if actor_class == 'ego_car' else 2)
      for point_index, point in enumerate(points):
        radius = 4 if actor_class == 'ego_car' else 3
        fill = color if point_index == len(points) - 1 else tuple(max(0, int(channel * 0.65)) for channel in color)
        draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=fill)
      if track['poses']:
        self._draw_oriented_box(draw, track['poses'][-1], track.get('extent', DEFAULT_EGO_EXTENT.tolist()), color,
                                width=3 if actor_class == 'ego_car' else 2)

    if candidates:
      self._draw_candidate_waypoints(draw, candidates[0])

    unsafe = any(int(candidate.get('will_collide', self.plan_safety_safe_label)) == self.plan_safety_unsafe_label
                 for candidate in candidates)
    label = 'unsafe' if unsafe else 'safe'
    title_color = (210, 30, 30) if unsafe else (20, 135, 55)
    draw.text((8, 8), f'{frame_key} {label} recorded dataset trajectories', fill=title_color)
    draw.text((8, 24), f'horizon={MAX_CHECKED_FRAMES_BEFORE_EVENT} saved frames, actors={len(tracks)}', fill=(35, 35, 35))
    if candidates:
      target_speed = float(candidates[0].get('target_speed', 0.0) or 0.0)
      draw.text((8, 40), f'target_speed={target_speed:.2f} m/s ({target_speed * 3.6:.1f} km/h)', fill=(20, 20, 20))

    image = self._append_rgb_panel(image, frame_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{frame_key}_{label}_dataset_trajectories.png'
    image.save(output_path)
    return output_path

  def _save_plan_safety_trajectory_visualizations(self, labeled_frames):
    if self.save_path is None or not labeled_frames:
      return

    unsafe_frame_keys = []
    safe_frame_keys = []
    for frame_key, candidates in labeled_frames.items():
      unsafe = any(int(candidate.get('will_collide', self.plan_safety_safe_label)) == self.plan_safety_unsafe_label
                   for candidate in candidates)
      if unsafe:
        unsafe_frame_keys.append(frame_key)
      else:
        safe_frame_keys.append(frame_key)

    rng_seed = sum(ord(char) for char in str(self.save_path))
    rng = random.Random(rng_seed)
    sampled_safe_keys = rng.sample(safe_frame_keys, min(TRAJECTORY_VIS_MAX_SAFE_FRAMES, len(safe_frame_keys)))
    output_dir = self.save_path / 'plan_safety_trajectory_visualizations'
    if output_dir.is_dir():
      shutil.rmtree(output_dir)
    for frame_key in sorted(unsafe_frame_keys) + sorted(sampled_safe_keys):
      self._render_trajectory_visualization(frame_key, labeled_frames[frame_key], output_dir)

  def _write_plan_safety_labels(self, results=None):
    if (
        self.save_path is None or not self.datagen or not self.generate_plan_safety_labels or
        not self.plan_safety_label_frames
    ):
      return

    route_had_collision = self._results_have_collision(results)
    collision_events = self._resolve_collision_events(results)
    collision_data_frame = collision_events[0]['frame'] if collision_events else None
    collision_frame_source = 'nearest_saved_measurement_to_collision_location' if collision_events else 'unresolved'
    collision_region_radius, collision_region_radius_source, ego_extent = self._resolve_collision_region_radius(
        collision_data_frame)

    labeled_frames = {}
    for frame_key, candidates in self.plan_safety_label_frames.items():
      frame = int(frame_key)
      if collision_data_frame is not None and frame >= collision_data_frame:
        if not any(frame < event['frame'] for event in collision_events):
          continue
      measurement = self._load_saved_measurement(frame)
      for candidate in candidates:
        safety_detail = None
        if measurement is not None:
          unsafe, safety_detail = self._evaluate_collision_events_reachability(
              measurement, frame, collision_events, collision_region_radius)
        else:
          unsafe = False
        candidate['will_collide'] = self.plan_safety_unsafe_label if unsafe else self.plan_safety_safe_label
        if safety_detail is not None:
          candidate['safety_label_detail'] = safety_detail
      labeled_frames[frame_key] = candidates

    labels = {
        'label_map': {
            'unsafe_sim_collision': self.plan_safety_unsafe_label,
            'safe': self.plan_safety_safe_label,
        },
        'source': 'carla_transfuser',
        'case_label': self.plan_safety_case_label,
        'route_had_collision': route_had_collision,
        'collision_data_frame': collision_data_frame,
        'collision_data_frames': [event['frame'] for event in collision_events],
        'collision_events': [{
            'frame': event['frame'],
            'location': event['location'],
            'location_distance': event['location_distance'],
            'source': event['source'],
        } for event in collision_events],
        'collision_data_frame_source': collision_frame_source,
        'collision_case_unsafe_criterion': (
            'within_10_frames_before_collision_future_collision_region_intersection_and_velocity_rollout_reachability_'
            'excluding_straight_waypoints_unless_collision_point_is_in_same_lane'),
        'max_checked_frames_before_event': MAX_CHECKED_FRAMES_BEFORE_EVENT,
        'straight_waypoint_max_lateral_spread': STRAIGHT_WAYPOINT_MAX_LATERAL_SPREAD,
        'straight_waypoint_max_heading_change_deg': STRAIGHT_WAYPOINT_MAX_HEADING_CHANGE_DEG,
        'collision_point_same_lane_lateral_margin': COLLISION_POINT_SAME_LANE_LATERAL_MARGIN,
        'sim_fps': DEFAULT_SIM_FPS,
        'data_save_freq': self.config.data_save_freq,
        'max_acceleration': DEFAULT_MAX_ACCELERATION,
        'max_deceleration': DEFAULT_MAX_DECELERATION,
        'rollout_dt': 0.05,
        'collision_region_radius': collision_region_radius,
        'collision_region_radius_source': collision_region_radius_source,
        'ego_extent': ego_extent,
        'pred_len': self.config.pred_len,
        'frames': labeled_frames,
    }
    with gzip.open(self.save_path / 'plan_safety_labels.json.gz', 'wt', encoding='utf-8') as outfile:
      ujson.dump(labels, outfile, indent=2)
    self._save_plan_safety_trajectory_visualizations(labeled_frames)

  def destroy(self, results=None):
    delete_route_folder = (
        self.delete_route_folder_without_collision and self.save_path is not None and
        not self._results_have_collision(results)
    )
    self._write_plan_safety_labels(results)
    super().destroy(results)
    route_folder = self.save_path
    if delete_route_folder and route_folder is not None and route_folder.exists():
      print(f'Deleting route folder without collision event: {route_folder}')
      shutil.rmtree(route_folder)
