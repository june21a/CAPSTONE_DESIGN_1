"""
Evaluation agent that saves model vision predictions together with privileged GT labels.

This is intended for local comparison/debug runs, not benchmark submission, because it
adds semantic/depth sensors and uses privileged world state for BEV labels.
"""

import pathlib
import json
import os

import cv2
import numpy as np
import torch

from sensor_agent import SensorAgent
import transfuser_utils as t_u
from birds_eye_view.chauffeurnet import ObsManager
from birds_eye_view.run_stop_sign import RunStopSign


def get_entry_point():
  return 'ComparisonAgent'


class ComparisonAgent(SensorAgent):
  """
  SensorAgent variant that records GT semantic/depth/BEV semantic labels next to model predictions.
  """

  SEMANTIC_CLASS_NAMES = ['unlabeled', 'vehicle', 'road', 'traffic_light', 'pedestrian', 'road_line', 'sidewalk']
  BEV_SEMANTIC_CLASS_NAMES = [
      'unlabeled', 'road', 'sidewalk', 'lane_marker', 'lane_marker_broken', 'stop_sign', 'traffic_light_green',
      'traffic_light_yellow', 'traffic_light_red', 'vehicle', 'walker'
  ]
  DETECTION_CLASS_NAMES = ['car', 'walker', 'traffic_light', 'stop_sign', 'emergency_vehicle']

  def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
    super().setup(path_to_conf_file, route_index, traffic_manager=traffic_manager)
    self.vision_task_gt_paths = None
    if os.environ.get('COMPARISON_FORCE_DEBUG', '1') == '1':
      self.config.debug = True
    self.gt_bev_manager = None
    self.gt_stop_sign_criteria = None
    self.gt_vehicle = None
    self._init_vision_task_gt_paths()

  def _init(self):
    super()._init()

    if self.save_path is None:
      return

    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider  # pylint: disable=import-outside-toplevel

    self.gt_vehicle = CarlaDataProvider.get_hero_actor()
    obs_config = {
        'width_in_pixels': self.config.lidar_resolution_width,
        'pixels_ev_to_bottom': self.config.lidar_resolution_height / 2.0,
        'pixels_per_meter': self.config.pixels_per_meter_collection,
        'history_idx': [-1],
        'scale_bbox': True,
        'scale_mask_col': 1.0,
        'map_folder': 'maps_2ppm_cv',
    }
    self.gt_stop_sign_criteria = RunStopSign(self.gt_vehicle.get_world())
    self.gt_bev_manager = ObsManager(obs_config, self.config)
    self.gt_bev_manager.attach_ego_vehicle(self.gt_vehicle, criteria_stop=self.gt_stop_sign_criteria)

  def sensors(self):
    sensors = super().sensors()
    if self.save_path is None:
      return sensors

    sensors += [{
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
        'id': 'gt_semantics'
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
        'id': 'gt_depth'
    }]
    return sensors

  @torch.inference_mode()
  def tick(self, input_data):
    tick_data = super().tick(input_data)

    if 'gt_semantics' in input_data:
      tick_data['gt_semantic'] = input_data['gt_semantics'][1][:, :, 2].astype(np.uint8)

    if 'gt_depth' in input_data:
      depth = input_data['gt_depth'][1][:, :, :3]
      tick_data['gt_depth'] = (t_u.convert_depth(depth) * 255.0 + 0.5).astype(np.uint8)

    if self.gt_bev_manager is not None and self.gt_vehicle is not None:
      self.gt_stop_sign_criteria.tick(self.gt_vehicle)
      gt_bev = self.gt_bev_manager.get_observation(close_traffic_lights=None)
      tick_data['gt_bev_semantic'] = gt_bev['bev_semantic_classes'].astype(np.uint8)
      tick_data['gt_detection_boxes'] = self._get_gt_detection_boxes()

    return tick_data

  @staticmethod
  def _per_class_accuracy(pred_indices, gt_indices, class_names):
    per_class = {}
    for class_id, class_name in enumerate(class_names):
      class_mask = gt_indices == class_id
      pixel_count = int(np.sum(class_mask))
      if pixel_count == 0:
        accuracy = None
        correct = 0
      else:
        correct = int(np.sum(pred_indices[class_mask] == class_id))
        accuracy = float(correct / pixel_count)
      per_class[class_name] = {
          'accuracy': accuracy,
          'correct': correct,
          'total': pixel_count,
      }
    return per_class

  @staticmethod
  def _per_class_dice(pred_indices, gt_indices, class_names):
    per_class = {}
    for class_id, class_name in enumerate(class_names):
      pred_mask = pred_indices == class_id
      gt_mask = gt_indices == class_id
      pred_count = int(np.sum(pred_mask))
      gt_count = int(np.sum(gt_mask))
      denominator = pred_count + gt_count
      if denominator == 0:
        dice = None
        intersection = 0
      else:
        intersection = int(np.sum(pred_mask & gt_mask))
        dice = float((2.0 * intersection) / denominator)
      per_class[class_name] = {
          'dice': dice,
          'intersection': intersection,
          'pred_total': pred_count,
          'gt_total': gt_count,
      }
    return per_class

  def _get_gt_detection_boxes(self):
    if self.gt_vehicle is None:
      return []

    ego_transform = self.gt_vehicle.get_transform()
    ego_matrix = np.array(ego_transform.get_matrix())
    ego_yaw = np.deg2rad(ego_transform.rotation.yaw)
    ego_location = self.gt_vehicle.get_location()
    gt_boxes = []

    actors = self.gt_vehicle.get_world().get_actors()
    for vehicle in actors.filter('*vehicle*'):
      if vehicle.id == self.gt_vehicle.id:
        continue
      if vehicle.get_location().distance(ego_location) > self.config.bb_save_radius:
        continue

      vehicle_transform = vehicle.get_transform()
      relative_pos = t_u.get_relative_transform(ego_matrix, np.array(vehicle_transform.get_matrix()))
      yaw = t_u.normalize_angle(np.deg2rad(vehicle_transform.rotation.yaw) - ego_yaw)
      extent = vehicle.bounding_box.extent
      class_id = 0
      if vehicle.attributes.get('role_name') == 'scenario' and vehicle.type_id in [
          'vehicle.dodge.charger_police_2020', 'vehicle.dodge.charger_police', 'vehicle.ford.ambulance',
          'vehicle.carlamotors.firetruck'
      ]:
        class_id = 4
      gt_boxes.append([relative_pos[0], relative_pos[1], extent.x, extent.y, yaw, class_id])

    for walker in actors.filter('*walker*'):
      if walker.get_location().distance(ego_location) > self.config.bb_save_radius:
        continue

      walker_transform = walker.get_transform()
      relative_pos = t_u.get_relative_transform(ego_matrix, np.array(walker_transform.get_matrix()))
      yaw = t_u.normalize_angle(np.deg2rad(walker_transform.rotation.yaw) - ego_yaw)
      extent = walker.bounding_box.extent
      gt_boxes.append([relative_pos[0], relative_pos[1], extent.x, extent.y, yaw, 1])

    return gt_boxes

  @staticmethod
  def _average_precision(recalls, precisions):
    if len(recalls) == 0:
      return 0.0

    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for idx in range(len(precisions) - 1, 0, -1):
      precisions[idx - 1] = max(precisions[idx - 1], precisions[idx])
    changing_points = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[changing_points + 1] - recalls[changing_points]) * precisions[changing_points + 1]))

  def _detection_map(self, pred_boxes, gt_boxes, iou_thresholds=(0.3, 0.5, 0.7)):
    pred_boxes = [] if pred_boxes is None else [np.asarray(box, dtype=np.float32) for box in pred_boxes]
    gt_boxes = [] if gt_boxes is None else [np.asarray(box, dtype=np.float32) for box in gt_boxes]
    metrics = {}

    for iou_threshold in iou_thresholds:
      threshold_key = f'{iou_threshold:.2f}'
      per_class = {}
      class_aps = []
      for class_id, class_name in enumerate(self.DETECTION_CLASS_NAMES[:self.config.num_bb_classes]):
        class_preds = [box for box in pred_boxes if int(box[7]) == class_id]
        class_gts = [box for box in gt_boxes if int(box[5]) == class_id]
        class_preds = sorted(class_preds, key=lambda box: float(box[-1]) if len(box) > 8 else 1.0, reverse=True)
        matched_gt = set()
        tp = []
        fp = []

        for pred_box in class_preds:
          best_iou = 0.0
          best_gt_idx = None
          for gt_idx, gt_box in enumerate(class_gts):
            if gt_idx in matched_gt:
              continue
            iou = t_u.iou_bbs(pred_box[:5], gt_box[:5])
            if iou > best_iou:
              best_iou = iou
              best_gt_idx = gt_idx

          if best_gt_idx is not None and best_iou >= iou_threshold:
            matched_gt.add(best_gt_idx)
            tp.append(1.0)
            fp.append(0.0)
          else:
            tp.append(0.0)
            fp.append(1.0)

        if len(class_gts) == 0:
          ap = None
          recall = None
          precision = None
        else:
          tp_cum = np.cumsum(tp)
          fp_cum = np.cumsum(fp)
          recalls = tp_cum / max(len(class_gts), 1)
          precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)
          ap = self._average_precision(recalls, precisions)
          recall = float(recalls[-1]) if len(recalls) > 0 else 0.0
          precision = float(precisions[-1]) if len(precisions) > 0 else 0.0
          class_aps.append(ap)

        per_class[class_name] = {
            'ap': ap,
            'precision': precision,
            'recall': recall,
            'tp': int(np.sum(tp)),
            'fp': int(np.sum(fp)),
            'gt_total': len(class_gts),
            'pred_total': len(class_preds),
        }

      metrics[threshold_key] = {
          'map': None if len(class_aps) == 0 else float(np.mean(class_aps)),
          'per_class': per_class,
      }

    return metrics

  def _save_vision_task_data(self, tick_data, lidar_bev, pred_semantic, pred_bev_semantic, pred_depth, pred_bb):
    super()._save_vision_task_data(tick_data, lidar_bev, pred_semantic, pred_bev_semantic, pred_depth, pred_bb)

    if self.save_path is None or not self.collect_sensor_data or not self.vision_task_visualization:
      return
    if self.step % self.attention_save_freq != 0:
      return

    frame_id = f'{self.step:04}'
    if self.vision_task_gt_paths is None:
      self._init_vision_task_gt_paths()
      if self.vision_task_gt_paths is None:
        return

    semantic_path = self.vision_task_gt_paths['semantic']
    bev_semantic_path = self.vision_task_gt_paths['bev_semantic']
    depth_path = self.vision_task_gt_paths['depth']
    metrics_path = self.vision_task_gt_paths['metrics']

    metrics = {}

    if 'gt_semantic' in tick_data:
      gt_semantic_raw = tick_data['gt_semantic']
      converter = np.array(self.config.converter, dtype=np.uint8)
      gt_semantic = converter[np.clip(gt_semantic_raw, 0, len(converter) - 1)]
      np.savez_compressed(semantic_path / f'{frame_id}.npz', semantic=gt_semantic, semantic_raw=gt_semantic_raw)
      palette = np.array(self.config.classes_list, dtype=np.uint8)
      gt_semantic_image = palette[np.clip(gt_semantic, 0, len(palette) - 1)]
      cv2.imwrite(str(semantic_path / f'{frame_id}.png'), gt_semantic_image)

      if pred_semantic is not None:
        pred_semantic_indices = torch.argmax(pred_semantic[0], dim=0).detach().cpu().numpy().astype(np.uint8)
        gt_for_pred = cv2.resize(gt_semantic,
                                 dsize=(pred_semantic_indices.shape[1], pred_semantic_indices.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
        metrics['semantic_pixel_accuracy'] = float(np.mean(pred_semantic_indices == gt_for_pred))
        metrics['semantic_per_class_accuracy'] = self._per_class_accuracy(pred_semantic_indices, gt_for_pred,
                                                                          self.SEMANTIC_CLASS_NAMES)
        metrics['semantic_per_class_dice'] = self._per_class_dice(pred_semantic_indices, gt_for_pred,
                                                                  self.SEMANTIC_CLASS_NAMES)
        valid_dice = [
            class_metrics['dice'] for class_metrics in metrics['semantic_per_class_dice'].values()
            if class_metrics['dice'] is not None
        ]
        metrics['semantic_mean_dice'] = None if len(valid_dice) == 0 else float(np.mean(valid_dice))

    if 'gt_bev_semantic' in tick_data:
      gt_bev_semantic_raw = tick_data['gt_bev_semantic']
      converter = np.array(self.config.bev_converter, dtype=np.uint8)
      gt_bev_semantic = converter[np.clip(gt_bev_semantic_raw, 0, len(converter) - 1)]
      np.savez_compressed(bev_semantic_path / f'{frame_id}.npz', bev_semantic=gt_bev_semantic)
      converter = np.array(self.config.bev_classes_list, dtype=np.uint8)
      gt_bev_image = converter[np.clip(gt_bev_semantic, 0, len(converter) - 1)]
      cv2.imwrite(str(bev_semantic_path / f'{frame_id}.png'), cv2.cvtColor(gt_bev_image, cv2.COLOR_RGB2BGR))

      if pred_bev_semantic is not None:
        pred_bev_indices = torch.argmax(pred_bev_semantic[0], dim=0).detach().cpu().numpy().astype(np.uint8)
        gt_for_pred = cv2.resize(gt_bev_semantic,
                                 dsize=(pred_bev_indices.shape[1], pred_bev_indices.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
        metrics['bev_semantic_pixel_accuracy'] = float(np.mean(pred_bev_indices == gt_for_pred))
        metrics['bev_semantic_per_class_accuracy'] = self._per_class_accuracy(pred_bev_indices, gt_for_pred,
                                                                              self.BEV_SEMANTIC_CLASS_NAMES)
        metrics['bev_semantic_per_class_dice'] = self._per_class_dice(pred_bev_indices, gt_for_pred,
                                                                      self.BEV_SEMANTIC_CLASS_NAMES)
        valid_dice = [
            class_metrics['dice'] for class_metrics in metrics['bev_semantic_per_class_dice'].values()
            if class_metrics['dice'] is not None
        ]
        metrics['bev_semantic_mean_dice'] = None if len(valid_dice) == 0 else float(np.mean(valid_dice))

    if 'gt_depth' in tick_data:
      gt_depth = tick_data['gt_depth']
      np.savez_compressed(depth_path / f'{frame_id}.npz', depth=gt_depth)
      cv2.imwrite(str(depth_path / f'{frame_id}.png'), cv2.applyColorMap(gt_depth, cv2.COLORMAP_TURBO))

      if pred_depth is not None:
        pred_depth_image = pred_depth[0].detach().float().cpu().numpy()
        pred_depth_image = cv2.resize(pred_depth_image,
                                      dsize=(gt_depth.shape[1], gt_depth.shape[0]),
                                      interpolation=cv2.INTER_LINEAR)
        gt_depth_float = gt_depth.astype(np.float32) / 255.0
        metrics['depth_mae'] = float(np.mean(np.abs(pred_depth_image - gt_depth_float)))

    if 'gt_detection_boxes' in tick_data and pred_bb is not None:
      metrics['detection_map'] = self._detection_map(pred_bb, tick_data['gt_detection_boxes'])

    if metrics:
      with open(metrics_path / f'{frame_id}.json', 'w', encoding='utf-8') as outfile:
        json.dump(metrics, outfile, indent=2)

  def _init_vision_task_gt_paths(self):
    self.vision_task_gt_paths = None
    if self.save_path is None or not self.collect_sensor_data or not self.vision_task_visualization:
      return

    gt_path = pathlib.Path(self.save_path) / 'sensor_data' / 'vision_tasks_gt'
    self.vision_task_gt_paths = {
        'semantic': gt_path / 'semantic',
        'bev_semantic': gt_path / 'bev_semantic',
        'depth': gt_path / 'depth',
        'metrics': gt_path / 'metrics',
    }
    for path in self.vision_task_gt_paths.values():
      path.mkdir(parents=True, exist_ok=True)
