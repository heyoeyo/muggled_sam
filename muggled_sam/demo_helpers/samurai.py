#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import cv2
import numpy as np

from .bounding_boxes import get_one_mask_bounding_box, box_xy1xy2_to_xywh, box_xywh_to_xy1xy2, get_2box_iou

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SimpleSamurai:
    """
    Simplified interpretation of the SAM mask post-processing steps described in the paper:
        "SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory"
        By: Cheng-Yen Yang, Hsiang-Wei Huang, Wenhao Chai, Zhongyu Jiang, Jenq-Neng Hwang
        @ https://arxiv.org/abs/2411.11922

    The basic idea is to include an additional tracking model (a kalman filter) which estimates the
    bounding box of mask predictions from the SAMv2 model. These box predictions are then used to
    influence which of the SAM mask predictions should be used at each time step. The original SAMv2
    model used it's own internal IoU prediction (called 'affinity' in the SAMURAI paper), while SAMURAI
    uses a weighted combination of the SAM IoU along with the IoU between the kalman filter box prediction
    and candidate masks. SAMURAI also ignores memory encodings based on additional scoring thresholds.

    The basic usage is to initialize this class using the initial mask prediction for a tracked object,
    then during tracking, call 'step_video_masking' from this class, rather than using the
    same function on the original SAM2 class.

    This implementation is closer to the implementation described in the paper itself, rather
    than the code associated with the paper, though it does not match either exactly.
    One major difference is that this implementation tracks bounding box values: [x,y,w,h]
    as described in the paper, while the original code tracks: [x,y,ar,h].
    There are also significant differences in process & measurement noise models.
    The original code can be found here:
    https://github.com/yangchris11/samurai/blob/76ba195984892b0d1e3db5d9c9f90bb62175680a/sam2/sam2/utils/kalman_filter.py
    """

    # Weights & thresholds used in samurai algorithm
    alpha_kf = 0.15
    threshold_objscore = 0.0
    threshold_affinity = 0.5
    threshold_kf = 0.5

    # Kalman filter noise scaling factors, these are just kind of made up
    # -> Tracking is not especially sensitive to these values
    # -> For more optimal tracking, these would need to be tuned to match the statistics of the objects being tracked
    value_noise_base = 0.25
    velo_scale_base = 0.1

    # .................................................................................................................

    def __init__(self, initial_mask: Tensor, video_framerate=1, smoothness=0.5):

        # Set up kalman filter to track object bounding box, using state vector: [x, y, w, h, vx, vy, vw, vh]
        # -> State vector is a set of 4 values and corresponding velocities (rates of change)
        num_state_params, num_measured_params = 8, 4
        self._kalman = cv2.KalmanFilter(num_state_params, num_measured_params)
        floattype = np.float32

        # Values update with velocity, velocities are estimated as constant
        # eq: S(t) = T * S(t-1) + noise, where T is matrix below, S is state vector: [x,y,w,h,vx,vy,vw,vh]
        # (this matrix is called the 'state transition model' on kalman filter wikipedia entry)
        dt = 1 / video_framerate
        self._kalman.transitionMatrix = np.array(
            [
                [1, 0, 0, 0, dt, 0, 0, 0],
                [0, 1, 0, 0, 0, dt, 0, 0],
                [0, 0, 1, 0, 0, 0, dt, 0],
                [0, 0, 0, 1, 0, 0, 0, dt],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=floattype,
        )

        # Assume we measure values directly but can't measure velocities
        # eq: Measurement(t) = M * S'(t) + noise, where M is matrix below, S' is the unknown true state
        # (this matrix is called the 'observation model' on kalman wikipedia entry)
        self._kalman.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ],
            dtype=floattype,
        )

        # Assume all values have equal variance, all velocities have equal variance
        # -> This is a bit silly but simple to specify
        # -> This matrix models the inherent variability in the box position/size and rate of change
        value_var = self.value_noise_base
        velo_var = value_var * self.velo_scale_base
        self._kalman.processNoiseCov = np.diag([*[value_var] * 4, *[velo_var] * 4]).astype(floattype)

        # Assume same measurement noise on all values because it's easy to specify
        # -> This noise models the inherent error/variability in the measurements
        # -> Here we scale to a number between 0 and near-infinity in an arbitrary way (tangent function)
        # -> Higher smoothness values make the measurements (i.e. the 'real' mask bounding boxes)
        #    less trustworthy for kalman predictions, so predictions will lag fast mask changes
        smoothness_as_angle = np.clip(smoothness, 0, 0.995) * np.pi * 0.5
        measure_var = value_var * (np.tan(smoothness_as_angle) ** 2)
        self._kalman.measurementNoiseCov = np.diag([*[measure_var] * 4]).astype(floattype)

        # Initialize kalman filter
        is_valid_mask, initial_mask_box = get_one_mask_bounding_box(initial_mask)
        assert is_valid_mask, "Initial mask is empty!"
        init_xywh = box_xy1xy2_to_xywh(*initial_mask_box)
        init_velos = [0] * 4
        init_state = np.array([*init_xywh, *init_velos], dtype=floattype)
        self._kalman.statePre = init_state
        self._kalman.statePost = init_state

    # .................................................................................................................

    def get_best_decoder_results(
        self, mask_predictions: Tensor, iou_predictions: Tensor, exclude_0th_index=True
    ) -> tuple[int, float, tuple]:
        """
        Function used to pick the best mask prediction for tracking.
        Uses a kalman filter to predict where the object bounding box should be
        based on previous frames. The masks are scored based one the overlap
        with the kalman prediction. The 'best' mask is then selected based on
        a weighted sum of the kalman filter score and original SAM IoU score.

        Returns:
            best_mask_index, best_iou, kalman_box_xy1xy2

        - The returned kalman box is formatted as: [(x1, y1), (x2, y2)]
          in 'pixel' units, matching the mask prediction sizing
        """

        # Make sure we don't get batched data
        batch_size, num_masks = mask_predictions.shape[0:2]
        assert batch_size == 1, "Error! No support for batched inputs. Run .update(...) over the batch instead"

        # Get bounding box of each mask
        mask_xy1xy2_list = [get_one_mask_bounding_box(mask_predictions[0, midx]) for midx in range(num_masks)]

        # Compute IoU of each box vs. the kalman prediction
        kal_predict_as_xywh = self._kalman.predict()[:4]
        kal_predict_as_xy1xy2 = box_xywh_to_xy1xy2(*kal_predict_as_xywh)
        kalman_ious = np.zeros(num_masks, dtype=np.float32)
        for midx, (_, mask_xy1xy2) in enumerate(mask_xy1xy2_list):
            kalman_ious[midx] = get_2box_iou(mask_xy1xy2, kal_predict_as_xy1xy2)

        # Perform weighted mask index selection scoring as per samurai paper (see eq. 7)
        affinity_scores = iou_predictions[0].float().cpu().numpy()
        weighted_samurai_score = kalman_ious * self.alpha_kf + affinity_scores * (1.0 - self.alpha_kf)
        if exclude_0th_index:
            weighted_samurai_score[0] = -100

        # Pick the highest score between weighted samurai + original sam predictions
        best_idx = np.argmax(weighted_samurai_score)
        best_iou = weighted_samurai_score[best_idx]

        # Get the best mask bounding box in terms of kalman filter state variables, for measurement update
        is_valid_mask, best_box_as_xy1xy2 = mask_xy1xy2_list[best_idx]
        if is_valid_mask:
            best_box_as_xywh = box_xy1xy2_to_xywh(*best_box_as_xy1xy2)
            kal_measurement = np.array([best_box_as_xywh], dtype=np.float32).T
            self._kalman.correct(kal_measurement)

        return best_idx, best_iou, kal_predict_as_xy1xy2

    # .................................................................................................................

    def step_video_masking(
        self,
        sammodel,
        encoded_image_features_list: list[Tensor],
        prompt_memory_encodings: list[Tensor],
        prompt_object_pointers: list[Tensor],
        previous_memory_encodings: list[Tensor],
        previous_object_pointers: list[Tensor],
    ) -> [bool, Tensor, Tensor, Tensor, tuple[tuple, tuple]]:
        """
        Variant of original SAMv2 video tracking function, instead of using
        SAMv2 IoU predictions to select mask during tracking, this function
        uses SAMURAI to select the best mask.

        Returns:
            is_ok_memory, best_mask_prediction, memory_encoding, best_object_point, xy1xy2_kalman_prediction

        - The returned kalman box is formatted as: [(x1, y1), (x2, y2)] in normalized (0 to 1) units
        """

        with torch.inference_mode():

            # Encode image features with previous memory encodings & object pointer data
            lowres_imgenc, *hires_imgenc = encoded_image_features_list
            memfused_encimg = sammodel.memory_image_fusion(
                lowres_imgenc,
                prompt_memory_encodings,
                prompt_object_pointers,
                previous_memory_encodings,
                previous_object_pointers,
            )

            # Run mask decoder on memory-fused features
            patch_grid_hw = memfused_encimg.shape[2:]
            grid_posenc = sammodel.coordinate_encoder.get_grid_position_encoding(patch_grid_hw)
            mask_preds, iou_preds, obj_ptrs, obj_score = sammodel.mask_decoder(
                [memfused_encimg, *hires_imgenc],
                sammodel.prompt_encoder.create_video_no_prompt_encoding(),
                grid_posenc,
                mask_hint=None,
                blank_promptless_output=False,
            )

            # Use samurai predictions to pick the best mask, instead of just using SAM IoUs
            # (this is the main difference between SAMv2 vs. SAMURAI)
            best_mask_idx, best_samurai_iou, xy1xy2_kal_px = self.get_best_decoder_results(mask_preds, iou_preds)
            best_mask_pred = mask_preds[:, [best_mask_idx], ...]
            best_iou_pred = iou_preds[:, [best_mask_idx], ...]
            best_obj_ptr = obj_ptrs[:, [best_mask_idx], ...]

            # Encode new memory features
            # -> we could skip this if memory check comes back bad
            # -> include anyways for the sake of debugging/inspection
            is_ok_mem = self._check_memory_ok(obj_score, best_iou_pred, best_samurai_iou)
            memory_encoding = sammodel.memory_encoder(lowres_imgenc, best_mask_pred, obj_score)

            # Normalize box prediction coords. for easier usage
            _, _, mask_h, mask_w = best_mask_pred.shape
            xy_norm_scale = 1.0 / np.float32((mask_w - 1, mask_h - 1))
            xy1xy2_kal_norm = [xy * xy_norm_scale for xy in xy1xy2_kal_px]

        return is_ok_mem, best_mask_pred, memory_encoding, best_obj_ptr, xy1xy2_kal_norm

    # .................................................................................................................

    def _check_memory_ok(self, object_score, iou_score, kf_score) -> bool:
        """
        Helper used to decide if a memory encoding should be stored for re-use
        Storage is based on scoring passing several thresholds, similar to
        the original SAMv2, but with extra parameters.

        See section 4.2 and equation (9) from the SAMURAI paper for more details.
        """

        ok_obj = object_score > self.threshold_objscore
        ok_iou = iou_score > self.threshold_affinity
        ok_kf = kf_score > self.threshold_kf

        return ok_obj and ok_iou and ok_kf

    # .................................................................................................................

    @staticmethod
    def draw_box_prediction(frame, xy1xy2_kalman_prediction, line_color=(0, 255, 255), line_thickness=2):
        """Helper used to visualize kalman filter box predictions (from running video steps)"""
        frame_h, frame_w = frame.shape[0:2]
        xy_scale = np.float32((frame_w - 1, frame_h - 1))
        xy1, xy2 = [np.int32(np.round(xy_kal * xy_scale)).tolist() for xy_kal in xy1xy2_kalman_prediction]
        return cv2.rectangle(frame, xy1, xy2, line_color, line_thickness)
