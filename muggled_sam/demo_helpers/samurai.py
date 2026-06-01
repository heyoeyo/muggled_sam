#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np

from .bounding_boxes import get_one_mask_bounding_box, box_xy1xy2_to_xywh, box_xywh_to_xy1xy2, get_2box_iou

# For type hints
from torch import Tensor
from numpy import ndarray


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class MuggledSAMURAI:
    """
    Simplified interpretation of the SAM mask post-processing steps described in the paper:
        "SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory"
        By: Cheng-Yen Yang, Hsiang-Wei Huang, Wenhao Chai, Zhongyu Jiang, Jenq-Neng Hwang
        @ https://arxiv.org/abs/2411.11922

    The basic idea is to include an additional tracking model (a kalman filter) which estimates the
    bounding box of mask predictions from the SAM model. These box predictions are then used to
    influence which of the SAM mask predictions should be used at each time step. The original SAM
    model uses it's own internal IoU prediction (called 'affinity' in the SAMURAI paper), while SAMURAI
    uses a weighted combination of the SAM IoU along with the IoU between the kalman filter box prediction
    and candidate masks. SAMURAI also ignores memory encodings based on additional scoring thresholds.

    This implementation is closer to the implementation described in the paper itself, rather
    than the code associated with the paper, though it does not match either exactly.
    One major difference is that this implementation tracks bounding box values: [x,y,w,h]
    as described in the paper, while the original code tracks: [x,y,aspect-ratio,h].
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

    def __call__(
        self,
        mask_predictions_bnhw: Tensor,
        iou_predictions_bn: Tensor,
        object_score_b: Tensor,
    ) -> tuple[bool, int, tuple[ndarray, ndarray]]:
        """Convenience wrapper around .update(...) function"""
        return self.update(mask_predictions_bnhw, iou_predictions_bn, object_score_b)

    # .................................................................................................................

    def update(
        self,
        mask_predictions_bnhw: Tensor,
        iou_predictions_bn: Tensor,
        object_score_b: Tensor,
    ) -> tuple[bool, int, tuple[ndarray, ndarray]]:
        """
        Main SAMURAI function. This is meant to be called after predicting masks
        on a video frame, but before encoding/storing memory data.

        This function is used to decide which (if any) mask should be stored,
        as an alternative to the original SAM approach of simply picking the
        mask with the highest IoU prediction. It works be independently
        predicting the bounding box of the mask (using a Kalman filter)
        which is then compared to the SAM mask predictions and used to
        select the best match. See section 4.2 and equation (9) from
        the SAMURAI paper for more details.

        The inputs are expected to be the unfiltered results from the SAM model,
        that is the masks should have shape: BxNxHxW, IoU shaped: BxN, score shaped: B

        Returns:
            is_ok_mask, best_mask_index, box_xy1xy2_prediction
            -> is_ok_mask is True if the mask is 'good enough' to use for memory encoding
            -> The best_mask_index is the index of the mask predictions to use (according to SAMURAI)
            -> box_xy1xy2_prediction is formatted as: [(x1, y1), (x2, y2)] in normalized (0 to 1) units
        """

        # Use samurai predictions to pick the best mask, instead of just using SAM IoUs
        best_mask_idx, best_samurai_iou, xy1xy2_kal_px = self._step_kalman(mask_predictions_bnhw, iou_predictions_bn)
        best_iou_pred = iou_predictions_bn[:, [best_mask_idx], ...]

        # Decide if mask should be used for memory encoding
        ok_obj = object_score_b > self.threshold_objscore
        ok_iou = best_iou_pred > self.threshold_affinity
        ok_kf = best_samurai_iou > self.threshold_kf
        is_ok_mask = ok_obj and ok_iou and ok_kf

        # Normalize box prediction coords. for easier usage
        _, _, mask_h, mask_w = mask_predictions_bnhw.shape
        xy_norm_scale = 1.0 / np.float32((mask_w - 1, mask_h - 1))
        xy1xy2_kal_norm = [xy * xy_norm_scale for xy in xy1xy2_kal_px]

        return is_ok_mask, best_mask_idx, xy1xy2_kal_norm

    # .................................................................................................................

    def _step_kalman(
        self, mask_predictions: Tensor, iou_predictions: Tensor, exclude_0th_index=True
    ) -> tuple[int, float, tuple]:
        """
        Function used to advance the SAMURAI tracking predictions.
        Uses a kalman filter to predict where the object bounding box *should*
        be based on previous frames. The masks are scored based on the overlap
        with the kalman prediction. The 'best' mask is then selected based on
        a weighted sum of the kalman filter score and original SAM IoU score.

        Returns:
            best_mask_index, best_iou, kalman_box_xy1xy2

        - The returned kalman box is formatted as: [(x1, y1), (x2, y2)]
          in 'pixel' units, matching the mask prediction sizing
        """

        # Make sure we don't get batched data
        batch_size, num_masks = mask_predictions.shape[0:2]
        assert batch_size == 1, "Error, no support for batched inputs! Each batch item needs it's own SAMURAI instance"

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
        best_samurai_iou = weighted_samurai_score[best_idx]

        # Get the best mask bounding box in terms of kalman filter state variables, for measurement update
        is_valid_mask, best_box_as_xy1xy2 = mask_xy1xy2_list[best_idx]
        if is_valid_mask:
            best_box_as_xywh = box_xy1xy2_to_xywh(*best_box_as_xy1xy2)
            kal_measurement = np.array([best_box_as_xywh], dtype=np.float32).T
            self._kalman.correct(kal_measurement)

        return best_idx, best_samurai_iou, kal_predict_as_xy1xy2

    # .................................................................................................................

    @staticmethod
    def draw_box_prediction(frame, xy1xy2_kalman_prediction, line_color=(0, 255, 255), line_thickness=2):
        """Helper used to visualize kalman filter box predictions (from running video steps)"""
        frame_h, frame_w = frame.shape[0:2]
        xy_scale = np.float32((frame_w - 1, frame_h - 1))
        xy1, xy2 = [np.int32(np.round(xy_kal * xy_scale)).tolist() for xy_kal in xy1xy2_kalman_prediction]
        return cv2.rectangle(frame, xy1, xy2, line_color, line_thickness)
