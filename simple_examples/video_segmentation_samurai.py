#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This is a hack to make this script work from outside the root project folder (without requiring install)
try:
    import lib  # NOQA
except ModuleNotFoundError:
    import os
    import sys

    parent_folder = os.path.dirname(os.path.dirname(__file__))
    if "lib" in os.listdir(parent_folder):
        sys.path.insert(0, parent_folder)
    else:
        raise ImportError("Can't find path to lib folder!")

from time import perf_counter
from collections import deque
import cv2
import numpy as np
import torch
from lib.v2_sam.make_sam_v2 import make_samv2_from_original_state_dict
from lib.demo_helpers.bounding_boxes import get_one_mask_bounding_box, box_xy1xy2_to_xywh, box_xywh_to_xy1xy2

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% SAMURAI Implementation


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

    This implementation is closer to the implementation described in the paper itself, rather
    than the code associated with the paper, though it does not match either exactly.
    One major difference is that this implementation tracks bounding box values: [x,y,w,h]
    as described in the paper, while the original code tracks: [x,y,ar,h].
    There are also significant differences in process & measurement noise models.
    The original code can be found here:
    https://github.com/yangchris11/samurai/blob/76ba195984892b0d1e3db5d9c9f90bb62175680a/sam2/sam2/utils/kalman_filter.py
    """

    # Kalman filter noise scaling factors, these are just kind of made up
    # -> Tracking is not especially sensitive to these values
    # -> For more optimal tracking, these would need to be tuned to match the statistics of the objects being tracked
    value_noise_base = 0.25
    velo_scale_base = 0.1

    # Weights & thresholds used in samurai algorithm
    alpha_kf = 0.15
    threshold_objscore = 0.0
    threshold_affinity = 0.5
    threshold_kf = 0.5

    def __init__(self, initial_mask, video_framerate=1, smoothness=0.5):

        # Set up base kalman filter, using state vector: [x, y, w, h, vx, vy, vw, vh]
        # -> State vectors is a set of 4 values and corresponding velocities (rates of change)
        # -> We assume we can only measure the values
        num_state_params, num_measured_params = 8, 4
        self._kalman = cv2.KalmanFilter(num_state_params, num_measured_params)
        floattype = np.float32

        # Values update with velocity, velocities are estimated as constant
        # eq: S(t) = T * S(t-1), where T is matrix below, S is state vector: [x,y,w,h,vx,vy,vw,vh]
        # (this matrix is called the 'state transition model' on wikipedia entry)
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
        # eq: Measurement(t) = M * S'(t), where M is matrix below, S' is true state we're estimating
        # (this matrix is called the 'observation model' on wikipedia entry)
        self._kalman.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ],
            dtype=floattype,
        )

        # Also assume all values have equal variance, all velocities have equal variance
        # -> This is a bit silly but simple to specify
        value_var = self.value_noise_base
        velo_var = value_var * self.velo_scale_base
        self._kalman.processNoiseCov = np.diag([*[value_var] * 4, *[velo_var] * 4]).astype(floattype)

        # Assume same measurement noise on all values because it's easy to specify
        # -> Smoothness is scaled a number between 0 and near-infinity in an arbitrary way (using tangent function)
        # -> Higher smoothness values make the measurements (i.e. the 'real' mask bounding boxes)
        #    less trustworthy for kalman predictions, so predictions will lag fast mask changes
        smoothness = np.clip(smoothness, 0, 0.995)
        measure_var = value_var * (np.tan(smoothness * np.pi * 0.5) ** 2)
        self._kalman.measurementNoiseCov = np.diag([*[measure_var] * 4]).astype(floattype)

        # Initialize kalman filter
        is_valid_mask, initial_mask_box = get_one_mask_bounding_box(initial_mask)
        assert is_valid_mask, "Initial mask is empty!"
        init_xywh = box_xy1xy2_to_xywh(*initial_mask_box)
        init_velos = [0] * 4
        init_state = np.array([*init_xywh, *init_velos], dtype=floattype)
        self._kalman.statePre = init_state
        self._kalman.statePost = init_state

    def get_best_decoder_results(
        self, mask_predictions, iou_predictions, exclude_0th_index=True
    ) -> tuple[int, float, tuple[tuple, tuple]]:
        """
        Function used to pick the best mask prediction for tracking.
        Uses a kalman filter to predict where the object bounding box should be
        based on previous frames. The masks are scored based one the overlap
        with the kalman prediction. The 'best' mask is then selected based on
        a weighted sum of the kalman filter score and original SAM IoU score.

        Returns:
            best_mask_index, best_iou, kalman_box_xy1xy2

        - The returned kalman box is formatted as: [(x1, y1), (x2, y2)]
        """

        # Make sure we don't get batched data
        batch_size, num_masks = mask_predictions.shape[0:2]
        assert batch_size == 1, "Error! No support for batched inputs. Run .update(...) over the batch instead"

        # Get bounding box of each mask
        mask_xy1xy2_list = [get_one_mask_bounding_box(mask_predictions[0, midx]) for midx in range(num_masks)]

        # Compute IoU of each box vs. the kalman prediction
        kal_predict_as_xywh = self._kalman.predict()[:4]
        (x1_kal, y1_kal), (x2_kal, y2_kal) = box_xywh_to_xy1xy2(*kal_predict_as_xywh)
        kalman_ious = np.zeros(num_masks, dtype=np.float32)
        for midx, (_, ((x1, y1), (x2, y2))) in enumerate(mask_xy1xy2_list):

            # Compute intersection area between kalman predicted box & mask box
            inter_w = min(x2, x2_kal) - max(x1, x1_kal)
            inter_h = min(y2, y2_kal) - max(y1, y1_kal)
            inter_area = max(0, inter_w) * max(0, inter_h)

            # Compute union of areas
            kal_area = (x2_kal - x1_kal) * (y2_kal - y1_kal)
            mask_area = (x2 - x1) * (y2 - y1)
            union_area = kal_area + mask_area - inter_area

            # Record the IoU for each mask, so we can check which is best
            kalman_mask_iou = inter_area / union_area if union_area > 0.001 else 0
            kalman_ious[midx] = kalman_mask_iou

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

        # Include kalman prediction in output, for debugging/visualization
        kalman_prediction_xy1xy2 = (np.array((x1_kal, y1_kal)), np.array((x2_kal, y2_kal)))
        return best_idx, best_iou, kalman_prediction_xy1xy2

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
        """

        with torch.inference_mode():

            # Encode image features with previous memory encodings & object pointer data
            lowres_imgenc, *hires_imgenc = encoded_image_features_list
            memfused_encimg = sammodel.memory_fusion(
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
            best_mask_idx, best_samurai_iou, xy1xy2_kal = self.get_best_decoder_results(mask_preds, iou_preds)
            best_mask_pred = mask_preds[:, [best_mask_idx], ...]
            best_iou_pred = iou_preds[:, [best_mask_idx], ...]
            best_obj_ptr = obj_ptrs[:, [best_mask_idx], ...]

            # Encode new memory features
            # -> we could skip this if memory check comes back bad
            # -> but included here anyways for the sake of debugging/inspection
            memory_encoding = sammodel.memory_encoder(lowres_imgenc, best_mask_pred, obj_score)

        is_ok_mem = self._check_memory_ok(obj_score, best_iou_pred, best_samurai_iou)

        return is_ok_mem, best_mask_pred, memory_encoding, best_obj_ptr, xy1xy2_kal

    def _check_memory_ok(self, object_score, iou_score, kf_score) -> bool:
        """
        Helper used to decide if a memory encoding should be store for re-use
        Storage is based on scoring passing several thresholds, similar to
        the original SAMv2, but with extra parameters.
        """

        ok_obj = object_score > self.threshold_objscore
        ok_iou = iou_score > self.threshold_affinity
        ok_kf = kf_score > self.threshold_kf

        return ok_obj and ok_iou and ok_kf


# ---------------------------------------------------------------------------------------------------------------------
# %% Demo

# Define pathing & device usage
initial_frame_index = 0
video_path = "/path/to/video.mp4"
model_path = "/path/to/samv2_model.pth"
device, dtype = "cpu", torch.float32
if torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16

# Define prompts using xy coordinates normalized between 0 and 1
boxes_tlbr_norm_list = []  # Example:  [[(0.25, 0.25), (0.75, 0.75)]]
fg_xy_norm_list = [(0.5, 0.5)]
bg_xy_norm_list = []
imgenc_config_dict = {"max_side_length": 1024, "use_square_sizing": True}

# Read first frame
vcap = cv2.VideoCapture(video_path)
vcap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)  # See: https://github.com/opencv/opencv/issues/26795
vcap.set(cv2.CAP_PROP_POS_FRAMES, initial_frame_index)
ok_frame, first_frame = vcap.read()
assert ok_frame, f"Could not read frames from video: {video_path}"

# Set up model
print("Loading model...")
model_config_dict, sammodel = make_samv2_from_original_state_dict(model_path)
sammodel.to(device=device, dtype=dtype)

# Use initial prompt to begin segmenting an object
init_encoded_img, _, _ = sammodel.encode_image(first_frame, **imgenc_config_dict)
init_mask, init_mem, init_ptr = sammodel.initialize_video_masking(
    init_encoded_img, boxes_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list, mask_index_select=None
)

# Set up data storage for prompted object (repeat this for each unique object)
prompt_mems = deque([init_mem])
prompt_ptrs = deque([init_ptr])
prev_mems = deque([], maxlen=6)
prev_ptrs = deque([], maxlen=15)

# Initialize SAMURAI
samurai = SimpleSamurai(init_mask)

# Process video frames
stack_func = np.hstack if first_frame.shape[0] > first_frame.shape[1] else np.vstack
close_keycodes = {27, ord("q")}  # Esc or q to close
try:
    is_webcam = isinstance(video_path, int)
    total_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 100_000
    for frame_idx in range(1 + initial_frame_index, total_frames):

        # Read frames
        ok_frame, frame = vcap.read()
        if not ok_frame:
            break

        # Process video frames with model & added SAMURAI post-processing
        t1 = perf_counter()
        encoded_imgs_list, _, _ = sammodel.encode_image(frame, **imgenc_config_dict)
        is_mem_ok, best_mask_pred, mem_enc, obj_ptr, xy1xy2_kal = samurai.step_video_masking(
            sammodel, encoded_imgs_list, prompt_mems, prompt_ptrs, prev_mems, prev_ptrs
        )
        t2 = perf_counter()
        print(f"Took {round(1000 * (t2 - t1))} ms")

        # Only store memory that is considered valid by SAMURAI
        if is_mem_ok:
            prev_mems.appendleft(mem_enc)
            prev_ptrs.appendleft(obj_ptr)
        else:
            print("SAMURAI rejected memory!")

        # Create mask for display
        dispres_mask = torch.nn.functional.interpolate(
            best_mask_pred,
            size=frame.shape[0:2],
            mode="bilinear",
            align_corners=False,
        )
        disp_mask = ((dispres_mask > 0.0).byte() * 255).cpu().numpy().squeeze()
        disp_mask = cv2.cvtColor(disp_mask, cv2.COLOR_GRAY2BGR)

        # Draw SAMURAI kalman filter prediction over top of selected mask (just for display/debugging!)
        w_disp, w_mask = dispres_mask.shape[3], best_mask_pred.shape[3]
        h_disp, h_mask = dispres_mask.shape[2], best_mask_pred.shape[2]
        xyscale = np.float32((w_disp / w_mask, h_disp / h_mask))
        xy1_kal, xy2_kal = xy1xy2_kal
        xy1, xy2 = [np.int32(np.round(xy_kal * xyscale)).tolist() for xy_kal in xy1xy2_kal]
        mask_color = (0, 255, 0) if is_mem_ok else (0, 0, 255)
        disp_mask = cv2.rectangle(disp_mask, xy1, xy2, mask_color, 2)

        # Show frame and mask, side-by-side
        sidebyside = stack_func((frame, disp_mask))
        cv2.imshow("SAMURAI Segmentation Result - q to quit", cv2.resize(sidebyside, dsize=None, fx=0.5, fy=0.5))
        keypress = cv2.waitKey(1) & 0xFF
        if keypress in close_keycodes:
            break

except KeyboardInterrupt:
    print("Closed by ctrl+c!")

except Exception as err:
    raise err

finally:
    vcap.release()
    cv2.destroyAllWindows()
