from .operators import *
import torch, json, pandas, os


def draw_pose_on_canvas(canvas, keypoints, subset=None, draw_labels=True, is_normalized=None):
    import cv2, math, numpy as np
    H, W, C = canvas.shape
    
    # Colors for different body parts (BGR format)
    POSE_COLORS = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
        [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
        [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
        [255, 0, 255], [255, 0, 170], [255, 0, 85]
    ]
    
    candidate = keypoints.copy() # (18, 2)
    # print(np.max(keypoints), np.min(keypoints), keypoints.shape, keypoints);assert 0
    
    if subset is None:
        assert 0
        if is_normalized is None:
            valid_kps = keypoints[np.logical_and(keypoints[:, 0] > 0, keypoints[:, 1] > 0)]
            max_coord = np.max(valid_kps) if len(valid_kps) > 0 else 0
            is_normalized = max_coord <= 1.5 
            
        subset_arr = np.zeros((1, 18)) - 1
        for i in range(18):
            x, y = keypoints[i]
            valid = False
            if is_normalized:
                if 0 <= x <= 1 and 0 <= y <= 1 and (x > 0 or y > 0):
                    valid = True
            else:
                if x > 0 and y > 0:
                    valid = True
            if valid:
                subset_arr[0][i] = i
    else:
        subset_arr = np.array(subset)
        if subset_arr.ndim == 1:
            subset_arr = subset_arr[np.newaxis, :]
            
        # print(is_normalized is None, is_normalized) # False Fasle
        if is_normalized is None:
            valid_indices = subset_arr[0] != -1
            if np.any(valid_indices):
                valid_mask = np.zeros(18, dtype=bool)
                for i in range(18):
                    if subset_arr[0][i] != -1:
                        valid_mask[i] = True
                valid_kps = candidate[valid_mask]
                print(len(candidate), len(valid_kps));assert 0
                max_coord = np.max(valid_kps) if len(valid_kps) > 0 else 0
                is_normalized = max_coord <= 1.5
            else:
                is_normalized = True

    stickwidth = 4
    limbSeq = [
        [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9],
        [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16],
        [0, 15], [15, 17], [2, 16], [5, 17]
    ]
    
    for i in range(len(limbSeq)):
        for n in range(len(subset_arr)):
            limb_indices = np.array(limbSeq[i])  # e.g., [1, 2]
            # Check if both keypoints of this limb are valid
            if subset_arr[n][limb_indices[0]] == -1 or subset_arr[n][limb_indices[1]] == -1:
                continue
            
            # Use limb_indices directly to access candidate (not subset values)
            # if is_normalized:
            #     Y = candidate[limb_indices, 0] * float(W)
            #     X = candidate[limb_indices, 1] * float(H)
            # else:
            Y = candidate[limb_indices, 0] * float(W)
            X = candidate[limb_indices, 1] * float(H)
                
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )
            cv2.fillConvexPoly(canvas, polygon, POSE_COLORS[i % len(POSE_COLORS)])

    # Debug: count valid points and collect their coordinates
    valid_points_count = 0
    valid_points_coords = []
    
    for i in range(18):
        for n in range(len(subset_arr)):
            # Check if keypoint i is valid (subset[i] != -1)
            if subset_arr[n][i] == -1:
                continue
            
            # Use i directly to access candidate (candidate is already indexed by keypoint index)
            x, y = candidate[i][0:2]
            
            if is_normalized:
                px = int(x * W)
                py = int(y * H)
            else:
                px = int(x)
                py = int(y)
            
            # Debug: collect valid point info
            valid_points_count += 1
            valid_points_coords.append((i, px, py))
                
            cv2.circle(canvas, (px, py), 4, POSE_COLORS[i % len(POSE_COLORS)], thickness=-1)
                        
    return canvas


def draw_3d_keypoints_on_canvas(canvas, keypoints, draw_labels=True):
    """
    Draw 134-point SMPL+DWPose keypoints on canvas.
    keypoints: (134, 2) or (134, 3) where [:, :2] are normalized coords [0,1], [:, 2] is validity.
    Color scheme: Red=SMPL joints (24), Blue=face (68), Yellow=hands (42)
    """
    import cv2, numpy as np
    H, W, C = canvas.shape

    NUM_SMPL = 24
    NUM_FACE = 68
    # NUM_HAND_L = 21, NUM_HAND_R = 21

    # SMPL skeleton limb connections (0-indexed, 24 joints)
    SMPL_LIMBS = [
        (0, 1), (0, 2), (0, 3),       # pelvis -> left hip, right hip, spine1
        (1, 4), (2, 5), (3, 6),       # hips -> knees, spine1 -> spine2
        (4, 7), (5, 8), (6, 9),       # knees -> ankles, spine2 -> spine3
        (7, 10), (8, 11), (9, 12),    # ankles -> feet, spine3 -> neck
        (12, 13), (12, 14),            # neck -> head, neck -> left shoulder area
        (9, 13), (9, 14),             # spine3 -> shoulders
        (13, 16), (14, 17),           # shoulders -> elbows
        (16, 18), (17, 19),           # elbows -> wrists
        (18, 20), (19, 21),           # wrists -> hands
    ]

    # Face contour connections (simplified: jaw + eyebrows + nose + eyes + mouth)
    FACE_JAW = list(zip(range(0, 16), range(1, 17)))  # 0-16
    FACE_EYEBROW_L = list(zip(range(17, 21), range(18, 22)))  # 17-21
    FACE_EYEBROW_R = list(zip(range(22, 26), range(23, 27)))  # 22-26
    FACE_NOSE = list(zip(range(27, 30), range(28, 31)))  # 27-30
    FACE_NOSE_TIP = list(zip(range(31, 35), range(32, 36)))  # 31-35
    FACE_EYE_L = list(zip(range(36, 41), range(37, 42))) + [(41, 36)]  # 36-41
    FACE_EYE_R = list(zip(range(42, 47), range(43, 48))) + [(47, 42)]  # 42-47
    FACE_MOUTH_O = list(zip(range(48, 59), range(49, 60))) + [(59, 48)]  # 48-59
    FACE_MOUTH_I = list(zip(range(60, 67), range(61, 68))) + [(67, 60)]  # 60-67
    FACE_LIMBS = FACE_JAW + FACE_EYEBROW_L + FACE_EYEBROW_R + FACE_NOSE + FACE_NOSE_TIP + FACE_EYE_L + FACE_EYE_R + FACE_MOUTH_O + FACE_MOUTH_I

    # Hand connections (21 points each, standard hand skeleton)
    HAND_LIMBS_BASE = [
        (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),       # index
        (0, 9), (9, 10), (10, 11), (11, 12),  # middle
        (0, 13), (13, 14), (14, 15), (15, 16),# ring
        (0, 17), (17, 18), (18, 19), (19, 20),# pinky
        (5, 9), (9, 13), (13, 17),             # palm
    ]

    has_validity = keypoints.shape[1] >= 3

    def is_valid(idx):
        if has_validity:
            return keypoints[idx, 2] > 0.5
        x, y = keypoints[idx, 0], keypoints[idx, 1]
        return (x > 0 or y > 0) and (0 <= x <= 1) and (0 <= y <= 1)

    def px(idx):
        return int(keypoints[idx, 0] * W), int(keypoints[idx, 1] * H)

    def draw_limbs(limbs, offset, color, thickness=2):
        for (a, b) in limbs:
            ai, bi = a + offset, b + offset
            if ai >= len(keypoints) or bi >= len(keypoints):
                continue
            if not (is_valid(ai) and is_valid(bi)):
                continue
            p1, p2 = px(ai), px(bi)
            cv2.line(canvas, p1, p2, color, thickness)

    def draw_points(start, count, color, radius):
        for i in range(start, min(start + count, len(keypoints))):
            if not is_valid(i):
                continue
            p = px(i)
            cv2.circle(canvas, p, radius, color, -1)

    # Draw SMPL skeleton limbs (red)
    draw_limbs(SMPL_LIMBS, 0, (255, 0, 0), 2)
    # Draw face limbs (blue)
    draw_limbs(FACE_LIMBS, NUM_SMPL, (0, 100, 255), 1)
    # Draw left hand limbs (yellow)
    draw_limbs(HAND_LIMBS_BASE, NUM_SMPL + NUM_FACE, (255, 255, 0), 1)
    # Draw right hand limbs (yellow)
    draw_limbs(HAND_LIMBS_BASE, NUM_SMPL + NUM_FACE + 21, (255, 255, 0), 1)

    # Draw keypoints
    draw_points(0, NUM_SMPL, (255, 0, 0), 5)                       # SMPL joints: red
    draw_points(NUM_SMPL, NUM_FACE, (0, 100, 255), 2)              # Face: blue
    draw_points(NUM_SMPL + NUM_FACE, 21, (255, 255, 0), 3)         # Left hand: yellow
    draw_points(NUM_SMPL + NUM_FACE + 21, 21, (255, 255, 0), 3)    # Right hand: yellow

    return canvas


def draw_pose2_keypoints_on_canvas(canvas, keypoints, draw_labels=True):
    """
    Draw 128-point DWPose full-body keypoints on canvas.
    keypoints: (128, 2) or (128, 3) where [:, :2] are normalized coords [0,1], [:, 2] is validity.
    Layout: 18 body + 68 face + 42 hands (21 left + 21 right)
    Color scheme: Green=body joints (18), Blue=face (68), Yellow=hands (42)
    """
    import cv2, numpy as np
    H, W, C = canvas.shape

    NUM_BODY = 18
    NUM_FACE = 68
    # NUM_HAND_L = 21, NUM_HAND_R = 21

    # DWPose 18-joint body limb connections (0-indexed)
    BODY_LIMBS = [
        (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9),
        (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16),
        (0, 15), (15, 17)
    ]

    # Face contour connections (same as 3D version)
    FACE_JAW = list(zip(range(0, 16), range(1, 17)))
    FACE_EYEBROW_L = list(zip(range(17, 21), range(18, 22)))
    FACE_EYEBROW_R = list(zip(range(22, 26), range(23, 27)))
    FACE_NOSE = list(zip(range(27, 30), range(28, 31)))
    FACE_NOSE_TIP = list(zip(range(31, 35), range(32, 36)))
    FACE_EYE_L = list(zip(range(36, 41), range(37, 42))) + [(41, 36)]
    FACE_EYE_R = list(zip(range(42, 47), range(43, 48))) + [(47, 42)]
    FACE_MOUTH_O = list(zip(range(48, 59), range(49, 60))) + [(59, 48)]
    FACE_MOUTH_I = list(zip(range(60, 67), range(61, 68))) + [(67, 60)]
    FACE_LIMBS = FACE_JAW + FACE_EYEBROW_L + FACE_EYEBROW_R + FACE_NOSE + FACE_NOSE_TIP + FACE_EYE_L + FACE_EYE_R + FACE_MOUTH_O + FACE_MOUTH_I

    # Hand connections (21 points each)
    HAND_LIMBS_BASE = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]

    has_validity = keypoints.shape[1] >= 3

    def is_valid(idx):
        if idx >= len(keypoints):
            return False
        if has_validity:
            return keypoints[idx, 2] > 0.5
        x, y = keypoints[idx, 0], keypoints[idx, 1]
        return (x > 0 or y > 0) and (0 <= x <= 1) and (0 <= y <= 1)

    def px(idx):
        return int(keypoints[idx, 0] * W), int(keypoints[idx, 1] * H)

    def draw_limbs(limbs, offset, color, thickness=2):
        for (a, b) in limbs:
            ai, bi = a + offset, b + offset
            if ai >= len(keypoints) or bi >= len(keypoints):
                continue
            if not (is_valid(ai) and is_valid(bi)):
                continue
            p1, p2 = px(ai), px(bi)
            cv2.line(canvas, p1, p2, color, thickness)

    def draw_points(start, count, color, radius):
        for i in range(start, min(start + count, len(keypoints))):
            if not is_valid(i):
                continue
            p = px(i)
            cv2.circle(canvas, p, radius, color, -1)

    # Draw body skeleton limbs (green)
    draw_limbs(BODY_LIMBS, 0, (0, 255, 0), 2)
    # Draw face limbs (blue)
    draw_limbs(FACE_LIMBS, NUM_BODY, (0, 100, 255), 1)
    # Draw left hand limbs (yellow)
    draw_limbs(HAND_LIMBS_BASE, NUM_BODY + NUM_FACE, (255, 255, 0), 1)
    # Draw right hand limbs (yellow)
    draw_limbs(HAND_LIMBS_BASE, NUM_BODY + NUM_FACE + 21, (255, 255, 0), 1)

    # Draw keypoints
    draw_points(0, NUM_BODY, (0, 255, 0), 5)                        # Body joints: green
    draw_points(NUM_BODY, NUM_FACE, (0, 100, 255), 2)               # Face: blue
    draw_points(NUM_BODY + NUM_FACE, 21, (255, 255, 0), 3)          # Left hand: yellow
    draw_points(NUM_BODY + NUM_FACE + 21, 21, (255, 255, 0), 3)     # Right hand: yellow

    return canvas


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
        sanity_check=False,  # Add sanity check flag
        sanity_check_dir="sanity_check_output",  # Output directory for sanity check
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.sanity_check = sanity_check
        self.sanity_check_dir = sanity_check_dir
        if self.sanity_check:
            os.makedirs(self.sanity_check_dir, exist_ok=True)
        self.load_metadata(metadata_path)
    
    def sanity_check_save_data(self, data, data_id):
        """
        Save and visualize all data fields for sanity check.
        Supports: images, videos (list of images), tensors, keypoints, text, etc.
        """
        import numpy as np
        import imageio
        from PIL import Image
        
        # Create a subdirectory for this data point
        data_dir = os.path.join(self.sanity_check_dir, f"data_{data_id}")
        os.makedirs(data_dir, exist_ok=True)
        
        # Save metadata as JSON
        metadata_to_save = {}
        
        for key, value in data.items():
            try:
                if value is None:
                    metadata_to_save[key] = "None"
                    continue
                
                # Handle PIL Image
                if isinstance(value, Image.Image):
                    save_path = os.path.join(data_dir, f"{key}.png")
                    value.save(save_path)
                    metadata_to_save[key] = f"Image saved: {key}.png, size={value.size}"
                
                # Handle list of PIL Images (video frames)
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], Image.Image):
                    # Save as video
                    frames = [np.array(img) for img in value]
                    video_path = os.path.join(data_dir, f"{key}.mp4")
                    imageio.mimsave(video_path, frames, fps=24)
                    # Also save first, middle, last frames as images
                    value[0].save(os.path.join(data_dir, f"{key}_frame_0.png"))
                    value[len(value)//2].save(os.path.join(data_dir, f"{key}_frame_mid.png"))
                    value[-1].save(os.path.join(data_dir, f"{key}_frame_last.png"))
                    metadata_to_save[key] = f"Video saved: {key}.mp4, num_frames={len(value)}, size={value[0].size}"
                
                # Handle torch.Tensor
                elif isinstance(value, torch.Tensor):
                    tensor_info = f"Tensor: shape={list(value.shape)}, dtype={value.dtype}"
                    
                    # Depth keypoints: [T, N, 2] where [:,:,0]=depth, [:,:,1]=validity
                    if key == "depth_keypoints" and value.ndim == 3 and value.shape[2] == 2:
                        dk = value.cpu().numpy()
                        np.save(os.path.join(data_dir, f"{key}.npy"), dk)
                        T_dk, N_dk, _ = dk.shape
                        depth_vals = dk[:, :, 0]
                        valid_mask = dk[:, :, 1] > 0.5
                        total_points = valid_mask.size
                        valid_count = valid_mask.sum()
                        valid_ratio = valid_count / max(total_points, 1)
                        valid_depths = depth_vals[valid_mask]
                        if len(valid_depths) > 0:
                            d_min, d_max, d_mean = valid_depths.min(), valid_depths.max(), valid_depths.mean()
                        else:
                            d_min, d_max, d_mean = 0.0, 0.0, 0.0
                        tensor_info += (f" -> Depth keypoints: T={T_dk}, N={N_dk}, "
                                        f"valid_ratio={valid_ratio:.2%}, "
                                        f"depth_range=[{d_min:.4f}, {d_max:.4f}], "
                                        f"depth_mean={d_mean:.4f}")
                        tensor_info += f", raw saved: {key}.npy"
                        
                        # Visualize depth keypoints overlaid on control video
                        vis_frames = self._visualize_depth_keypoints(dk, data=data)
                        if vis_frames:
                            video_path = os.path.join(data_dir, f"{key}_vis.mp4")
                            imageio.mimsave(video_path, vis_frames, fps=24)
                            tensor_info += f", depth vis saved: {key}_vis.mp4"
                    
                    # Pre-computed depth indices: [f, h, w] int64 from LoadDepthKeypoints2
                    elif key == "depth_keypoints2" and value.ndim == 3:
                        di = value.cpu().numpy()  # [f, h, w] int64 depth indices
                        np.save(os.path.join(data_dir, f"{key}.npy"), di)
                        f_dm, h_dm, w_dm = di.shape
                        d_min, d_max, d_mean = di.min(), di.max(), di.mean()
                        tensor_info += (f" -> Depth indices (pre-quantized): f={f_dm}, h={h_dm}, w={w_dm}, "
                                        f"index_range=[{d_min}, {d_max}], "
                                        f"index_mean={d_mean:.1f}")
                        tensor_info += f", raw saved: {key}.npy"
                    
                    # If it looks like keypoints (T, N, D) where N=18 (DWPose) or N=128 (DWPose full-body) or N=134 (SMPL+DWPose)
                    elif value.ndim == 3 and value.shape[1] in (18, 128, 134) and value.shape[2] >= 3:
                        kps = value.cpu().numpy()
                        # Save keypoints video with background
                        vis_frames_with_bg = self._visualize_keypoints(kps, data=data, use_background=True)
                        if vis_frames_with_bg:
                            video_path = os.path.join(data_dir, f"{key}_vis_with_bg.mp4")
                            imageio.mimsave(video_path, vis_frames_with_bg, fps=24)
                            tensor_info += f" -> Keypoints video with bg saved: {key}_vis_with_bg.mp4"
                        # Save keypoints video without background
                        vis_frames_no_bg = self._visualize_keypoints(kps, data=data, use_background=False)
                        if vis_frames_no_bg:
                            video_path = os.path.join(data_dir, f"{key}_vis_no_bg.mp4")
                            imageio.mimsave(video_path, vis_frames_no_bg, fps=24)
                            tensor_info += f", Keypoints video without bg saved: {key}_vis_no_bg.mp4"
                        
                        # Also save raw keypoints as numpy
                        np.save(os.path.join(data_dir, f"{key}.npy"), kps)
                        tensor_info += f", raw saved: {key}.npy"
                    
                    # If it looks like an image tensor (C, H, W) or (H, W, C)
                    elif value.ndim == 3:
                        if value.shape[0] in [1, 3, 4]:  # (C, H, W)
                            img_np = value.permute(1, 2, 0).cpu().numpy()
                        else:  # (H, W, C)
                            img_np = value.cpu().numpy()
                        
                        if img_np.max() <= 1.0:
                            img_np = (img_np * 255).astype(np.uint8)
                        else:
                            img_np = img_np.astype(np.uint8)
                        
                        if img_np.shape[-1] == 1:
                            img_np = img_np.squeeze(-1)
                        
                        save_path = os.path.join(data_dir, f"{key}.png")
                        Image.fromarray(img_np).save(save_path)
                        tensor_info += f" -> Image saved: {key}.png"
                    
                    # If it looks like a video tensor (T, C, H, W)
                    elif value.ndim == 4:
                        T, C, H, W = value.shape
                        if C in [1, 3, 4]:
                            frames = []
                            for t in range(T):
                                frame = value[t].permute(1, 2, 0).cpu().numpy()
                                if frame.max() <= 1.0:
                                    frame = (frame * 255).astype(np.uint8)
                                else:
                                    frame = frame.astype(np.uint8)
                                if frame.shape[-1] == 1:
                                    frame = frame.squeeze(-1)
                                frames.append(frame)
                            video_path = os.path.join(data_dir, f"{key}.mp4")
                            imageio.mimsave(video_path, frames, fps=24)
                            tensor_info += f" -> Video saved: {key}.mp4"
                    
                    metadata_to_save[key] = tensor_info
                
                # Handle numpy array
                elif isinstance(value, np.ndarray):
                    np_info = f"ndarray: shape={list(value.shape)}, dtype={value.dtype}"
                    np.save(os.path.join(data_dir, f"{key}.npy"), value)
                    np_info += f" -> saved: {key}.npy"
                    
                    # Visualize if it looks like keypoints (18-point DWPose, 128-point DWPose full-body, or 134-point SMPL+DWPose)
                    if value.ndim == 3 and value.shape[1] in (18, 128, 134):
                        # Save keypoints video with background
                        vis_frames_with_bg = self._visualize_keypoints(value, data=data, use_background=True)
                        if vis_frames_with_bg:
                            video_path = os.path.join(data_dir, f"{key}_vis_with_bg.mp4")
                            imageio.mimsave(video_path, vis_frames_with_bg, fps=24)
                            np_info += f", keypoints video with bg: {key}_vis_with_bg.mp4"
                        # Save keypoints video without background
                        vis_frames_no_bg = self._visualize_keypoints(value, data=data, use_background=False)
                        if vis_frames_no_bg:
                            video_path = os.path.join(data_dir, f"{key}_vis_no_bg.mp4")
                            imageio.mimsave(video_path, vis_frames_no_bg, fps=24)
                            np_info += f", keypoints video without bg: {key}_vis_no_bg.mp4"
                    
                    metadata_to_save[key] = np_info
                
                # Handle string (text prompt, path, etc.)
                elif isinstance(value, str):
                    metadata_to_save[key] = f"String: {value[:500]}{'...' if len(value) > 500 else ''}"
                
                # Handle numeric types
                elif isinstance(value, (int, float)):
                    metadata_to_save[key] = f"Number: {value}"
                
                # Handle list of other types
                elif isinstance(value, list):
                    metadata_to_save[key] = f"List: len={len(value)}, types={[type(v).__name__ for v in value[:5]]}"
                
                # Handle dict
                elif isinstance(value, dict):
                    metadata_to_save[key] = f"Dict: keys={list(value.keys())}"
                
                else:
                    metadata_to_save[key] = f"Unknown type: {type(value).__name__}"
                    
            except Exception as e:
                metadata_to_save[key] = f"Error saving {key}: {str(e)}"
        
        # Save metadata
        with open(os.path.join(data_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"[Sanity Check] Data {data_id} saved to {data_dir}")
    
    def _visualize_keypoints(self, kps, data=None, canvas_size=(512, 512), use_background=True):
        """
        Helper function to visualize keypoints as video frames.
        Supports both 18-point DWPose and 134-point SMPL+DWPose keypoints.
        
        Args:
            kps: keypoints array with shape (T, N, D) where N=18 or N=134,
                 [:, :, :2] is coordinates, [:, :, 2] is validity (optional)
            data: optional data dict to get background video from control_video
            canvas_size: default canvas size if no background video is found
            use_background: if True, use control_video as background; if False, use black background
        """
        import numpy as np
        import cv2
        
        H, W = canvas_size
        num_keypoints = kps.shape[1]  # 18 or 134
        vis_frames = []
        background_frames = None
        
        try:
            # Try to find background video for visualization
            if data is not None:
                for bg_key in ['control_video', 'control_video_3dpose', 'control_video2']:
                    if bg_key in data:
                        bg_data = data[bg_key]
                        if isinstance(bg_data, list) and len(bg_data) > 0:
                            # List of PIL Images
                            if hasattr(bg_data[0], 'size'):
                                W, H = bg_data[0].size
                                if use_background:
                                    background_frames = [np.array(img) for img in bg_data]
                                break
                        elif isinstance(bg_data, torch.Tensor) and bg_data.ndim >= 4:
                            # Tensor (T, C, H, W)
                            if bg_data.ndim == 4:
                                H, W = bg_data.shape[-2], bg_data.shape[-1]
                                if use_background:
                                    bg_np = bg_data.permute(0, 2, 3, 1).cpu().numpy()
                                    if bg_np.max() <= 1.0:
                                        bg_np = (bg_np * 255).astype(np.uint8)
                                    else:
                                        bg_np = bg_np.astype(np.uint8)
                                    background_frames = list(bg_np)
                                break
            
            for frame_idx in range(len(kps)):
                # Prepare canvas
                if use_background and background_frames is not None and frame_idx < len(background_frames):
                    canvas = background_frames[frame_idx].copy()
                    if canvas.shape[0] != H or canvas.shape[1] != W:
                        canvas = cv2.resize(canvas, (W, H))
                else:
                    canvas = np.zeros((H, W, 3), dtype=np.uint8)
                
                frame_kps = kps[frame_idx]
                
                if num_keypoints == 134:
                    # 134-point SMPL+DWPose: use dedicated drawing function
                    # frame_kps[:, :2] are normalized coords [0,1], pass as-is
                    canvas = draw_3d_keypoints_on_canvas(canvas, frame_kps)
                elif num_keypoints == 128:
                    # 128-point DWPose full-body: 18 body + 68 face + 42 hands
                    canvas = draw_pose2_keypoints_on_canvas(canvas, frame_kps)
                else:
                    # 18-point DWPose: use original drawing function
                    candidate = frame_kps[:, :2]  # (18, 2)
                    
                    if frame_kps.shape[1] >= 3:
                        subset_raw = frame_kps[:, 2]  # (18,)
                        subset = np.where(subset_raw > 0, np.arange(18), -1).astype(float)
                    else:
                        subset = np.arange(18).astype(float)
                    
                    canvas = draw_pose_on_canvas(canvas, candidate, subset=subset, is_normalized=False)
                
                # Add frame info
                label = f"Frame: {frame_idx} ({num_keypoints}pts)"
                cv2.putText(canvas, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Convert BGR to RGB for imageio
                vis_frames.append(canvas[:, :, ::-1])
                
        except Exception as e:
            print(f"Error visualizing keypoints: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        return vis_frames
    
    def _visualize_depth_keypoints(self, dk, data=None, canvas_size=(512, 512)):
        """
        Visualize depth keypoints overlaid on control video with color-coded depth values.
        
        Args:
            dk: depth keypoints array [T, N, 2] where [:,:,0]=depth (0~1), [:,:,1]=validity
            data: optional data dict to get background video and spatial keypoints
            canvas_size: default canvas size if no background video is found
        
        Returns:
            list of RGB frames (numpy arrays) for video visualization
        """
        import numpy as np
        import cv2
        
        H, W = canvas_size
        T_dk, N_dk, _ = dk.shape
        vis_frames = []
        background_frames = None
        spatial_kps = None
        
        try:
            # Try to find background video and spatial keypoints
            if data is not None:
                for bg_key in ['control_video', 'control_video_3dpose', 'control_video2']:
                    if bg_key in data:
                        bg_data = data[bg_key]
                        if isinstance(bg_data, list) and len(bg_data) > 0 and hasattr(bg_data[0], 'size'):
                            W, H = bg_data[0].size
                            background_frames = [np.array(img) for img in bg_data]
                            break
                        elif isinstance(bg_data, torch.Tensor) and bg_data.ndim == 4:
                            H, W = bg_data.shape[-2], bg_data.shape[-1]
                            bg_np = bg_data.permute(0, 2, 3, 1).cpu().numpy()
                            if bg_np.max() <= 1.0:
                                bg_np = (bg_np * 255).astype(np.uint8)
                            else:
                                bg_np = bg_np.astype(np.uint8)
                            background_frames = list(bg_np)
                            break
                
                # Find spatial keypoints for position reference
                for kp_key in ['key_points', 'key_points2', '3d_key_points']:
                    if kp_key in data:
                        kp_val = data[kp_key]
                        if isinstance(kp_val, torch.Tensor):
                            spatial_kps = kp_val.cpu().numpy()
                        elif isinstance(kp_val, np.ndarray):
                            spatial_kps = kp_val
                        break
            
            for frame_idx in range(T_dk):
                # Prepare canvas
                if background_frames is not None and frame_idx < len(background_frames):
                    canvas = background_frames[frame_idx].copy()
                    if canvas.shape[0] != H or canvas.shape[1] != W:
                        canvas = cv2.resize(canvas, (W, H))
                else:
                    canvas = np.zeros((H, W, 3), dtype=np.uint8)
                
                depth_vals = dk[frame_idx, :, 0]   # [N]
                valid_mask = dk[frame_idx, :, 1] > 0.5  # [N]
                
                # Draw depth values at keypoint positions
                if spatial_kps is not None and frame_idx < len(spatial_kps):
                    frame_kps = spatial_kps[frame_idx]  # [N_kp, D]
                    N_draw = min(N_dk, frame_kps.shape[0])
                    
                    for n in range(N_draw):
                        if not valid_mask[n]:
                            continue
                        if frame_kps.shape[1] >= 3 and frame_kps[n, 2] < 0.5:
                            continue
                        
                        x = int(frame_kps[n, 0] * W)
                        y = int(frame_kps[n, 1] * H)
                        if x < 0 or y < 0 or x >= W or y >= H:
                            continue
                        
                        # Color-code by depth: blue (near, 0) -> red (far, 1)
                        d = float(depth_vals[n])
                        r = int(d * 255)
                        b = int((1 - d) * 255)
                        g = 50
                        color = (r, g, b)
                        
                        cv2.circle(canvas, (x, y), 4, color, -1)
                        cv2.circle(canvas, (x, y), 4, (255, 255, 255), 1)
                        
                        # Draw depth value label for body keypoints only (first 18)
                        if n < 18:
                            label = f"{d:.2f}"
                            cv2.putText(canvas, label, (x + 5, y - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Add frame info
                valid_count = int(valid_mask.sum())
                valid_depths = depth_vals[valid_mask]
                if len(valid_depths) > 0:
                    d_range = f"[{valid_depths.min():.2f}, {valid_depths.max():.2f}]"
                else:
                    d_range = "N/A"
                label = f"Frame {frame_idx}: {valid_count}/{N_dk} valid, depth {d_range}"
                cv2.putText(canvas, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Convert BGR to RGB for imageio
                vis_frames.append(canvas[:, :, ::-1])
                
        except Exception as e:
            print(f"Error visualizing depth keypoints: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        return vis_frames
    
    def _visualize_depth_map(self, dm, data=None):
        """
        Visualize full depth map [T, H, W] as a color-coded depth video.
        Blue = near (0), Red = far (1). Optionally side-by-side with control video.
        
        Args:
            dm: depth map array [T, H, W] with normalized depth values [0, 1]
            data: optional data dict to get background video for side-by-side comparison
        
        Returns:
            list of RGB frames (numpy arrays) for video visualization
        """
        import numpy as np
        import cv2
        
        T_dm, H_dm, W_dm = dm.shape
        vis_frames = []
        background_frames = None
        
        try:
            # Try to find background video for side-by-side comparison
            if data is not None:
                for bg_key in ['control_video', 'control_video_3dpose', 'control_video2']:
                    if bg_key in data:
                        bg_data = data[bg_key]
                        if isinstance(bg_data, list) and len(bg_data) > 0 and hasattr(bg_data[0], 'size'):
                            background_frames = [np.array(img) for img in bg_data]
                            break
                        elif isinstance(bg_data, torch.Tensor) and bg_data.ndim == 4:
                            bg_np = bg_data.permute(0, 2, 3, 1).cpu().numpy()
                            if bg_np.max() <= 1.0:
                                bg_np = (bg_np * 255).astype(np.uint8)
                            else:
                                bg_np = bg_np.astype(np.uint8)
                            background_frames = list(bg_np)
                            break
            
            for frame_idx in range(T_dm):
                depth_frame = dm[frame_idx]  # [H, W] float [0, 1]
                
                # Convert depth to color map: COLORMAP_JET (blue=near, red=far)
                depth_uint8 = (depth_frame * 255).clip(0, 255).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)  # BGR
                
                # Add frame info text
                d_min, d_max, d_mean = depth_frame.min(), depth_frame.max(), depth_frame.mean()
                label = f"Frame {frame_idx}: depth [{d_min:.3f}, {d_max:.3f}], mean={d_mean:.3f}"
                cv2.putText(depth_color, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Side-by-side with background if available
                if background_frames is not None and frame_idx < len(background_frames):
                    bg = background_frames[frame_idx]
                    if bg.shape[:2] != (H_dm, W_dm):
                        bg = cv2.resize(bg, (W_dm, H_dm))
                    # Convert bg from RGB to BGR for consistent cv2 processing
                    bg_bgr = bg[:, :, ::-1]
                    canvas = np.concatenate([bg_bgr, depth_color], axis=1)  # side-by-side
                else:
                    canvas = depth_color
                
                # Convert BGR to RGB for imageio
                vis_frames.append(canvas[:, :, ::-1])
                
        except Exception as e:
            print(f"Error visualizing depth map: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        return vis_frames
    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])
        
    @staticmethod
    def default_video_operator_keep_original_ratio(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor, keep_original_ratio=True) >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor, keep_original_ratio=True),
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor, keep_original_ratio=True),
                )),
            ])),
        ])

    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path, keep_default_na=False)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def __getitem__(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
        else:
            data = self.data[data_id % len(self.data)].copy()
            for key in self.data_file_keys:
                if key in data:
                    if key in self.special_operator_map:
                        data[key] = self.special_operator_map[key](data[key])
                    elif key in self.data_file_keys:
                        data[key] = self.main_data_operator(data[key])

        # Sanity check: save all data for visualization
        if self.sanity_check:
            self.sanity_check_save_data(data, data_id)
            data["_sanity_check_data_id"] = data_id
                    
        return data

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
