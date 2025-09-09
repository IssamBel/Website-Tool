import shutil
import os
from itertools import product
import zipfile
import uuid
from typing import List

from PIL import Image
import cv2
import numpy as np
from rembg import remove
from moviepy.editor import VideoFileClip



# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------
def generate_random_name(length=9):
    return ''.join(uuid.uuid4().hex[:length])

def remove_background(input_path, output_path="output.png", width=494, height=505):
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not load image: {input_path}")

    _, encoded_img = cv2.imencode(".png", img)
    result = remove(encoded_img.tobytes())
    result_array = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_UNCHANGED)
    resized = cv2.resize(result_array, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_path, resized)
    return output_path

def convert_to_webm(input_path, output_path):
    clip = VideoFileClip(input_path)
    clip.write_videofile(output_path, codec="libvpx", audio_codec="libvorbis")
    return output_path

def create_spinning_character_video(
    background_image_path,
    character_image_path,
    output_path="character.mp4",
    outline_thickness=5,
    num_turns=2,
    duration=5,
    fps=30,
    character_size=(300, 300),
    canvas_size_ratio=1.2,
    video_size=(540, 960),
    text="Hello everyone!",
    text_duration=0.2,
    font_scale=1.5,
    font_thickness=5,
):
    w, h = video_size
    frames = int(duration * fps)
    text_frames = int(text_duration * fps)

    bg_img = Image.open(background_image_path).convert("RGB")
    bg_img = bg_img.resize((w, h), Image.LANCZOS)
    temp_bg_path = "temp_bg.png"
    bg_img.save(temp_bg_path, "PNG")
    bg = cv2.imread(temp_bg_path)

    img = cv2.imread(character_image_path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, character_size)

    black_char = img.copy()
    black_char[:, :, :3] = 0
    alpha_mask = black_char[:, :, 3]
    kernel = np.ones((outline_thickness, outline_thickness), np.uint8)
    outline_mask = cv2.dilate(alpha_mask, kernel, iterations=1)
    outline = np.zeros_like(black_char)
    outline[:, :, 0:3] = 255
    outline[:, :, 3] = outline_mask

    diag = int(np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2) * canvas_size_ratio)
    canvas_size = diag
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    cx, cy = w // 2, h // 2

    for i in range(frames):
        angle = 360 * num_turns * i / frames
        big_canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
        y_offset = (canvas_size - img.shape[0]) // 2
        x_offset = (canvas_size - img.shape[1]) // 2
        big_canvas[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img
        M = cv2.getRotationMatrix2D((canvas_size // 2, canvas_size // 2), angle, 1)
        rotated = cv2.warpAffine(big_canvas, M, (canvas_size, canvas_size), borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))

        frame = bg.copy()
        y1_s = cy - black_char.shape[0] // 2
        y2_s = y1_s + black_char.shape[0]
        x1_s = cx - black_char.shape[1] // 2
        x2_s = x1_s + black_char.shape[1]
        alpha_s = outline[:, :, 3] / 255.0
        for c in range(3):
            frame[y1_s:y2_s, x1_s:x2_s, c] = alpha_s * outline[:, :, c] + (1 - alpha_s) * frame[y1_s:y2_s, x1_s:x2_s, c]

        alpha_b = black_char[:, :, 3] / 255.0
        for c in range(3):
            frame[y1_s:y2_s, x1_s:x2_s, c] = alpha_b * black_char[:, :, c] + (1 - alpha_b) * frame[y1_s:y2_s, x1_s:x2_s, c]

        y1 = cy - canvas_size // 2
        y2 = y1 + canvas_size
        x1 = cx - canvas_size // 2
        x2 = x1 + canvas_size
        y1_crop = max(0, y1)
        x1_crop = max(0, x1)
        y2_crop = min(h, y2)
        x2_crop = min(w, x2)
        y1_rot = max(0, -y1)
        x1_rot = max(0, -x1)
        y2_rot = y1_rot + (y2_crop - y1_crop)
        x2_rot = x1_rot + (x2_crop - x1_crop)

        alpha = rotated[y1_rot:y2_rot, x1_rot:x2_rot, 3] / 255.0
        for c in range(3):
            frame[y1_crop:y2_crop, x1_crop:x2_crop, c] = alpha * rotated[y1_rot:y2_rot, x1_rot:x2_rot, c] + \
                                                          (1 - alpha) * frame[y1_crop:y2_crop, x1_crop:x2_crop, c]

        angle_in_turn = angle % 360
        if angle_in_turn < (360 / frames) * text_frames:
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            text_x = cx - text_w // 2
            text_y = y1_s - 20
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                        font_thickness + 2, cv2.LINE_AA)
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                        font_thickness, cv2.LINE_AA)

        out.write(frame)

    out.release()
    return output_path
