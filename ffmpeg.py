import numpy as np
from PIL import Image, ImageDraw, ImageFont
import subprocess
import time
from scipy.ndimage import binary_dilation

def create_spinning_character_video_ffmpeg(
    background_image_path,
    character_image_path,
    output_path="character_video.mp4",
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
    font_thickness=5
):
    """
    Generates a spinning character video with text overlay using FFmpeg piping.
    Returns nothing, outputs the video to output_path.
    """
    start_time = time.time()
    w, h = video_size
    total_frames = int(duration * fps)
    text_frames = int(text_duration * fps)

    # Load background
    bg_img = Image.open(background_image_path).convert("RGB").resize((w, h), Image.LANCZOS)
    bg_np = np.array(bg_img)

    # Load character
    char_img = Image.open(character_image_path).convert("RGBA").resize(character_size)
    char_np = np.array(char_img)

    # Prepare static black character + white outline (underneath)
    black_char = char_np.copy()
    black_char[:, :, :3] = 0  # black
    alpha_mask = black_char[:, :, 3]
    outline_mask = binary_dilation(alpha_mask, iterations=outline_thickness).astype(np.uint8) * 255
    outline = np.zeros_like(black_char)
    outline[:, :, :3] = 255  # white outline
    outline[:, :, 3] = outline_mask

    # Compute canvas for rotating character
    diag = int(np.sqrt(char_np.shape[0] ** 2 + char_np.shape[1] ** 2) * canvas_size_ratio)
    canvas_size = diag
    cx, cy = w // 2, h // 2

    # FFmpeg command
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Prepare font
    try:
        font_size = int(40 * font_scale)
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Precompute static black character with white outline
    base_frame = bg_np.copy()
    rh, rw = black_char.shape[:2]
    y1, x1 = cy - rh // 2, cx - rw // 2
    y2, x2 = y1 + rh, x1 + rw
    alpha = outline[:, :, 3:4] / 255.0
    base_frame[y1:y2, x1:x2, :3] = alpha * outline[:, :, :3] + (1 - alpha) * base_frame[y1:y2, x1:x2, :3]
    alpha_black = black_char[:, :, 3:4] / 255.0
    base_frame[y1:y2, x1:x2, :3] = alpha_black * black_char[:, :, :3] + (1 - alpha_black) * base_frame[y1:y2, x1:x2, :3]

    # Generate frames
    for i in range(total_frames):
        angle = 360 * num_turns * i / total_frames

        # Rotate character
        char_rot = Image.fromarray(char_np).rotate(angle, expand=True)
        char_rot_np = np.array(char_rot)
        rh, rw = char_rot_np.shape[:2]
        y1_r = cy - rh // 2
        x1_r = cx - rw // 2
        y2_r, x2_r = y1_r + rh, x1_r + rw
        y1_c, x1_c = max(0, y1_r), max(0, x1_r)
        y2_c, x2_c = min(h, y2_r), min(w, x2_r)
        y1_rot, x1_rot = y1_c - y1_r, x1_c - x1_r
        y2_rot, x2_rot = y2_c - y1_r, x2_c - x1_r

        frame = base_frame.copy()
        alpha = char_rot_np[y1_rot:y2_rot, x1_rot:x2_rot, 3:4] / 255.0
        frame[y1_c:y2_c, x1_c:x2_c, :3] = alpha * char_rot_np[y1_rot:y2_rot, x1_rot:x2_rot, :3] + \
                                          (1 - alpha) * frame[y1_c:y2_c, x1_c:x2_c, :3]

        # Text overlay above character
        angle_in_turn = angle % 360
        if angle_in_turn < 360 / total_frames * text_frames:
            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w, text_h = bbox[2], bbox[3]
            text_x = cx - text_w // 2
            text_y = cy - char_np.shape[0] // 2 - text_h - 5
            # Draw white outline
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    draw.text((text_x + dx, text_y + dy), text, font=font, fill=(255,255,255))
            # Draw black text
            draw.text((text_x, text_y), text, font=font, fill=(0,0,0))
            frame = np.array(img_pil)

        proc.stdin.write(frame.astype(np.uint8).tobytes())

    proc.stdin.close()
    proc.wait()
    print(f"FFmpeg video generated in {time.time() - start_time:.2f} seconds")
    return output_path




