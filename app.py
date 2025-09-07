import streamlit as st
import os
import cv2
import numpy as np
from rembg import remove
from PIL import Image
from itertools import product
import zipfile
import secrets
import string
from moviepy.editor import VideoFileClip

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="🎬 Multi-Combo Character Video Generator", layout="wide")

# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------
def generate_random_name(length=9):
    return ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(length))

def convert_to_webm(input_path, output_path):
    clip = VideoFileClip(input_path)
    clip.write_videofile(output_path, codec="libvpx", audio_codec="libvorbis")
    return output_path

def remove_background(input_path, output_path="output.png", width=494, height=505):
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"❌ Could not load image: {input_path}")

    _, encoded_img = cv2.imencode(".png", img)
    result = remove(encoded_img.tobytes())
    result_array = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_UNCHANGED)
    resized = cv2.resize(result_array, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_path, resized)
    return output_path

def create_spinning_character_video(
    background_image_path,
    character_image_path,
    output_path="character_tiktok_custom.mp4",
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

    # Safe background load + resize
    try:
        bg_img = Image.open(background_image_path).convert("RGB")
        bg_img = bg_img.resize((w, h), Image.LANCZOS)
        temp_bg_path = "temp_bg.png"
        bg_img.save(temp_bg_path, "PNG")
        bg = cv2.imread(temp_bg_path)
    except Exception as e:
        raise ValueError(f"❌ Could not load background: {background_image_path} ({e})")

    # Character load
    img = cv2.imread(character_image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"❌ Could not load character: {character_image_path}")
    img = cv2.resize(img, character_size)

    black_char = img.copy()
    black_char[:,:,:3] = 0
    alpha_mask = black_char[:,:,3]
    kernel = np.ones((outline_thickness, outline_thickness), np.uint8)
    outline_mask = cv2.dilate(alpha_mask, kernel, iterations=1)
    outline = np.zeros_like(black_char)
    outline[:,:,0:3] = 255
    outline[:,:,3] = outline_mask

    diag = int(np.sqrt(img.shape[0]**2 + img.shape[1]**2) * canvas_size_ratio)
    canvas_size = diag
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    cx, cy = w // 2, h // 2

    for i in range(frames):
        angle = 360 * num_turns * i / frames

        big_canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
        y_offset = (canvas_size - img.shape[0]) // 2
        x_offset = (canvas_size - img.shape[1]) // 2
        big_canvas[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img

        M = cv2.getRotationMatrix2D((canvas_size//2, canvas_size//2), angle, 1)
        rotated = cv2.warpAffine(big_canvas, M, (canvas_size, canvas_size),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

        frame = bg.copy()

        y1_s = cy - black_char.shape[0]//2
        y2_s = y1_s + black_char.shape[0]
        x1_s = cx - black_char.shape[1]//2
        x2_s = x1_s + black_char.shape[1]
        alpha_s = outline[:,:,3] / 255.0
        for c in range(3):
            frame[y1_s:y2_s, x1_s:x2_s, c] = alpha_s*outline[:,:,c] + (1-alpha_s)*frame[y1_s:y2_s, x1_s:x2_s, c]

        alpha_b = black_char[:,:,3] / 255.0
        for c in range(3):
            frame[y1_s:y2_s, x1_s:x2_s, c] = alpha_b*black_char[:,:,c] + (1-alpha_b)*frame[y1_s:y2_s, x1_s:x2_s, c]

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
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), font_thickness+2, cv2.LINE_AA)
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), font_thickness, cv2.LINE_AA)

        out.write(frame)

    out.release()
    return output_path

# -----------------------------
# FOLDERS
# -----------------------------
st.title("🎬 Multi-Combo Character Video Generator")
st.markdown(
    """
    Welcome! This tool allows you to **create spinning character videos** by combining your uploaded 
    backgrounds and characters. You can customize the number of spins, duration, overlay text, and 
    other video parameters.  

    **How to use:**
    1. Upload background images and character images in the sidebar.
    2. Select the backgrounds and characters you want to include.
    3. Adjust video parameters like spins, duration, and text overlay.
    4. Click **Generate All Videos** to create previews and download individual videos or a ZIP of all combinations.
    
    ⚠️ Make sure to upload at least one background and one character to generate videos.
    """
)
background_folder = "backgrounds"
character_folder = "characters"
output_folder = "videos"
os.makedirs(background_folder, exist_ok=True)
os.makedirs(character_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# UPLOAD FILES
# -----------------------------
st.sidebar.header("📂 Upload Files")
bg_files = st.sidebar.file_uploader("Upload Background(s)", type=["jpg","png"], accept_multiple_files=True)
char_files = st.sidebar.file_uploader("Upload Character(s)", type=["jpg","png"], accept_multiple_files=True)

# Save background uploads
if bg_files:
    for file in bg_files:
        file_path = os.path.join(background_folder, file.name)
        img = Image.open(file).convert("RGB")
        safe_path = os.path.splitext(file_path)[0] + ".png"
        img.save(safe_path, "PNG")

# Save character uploads AND auto remove background
if char_files:
    for file in char_files:
        file_path = os.path.join(character_folder, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
        # Automatically remove background
        no_bg_path = os.path.join(character_folder, f"no_bg_{file.name}.png")
        remove_background(file_path, no_bg_path)

# -----------------------------
# LOAD FILE LISTS
# -----------------------------
bg_list = [f for f in os.listdir(background_folder) if f.lower().endswith((".jpg",".png"))]
char_list = [f for f in os.listdir(character_folder) if f.lower().startswith("no_bg_")]

# -----------------------------
# BACKGROUND MULTI-SELECTION
# -----------------------------
st.subheader("🖼️ Background Selection")
selected_bgs = []

bg_count = len(bg_list)
if bg_count == 0:
    st.warning("⚠️ No background images uploaded.")
else:
    cols_per_row = min(bg_count, 6)  # max 6 columns
    
    # Dynamically set image size
    if bg_count <= 2:
        img_size = 150
    elif bg_count <= 4:
        img_size = 120
    else:
        img_size = 80
    
    for i in range(0, bg_count, cols_per_row):
        row_cols = st.columns(cols_per_row, gap="small")
        for j, bg in enumerate(bg_list[i:i+cols_per_row]):
            img_path = os.path.join(background_folder, bg)
            img = Image.open(img_path)
            if img.mode in ("RGBA", "LA"):
                img = img.convert("RGB")
            img = img.resize((img_size, img_size))
            with row_cols[j]:
                st.image(img, width=img_size)  # explicitly set width
                if st.checkbox("Select", value=True, key=f"bg_{bg}"):
                    selected_bgs.append(bg)
    
    st.markdown(f"**Selected Backgrounds:** {len(selected_bgs)}" if selected_bgs else "**Selected Backgrounds:** None")

# -----------------------------
# CHARACTER MULTI-SELECTION
# -----------------------------
st.subheader("🖼️ Character Selection")
selected_chars = []

char_count = len(char_list)
if char_count == 0:
    st.warning("⚠️ No Character images uploaded.")
else:
    cols_per_row = min(char_count, 6)
    
    # Dynamically set image size
    if char_count <= 2:
        img_size = 150
    elif char_count <= 4:
        img_size = 120
    else:
        img_size = 80
    
    for i in range(0, char_count, cols_per_row):
        row_cols = st.columns(cols_per_row, gap="small")
        for j, c in enumerate(char_list[i:i+cols_per_row]):
            img_path = os.path.join(character_folder, c)
            img = Image.open(img_path)
            if img.mode in ("RGBA", "LA"):
                img = img.convert("RGB")
            img = img.resize((img_size, img_size))
            with row_cols[j]:
                st.image(img, width=img_size)  # explicitly set width
                if st.checkbox("Select", value=True, key=f"char_{c}"):
                    selected_chars.append(c)
    
    st.markdown(f"**Selected Characters:** {len(selected_chars)}" if selected_chars else "**Selected Characters:** None")

# -----------------------------
# VIDEO PARAMETERS
# -----------------------------
st.sidebar.header("⚙️ Video Parameters")
text = st.sidebar.text_input("Overlay Text", "Hello everyone!")
outline_thickness = st.sidebar.slider("Outline Thickness", 1, 20, 5)
num_turns = st.sidebar.slider("Number of Spins", 1, 10, 2)
duration = st.sidebar.slider("Duration (seconds)", 2, 15, 5)
fps = st.sidebar.slider("FPS", 10, 60, 30)
character_size = st.sidebar.slider("Character Size", 100, 600, 300)
font_scale = st.sidebar.slider("Font Scale", 0.5, 3.0, 1.5, step=0.1)
font_thickness = st.sidebar.slider("Font Thickness", 1, 10, 5)

# Initialize session_state
if "generated_videos" not in st.session_state:
    st.session_state.generated_videos = []

if "col_index" not in st.session_state:
    st.session_state.col_index = 0

videos_per_row = 5

# -----------------------------
# Generate videos
# -----------------------------
if st.button("Generate All Videos"):
    for bg_file, char_file in product(selected_bgs, selected_chars):
        bg_path = os.path.join(background_folder, bg_file)
        char_path = os.path.join(character_folder, char_file)
        bg_name = os.path.splitext(bg_file)[0]
        char_name = os.path.splitext(char_file)[0].replace("no_bg_", "")
        output_path = os.path.join(output_folder, f"{bg_name}_{char_name}.mp4")
        preview_path = os.path.join(output_folder, f"{bg_name}_{char_name}_preview.webm")

        # Skip if already generated
        if any(v["output"] == output_path for v in st.session_state.generated_videos):
            continue

        try:
            create_spinning_character_video(
                bg_path, char_path,
                output_path=output_path,
                outline_thickness=outline_thickness,
                num_turns=num_turns,
                duration=duration,
                fps=fps,
                character_size=(character_size, character_size),
                text=text,
                font_scale=font_scale,
                font_thickness=font_thickness
            )
            convert_to_webm(output_path, preview_path)

            # Store in session_state
            st.session_state.generated_videos.append({
                "output": output_path,
                "preview": preview_path
            })

        except Exception as e:
            st.error(f"❌ Failed {bg_file} + {char_file}: {e}")

# -----------------------------
# Display videos and downloads
# -----------------------------
row_cols = st.columns(videos_per_row)
for i, vid in enumerate(st.session_state.generated_videos):
    with row_cols[i % videos_per_row]:
        st.video(vid["preview"], width=200)
        random_download_name = generate_random_name() + ".mp4"
        with open(vid["output"], "rb") as f:
            st.download_button("⬇️ Download", data=f, file_name=random_download_name, mime="video/mp4")

# -----------------------------
# ZIP download
# -----------------------------
if st.session_state.generated_videos:
    zip_name = generate_random_name() + ".zip"
    zip_path = os.path.join(output_folder, zip_name)

    with zipfile.ZipFile(zip_path, "w") as zipf:
        for vid in st.session_state.generated_videos:
            if os.path.exists(vid["output"]):
                zipf.write(vid["output"], os.path.basename(vid["output"]))

    with open(zip_path, "rb") as f:
        st.download_button("Download All Videos", data=f, file_name=zip_name, mime="application/zip")


