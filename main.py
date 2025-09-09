# Import necessary modules from FastAPI and other libraries.
# We've added 'Request' to access client information and 'SecureCookieMiddleware' for session management.
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.concurrency import run_in_threadpool
import os, shutil, uuid
import logging
import asyncio
import datetime

# Import the actual functions from your modules
from vide import remove_background, convert_to_webm
from ffmpeg import create_spinning_character_video_ffmpeg

# Set up logging for better debugging and request tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Add session middleware for user tracking.
# The 'secret_key' is crucial for securing the session cookie.
# For production, this should be a random, long string loaded from an environment variable.
app.add_middleware(SessionMiddleware, secret_key="your-super-secret-key-goes-here")

# --- Background Task for File Cleanup ---
async def cleanup_old_files():
    """
    A background task that periodically removes old files from the upload and video directories.
    This prevents the server from filling up with old generated content.
    """
    CLEANUP_INTERVAL_SECONDS = 3600  # Run cleanup every hour
    FILE_LIFETIME_SECONDS = 3600    # Delete files older than 1 hour

    while True:
        logger.info("Starting cleanup of old files...")
        now = datetime.datetime.now()
        
        folders = [UPLOAD_FOLDER, OUTPUT_FOLDER]
        for folder in folders:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path):
                        # Get the last modification time of the file
                        mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                        if (now - mod_time).total_seconds() > FILE_LIFETIME_SECONDS:
                            os.remove(file_path)
                            logger.info(f"Deleted old file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting file {file_path}: {e}")
        
        logger.info("Cleanup complete. Next run in 1 hour.")
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)

# Schedule the cleanup task to run on application startup
@app.on_event("startup")
async def start_cleanup_task():
    asyncio.create_task(cleanup_old_files())

@app.post("/generate_video/")
async def generate_video_ffmpeg(
    request: Request,
    background: UploadFile = File(...),
    character: UploadFile = File(...),
    overlay_text: str = Form("Hello!"),
    outline_thickness: int = Form(5),
    num_turns: int = Form(2),
    duration: int = Form(5),
    fps: int = Form(30),
    character_size: int = Form(300),
    font_scale: float = Form(1.5),
    font_thickness: int = Form(5)
):
    """
    This endpoint generates a spinning character video.
    It now tracks requests using a unique video_id and logs user session and IP address.
    """
    # Use a unique ID for each video generation task to avoid conflicts.
    video_id = uuid.uuid4().hex

    # Get user information from the request object.
    # request.client.host provides the client's IP address.
    client_ip = request.client.host
    # request.session is a dictionary-like object from the SessionMiddleware.
    # You can store data specific to the user's session here.
    # We'll use a session counter to track how many times a user has used the tool.
    session_id = request.session.get("session_id")
    if not session_id:
        session_id = uuid.uuid4().hex
        request.session["session_id"] = session_id
        request.session["request_count"] = 1
    else:
        request.session["request_count"] += 1
    
    logger.info(f"Received request from IP: {client_ip} | Session ID: {session_id} | Request count: {request.session['request_count']} | Video ID: {video_id}")

    # Save uploaded files. Using unique filenames prevents conflicts.
    bg_path = os.path.join(UPLOAD_FOLDER, f"{video_id}_bg.png")
    char_path = os.path.join(UPLOAD_FOLDER, f"{video_id}_char.png")

    # Use run_in_threadpool to run blocking I/O operations in a background thread.
    # This prevents the main event loop from being blocked and keeps the API responsive.
    await run_in_threadpool(shutil.copyfileobj, background.file, open(bg_path, "wb"))
    await run_in_threadpool(shutil.copyfileobj, character.file, open(char_path, "wb"))

    # Remove background from character image
    no_bg_path = os.path.join(UPLOAD_FOLDER, f"{video_id}_no_bg.png")
    
    # For a production application, heavy tasks like this should also be offloaded
    # to a thread pool or a separate worker process to not block the main event loop.
    # We are calling the function from vide.py here.
    await run_in_threadpool(remove_background, char_path, no_bg_path, width=character_size, height=character_size)
    
    # Output video paths
    output_mp4 = os.path.join(OUTPUT_FOLDER, f"{video_id}.mp4")
    output_webm = os.path.join(OUTPUT_FOLDER, f"{video_id}.webm")

    # Generate FFmpeg video. We are calling the function from ffmpeg.py here.
    await run_in_threadpool(
        create_spinning_character_video_ffmpeg,
        background_image_path=bg_path,
        character_image_path=no_bg_path,
        output_path=output_mp4,
        outline_thickness=outline_thickness,
        num_turns=num_turns,
        duration=duration,
        fps=fps,
        character_size=(character_size, character_size),
        text=overlay_text,
        font_scale=font_scale,
        font_thickness=font_thickness
    )

    # Convert to webm. We are calling the function from vide.py here.
    await run_in_threadpool(convert_to_webm, output_mp4, output_webm)

    if not os.path.exists(output_webm):
        raise HTTPException(status_code=500, detail="Video generation failed.")

    logger.info(f"Video generation complete for Video ID: {video_id}")
    return {"video_id": video_id}

@app.get("/video/{video_id}")
async def get_video_ffmpeg(video_id: str, format: str = "webm"):
    """
    Serves the generated video file.
    """
    ext = "mp4" if format.lower() == "mp4" else "webm"
    video_path = os.path.join(OUTPUT_FOLDER, f"{video_id}.{ext}")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video_path, media_type=f"video/{ext}", filename=f"{video_id}.{ext}")



#uvicorn main:app --workers 4 --host 0.0.0.0 --port 8000
