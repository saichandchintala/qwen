# -*- coding: utf-8 -*-
"""
Working code - tested on 90 sec video. 
Mini-run (2 seconds) → per-second multimodal embeddings with Qwen3-Omni (MoviePy v2)

Key fixes:
- Use absolute paths for segments
- Convert segment paths to proper file URIs: Path(...).resolve().as_uri()
- Remove 'verbose'/'logger' args from write_videofile (MoviePy v2)
- Talker disabled + USE_AUDIO_IN_VIDEO=True
- Error handling + tqdm progress bars
- Read transcript from CSV file for each second

Refs:
- MoviePy v2 import & examples: https://pypi.org/project/moviepy/            # top-level import, v2 API
- qwen-omni-utils file path/URI usage: https://pypi.org/project/qwen-omni-utils/  # 'file:///path/to/...'
"""

import os
import math
import logging
from typing import List, Dict
from pathlib import Path
import time
import gc

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from moviepy import VideoFileClip  # MoviePy v2 import (not moviepy.editor)

from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

# ----------------------------
# Configuration
# ----------------------------
MODEL_PATH = "/storage/scratch/saichandc/Qwen3-Omni-30B-A3B-Thinking"   # or "Qwen/Qwen3-Omni-30B-A3B-Thinking"
VIDEO_PATH = "/storage/home/saichandc/video/first_debate_first.mp4"                                  # <-- change to your video path
TRANSCRIPT_CSV = "/storage/home/saichandc/qwen/transcript_by_second_first_debate.csv"
OUT_DIR = "./tmp_slices_v2"
SAVE_EMBEDDINGS_NPY = "./second_level_embeddings_first_debate_with_transcript.npy"

USE_AUDIO_IN_VIDEO = True
BATCH_SIZE = 2
MAX_SECS = None     # only first 2 seconds now; set None for full length later

LOG_LEVEL = logging.INFO
os.makedirs(OUT_DIR, exist_ok=True)
OUT_DIR = str(Path(OUT_DIR).resolve())  # <-- absolutize output dir

logging.basicConfig(level=LOG_LEVEL, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("qwen_embeddings")

# ----------------------------
# Load transcript from CSV
# ----------------------------
def load_transcript(csv_path: str) -> Dict[int, str]:
    """
    Load transcript CSV and return a dict mapping second -> text.
    Expected CSV format: columns like 'second', 'text' (adjust as needed)
    """
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded transcript CSV with shape: {df.shape}")
        logger.info(f"CSV columns: {df.columns.tolist()}")
        
        # Try to identify the correct columns
        # Common column names: 'second', 'time', 'sec', 'timestamp' for time
        # and 'text', 'transcript', 'content' for text
        time_col = None
        text_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['second', 'sec', 'time', 'seconds']:
                time_col = col
            if col_lower in ['text', 'transcript', 'content', 'speech']:
                text_col = col
        
        if time_col is None:
            # Assume first column is time
            time_col = df.columns[0]
            logger.warning(f"Time column not found, using first column: {time_col}")
        
        if text_col is None:
            # Assume second column is text
            text_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            logger.warning(f"Text column not found, using column: {text_col}")
        
        # Create dictionary mapping second to text
        transcript_dict = {}
        for _, row in df.iterrows():
            sec = int(row[time_col])
            text = str(row[text_col]) if pd.notna(row[text_col]) else ""
            transcript_dict[sec] = text
        
        logger.info(f"Loaded transcript for {len(transcript_dict)} seconds")
        return transcript_dict
    
    except Exception as e:
        logger.exception("Failed to load transcript CSV.")
        raise SystemExit(e)

# ----------------------------
# Helper: masked mean pooling
# ----------------------------
def pooled_last_hidden_state(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
    summed = (last_hidden * mask).sum(dim=1)     # [B, H]
    denom = mask.sum(dim=1).clamp(min=1e-6)      # [B, 1]
    return summed / denom

# ----------------------------
# Load Qwen model & processor
# ----------------------------
def load_qwen():
    try:
        # Explicitly set device_map to use GPU
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_PATH, 
            torch_dtype=torch.bfloat16,  # Use bfloat16 for better GPU performance
            device_map="auto", 
            trust_remote_code=True,
        )
        processor = Qwen3OmniMoeProcessor.from_pretrained(
            MODEL_PATH, 
            trust_remote_code=True,
        )
        
        # Log GPU usage
        if torch.cuda.is_available():
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        else:
            logger.warning("GPU not available! Model will run on CPU (slow).")
            
    except Exception as e:
        logger.exception("Failed to load Qwen model/processor.")
        raise SystemExit(e)

    # Save ~2GB if you don't need audio outputs
    try:
        model.disable_talker()
        logger.info("Talker disabled (no audio generation).")
    except Exception as e:
        logger.warning("Could not disable Talker explicitly; continuing without audio generation. Details: %s", e)

    return model, processor

# ----------------------------
# Slice video → 1-second MP4 segments (MoviePy v2)
# ----------------------------
# def slice_video_v2(video_path: str, out_dir: str, max_secs: int | None) -> List[Dict]:
#     """
#     Export 1-second segments with MoviePy v2 `.subclipped`.
#     Return [{"sec": t, "video_uri": file:///..., "video_path": /abs/path.mp4}, ...]
#     """
#     vp = Path(video_path).resolve()
#     if not vp.exists():
#         raise FileNotFoundError(f"Video not found: {vp}")

#     try:
#         clip = VideoFileClip(str(vp))
#     except Exception as e:
#         logger.exception("Failed to open video with MoviePy v2.")
#         raise

#     duration = float(clip.duration or 0.0)
#     total_secs = math.ceil(duration)

#     if max_secs is None:
#         n_secs = total_secs
#         logger.info("Processing FULL length: %d second(s).", n_secs)
#     else:
#         n_secs = min(max_secs, total_secs)
#         logger.info("Processing FIRST %d second(s) (of %d).", n_secs, total_secs)

#     items: List[Dict] = []
#     logger.info("Slicing %s → %d 1-second segment(s)...", vp.name, n_secs)

#     fps = getattr(clip, "fps", None)

#     for t in tqdm(range(n_secs), desc="Slicing seconds", unit="s"):
#         start, end = t, min(t + 1, duration)
#         if end <= start:
#             continue
#         try:
#             sub = clip.subclipped(start, end)  # MoviePy v2 API
#             out_file = Path(out_dir) / f"seg_{t:06d}.mp4"

#             # IMPORTANT: write without 'verbose'/'logger' (v2); keep audio
#             if fps is not None:
#                 sub.write_videofile(str(out_file), fps=fps, audio=True, codec="libx264", audio_codec="aac")
#             else:
#                 sub.write_videofile(str(out_file), audio=True, codec="libx264", audio_codec="aac")

#             sub.close()

#             # Build proper absolute path + file URI
#             abs_path = out_file.resolve()
#             items.append({
#                 "sec": t,
#                 "video_path": str(abs_path),
#                 "video_uri": abs_path.as_uri(),  # e.g., file:///storage/home/...
#             })
#         except Exception as se:
#             logger.error("Failed to write segment %d: %s", t, se)
#             continue

#     clip.close()
#     if not items:
#         raise RuntimeError("No segments produced. Check the video and codecs.")
#     logger.info("Created %d segment(s) in %s", len(items), out_dir)
#     return items

import gc

import subprocess
import gc

def slice_video_v2(video_path: str, out_dir: str, max_secs: int | None) -> List[Dict]:
    """
    Export 1-second segments using ffmpeg directly (avoids MoviePy subprocess issues)
    """
    vp = Path(video_path).resolve()
    if not vp.exists():
        raise FileNotFoundError(f"Video not found: {vp}")
    
    # Get video duration using ffprobe
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
             '-of', 'default=noprint_wrappers=1:nokey=1', str(vp)],
            capture_output=True,
            text=True,
            check=True
        )
        duration = float(result.stdout.strip())
        total_secs = int(duration)
    except Exception as e:
        logger.exception("Failed to get video duration with ffprobe")
        raise

    if max_secs is None:
        n_secs = total_secs
        logger.info("Processing FULL length: %d second(s).", n_secs)
    else:
        n_secs = min(max_secs, total_secs)
        logger.info("Processing FIRST %d second(s) (of %d).", n_secs, total_secs)

    items: List[Dict] = []
    failed_segments = []
    logger.info("Slicing %s → %d 1-second segment(s)...", vp.name, n_secs)
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for t in tqdm(range(n_secs), desc="Slicing seconds", unit="s"):
        out_file = Path(out_dir) / f"seg_{t:06d}.mp4"
        
        # Skip if already exists (useful for resuming)
        if out_file.exists() and out_file.stat().st_size > 1000:  # at least 1KB
            abs_path = out_file.resolve()
            items.append({
                "sec": t,
                "video_path": str(abs_path),
                "video_uri": abs_path.as_uri(),
            })
            continue
        
        try:
            # Use ffmpeg to extract 1-second segment
            cmd = [
                'ffmpeg', '-y',  # overwrite
                '-ss', str(t),  # start time
                '-i', str(vp),  # input file
                '-t', '1',  # duration (1 second)
                '-c:v', 'libx264',  # video codec
                '-c:a', 'aac',  # audio codec
                '-strict', 'experimental',
                '-loglevel', 'error',  # only show errors
                str(out_file)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                stderr_msg = result.stderr[-300:] if result.stderr else "No error message"
                logger.error(f"Segment {t} ffmpeg error: {stderr_msg}")
                failed_segments.append(t)
                continue
            
            # Verify file was created and has reasonable size
            if not out_file.exists() or out_file.stat().st_size < 100:
                logger.error(f"Segment {t}: file not created or too small")
                failed_segments.append(t)
                continue
            
            abs_path = out_file.resolve()
            items.append({
                "sec": t,
                "video_path": str(abs_path),
                "video_uri": abs_path.as_uri(),
            })
            
            # Periodic cleanup
            if t % 100 == 0 and t > 0:
                gc.collect()
                logger.info(f"Progress: {len(items)}/{t+1} segments created")
                
        except subprocess.TimeoutExpired:
            logger.error(f"Segment {t}: timeout (30s)")
            failed_segments.append(t)
        except Exception as e:
            logger.exception(f"Segment {t}: unexpected error")
            failed_segments.append(t)

    # Final report
    logger.info(f"Slicing complete: {len(items)}/{n_secs} successful")
    if failed_segments:
        logger.warning(f"Failed {len(failed_segments)} segments: {failed_segments[:30]}...")
        logger.warning(f"Success rate: {100*len(items)/n_secs:.1f}%")
    
    if not items:
        raise RuntimeError("No segments produced. Check video file and ffmpeg.")
    
    logger.info("Created %d segment(s) in %s", len(items), out_dir)
    return items

# ----------------------------
# Build conversations & extract embeddings
# ----------------------------
def extract_second_level_embeddings(
    model, 
    processor, 
    sec_items: List[Dict], 
    transcript_dict: Dict[int, str]
) -> np.ndarray:
    failed_seconds = []
    embeds_all = []

    logger.info("Building per-second conversations and extracting embeddings...")
    for i in tqdm(range(0, len(sec_items), BATCH_SIZE), desc="Embedding batches", unit="batch"):
        batch = sec_items[i:i + BATCH_SIZE]

        conversations = []
        for it in batch:
            # Get text for this second from transcript, or use empty string if not found
            sec = it["sec"]
            text = transcript_dict.get(sec, "")
            
            # Log first few to verify
            if i < 3:
                if text:
                    logger.info(f"Second {sec}: Using text: {text[:100]}...")
                else:
                    logger.info(f"Second {sec}: Using empty string (no text in transcript)")
            
            conversations.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": it["video_uri"]},
                        {"type": "text",  "text": text},
                    ],
                }
            ])

        try:
            text = processor.apply_chat_template(
                conversations, add_generation_prompt=False, tokenize=False
            )
            audios, images, videos = process_mm_info(
                conversations, use_audio_in_video=USE_AUDIO_IN_VIDEO
            )
            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=USE_AUDIO_IN_VIDEO,
            ).to(model.device).to(model.dtype)
        except Exception as e:
            logger.error("Processor/prep failed for batch starting at sec %d: %s", batch[0]["sec"], e)
            # Helpful debug: print the exact URI/path we passed
            for dbg in batch:
                logger.error("   video_uri=%s | exists=%s",
                             dbg["video_uri"], Path(dbg["video_path"]).exists())
            failed_seconds.extend([it["sec"] for it in batch])
            continue

        try:
            with torch.no_grad():
                # forward pass using the processor to match multimodal input spec
                outputs = model.thinker(**inputs, output_hidden_states=True, return_dict=True)
                last_hidden = outputs.hidden_states[-1]
                pooled = pooled_last_hidden_state(last_hidden, inputs["attention_mask"])

                embeds_all.append(pooled.detach().cpu().numpy())
        except Exception as e:
            logger.error("Model forward failed for batch starting at sec %d: %s", batch[0]["sec"], e)
            failed_seconds.extend([it["sec"] for it in batch])
            continue

    if not embeds_all:
        raise RuntimeError("No embeddings produced; all batches failed.")

    embeds = np.concatenate(embeds_all, axis=0)
    if failed_seconds:
        logger.warning("Failed seconds (skipped): %s", failed_seconds)
    else:
        logger.info("All seconds processed successfully.")
    return embeds

# ----------------------------
# Entry point
# ----------------------------
def main():
    start_total = time.time()

    # ---- Load transcript ----
    t_transcript = time.time()
    transcript_dict = load_transcript(TRANSCRIPT_CSV)
    logger.info("Transcript loaded in %.2f sec", time.time() - t_transcript)

    # ---- Load model ----
    t0 = time.time()
    model, processor = load_qwen()
    logger.info("Model + processor loaded in %.2f sec", time.time() - t0)

    # ---- Slice video ----
    t1 = time.time()
    try:
        sec_items = slice_video_v2(VIDEO_PATH, OUT_DIR, max_secs=MAX_SECS)
    except Exception as e:
        logger.exception("Slicing failed.")
        raise SystemExit(e)
    logger.info("Video slicing completed in %.2f sec", time.time() - t1)

    # ---- Extract embeddings ----
    t2 = time.time()
    try:
        embeds = extract_second_level_embeddings(model, processor, sec_items, transcript_dict)
    except Exception as e:
        logger.exception("Embedding extraction failed.")
        raise SystemExit(e)
    logger.info("Embedding extraction completed in %.2f sec", time.time() - t2)

    # ---- Save embeddings ----
    np.save(SAVE_EMBEDDINGS_NPY, embeds)
    logger.info("Saved embeddings: shape=%s -> %s", embeds.shape, SAVE_EMBEDDINGS_NPY)

    # ---- Total time ----
    logger.info("Total runtime: %.2f sec (%.2f min)", time.time() - start_total, (time.time() - start_total)/60)

if __name__ == "__main__":
    main()
