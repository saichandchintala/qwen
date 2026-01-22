# -*- coding: utf-8 -*-
"""
Working code - tested on 90 sec video. 
Creates SEPARATE embeddings for:
1. Audio+Video (multimodal)
2. Text only (from transcript)

Key fixes:
- Use ffmpeg directly for video slicing (avoids MoviePy subprocess issues)
- Separate embedding extraction for multimodal vs text-only
- GPU optimized
"""

import os
import math
import logging
from typing import List, Dict
from pathlib import Path
import time
import gc
import subprocess

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

# ----------------------------
# Configuration
# ----------------------------
MODEL_PATH = "/storage/scratch/saichandc/Qwen3-Omni-30B-A3B-Thinking"
VIDEO_PATH = "/storage/home/saichandc/video/first_debate_first.mp4"
TRANSCRIPT_CSV = "/storage/home/saichandc/qwen/transcript_by_second_first_debate.csv"
OUT_DIR = "./tmp_slices_v2"

# Separate output files
SAVE_EMBEDDINGS_NPY_AUDIO_VIDEO = "./second_level_embeddings_first_debate_audio_video.npy"
SAVE_EMBEDDINGS_NPY_TEXT_ONLY = "./second_level_embeddings_first_debate_text_only.npy"

USE_AUDIO_IN_VIDEO = True
BATCH_SIZE = 2
MAX_SECS = None

LOG_LEVEL = logging.INFO
os.makedirs(OUT_DIR, exist_ok=True)
OUT_DIR = str(Path(OUT_DIR).resolve())

logging.basicConfig(level=LOG_LEVEL, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("qwen_embeddings")

# ----------------------------
# Load transcript from CSV
# ----------------------------
def load_transcript(csv_path: str) -> Dict[int, str]:
    """
    Load transcript CSV and return a dict mapping second -> text.
    """
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded transcript CSV with shape: {df.shape}")
        logger.info(f"CSV columns: {df.columns.tolist()}")
        
        time_col = None
        text_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['second', 'sec', 'time', 'seconds']:
                time_col = col
            if col_lower in ['text', 'transcript', 'content', 'speech']:
                text_col = col
        
        if time_col is None:
            time_col = df.columns[0]
            logger.warning(f"Time column not found, using first column: {time_col}")
        
        if text_col is None:
            text_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            logger.warning(f"Text column not found, using column: {text_col}")
        
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
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom

# ----------------------------
# Load Qwen model & processor
# ----------------------------
def load_qwen():
    try:
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_PATH, 
            torch_dtype=torch.bfloat16,
            device_map="auto", 
            trust_remote_code=True,
        )
        processor = Qwen3OmniMoeProcessor.from_pretrained(
            MODEL_PATH, 
            trust_remote_code=True,
        )
        
        if torch.cuda.is_available():
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        else:
            logger.warning("GPU not available! Model will run on CPU (slow).")
            
    except Exception as e:
        logger.exception("Failed to load Qwen model/processor.")
        raise SystemExit(e)

    try:
        model.disable_talker()
        logger.info("Talker disabled (no audio generation).")
    except Exception as e:
        logger.warning("Could not disable Talker explicitly; continuing without audio generation. Details: %s", e)

    return model, processor

# ----------------------------
# Slice video with ffmpeg
# ----------------------------
def slice_video_v2(video_path: str, out_dir: str, max_secs: int | None) -> List[Dict]:
    """
    Export 1-second segments using ffmpeg directly
    """
    vp = Path(video_path).resolve()
    if not vp.exists():
        raise FileNotFoundError(f"Video not found: {vp}")
    
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
        
        # Skip if already exists
        if out_file.exists() and out_file.stat().st_size > 1000:
            abs_path = out_file.resolve()
            items.append({
                "sec": t,
                "video_path": str(abs_path),
                "video_uri": abs_path.as_uri(),
            })
            continue
        
        try:
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(t),
                '-i', str(vp),
                '-t', '1',
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-strict', 'experimental',
                '-loglevel', 'error',
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
            
            if t % 100 == 0 and t > 0:
                gc.collect()
                logger.info(f"Progress: {len(items)}/{t+1} segments created")
                
        except subprocess.TimeoutExpired:
            logger.error(f"Segment {t}: timeout (30s)")
            failed_segments.append(t)
        except Exception as e:
            logger.exception(f"Segment {t}: unexpected error")
            failed_segments.append(t)

    logger.info(f"Slicing complete: {len(items)}/{n_secs} successful")
    if failed_segments:
        logger.warning(f"Failed {len(failed_segments)} segments: {failed_segments[:30]}...")
    
    if not items:
        raise RuntimeError("No segments produced. Check video file and ffmpeg.")
    
    logger.info("Created %d segment(s) in %s", len(items), out_dir)
    return items

# ----------------------------
# Extract AUDIO+VIDEO embeddings (NO TEXT)
# ----------------------------
def extract_audio_video_embeddings(
    model, 
    processor, 
    sec_items: List[Dict]
) -> np.ndarray:
    """
    Extract embeddings from audio+video only (no text prompt)
    """
    failed_seconds = []
    embeds_all = []

    logger.info("Extracting AUDIO+VIDEO embeddings (no text)...")
    for i in tqdm(range(0, len(sec_items), BATCH_SIZE), desc="Audio+Video batches", unit="batch"):
        batch = sec_items[i:i + BATCH_SIZE]

        conversations = []
        for it in batch:
            # NO TEXT - just video/audio
            conversations.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": it["video_uri"]},
                        {"type": "text",  "text": ""},  # Empty text
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
            logger.error("Processor/prep failed for audio+video batch at sec %d: %s", batch[0]["sec"], e)
            failed_seconds.extend([it["sec"] for it in batch])
            continue

        try:
            with torch.no_grad():
                outputs = model.thinker(**inputs, output_hidden_states=True, return_dict=True)
                last_hidden = outputs.hidden_states[-1]
                pooled = pooled_last_hidden_state(last_hidden, inputs["attention_mask"])
                embeds_all.append(pooled.detach().cpu().numpy())
        except Exception as e:
            logger.error("Model forward failed for audio+video batch at sec %d: %s", batch[0]["sec"], e)
            failed_seconds.extend([it["sec"] for it in batch])
            continue

    if not embeds_all:
        raise RuntimeError("No audio+video embeddings produced; all batches failed.")

    embeds = np.concatenate(embeds_all, axis=0)
    if failed_seconds:
        logger.warning("Audio+video failed seconds: %s", failed_seconds)
    else:
        logger.info("All audio+video seconds processed successfully.")
    return embeds

# ----------------------------
# Extract TEXT-ONLY embeddings
# ----------------------------
def extract_text_only_embeddings(
    model,
    processor,
    sec_items: List[Dict],
    transcript_dict: Dict[int, str]
) -> np.ndarray:
    """
    Extract embeddings from text only (no video/audio)
    """
    failed_seconds = []
    embeds_all = []

    logger.info("Extracting TEXT-ONLY embeddings...")
    for i in tqdm(range(0, len(sec_items), BATCH_SIZE), desc="Text-only batches", unit="batch"):
        batch = sec_items[i:i + BATCH_SIZE]

        conversations = []
        for it in batch:
            sec = it["sec"]
            text = transcript_dict.get(sec, "")
            
            # Log first few
            if i < 3:
                logger.info(f"Text-only second {sec}: {text[:100] if text else '[empty]'}...")
            
            # TEXT ONLY - no video/audio
            conversations.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text if text else " "},  # At least one space if empty
                    ],
                }
            ])

        try:
            text = processor.apply_chat_template(
                conversations, add_generation_prompt=False, tokenize=False
            )
            # No audio/video processing
            inputs = processor(
                text=text,
                return_tensors="pt",
                padding=True,
            ).to(model.device).to(model.dtype)
        except Exception as e:
            logger.error("Processor/prep failed for text batch at sec %d: %s", batch[0]["sec"], e)
            failed_seconds.extend([it["sec"] for it in batch])
            continue

        try:
            with torch.no_grad():
                outputs = model.thinker(**inputs, output_hidden_states=True, return_dict=True)
                last_hidden = outputs.hidden_states[-1]
                pooled = pooled_last_hidden_state(last_hidden, inputs["attention_mask"])
                embeds_all.append(pooled.detach().cpu().numpy())
        except Exception as e:
            logger.error("Model forward failed for text batch at sec %d: %s", batch[0]["sec"], e)
            failed_seconds.extend([it["sec"] for it in batch])
            continue

    if not embeds_all:
        raise RuntimeError("No text embeddings produced; all batches failed.")

    embeds = np.concatenate(embeds_all, axis=0)
    if failed_seconds:
        logger.warning("Text-only failed seconds: %s", failed_seconds)
    else:
        logger.info("All text-only seconds processed successfully.")
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

    # ---- Slice video ----
    t1 = time.time()
    try:
        sec_items = slice_video_v2(VIDEO_PATH, OUT_DIR, max_secs=MAX_SECS)
    except Exception as e:
        logger.exception("Slicing failed.")
        raise SystemExit(e)
    logger.info("Video slicing completed in %.2f sec", time.time() - t1)

    # ---- Load model ----
    t0 = time.time()
    model, processor = load_qwen()
    logger.info("Model + processor loaded in %.2f sec", time.time() - t0)
    
    # ---- Extract AUDIO+VIDEO embeddings ----
    logger.info("\n" + "="*70)
    logger.info("PHASE 1: Extracting AUDIO+VIDEO embeddings")
    logger.info("="*70)
    t2 = time.time()
    try:
        embeds_audio_video = extract_audio_video_embeddings(model, processor, sec_items)
    except Exception as e:
        logger.exception("Audio+video embedding extraction failed.")
        raise SystemExit(e)
    logger.info("Audio+video embedding extraction completed in %.2f sec", time.time() - t2)

    # Save audio+video embeddings
    np.save(SAVE_EMBEDDINGS_NPY_AUDIO_VIDEO, embeds_audio_video)
    logger.info("Saved audio+video embeddings: shape=%s -> %s", 
                embeds_audio_video.shape, SAVE_EMBEDDINGS_NPY_AUDIO_VIDEO)

    # ---- Extract TEXT-ONLY embeddings ----
    logger.info("\n" + "="*70)
    logger.info("PHASE 2: Extracting TEXT-ONLY embeddings")
    logger.info("="*70)
    t3 = time.time()
    try:
        embeds_text_only = extract_text_only_embeddings(model, processor, sec_items, transcript_dict)
    except Exception as e:
        logger.exception("Text-only embedding extraction failed.")
        raise SystemExit(e)
    logger.info("Text-only embedding extraction completed in %.2f sec", time.time() - t3)

    # Save text-only embeddings
    np.save(SAVE_EMBEDDINGS_NPY_TEXT_ONLY, embeds_text_only)
    logger.info("Saved text-only embeddings: shape=%s -> %s", 
                embeds_text_only.shape, SAVE_EMBEDDINGS_NPY_TEXT_ONLY)

    # ---- Summary ----
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"Total video segments: {len(sec_items)}")
    logger.info(f"Audio+video embeddings: {embeds_audio_video.shape}")
    logger.info(f"Text-only embeddings: {embeds_text_only.shape}")
    logger.info(f"Total runtime: {time.time() - start_total:.2f} sec ({(time.time() - start_total)/60:.2f} min)")
    logger.info("="*70)

if __name__ == "__main__":
    main()