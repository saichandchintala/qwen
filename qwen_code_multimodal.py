# import os
# import torch
# import numpy as np
# from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
# from qwen_omni_utils import process_mm_info
# from tqdm import tqdm
# import subprocess

# MODEL_PATH = "/storage/scratch/saichandc/Qwen3-Omni-30B-A3B-Thinking"
# VIDEO_PATH = "/storage/home/saichandc/video/90secvideo.mp4"
# USE_AUDIO_IN_VIDEO = True
# FIXED_TEXT = "Commission on Presidential"

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

import numpy as np
import torch
from tqdm import tqdm
from moviepy import VideoFileClip  # MoviePy v2 import (not moviepy.editor)

from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

# ----------------------------
# Configuration
# ----------------------------
MODEL_PATH = "/storage/scratch/saichandc/Qwen3-Omni-30B-A3B-Thinking"   # or "Qwen/Qwen3-Omni-30B-A3B-Thinking"
VIDEO_PATH = "/storage/home/saichandc/video/90secvideo.mp4"                                  # <-- change to your video path
OUT_DIR = "./tmp_slices_v2"
SAVE_EMBEDDINGS_NPY = "./second_level_embeddings.npy"

PLACEHOLDER_TEXT = "Process this 1-second slice."
USE_AUDIO_IN_VIDEO = True
BATCH_SIZE = 2
MAX_SECS = None     # only first 2 seconds now; set None for full length later

LOG_LEVEL = logging.INFO
os.makedirs(OUT_DIR, exist_ok=True)
OUT_DIR = str(Path(OUT_DIR).resolve())  # <-- absolutize output dir

logging.basicConfig(level=LOG_LEVEL, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("qwen_embeddings")

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
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_PATH, dtype="auto", device_map="auto", trust_remote_code=True,
        )
        processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True,)
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
def slice_video_v2(video_path: str, out_dir: str, max_secs: int | None) -> List[Dict]:
    """
    Export 1-second segments with MoviePy v2 `.subclipped`.
    Return [{"sec": t, "video_uri": file:///..., "video_path": /abs/path.mp4}, ...]
    """
    vp = Path(video_path).resolve()
    if not vp.exists():
        raise FileNotFoundError(f"Video not found: {vp}")

    try:
        clip = VideoFileClip(str(vp))
    except Exception as e:
        logger.exception("Failed to open video with MoviePy v2.")
        raise

    duration = float(clip.duration or 0.0)
    total_secs = math.ceil(duration)

    if max_secs is None:
        n_secs = total_secs
        logger.info("Processing FULL length: %d second(s).", n_secs)
    else:
        n_secs = min(max_secs, total_secs)
        logger.info("Processing FIRST %d second(s) (of %d).", n_secs, total_secs)

    items: List[Dict] = []
    logger.info("Slicing %s → %d 1-second segment(s)...", vp.name, n_secs)

    fps = getattr(clip, "fps", None)

    for t in tqdm(range(n_secs), desc="Slicing seconds", unit="s"):
        start, end = t, min(t + 1, duration)
        if end <= start:
            continue
        try:
            sub = clip.subclipped(start, end)  # MoviePy v2 API
            out_file = Path(out_dir) / f"seg_{t:06d}.mp4"

            # IMPORTANT: write without 'verbose'/'logger' (v2); keep audio
            if fps is not None:
                sub.write_videofile(str(out_file), fps=fps, audio=True, codec="libx264", audio_codec="aac")
            else:
                sub.write_videofile(str(out_file), audio=True, codec="libx264", audio_codec="aac")

            sub.close()

            # Build proper absolute path + file URI
            abs_path = out_file.resolve()
            items.append({
                "sec": t,
                "video_path": str(abs_path),
                "video_uri": abs_path.as_uri(),  # e.g., file:///storage/home/...
            })
        except Exception as se:
            logger.error("Failed to write segment %d: %s", t, se)
            continue

    clip.close()
    if not items:
        raise RuntimeError("No segments produced. Check the video and codecs.")
    logger.info("Created %d segment(s) in %s", len(items), out_dir)
    return items

# ----------------------------
# Build conversations & extract embeddings
# ----------------------------
def extract_second_level_embeddings(model, processor, sec_items: List[Dict]) -> np.ndarray:
    failed_seconds = []
    embeds_all = []

    logger.info("Building per-second conversations and extracting embeddings...")
    for i in tqdm(range(0, len(sec_items), BATCH_SIZE), desc="Embedding batches", unit="batch"):
        batch = sec_items[i:i + BATCH_SIZE]

        conversations = []
        for it in batch:
            # You can use either 'video_uri' (file:///...) or 'video_path' (plain absolute path).
            # Both are accepted by qwen-omni-utils (docs show 'file:///...' explicitly).  # see PyPI page
            conversations.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": it["video_uri"]},
                        {"type": "text",  "text": PLACEHOLDER_TEXT},
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
                outputs = model.thinker(**inputs,output_hidden_states=True,return_dict=True)
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
        embeds = extract_second_level_embeddings(model, processor, sec_items)
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

