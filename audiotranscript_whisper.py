import whisper
import csv

def get_transcript_by_second(video_path, output_path="transcript_by_second_first_debate.csv"):
    # Load model with GPU
    model = whisper.load_model("large", device="cuda")  # Use GPU

    # Transcribe with word timestamps
    result = model.transcribe(video_path, word_timestamps=True)

    # Organize by second
    transcript_by_second = {}

    for segment in result["segments"]:
        if "words" in segment:
            for word_info in segment["words"]:
                second = int(word_info["start"])
                word = word_info["word"].strip()
                if second not in transcript_by_second:
                    transcript_by_second[second] = []
                transcript_by_second[second].append(word)

    # Prepare output for all seconds in the video
    max_second = max(transcript_by_second.keys()) if transcript_by_second else 0
    with open(output_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["second", "transcript"])
        for second in range(max_second + 1):
            words = transcript_by_second.get(second, [])
            text = " ".join(words) if words else ""
            writer.writerow([second, text])

    print(f"Transcript saved to {output_path}")
    return transcript_by_second

# Usage
transcript = get_transcript_by_second("/storage/home/saichandc/video/file1.mp4")
