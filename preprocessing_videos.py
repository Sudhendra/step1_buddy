import os
import json
from moviepy.editor import VideoFileClip
import whisper
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import nltk
import torch
nltk.download('punkt', quiet=True)

def extract_audio(video_path, audio_path):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)

def transcribe_audio(audio_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base").to(device)
    result = model.transcribe(audio_path)
    
    transcript = []
    for segment in result["segments"]:
        transcript.append({
            "text": segment["text"],
            "start_time": segment["start"],
            "end_time": segment["end"]
        })
    
    return transcript

def process_video(video_path, topic):
    video_filename = os.path.basename(video_path)
    audio_path = f"temp_{video_filename}.wav"
    
    with tqdm(total=3, desc=f"Processing {video_filename}", leave=False) as pbar:
        # Extract audio from video
        extract_audio(video_path, audio_path)
        pbar.update(1)
        
        # Transcribe audio
        transcript = transcribe_audio(audio_path)
        pbar.update(1)
        
        # Clean up temporary audio file
        os.remove(audio_path)
        
        # Create sentences from transcript
        sentences = []
        for t in transcript:
            for sentence in sent_tokenize(t["text"]):
                sentences.append({
                    "text": sentence,
                    "video_title": video_filename,
                    "timestamp": t["start_time"],
                    "video_path": video_path
                })
        pbar.update(1)
    
    return sentences

def process_topic(topic_dir, output_file):
    all_sentences = []
    video_files = [f for f in os.listdir(topic_dir) if f.endswith(".mp4")]
    
    for video_file in tqdm(video_files, desc=f"Processing videos in {os.path.basename(topic_dir)}"):
        video_path = os.path.join(topic_dir, video_file)
        all_sentences.extend(process_video(video_path, os.path.basename(topic_dir)))
    
    with open(output_file, 'w') as f:
        json.dump(all_sentences, f, indent=2)

def main():
    topics = ["immunology"]  # Add other topics as needed
    base_dir = "/home/kambhamettu.s/Projects/step_vqa/data"
    
    for topic in tqdm(topics, desc="Processing topics"):
        topic_dir = os.path.join(base_dir, topic)
        output_file = f"{topic.lower()}_videos.json"
        process_topic(topic_dir, output_file)
        print(f"Processed {topic}. Output saved to {output_file}")

if __name__ == "__main__":
    main()