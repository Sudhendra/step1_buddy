import os
import json
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

def extract_audio(video_path, audio_path):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)

def transcribe_audio(audio_path):
    r = sr.Recognizer()
    
    sound = AudioSegment.from_wav(audio_path)
    chunks = split_on_silence(sound, min_silence_len=700, silence_thresh=sound.dBFS-14, keep_silence=500)
    
    transcript = []
    for i, chunk in enumerate(tqdm(chunks, desc="Transcribing chunks", leave=False)):
        chunk_filename = f"chunk{i}.wav"
        chunk.export(chunk_filename, format="wav")
        with sr.AudioFile(chunk_filename) as source:
            audio = r.record(source)
        
        try:
            text = r.recognize_google(audio)
            transcript.append({
                "text": text,
                "start_time": sum(len(c) for c in chunks[:i]) / 1000,
                "end_time": sum(len(c) for c in chunks[:i+1]) / 1000
            })
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError:
            print("Could not request results")
        
        os.remove(chunk_filename)
    
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
    # , "GIT", "Cardio", "Neurology", "Endocrinology"
    topics = ["immunology"]
    base_dir = "/home/kambhamettu.s/Projects/step_vqa/data"
    
    for topic in tqdm(topics, desc="Processing topics"):
        topic_dir = os.path.join(base_dir, topic)
        output_file = f"{topic.lower()}_videos.json"
        process_topic(topic_dir, output_file)
        print(f"Processed {topic}. Output saved to {output_file}")

if __name__ == "__main__":
    main()
    # sent_tokenize("hello, how do u do")