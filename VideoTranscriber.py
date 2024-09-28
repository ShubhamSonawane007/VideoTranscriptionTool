import whisper
import os
import cv2
from moviepy.editor import ImageSequenceClip, AudioFileClip, VideoFileClip
from tqdm import tqdm
from tkinter import messagebox
from tkinter import filedialog
from tkinter import StringVar, END
import threading
import platform
import subprocess
import json
import pyaudio
from vosk import Model, KaldiRecognizer
import customtkinter as ctk
import re
import time

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class VideoTranscriber:
    def __init__(self, model_path, video_path):
        self.model = whisper.load_model(model_path, download_root='models/whisper-model')
        self.video_path = video_path
        self.audio_path = ''
        self.text_array = []
        self.fps = 0
        self.char_width = 0

    def transcribe_video(self):
        print('Transcribing video')
        result = self.model.transcribe(self.audio_path)
        text = result["segments"][0]["text"]
        textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret, frame = cap.read()
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.char_width = int(textsize[0] / len(text))

        for j in tqdm(result["segments"]):
            lines = []
            text = j["text"]
            end = j["end"]
            start = j["start"]
            total_frames = int((end - start) * self.fps)
            start = start * self.fps
            total_chars = len(text)
            words = text.split(" ")
            i = 0

            while i < len(words):
                words[i] = words[i].strip()
                if words[i] == "":
                    i += 1
                    continue
                length_in_pixels = len(words[i]) * self.char_width
                remaining_pixels = width - length_in_pixels
                line = words[i]

                while remaining_pixels > 0:
                    i += 1
                    if i >= len(words):
                        break
                    length_in_pixels = len(words[i]) * self.char_width
                    remaining_pixels -= length_in_pixels
                    if remaining_pixels < 0:
                        continue
                    else:
                        line += " " + words[i]

                line_array = [line, int(start) + 15, int(len(line) / total_chars * total_frames) + int(start) + 15]
                start = int(len(line) / total_chars * total_frames) + int(start)
                lines.append(line_array)
                self.text_array.append(line_array)

        cap.release()
        print('Transcription complete')

    def extract_audio(self):
        print('Extracting audio')
        audio_path = os.path.join(os.path.dirname(self.video_path), "audio.mp3")
        video = VideoFileClip(self.video_path)
        audio = video.audio 
        audio.write_audiofile(audio_path)
        self.audio_path = audio_path
        print('Audio extracted')

    def extract_frames(self, output_folder):
        print('Extracting frames')
        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        N_frames = 0

        # Reference width for scaling
        reference_width = 1280  

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            for i in self.text_array:
                if N_frames >= i[1] and N_frames <= i[2]:
                    text = i[0]
                    # Calculate scaling factor
                    scaling_factor = width / reference_width
                    # Adjust text size and thickness dynamically
                    font_scale = 0.8 * scaling_factor
                    thickness = int(2 * scaling_factor)
                    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    text_x = int((frame.shape[1] - text_size[0]) / 2)
                    text_y = int(height * 0.9)
                    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
                    break

            cv2.imwrite(os.path.join(output_folder, str(N_frames) + ".jpg"), frame)
            N_frames += 1

        cap.release()
        print('Frames extracted')


    def create_video(self, output_video_path):
        print('Creating video')
        image_folder = os.path.join(os.path.dirname(self.video_path), "frames")
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        self.extract_frames(image_folder)

        images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
        images.sort(key=lambda x: int(x.split(".")[0]))

        clip = ImageSequenceClip([os.path.join(image_folder, image) for image in images], fps=self.fps)
        audio = AudioFileClip(self.audio_path)
        clip = clip.set_audio(audio)
        clip.write_videofile(output_video_path)
        for img in images:
            os.remove(os.path.join(image_folder, img))
        os.rmdir(image_folder)
        os.remove(os.path.join(os.path.dirname(self.video_path), "audio.mp3"))

def post_process(text):
    # Capitalize first letter of sentences
        text = '. '.join(sentence.capitalize() for sentence in text.split('. '))

        # Add full stop at the end if missing
        if text and text[-1] not in '.!?':
            text += '.'

        # Correct common mistakes (expand as needed)
        corrections = {
            'i ': 'I ',
            'im ': "I'm ",
            'dont ': "don't ",
            'cant ': "can't ",
        }
        for wrong, right in corrections.items():
            text = re.sub(r'\b' + wrong, right, text, flags=re.IGNORECASE)

        return text

def post_process_hindi(text):
    # Capitalize first letter of sentences (if applicable in Hindi)
    text = '। '.join(sentence.capitalize() for sentence in text.split('। '))
    
    # Add full stop at the end if missing
    if text and text[-1] not in '।!?':
        text += '।'
    
    # Correct common mistakes (expand as needed)
    corrections = {
        'मै ': 'मैं ',
        'हे ': 'है ',
        # Add more Hindi-specific corrections
    }
    for wrong, right in corrections.items():
        text = re.sub(r'\b' + wrong, right, text)
    
    return text

class SubtitleGenerator:
    def __init__(self, text_display, source_var, language_var):
        self.text_display = text_display
        self.source_var = source_var
        self.language_var = language_var
        self.is_running = False
        self.audio_thread = None
        self.models = {
            "english": Model(os.path.abspath("models/vosk-model-en-in-0.5")),
            "hindi": Model(os.path.abspath("models/vosk-model-hi-0.22")),
        }
        self.rec = None 
        self.subtitle_file = "subtitles.txt"
        if os.path.exists(self.subtitle_file):
            os.remove(self.subtitle_file)
        self.full_text = ""
        self.stop_event = threading.Event()

    def start(self, start_button, stop_button):
        self.is_running = True
        self.stop_event.clear()
        start_button.configure(state=ctk.DISABLED)
        stop_button.configure(state=ctk.NORMAL)
        self.rec = KaldiRecognizer(self.models[self.language_var.get()], 44100)
        self.audio_thread = threading.Thread(target=self.capture_audio)
        self.audio_thread.start()

    def stop(self, start_button, stop_button):
        self.is_running = False
        self.stop_event.set()
        start_button.configure(state=ctk.NORMAL)
        stop_button.configure(state=ctk.DISABLED)
        if self.audio_thread:
            self.audio_thread.join(timeout=5)  # Wait up to 5 seconds for thread to finish
            if self.audio_thread.is_alive():
                print("Warning: Audio thread did not terminate properly.")


    def capture_audio(self):
        p = pyaudio.PyAudio()
        device_index = None
    
        # Automatically select default microphone
        if self.source_var.get() == "microphone":
            for i in range(p.get_device_count()):
                dev = p.get_device_info_by_index(i)
                # Check if the device is an input device (microphone)
                if dev['maxInputChannels'] > 0:
                    print(f"Found input device: {dev['name']} (Index {i})")
                    # Select the first input device or customize based on your preference
                    device_index = i
                    break
                
        # If no system audio device is selected, use the default microphone
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, 
                        input_device_index=device_index, frames_per_buffer=4000)
        stream.start_stream()
    
        try:
            while not self.stop_event.is_set():
                if not self.is_running:
                    break
                data = stream.read(2000, exception_on_overflow=False)
                if len(data) == 0:
                    break
                if self.rec.AcceptWaveform(data):
                    result = json.loads(self.rec.Result())
                    text = result.get("text", "")
                    self.display_subtitles(text, final=True)
                else:
                    partial_result = json.loads(self.rec.PartialResult())
                    partial_text = partial_result.get("partial", "")
                    self.display_subtitles(partial_text, final=False)
        except Exception as e:
            print(f"Error while capturing audio: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("Audio capture stopped")
    
    
    def display_subtitles(self, text, final=False):
        if final:
            if self.language_var.get() == "hindi":
                processed_text = post_process_hindi(text)
            else:
                processed_text = post_process(text)
            self.full_text += processed_text + " "
            with open(self.subtitle_file, "w", encoding="utf-8") as f:
                f.write(self.full_text.strip() + "\n")
        self.text_display.delete(1.0, END)
        self.text_display.insert(END, self.full_text + text)

class VideoTranscriberApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Transcriber & Real-Time Subtitle Generator")
        self.root.geometry("800x600")

        self.model_path = "base"
        self.output_video_path = ''
        self.text_display = ctk.CTkTextbox(root, wrap='word', font=('Arial', 14))
        self.source_var = StringVar(value="microphone")
        self.language_var = StringVar(value="english")
        self.subtitle_gen = SubtitleGenerator(self.text_display, self.source_var, self.language_var)

        # Create widgets
        self.file_label = ctk.CTkLabel(root, text="Video File:")
        self.file_entry = ctk.CTkEntry(root, width=500)
        self.browse_button = ctk.CTkButton(root, text="Browse", command=self.browse_file)
        self.start_transcription_button = ctk.CTkButton(root, text="Generate Video", command=self.start_transcription)
        self.play_button = ctk.CTkButton(root, text="Play Transcribed Video", command=self.play_video, state=ctk.DISABLED)
        self.progress = ctk.CTkProgressBar(root)
        self.progress.set(0)

        self.start_button = ctk.CTkButton(root, text="Start Transcription", command=self.start_subtitle_generation)
        self.stop_button = ctk.CTkButton(root, text="Stop Transcription", command=self.stop_subtitle_generation, state=ctk.DISABLED)
        self.microphone_radio = ctk.CTkRadioButton(root, text="Microphone", variable=self.source_var, value="microphone")
        # self.system_audio_radio = ctk.CTkRadioButton(root, text="System Audio", variable=self.source_var, value="system_audio")
        self.text_display_label = ctk.CTkLabel(root, text="Transcribed Text:")
        self.english_radio = ctk.CTkRadioButton(root, text="English", variable=self.language_var, value="english")
        self.hindi_radio = ctk.CTkRadioButton(root, text="Hindi", variable=self.language_var, value="hindi")

        # Layout widgets
        self.file_label.grid(row=0, column=0, padx=10, pady=10, sticky='w')
        self.file_entry.grid(row=0, column=1, padx=10, pady=10, sticky='ew')
        self.browse_button.grid(row=0, column=2, padx=10, pady=10, sticky='w')
        self.start_transcription_button.grid(row=1, column=1, padx=10, pady=10, sticky='ew')
        self.play_button.grid(row=1, column=2, padx=10, pady=10, sticky='w')
        self.progress.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky='ew')

        self.microphone_radio.grid(row=3, column=0, padx=10, pady=10, sticky='w')
        # self.system_audio_radio.grid(row=3, column=1, padx=10, pady=10, sticky='w')
        self.start_button.grid(row=4, column=0, padx=10, pady=10, sticky='w')
        self.stop_button.grid(row=4, column=1, padx=10, pady=10, sticky='w')
        self.text_display_label.grid(row=5, column=0, padx=10, pady=10, sticky='w')
        self.text_display.grid(row=6, column=0, columnspan=3, padx=10, pady=10, sticky='nsew')
        self.english_radio.grid(row=3, column=2, padx=10, pady=10, sticky='w')
        self.hindi_radio.grid(row=4, column=2, padx=10, pady=10, sticky='w')

        self.root.grid_rowconfigure(6, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")])
        if file_path:
            self.file_entry.delete(0, END)
            self.file_entry.insert(0, file_path)

    def start_transcription(self):
        video_path = self.file_entry.get()
        if not video_path:
            messagebox.showerror("Error", "Please select a video file.")
            return

        self.output_video_path = os.path.join(os.path.dirname(video_path), "output.mp4")
        self.video_transcriber = VideoTranscriber(self.model_path, video_path)
        self.progress.start()
        self.play_button.configure(state=ctk.DISABLED)

        def run_transcription():
            self.video_transcriber.extract_audio()
            self.video_transcriber.transcribe_video()
            self.video_transcriber.create_video(self.output_video_path)
            self.progress.set(1)
            self.progress.stop()
            self.play_button.configure(state=ctk.NORMAL)
            messagebox.showinfo("Success", "Transcription complete! Output video saved as 'output.mp4'.")

        threading.Thread(target=run_transcription).start()

    def get_audio_devices(self):
        # Retrieve audio input devices using pyaudio
        p = pyaudio.PyAudio()
        devices = []
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev['maxInputChannels'] > 0:  # Only list input devices
                devices.append(dev['name'])
        return devices

    def play_video(self):
        if self.output_video_path:
            if platform.system() == 'Darwin':  # macOS
                subprocess.call(('open', self.output_video_path))
            elif platform.system() == 'Windows':  # Windows
                os.startfile(self.output_video_path)
            else:  # Linux
                subprocess.call(('xdg-open', self.output_video_path))

    def start_subtitle_generation(self):
        self.subtitle_gen.start(self.start_button, self.stop_button)

    def stop_subtitle_generation(self):
        self.subtitle_gen.stop(self.start_button, self.stop_button)
        self.root.after(100, self.check_thread_status)

    def check_thread_status(self):
        if self.subtitle_gen.audio_thread and self.subtitle_gen.audio_thread.is_alive():
            self.root.after(100, self.check_thread_status)
        else:
            print("Transcription stopped successfully")
    

if __name__ == "__main__":
    root = ctk.CTk()
    app = VideoTranscriberApp(root)
    root.mainloop()
