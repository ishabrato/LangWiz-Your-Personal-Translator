import customtkinter as ctk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from PIL import ImageTk
import torch
import pyaudio
import wave
import numpy as np
import threading
import time
import sqlite3
import queue

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Load translation model
modelName = "llm/nllb"
whishperModel = "llm/whisper"
model = AutoModelForSeq2SeqLM.from_pretrained(f"{modelName}/model")
tokenizer = AutoTokenizer.from_pretrained(f"{modelName}/tokenizer")

# Load Whisper model
whisper_model = WhisperForConditionalGeneration.from_pretrained(f"{whishperModel}/model")
whisper_processor = WhisperProcessor.from_pretrained(f"{whishperModel}/processor")

def translate(input_txt, tgt_lang):
    inputs = tokenizer(input_txt, return_tensors="pt")
    translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang))
    result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return result

def transcribe_audio(audio):
    input_features = whisper_processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    predicted_ids = whisper_model.generate(input_features)
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]

langList = {
    "English": "eng_Latn",
    "Bengali": "ben_Beng",
    "Hindi": "hin_Deva",
    "Sanskrit": "san_Deva",
    "Gujarati": "guj_Gujr"
}

class EnhancedTranslatorUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("LangWiz")
        self.geometry("1000x700")
        self.iconpath = ImageTk.PhotoImage(file="assets/logo.jpg")
        self.wm_iconbitmap()
        self.iconphoto(False, self.iconpath)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(3, weight=1)

        self.db_queue = queue.Queue()
        self.setup_database()
        self.setup_ui()

    def setup_database(self):
        self.conn = sqlite3.connect('translation_history.db', check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS translations
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
             input_text TEXT,
             output_text TEXT,
             target_language TEXT,
             timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
        ''')
        self.conn.commit()

    def setup_ui(self):
        # Tabview
        self.tabview = ctk.CTkTabview(self.main_frame)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.tabview.add("Translator")
        self.tabview.add("History")

        # Translator Tab
        self.translator_tab = self.tabview.tab("Translator")
        self.translator_tab.grid_columnconfigure(0, weight=1)
        self.translator_tab.grid_rowconfigure(3, weight=1)

        # Header
        self.header = ctk.CTkLabel(self.translator_tab, text="LangWiz: Your Personal Translator", font=ctk.CTkFont(size=24, weight="bold"))
        self.header.grid(row=0, column=0, padx=20, pady=(0, 20))

        # Target language selection
        self.lang_frame = ctk.CTkFrame(self.translator_tab)
        self.lang_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 20))
        self.lang_frame.grid_columnconfigure(1, weight=1)

        self.tgt_lang_var = ctk.StringVar(value="Bengali")

        ctk.CTkLabel(self.lang_frame, text="Target Language:").grid(row=0, column=0, padx=(0, 10), pady=10)
        self.tgt_lang_menu = ctk.CTkOptionMenu(self.lang_frame, values=list(langList.keys()), variable=self.tgt_lang_var)
        self.tgt_lang_menu.grid(row=0, column=1, sticky="ew", padx=10, pady=10)

        # Text areas
        self.text_frame = ctk.CTkFrame(self.translator_tab)
        self.text_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 20))
        self.text_frame.grid_columnconfigure((0, 1), weight=1)
        self.text_frame.grid_rowconfigure(0, weight=1)

        # Input area with label
        self.input_label = ctk.CTkLabel(self.text_frame, text="Input Text:")
        self.input_label.grid(row=0, column=0, sticky="w", padx=(0, 10), pady=5)

        self.input_text = ctk.CTkTextbox(self.text_frame, height=200)
        self.input_text.grid(row=1, column=0, sticky="nsew", padx=(0, 10), pady=10)

        # Output area with label
        self.output_label = ctk.CTkLabel(self.text_frame, text="Output Text:")
        self.output_label.grid(row=0, column=1, sticky="w", padx=(10, 0), pady=5)

        self.output_text = ctk.CTkTextbox(self.text_frame, height=200)
        self.output_text.grid(row=1, column=1, sticky="nsew", padx=(10, 0), pady=10)

        # Buttons
        self.button_frame = ctk.CTkFrame(self.translator_tab)
        self.button_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 20))
        self.button_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.translate_button = ctk.CTkButton(self.button_frame, text="Translate", command=self.translate_text, font=ctk.CTkFont(size=16, weight="bold"))
        self.translate_button.grid(row=0, column=1, sticky="ew", padx=10, pady=10)

        self.clear_button = ctk.CTkButton(self.button_frame, text="Clear", command=self.clear_text, font=ctk.CTkFont(size=16))
        self.clear_button.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        self.copy_button = ctk.CTkButton(self.button_frame, text="Copy", command=self.copy_text, font=ctk.CTkFont(size=16))
        self.copy_button.grid(row=0, column=2, sticky="ew", padx=10, pady=10)

        self.record_button = ctk.CTkButton(self.button_frame, text="Record", command=self.toggle_recording, font=ctk.CTkFont(size=16))
        self.record_button.grid(row=0, column=3, sticky="ew", padx=10, pady=10)

        # Status
        self.status_label = ctk.CTkLabel(self.translator_tab, text="Ready", font=ctk.CTkFont(size=14))
        self.status_label.grid(row=4, column=0, sticky="w", padx=20, pady=(0, 10))

        self.is_recording = False

        # History Tab
        self.history_tab = self.tabview.tab("History")
        self.history_tab.grid_columnconfigure(0, weight=1)
        self.history_tab.grid_rowconfigure(0, weight=1)

        self.history_text = ctk.CTkTextbox(self.history_tab, height=400)
        self.history_text.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.history_text.configure(state="disabled")

        # Refresh history button
        self.refresh_button = ctk.CTkButton(self.history_tab, text="Refresh History", command=self.update_history, font=ctk.CTkFont(size=16))
        self.refresh_button.grid(row=1, column=0, sticky="ew", padx=20, pady=10)

        # Load initial history
        self.update_history()

        # Start the database worker
        self.after(100, self.process_db_queue)

    def translate_text(self):
        input_txt = self.input_text.get("1.0", "end-1c")
        tgt_lang = langList[self.tgt_lang_var.get()]

        if not input_txt:
            self.status_label.configure(text="Please enter text to translate")
            return

        self.status_label.configure(text="Translating...")
        self.translate_button.configure(state="disabled")

        def translate_thread():
            start_time = time.time()
            try:
                result = translate(input_txt, tgt_lang)
                self.output_text.delete("1.0", "end")
                self.output_text.insert("1.0", result)
                end_time = time.time()
                self.status_label.configure(text=f"Translation completed in {end_time - start_time:.2f} seconds")

                # Add to database queue
                self.db_queue.put(('insert', (input_txt, result, self.tgt_lang_var.get())))

                self.after(0, self.update_history)
            except Exception as e:
                self.status_label.configure(text=f"Error: {str(e)}")
            finally:
                self.translate_button.configure(state="normal")

        threading.Thread(target=translate_thread).start()

    def process_db_queue(self):
        try:
            while True:
                action, data = self.db_queue.get_nowait()
                if action == 'insert':
                    self.cursor.execute('''
                        INSERT INTO translations (input_text, output_text, target_language)
                        VALUES (?, ?, ?)
                    ''', data)
                    self.conn.commit()
                elif action == 'select':
                    self.cursor.execute('''
                        SELECT input_text, output_text, target_language, timestamp
                        FROM translations
                        ORDER BY timestamp DESC
                        LIMIT 50
                    ''')
                    history = self.cursor.fetchall()
                    self.after(0, lambda: self.update_history_display(history))
        except queue.Empty:
            pass
        finally:
            self.after(100, self.process_db_queue)

    def update_history(self):
        self.db_queue.put(('select', None))

    def update_history_display(self, history):
        self.history_text.configure(state="normal")
        self.history_text.delete("1.0", "end")
        for entry in history:
            self.history_text.insert("end", f"Input: {entry[0]}\nOutput: {entry[1]}\nLanguage: {entry[2]}\nTimestamp: {entry[3]}\n\n")
        self.history_text.configure(state="disabled")

    def clear_text(self):
        self.input_text.delete("1.0", "end")
        self.output_text.delete("1.0", "end")
        self.status_label.configure(text="Cleared")

    def copy_text(self):
        self.clipboard_clear()
        self.clipboard_append(self.output_text.get("1.0", "end-1c"))
        self.status_label.configure(text="Copied to clipboard")

    def toggle_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.record_button.configure(text="Stop Recording")
            self.status_label.configure(text="Recording...")
            threading.Thread(target=self.record_audio).start()
        else:
            self.is_recording = False
            self.record_button.configure(text="Record")

    def record_audio(self):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        RECORD_SECONDS = 5
        WAVE_OUTPUT_FILENAME = "output.wav"

        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        self.status_label.configure(text="Processing audio...")

        try:
            with wave.open(WAVE_OUTPUT_FILENAME, 'rb') as wf:
                audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                audio = audio.astype(np.float32) / 32768.0

            transcription = transcribe_audio(audio)
            self.input_text.delete("1.0", "end")
            self.input_text.insert("1.0", transcription)
            self.status_label.configure(text="Audio transcribed. Ready to translate.")
        except Exception as e:
            self.status_label.configure(text=f"Error in transcription: {str(e)}")

        self.is_recording = False
        self.record_button.configure(text="Record")

if __name__ == "__main__":
    app = EnhancedTranslatorUI()
    app.mainloop()