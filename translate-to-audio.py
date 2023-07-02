import queue
import concurrent.futures
import sounddevice as sd
from transformers import WhisperProcessor, WhisperForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import numpy as np
import soundfile as sf
from datasets import load_dataset
import torch
import textwrap
import threading

recording = []
is_recording = True
q = queue.Queue()

def clean_transcription(transcription):
    transcription_text = transcription[0]
    special_tokens = ['<|startoftranscript|>', '<|en|>', '<|transcribe|>', '<|notimestamps|>', '<|endoftext|>', '<|translate|>']
    for token in special_tokens:
        transcription_text = transcription_text.replace(token, '')
    cleaned_transcription = ' '.join(transcription_text.split())
    return cleaned_transcription


def record_audio(samplerate=16000):
    global recording
    global is_recording
    # Start recording
    print("Starting recording...")
    while is_recording:
        recording.extend(sd.rec(int(1 * samplerate), samplerate=samplerate, channels=2, dtype='float64'))
    print("Recording stopped.")
    # Save the recording to a file
    sf.write("recording.wav", np.array(recording), samplerate)
    return np.array(recording)

def transcribe_audio(recording, samplerate=16000):
    audio_mono = np.mean(recording, axis=1)
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="translate")
    input_features = processor(audio_mono, sampling_rate=samplerate, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    transcription = processor.batch_decode(predicted_ids)
    
    cleaned_transcription = clean_transcription(transcription)
    return cleaned_transcription

def translate_text(input_text, source_lang="English", target_lang="German"):
    model_name = 't5-large'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    text_chunks = textwrap.wrap(input_text, width=400)
    translated_text_chunks = []
    for text_chunk in text_chunks:
        inputs = tokenizer.encode("Translate English to German: " + text_chunk, return_tensors='pt', max_length=512, truncation=True)
        outputs = model.generate(inputs, max_length=400, num_beams=4, early_stopping=True)
        translated_text_chunk = tokenizer.decode(outputs[0])
        translated_text_chunks.append(translated_text_chunk)
    translated_text = ' '.join(translated_text_chunks)
    cleaned_text = translated_text.replace('<pad>', '').replace('</s>', '').strip()
    print(cleaned_text)
    return cleaned_text


def text_to_speech(totranslate):
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    inputs = processor(text=totranslate, return_tensors="pt")
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    with torch.no_grad():
        speech = vocoder(spectrogram)
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    sf.write("output.wav", speech.numpy(), samplerate=16000)

def main():
    global is_recording
    recording_thread = threading.Thread(target=record_audio)
    recording_thread.start()
    input("Press Enter to stop recording...")
    is_recording = False
    recording_thread.join()
    
    # Read the recording from the file
    recording, samplerate = sf.read("recording.wav")
    
    transcription = transcribe_audio(recording, samplerate)
    translated_text = translate_text(transcription)
    text_to_speech(translated_text)

if __name__ == "__main__":
    main()

