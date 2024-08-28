import os
# import librosa
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sys

BATCH_SIZE = 3
def split_audio(input_file, output_dir, max_duration=30, min_silence_len=500, silence_thresh=-50):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # sound = AudioSegment.from_file(input_file, format="m4a")
    # sound.export(output_dir + "/temp.wav", format="wav")


    # audio = AudioSegment.from_wav(output_dir + "/temp.wav")
    audio = AudioSegment.from_file(input_file, format="m4a")

    nonsilent_ranges = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)


    # current_time = 0
    # segment_index = 0
    segments = []
    segment_times = []

    for start, end in nonsilent_ranges:
        segment_length = (end - start) / 1000

        while segment_length > max_duration:
            new_end = start + (max_duration * 1000)
            segment = audio[start:new_end]
            segments.append(segment)
            segment_times.append(start / 1000)
            start = new_end
            segment_length -= max_duration

        segment = audio[start:end]
        segments.append(segment)
        segment_times.append(start / 1000)

    # save all segments
    for i, segment in enumerate(segments):
        output_file = os.path.join(output_dir, f"segment_{i}.wav")
        segment.export(output_file, format="wav")

    # os.remove(output_dir + "/temp.wav")

    return [os.path.join(output_dir, f"segment_{i}.wav") for i in range(len(segments))], segment_times

def transcribe_audio(audio_files, pipe):

    transcriptions = []

    batch_size = BATCH_SIZE
    for i in range(len(audio_files) // batch_size):
        files = audio_files[i*batch_size:(i+1)*batch_size]
            
        try:
            # transcribe audio files
            transcription = pipe(files, batch_size=batch_size)
            print(transcription)
            transcriptions.extend([x["text"] for x in transcription])

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error transcribing audio: {e}")

    files = audio_files[-1 * (len(audio_files) % batch_size):]
    print(transcription)
    try:
        # transcribe audio files
        transcription = pipe(files, batch_size=len(files))
        print(transcription)
        transcriptions.extend([x["text"] for x in transcription])
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error transcribing audio: {e}")
    return transcriptions


def generate_srt(transcriptions, start_times, durations, output_file):
    with open(output_file + "_plain_text.txt", 'w', encoding='utf-8') as f:
        for i, transcription in enumerate(transcriptions):
            f.write(f"{transcription}\n\n")


    with open(output_file + ".srt", 'w', encoding='utf-8') as f:
        for i, (transcription, start_time, duration) in enumerate(zip(transcriptions, start_times, durations)):
            start_time_str = format_time(start_time)
            end_time_str = format_time(start_time + duration)

            f.write(f"{i + 1}\n")
            f.write(f"{start_time_str} --> {end_time_str}\n")
            f.write(f"{transcription}\n\n")

# format timestamp in HH:MM:SS,mmm format
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"



def e2e(input_file, pipe):
    print("processing file: ", input_file)
    segment_dir = input_file + "_seg"
    output_file = segment_dir + "/" + "output"
    # split audio file by silence or max duration
    audio_files, segment_times = split_audio(input_file, segment_dir, max_duration=30)

    # get duration of each segment
    durations = [(len(AudioSegment.from_wav(file)) / 1000) for file in audio_files]

    transcriptions = transcribe_audio(audio_files, pipe)

    # generate srt and txt file
    generate_srt(transcriptions, segment_times, durations, output_file)

    print(f"SRT file generated: {output_file}")


if __name__ == "__main__":

        
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16

    model_id = "openai/whisper-large-v3"


    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    files = os.listdir('./Kiir')
    for file in files:
        if file.endswith('.m4a'):
            input_file = "./Kiir/" + file
            e2e(input_file, pipe)

    # input_file = sys.argv[1]
    # e2e(input_file, pipe)

