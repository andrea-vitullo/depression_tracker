import os
from pydub import AudioSegment

from utils import audio_utils, transcripts_utils
import my_config


for path, directories, files in os.walk(my_config.DIRECTORY):
    for file in files:
        if file.endswith(my_config.START_FORMAT):
            tempaudio, df, _, patient_id = transcripts_utils.get_participant_info(path)
            audio_utils.diarisation(tempaudio, df, path)

for path, directories, files in os.walk(my_config.DIRECTORY):
    combined = None
    participant_audio_filename = None
    audio_files_to_delete = []

    # Retrieve the original audio filename if exists
    for file in files:
        if file.endswith(my_config.START_FORMAT):
            participant_audio_filename = file.replace(my_config.START_FORMAT, my_config.FINAL_FORMAT)
            break

    if participant_audio_filename:
        print(f"Found participant audio file: {participant_audio_filename}")

        # Sort the audio split files based on the start time
        sorted_audio_files = sorted(
            [file for file in files if file.endswith(my_config.SPLIT_FORMAT)],
            key=lambda x: float(x.split('_')[0])
        )

        # Concatenate the sorted audio files
        for audio in sorted_audio_files:
            audio_path = os.path.join(path, audio)
            audio_files_to_delete.append(audio_path)
            audio_data = AudioSegment.from_wav(audio_path)
            combined = audio_data if combined is None else combined + audio_data

        # Export the combined audio to the final filename
        if combined:
            combined.export(os.path.join(path, participant_audio_filename), format="wav")
            print(f"Exported combined audio to {participant_audio_filename}")
        else:
            print(f"No audio segments found to combine for {participant_audio_filename}")

        # Delete the split files
        for audio_file in audio_files_to_delete:
            os.remove(audio_file)
    else:
        print("No participant start audio file found, skipping concatenation.")
