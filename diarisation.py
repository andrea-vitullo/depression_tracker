import os
from pydub import AudioSegment

from utils import audio_utils, transcripts_utils
import my_config

for path, directories, files in os.walk(my_config.DIRECTORY):
    for file in files:
        if file.endswith(my_config.CLEANED_FORMAT):
            tempaudio, df, _, patient_id = transcripts_utils.get_participant_info(path)
            audio_utils.diarisation(tempaudio, df, path)


for path, directories, files in os.walk(my_config.DIRECTORY):
    combined = None
    participant = None
    audio_files_to_delete = []

    for audio in files:
        if audio.endswith(my_config.CLEANED_FORMAT):
            participant = audio.replace(my_config.CLEANED_FORMAT, my_config.FINAL_FORMAT)
        if audio.endswith(my_config.SPLIT_FORMAT):
            audio_path = os.path.join(path, audio)
            audio_files_to_delete.append(audio_path)
            audio_data = AudioSegment.from_wav(audio_path)
            if combined is None:
                combined = audio_data
            else:
                combined += audio_data

    if combined is not None and participant is not None:
        combined.export(os.path.join(path, participant), format=my_config.FILE_FORMAT)
    else:
        print("No suitable files for concatenation and / or participant name found.")

    for audio_file in audio_files_to_delete:
        os.remove(audio_file)
