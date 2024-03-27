import os

import my_config
from utils import audio_utils


for path, directories, files in os.walk(my_config.DIRECTORY):
    for audio in files:
        if audio.endswith(my_config.FILE_FORMAT):
            ip, op = audio_utils.get_raw_audio(path, audio)
            audio_utils.remove_noise(ip, op)
