import os

import my_config
from utils import transcripts_utils

participant_response_times = {}

for p_path, p_direct, p_files in os.walk(my_config.DIRECTORY):
    for p_file in p_files:
        if p_file.endswith(my_config.START_FORMAT):
            print(f"Processing file: {p_file}")

            # Retrieve 'Participant' and 'Ellie' from transcript
            p_dfr, p_p, p_i = transcripts_utils.read_transcript(p_path)

            if p_dfr is None:
                print("No 'Participant' or 'Ellie' data found. Exiting.")
            else:
                print(f"'Participant' and 'Ellie' data found for file: {p_file}")
                print(p_dfr.head())

                # extract participant response times from the transcript
                response_times = transcripts_utils.extract_response_times(p_dfr)

                if response_times is not None:
                    # Check if the participant already has response times saved
                    if p_i in participant_response_times:
                        participant_response_times[p_i].extend(response_times)
                    else:
                        participant_response_times[p_i] = list(response_times)

                print(f"Response times: {response_times}")
