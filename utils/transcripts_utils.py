import os
import pandas as pd
from pydub import AudioSegment

import my_config


def read_transcript(directory):
    """
    Function to get the transcript including 'Participant' and 'Ellie'

    Args:
        directory: Directory to search for transcript files
    Returns:
        p_dframe: Part of the transcript including 'Participant' and 'Ellie' in a 2D dataframe
        p_path: Path to the transcript file
        p_id: Participant id of the transcript
    """
    for p_path, p_direct, p_files in os.walk(directory):
        p_dframe = None
        p_id = None
        for p_file in p_files:
            if p_file.endswith("_TRANSCRIPT.csv"):
                p_dframe = pd.read_csv(os.path.join(p_path, p_file), sep='\t')
                p_dframe = p_dframe[p_dframe.speaker.isin(['Participant', 'Ellie'])]
                if p_dframe.empty:
                    print(f"No 'Participant' or 'Ellie' data found in {os.path.join(p_path, p_file)}")
                    return None, None, None
                p_id = p_file[0:3]
        return p_dframe, p_path, p_id


def extract_response_times(df, epsilon=0.1):
    """
    Function to extract the response times from participant transcript

    Args:
        df: Participant transcript dataframe
        epsilon: Epsilon for excluding near 0 values
    Returns:
        None
    """
    # Getting all Ellie and Participant only conversation rows & and shifting them up for calculating response time
    convo_df = df[df['speaker'].isin(['Ellie', 'Participant'])].copy()
    convo_df[['next_start_time', 'next_speaker']] = convo_df[['start_time', 'speaker']].shift(-1)

    # Computing response times: for Ellie rows, when next speaker is Participant
    convo_df.loc[(convo_df['speaker'] == 'Ellie') & (convo_df['next_speaker'] == 'Participant'), 'response_time'] = \
        convo_df['next_start_time'] - convo_df['stop_time']

    # Removing NaN and zero values (or values near zero)
    valid_response_times = convo_df['response_time'][convo_df['response_time'] > epsilon].dropna().values

    if valid_response_times.size == 0:
        return None
    else:
        return valid_response_times


def get_participant_info(directory):
    """
    Function to get participant part of the transcript and temporary audio

    Args:
        directory: Directory to search for audio and transcript files
    Returns:
        p_dframe: Participant part of the transcript in a 2D dataframe
        p_tempaudio: Temporary audio audiosegment object
        p_path: Path to the temporary audio file
        p_id: Participant id of the transcript
    """
    for p_path, p_direct, p_files in os.walk(directory):
        p_tempaudio = None
        p_dframe = None
        p_id = None
        for p_file in p_files:
            if p_file.endswith("_TRANSCRIPT.csv"):
                p_dframe = pd.read_csv(os.path.join(p_path, p_file), sep='\t')
                p_dframe = p_dframe[p_dframe.speaker == 'Participant']
                if p_dframe.empty:
                    print(f"No participant data found in {os.path.join(p_path, p_file)}")
                    return
            elif p_file.endswith(my_config.START_FORMAT):
                p_tempaudio = AudioSegment.from_wav(os.path.join(p_path, p_file))
                p_id = p_file[0:3]

        return p_tempaudio, p_dframe, p_path, p_id
