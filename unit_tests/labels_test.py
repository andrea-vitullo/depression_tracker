import unittest
import glob

from my_config import AUDIO_TEST_DIRECTORY
from utils import utils


# Mock directories
mock_directories = [f'{AUDIO_TEST_DIRECTORY}/non_depressed', f'{AUDIO_TEST_DIRECTORY}/depressed']


class LoadFilesLabelsTest(unittest.TestCase):
    def test_load_files_labels(self):
        expected_files = glob.glob(f"{mock_directories[0]}/male/*.wav") + glob.glob(f"{mock_directories[0]}/female/*.wav") + glob.glob(f"{mock_directories[1]}/male/*.wav") + glob.glob(f"{mock_directories[1]}/female/*.wav")
        expected_labels = [0] * (len(glob.glob(f"{mock_directories[0]}/male/*.wav")) + len(glob.glob(f"{mock_directories[0]}/female/*.wav"))) + [1] * (len(glob.glob(f"{mock_directories[1]}/male/*.wav")) + len(glob.glob(f"{mock_directories[1]}/female/*.wav")))

        files, labels = utils.load_files_labels(mock_directories, [0, 1])

        self.assertCountEqual(files, expected_files, f"Expected {set(expected_files)}, but got {set(files)}")
        self.assertListEqual(labels, expected_labels, f"Expected {expected_labels}, but got {labels}")


if __name__ == '__main__':
    unittest.main()
