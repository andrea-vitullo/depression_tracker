import keras
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from my_config import FEATURE_SHAPES
from data_management.data_loader import DataLoader


class ModelTester:
    def __init__(self, model_path, loader):
        self.model = keras.models.load_model(model_path)
        self.data_loader = loader

    def test(self, threshold=0.5):
        predictions = []
        test_labels = []

        # Now we iterate over the test_dataset itself
        for inputs, labels in self.data_loader.test_dataset:
            preds = self.model.predict(inputs)
            predictions.extend(preds)
            test_labels.extend(labels.numpy())

        # Apply the threshold to convert probabilities into binary predictions
        predicted_classes = (np.array(predictions) > threshold).astype(int)

        assert len(test_labels) == len(predicted_classes), "Mismatch in number of true and predicted labels."

        cm = confusion_matrix(test_labels, predicted_classes)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=['Non-depressed', 'Depressed'],
                    yticklabels=['Non-depressed', 'Depressed'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.show()

        print(classification_report(test_labels, predicted_classes, target_names=['Non-depressed', 'Depressed']))


train_files = {
    # 'mfcc': './processed_audio_features/train_mfcc.h5',
    'chroma': './processed_audio_features/train_chroma.h5',
    # 'logmel': './processed_audio_features/train_logmel.h5',
    # 'spectrogram': './processed_audio_features/train_spectrogram.h5',
}
dev_files = {
    # 'mfcc': './processed_audio_features/dev_mfcc.h5',
    'chroma': './processed_audio_features/dev_chroma.h5',
    # 'logmel': './processed_audio_features/dev_logmel.h5',
    # 'spectrogram': './processed_audio_features/dev_spectrogram.h5',
}
test_files = {
    # 'mfcc': './processed_audio_features/test_mfcc.h5',
    'chroma': './processed_audio_features/test_chroma.h5',
    # 'logmel': './processed_audio_features/test_logmel.h5',
    # 'spectrogram': './processed_audio_features/test_spectrogram.h5',
}


data_loader = DataLoader(train_files, dev_files, test_files, FEATURE_SHAPES)
tester = ModelTester('model.keras', data_loader)
tester.test(threshold=0.528)
