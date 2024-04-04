import keras
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import my_config
from data_loader import DataLoader


class ModelTester:
    def __init__(self, model_path, loader):
        self.model = keras.models.load_model(model_path)
        self.data_loader = loader

    def test(self):
        predictions = self.model.predict(self.data_loader.test_generator)
        predicted_classes = np.argmax(predictions, axis=1)

        test_labels = []
        for _, labels in self.data_loader.test_generator:
            test_labels.extend(labels)

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


# data_loader = DataLoader(my_config.TRAIN_H5, my_config.DEV_H5, my_config.TEST_H5)
# tester = ModelTester('./model/1d_model.keras', data_loader)
# tester.test()
