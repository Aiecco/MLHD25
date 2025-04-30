from src.pipelines.RadiographPipeline import radiograph_pipeline
from src.utils.labels_ordering import train_labels, val_labels, test_labels


def main():
    train_path = 'data/Train/train_labels.csv'
    test_path = 'data/Test/test_labels.csv'
    val_path = 'data/Val/val_labels.csv'

    # Workspace preparation
    #train_labels(train_path)
    #test_labels(test_path)
    #val_labels(val_path)

    radiograph_pipeline(epochs=5)


if __name__ == '__main__':
    main()