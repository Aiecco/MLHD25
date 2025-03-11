from src.Utils.labels_ordering import train_labels, val_labels, test_labels


def main():
    train_path = 'data/Train/train_labels.csv'
    test_path = 'data/Test/test_labels.csv'
    val_path = 'data/Val/val_labels.csv'

    #train_labels(train_path)
    #test_labels(test_path)
    #val_labels(val_path)

    preprocessing()


if __name__ == '__main__':
    main()