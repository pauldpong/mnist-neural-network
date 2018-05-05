#  MNIST dataloader into numpy arrays

import gzip
import numpy as np


def load_data(trainimage_path="./train-images-idx3-ubyte.gz", trainlabel_path="./train-labels-idx1-ubyte.gz",
              testimage_path="./t10k-images-idx3-ubyte.gz", testlabel_path="./t10k-labels-idx1-ubyte.gz"):

    print("Unzipping .gz files...")
    training_images = gzip.open(trainimage_path, 'rb')
    training_labels = gzip.open(trainlabel_path, 'rb')
    test_images = gzip.open(testimage_path, 'rb')
    test_labels = gzip.open(testlabel_path, 'rb')

    try:
        print("Reading training data...")

        # Skipping over magic numbers
        training_images.read(4)
        training_labels.read(8)
        test_images.read(4)
        test_labels.read(8)

        # Accessing the numerical values
        num_images = int.from_bytes(training_images.read(4), byteorder="big")
        num_rows = int.from_bytes(training_images.read(4), byteorder="big")
        num_cols = int.from_bytes(training_images.read(4), byteorder="big")

        image_pixel_count = num_rows * num_cols

        # Numpy array with zeros
        image_train_set = np.zeros((num_images, image_pixel_count))  # Array: num_images by image_pixel_count
        label_train_set = np.zeros((num_images, 10))  # Array: num_images by 10

        # Filling the array, each row is one image, and each row has 784 columns with the corresponding pixel value
        for image in range(num_images):
            label = int.from_bytes(training_labels.read(1), byteorder="big")
            label_train_set[image][label] = 1.0
            for pixel in range(image_pixel_count):
                image_train_set[image][pixel] = int.from_bytes(training_images.read(1), byteorder="big")

        # Same process as training data, except labels are not stored as a vector but as a single entry
        print("Reading testing data...")

        num_test_images = int.from_bytes(test_images.read(4), byteorder="big")
        test_images.read(8)

        image_test_set = np.zeros((num_test_images, image_pixel_count))
        label_test_set = np.zeros((num_test_images, 1))

        for image in range(num_test_images):
            label_test_set[image] = int.from_bytes(test_labels.read(1), byteorder="big")
            for pixel in range(image_pixel_count):
                image_test_set[image][pixel] = int.from_bytes(test_images.read(1), byteorder="big")

    finally:
        training_images.close()
        training_labels.close()
        test_images.close()
        test_labels.close()

        # Depending on version, zip will either return a list(ver. 2) or an iterable object(ver. 3)
        training_data = zip(image_train_set, label_train_set)
        testing_data = zip(image_test_set, label_test_set)

        print("Finished, returned training_data, testing_data as iterable objects.")

    return training_data, testing_data


# Function to display training image (image = the array of 784 numbers, eg. image_train_set[0] would be the first image
def display(image, width=28, threshold=200):
    render = ''

    for i in range(len(image)):
        if i % width == 0:
            render = render + '\n'
        if image[i] > threshold:
            render = render + '#'
        else:
            render = render + '.'

    return render
