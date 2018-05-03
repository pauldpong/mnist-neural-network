import gzip
import numpy as np

def load_data():
    print("Unzipping...")
    training_images = gzip.open("./MNIST/trainimages.gz", 'rb')
    training_labels = gzip.open("./MNIST/trainlabels.gz", 'rb')

    try:
        print("Reading...")
        training_images.read(4)
        training_labels.read(8)

        num_images = int.from_bytes(training_images.read(4), byteorder="big")
        num_rows = int.from_bytes(training_images.read(4), byteorder="big")
        num_cols = int.from_bytes(training_images.read(4), byteorder="big")

        image_pixel_count = num_rows * num_cols

        image_data = np.zeros((num_images, image_pixel_count))
        image_label = np.zeros((num_images, 10))

        for image in range(2):
            label = int.from_bytes(training_labels.read(1), byteorder="big")
            image_label[image][label] = 1.0
            for pixel in range(image_pixel_count):
                image_data[image][pixel] = int.from_bytes(training_images.read(1), byteorder="big")

    finally:
        training_images.close()
        print(display(image_data[1]), image_label[1])
        print("Success")


def display(number):
    render = ''

    for i in range(len(number)):
        if i % 28 == 0:
            render = render + '\n'
        if number[i] > 200:
            render = render + '#'
        else:
            render = render + '.'

    return render


load_data()