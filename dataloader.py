import gzip
import numpy as np

def load_data():
    print("Unzipping...")
    training_images = gzip.open("./MNIST/trainimages.gz", 'rb')

    try:
        print("Reading...")
        training_images.read(4)

        num__images = int.from_bytes(training_images.read(4), byteorder="big")
        num_rows = int.from_bytes(training_images.read(4), byteorder="big")
        num_cols = int.from_bytes(training_images.read(4), byteorder="big")

        image_pixel_count = num_rows * num_cols

        image_data = np.zeros((num_rows, num_cols))

        for row in range(num_rows):
            for col in range(num_cols):
                image_data[row][col] = int.from_bytes(training_images.read(1), byteorder="big")

    finally:
        training_images.close()
        print(num__images, num_cols, num_rows, image_pixel_count)
        print(image_data)
        print("Success")

load_data()
