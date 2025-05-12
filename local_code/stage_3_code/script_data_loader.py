import pickle
import numpy as np

def load_data(dataset_name):
    if dataset_name not in ['ORL', 'CIFAR', 'MNIST']:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Choose from 'ORL', 'CIFAR', 'MNIST'.")

    # Open the dataset file
    with open(f'/content/drive/MyDrive/ECS189/ECS189G-Project/local_code/stage_3_code/{dataset_name}', 'rb') as f:
        data = pickle.load(f)

    # Convert the data into tensors
    def process_split(split):
        X = np.array([instance['image'] for instance in data[split]], dtype=np.float32)
        y = np.array([instance['label'] for instance in data[split]], dtype=np.int64)
        return X, y

    X_train, y_train = process_split('train')
    X_test, y_test = process_split('test')

    # FIX: Shift labels to start from 0
    y_train = y_train - y_train.min()
    y_test = y_test - y_test.min()

    # Normalize and reshape
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Determine input channels
    if dataset_name == 'MNIST':
        # shape: (N, H, W) -> (N, 1, H, W)
        X_train = X_train[:, np.newaxis, :, :]
        X_test = X_test[:, np.newaxis, :, :]
        input_channels = 1
    else:
        # shape: (N, H, W, C) -> (N, C, H, W)
        X_train = np.transpose(X_train, (0, 3, 1, 2))
        X_test = np.transpose(X_test, (0, 3, 1, 2))
        input_channels = 3

    num_classes = len(set(y_train))

    return X_train, y_train, X_test, y_test, input_channels, num_classes



# import pickle
# from matplotlib import pyplot as plt
#
# # loading ORL dataset
# if 1:
# 	f = open('ORL', 'rb')
# 	data = pickle.load(f)
# 	f.close()
# 	for instance in data['train']:
# 		image_matrix = instance['image']
# 		image_label = instance['label']
# 		plt.imshow(image_matrix)
# 		plt.show()
# 		print(image_matrix)
# 		print(image_label)
# 		# remove the following "break" code if you would like to see more image in the training set
# 		break
#
# 	for instance in data['test']:
# 		image_matrix = instance['image']
# 		image_label = instance['label']
# 		plt.imshow(image_matrix)
# 		plt.show()
# 		print(image_matrix)
# 		print(image_label)
# 		# remove the following "break" code if you would like to see more image in the testing set
# 		break
#
# # loading CIFAR-10 dataset
# if 0:
# 	f = open('CIFAR', 'rb')
# 	data = pickle.load(f)
# 	f.close()
# 	for instance in data['train']:
# 		image_matrix = instance['image']
# 		image_label = instance['label']
# 		plt.imshow(image_matrix)
# 		plt.show()
# 		print(image_matrix)
# 		print(image_label)
# 		# remove the following "break" code if you would like to see more image in the training set
# 		break
#
# 	for instance in data['test']:
# 		image_matrix = instance['image']
# 		image_label = instance['label']
# 		plt.imshow(image_matrix)
# 		plt.show()
# 		print(image_matrix)
# 		print(image_label)
# 		# remove the following "break" code if you would like to see more image in the testing set
# 		break
#
# # loading MNIST dataset
# if 0:
# 	f = open('MNIST', 'rb')
# 	data = pickle.load(f)
# 	f.close()
# 	for instance in data['train']:
# 		image_matrix = instance['image']
# 		image_label = instance['label']
# 		plt.imshow(image_matrix, cmap='gray')
# 		plt.show()
# 		print(image_matrix)
# 		print(image_label)
# 		# remove the following "break" code if you would like to see more image in the training set
# 		break
#
# 	for instance in data['test']:
# 		image_matrix = instance['image']
# 		image_label = instance['label']
# 		plt.imshow(image_matrix, cmap='gray')
# 		plt.show()
# 		print(image_matrix)
# 		print(image_label)
# 		# remove the following "break" code if you would like to see more image in the testing set
# 		break
#
#
#