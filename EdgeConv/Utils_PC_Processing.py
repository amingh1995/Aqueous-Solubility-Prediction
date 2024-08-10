from sklearn.preprocessing import MinMaxScaler
import numpy as np

import matplotlib.pyplot as plt

def normalize_train_test_xyz_esp(pc_train, pc_test):
    """
    Normalize the training and testing data using MinMaxScaler.

    Parameters:
    - pc_train: numpy array, training data of shape (N_train, points, F(features)).
    - pc_test: numpy array, testing data of shape (N_test, points, F(features)).

    Returns:
    - pc_train_normalized: numpy array, normalized training data with the same shape as pc_train.
    - pc_test_normalized: numpy array, normalized testing data with the same shape as pc_test.
    """
    # Reshape the point cloud data to concatenate all features
    pc_train_reshaped = pc_train.reshape(-1, pc_train.shape[-1])
    pc_test_reshaped = pc_test.reshape(-1, pc_test.shape[-1])

    # Separate the x, y, z coordinates and ESP value
    xyz_train = pc_train_reshaped[:, :3]
    esp_train = pc_train_reshaped[:, 3]

    xyz_test = pc_test_reshaped[:, :3]
    esp_test = pc_test_reshaped[:, 3]

    # Use MinMaxScaler to normalize x, y, z coordinates together
    scaler_xyz = MinMaxScaler()
    xyz_train_normalized = scaler_xyz.fit_transform(xyz_train)
    xyz_test_normalized = scaler_xyz.transform(xyz_test)

    # Use MinMaxScaler to normalize the ESP values separately
    scaler_esp = MinMaxScaler()
    esp_train_normalized = scaler_esp.fit_transform(esp_train.reshape(-1, 1)).flatten()
    esp_test_normalized = scaler_esp.transform(esp_test.reshape(-1, 1)).flatten()

    # Concatenate the normalized features back
    pc_train_normalized = np.column_stack((xyz_train_normalized, esp_train_normalized))
    pc_test_normalized = np.column_stack((xyz_test_normalized, esp_test_normalized))

    # Reshape the normalized data back to the original shape
    pc_train_normalized = pc_train_normalized.reshape(pc_train.shape)
    pc_test_normalized = pc_test_normalized.reshape(pc_test.shape)

    return pc_train_normalized, pc_test_normalized

# Example usage:
# PC_train_normalized, PC_test_normalized = normalize_train_test(PC_train, PC_test)



def plot_comparison(index, original_point_cloud, normalized_point_cloud):
    """
    Plot a comparison between the original and normalized point clouds.

    Parameters:
    - index: int, the index of the example to plot.
    - pc_org: numpy array, original point cloud of shape (points, features).
    - pc_normalized: numpy array, normalized point cloud of shape (points, features).
    """
    
    pc_org = original_point_cloud[index]
    pc_normalized = normalized_point_cloud[index]
    
    # Plot original point cloud
    fig = plt.figure(figsize=(12, 6))

    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(pc_org[:, 0], pc_org[:, 1], pc_org[:, 2], c=pc_org[:, 3], cmap='viridis')
    ax.set_title('Original Point Cloud')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Plot normalized point cloud
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(pc_normalized[:, 0], pc_normalized[:, 1], pc_normalized[:, 2], c=pc_normalized[:, 3], cmap='viridis')
    ax.set_title('Normalized Point Cloud')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

    plt.show()
    
    
# Example usage:
# plot_comparison(100, PC_train[], PC_train_normalized[])



# Define a function to apply random rotations to point cloud data
def apply_random_rotation(point_cloud):
    # Generate random angles for rotation around x, y, and z axes
    theta_x = np.random.uniform(low=0.0, high=2*np.pi)
    theta_y = np.random.uniform(low=0.0, high=2*np.pi)
    theta_z = np.random.uniform(low=0.0, high=2*np.pi)
    
    # Compute the rotation matrices around each axis
    rotation_matrix_x = np.array([[1, 0, 0],
                                  [0, np.cos(theta_x), -np.sin(theta_x)],
                                  [0, np.sin(theta_x), np.cos(theta_x)]])
    
    rotation_matrix_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                                  [0, 1, 0],
                                  [-np.sin(theta_y), 0, np.cos(theta_y)]])
    
    rotation_matrix_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                                  [np.sin(theta_z), np.cos(theta_z), 0],
                                  [0, 0, 1]])
    
    # Apply the rotations to the x, y, and z coordinates of the point cloud
    point_cloud_xyz = point_cloud[:, :3]
    rotated_xyz = np.dot(rotation_matrix_z, np.dot(rotation_matrix_y, np.dot(rotation_matrix_x, point_cloud_xyz.T))).T
    
    # Concatenate the rotated x, y, and z coordinates with the unchanged additional feature values
    rotated_point_cloud = np.concatenate([rotated_xyz, point_cloud[:, 3:]], axis=1)
    
    return rotated_point_cloud


# Generate some example point cloud data

def augment_rot(poin_cloud_dataset):
    
    # Apply random rotations to the point cloud data
    augmented_mols_1 = np.zeros_like(poin_cloud_dataset)
    for i in range(poin_cloud_dataset.shape[0]):
        augmented_mols_1[i] = apply_random_rotation(poin_cloud_dataset[i])

    augmented_mols_2 = np.zeros_like(poin_cloud_dataset)
    for i in range(poin_cloud_dataset.shape[0]):
        augmented_mols_2[i] = apply_random_rotation(poin_cloud_dataset[i])


    augmented_mols_3 = np.zeros_like(poin_cloud_dataset)
    #for i in range(poin_cloud_dataset.shape[0]):
    #    augmented_mols_3[i] = apply_random_rotation(poin_cloud_dataset[i])
    
    #Augmented_molecules = np.concatenate((poin_cloud_dataset, augmented_APIs_1, augmented_APIs_2 ), axis = 0)

    Augmented_molecules = np.concatenate((poin_cloud_dataset, augmented_mols_1), axis = 0)
    return Augmented_molecules




