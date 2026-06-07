import numpy as np


def generate_spiral(samples_per_class=100, num_classes=3, noise=0.15):
    
    """ 
    First before starting to code, let's understand what a spiral dataset is. 
    A spiral dataset consists of multiple classes of data points arranged in a spiral pattern. 
    Each class forms its own spiral arm, and the data points are distributed along these arms with some added noise for realism. 
    That is going to serve as the dataset for our future neural network model.

    """

    """ 
    For the three arguments of this function we are going to use 3 parameters, they being:
    samples_per_class: Number of data points per class.
    num_classes: Number of spiral arms, in that case, classes.
    noise: Standard deviation of Gaussian noise added to the data, as if introducing some randomness to make the dataset more realistic and challenging for the model to learn.

    """


    """
    For the returns of this function we are going to return two variables, they being:
    X: The feature of matrix of shape and their coordinates (samples_per_class * num_classes, 2).
    y: Label vector of shape, being the vector of class labels (samples_per_class * num_classes,).
   
   """

#We first calculate the total number of samples by multiplying the number of samples per class with the number of classes.

    total_samples = samples_per_class * num_classes

#Then we initialize the feature matrix X with zeros, having a shape of (total_samples, 2) to hold the x and y coordinates of each data point.

    X = np.zeros((total_samples, 2))

#We also initialize the label vector y with zeros, having a shape of (total_samples,) to hold the class labels for each data point.

    y = np.zeros(total_samples, dtype=int)

#Next, we loop through each class index to generate the spiral arms. For each class, we calculate the start and end indices for that class's data points in the feature matrix and label vector.

    for class_idx in range(num_classes):
        start = class_idx * samples_per_class
        end = start + samples_per_class

#   We then generate the radius r and angle theta for the spiral arm corresponding to the current class. The radius increases linearly from 0 to 1, while the angle is calculated to create a spiral pattern with added noise.

        r = np.linspace(0.0, 1.0, samples_per_class)

#   For the angle theta, we create a linear spaace between each class of 4 radians and andd some random noise to it to make the dataset more realistic and challenging for the model to learn. The noise is generated 
#   using a normal distribution with a standard deviation specified by the noise parameter, in that case 0.15.
        theta = (
            np.linspace(class_idx * 4, (class_idx + 1) * 4, samples_per_class)
            + np.random.randn(samples_per_class) * noise
        )
#  Finally, we calculate the x and y coordinates for the data points in the current class using the radius and angle. 
#   The x coordinate is calculated as r * sin(theta) and the y coordinate is calculated as r * cos(theta). We also assign the class label to the corresponding indices in the label vector y.

        X[start:end, 0] = r * np.sin(theta)
        X[start:end, 1] = r * np.cos(theta)
        y[start:end] = class_idx

    return X, y
