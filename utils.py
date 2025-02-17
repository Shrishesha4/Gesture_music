import numpy as np

def calculate_distance(landmark1, landmark2):
    """
    Calculates the Euclidean distance between two landmarks.

    Args:
        landmark1: The first landmark.
        landmark2: The second landmark.

    Returns:
        The distance between the two landmarks.
    """
    return np.sqrt((landmark1.x - landmark2.x)**2 + (landmark1.y - landmark2.y)**2)
