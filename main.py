import cv2
import dlib
import numpy as np
import pandas as pd
import math

# Load pre-trained models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def load_image(image_path):
    """Loads an image from a given path."""
    return cv2.imread(image_path)

def preprocess_image(img):
    """Preprocess the image for better landmark detection."""
    # Convert to grayscale
    pre_processed_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Histogram Equalization
    pre_processed_image = cv2.equalizeHist(pre_processed_image)

    return pre_processed_image

def detect_faces_and_landmarks(gray):
    """Detects faces and landmarks in the grayscale image and returns a dictionary of landmark coordinates."""
    faces = detector(gray)
    landmarks_dict = {}

    if len(faces) == 0:
        print("Face not detected.")
        return landmarks_dict, faces

    # Process only the first detected face
    landmarks = predictor(gray, faces[0])
    for i in range(68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        landmarks_dict[i] = (x, y)

    return landmarks_dict, faces

def detect_hairline(img, landmarks, forehead_y, x_forehead):
    # Get the forehead midpoint for scanning upward
    y_forehead = forehead_y
    
    # Create a mask for the skin color range
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Calculate the mean HSV values in the forehead region
    forehead_region = hsv[y_forehead - 5:y_forehead + 5, x_forehead - 5:x_forehead + 5]
    mean_color = np.mean(forehead_region, axis=(0, 1))

    # Define a dynamic skin color range based on the forehead region's mean color
    lower_skin = np.array([max(mean_color[0] - 15, 0), max(mean_color[1] - 50, 0), max(mean_color[2] - 50, 0)], dtype=np.uint8)
    upper_skin = np.array([min(mean_color[0] + 15, 180), min(mean_color[1] + 50, 255), min(mean_color[2] + 50, 255)], dtype=np.uint8)

    # Create a mask for skin color range
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Apply Gaussian blur to the skin mask
    blurred_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)

    # Use Canny edge detection to find hairline edges
    edges = cv2.Canny(blurred_mask, 100, 200)

    # Scan upward from the forehead to detect where the skin ends (hairline)
    for y in range(y_forehead, 0, -1):  # Moving upward from the forehead
        if edges[y, x_forehead] > 0:  # Check for an edge (potential hairline)
            y_hairline = y
            break
    else:
        y_hairline = y_forehead - 50  # Default if no hairline is found

    return y_hairline

def add_hairline_to_landmarks(img, landmarks_dict):
    """Detects hairline and adds it as the 68th point in landmarks_dict."""
    # Get midpoint between eyebrows for the forehead region
    x_forehead = int((landmarks_dict[19][0] + landmarks_dict[24][0]) / 2)
    y_forehead = landmarks_dict[19][1] - 20

    # Detect hairline and add it as the 68th point in landmarks_dict
    y_hairline = detect_hairline(img, landmarks_dict, y_forehead, x_forehead)
    landmarks_dict[68] = (x_forehead, y_hairline)

def draw_landmarks(img, landmarks_dict):
    """Draws all landmarks, including the hairline if present, on the image."""
    for i, (x, y) in landmarks_dict.items():
        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)

    return img

def compute_feature_descriptor(landmarks_dict):
    """Computes a feature vector based on geometric features"""
    def euclidean_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    # Feature F1: Ratio of the height of a face to the width
    F1 = euclidean_distance(landmarks_dict[8], landmarks_dict[68]) / euclidean_distance(landmarks_dict[0], landmarks_dict[16])

    # Feature F2: Ratio of the distance between both sides of the jaws to the width of the face
    F2 = euclidean_distance(landmarks_dict[4], landmarks_dict[12]) / euclidean_distance(landmarks_dict[0], landmarks_dict[16])

    # Feature F3: Ratio of the distance between the chin and the bottom of the mouth to the distance between both sides of the jaws
    F3 = euclidean_distance(landmarks_dict[8], landmarks_dict[57]) / euclidean_distance(landmarks_dict[4], landmarks_dict[12])

    # Features F4 to F11: Angles between facial boundary and chin point for points 0 to 7
    chin_point = landmarks_dict[8]
    angles_F0_to_F7 = []
    for i in range(0, 8):
        point = landmarks_dict[i]
        angle = math.degrees(math.atan2(chin_point[1] - point[1], chin_point[0] - point[0]))
        angles_F0_to_F7.append(angle)

    # Features F12 to F19: Angles between facial boundary and chin point for points 9 to 16
    angles_F9_to_F16 = []
    for i in range(9, 17):
        point = landmarks_dict[i]
        angle = math.degrees(math.atan2(chin_point[1] - point[1], chin_point[0] - point[0]))
        angles_F9_to_F16.append(angle)

    # Combine all features into a single list
    feature_vector = [F1, F2, F3] + angles_F0_to_F7 + angles_F9_to_F16

    return feature_vector

def main():
    #image = load_image(r"C:\Users\rayaa\Desktop\testing python\cv_ml_project\heart\img_no_42.jpg")
    image = load_image("combined_dataset/oblong_71.jpg")
    pre_processed_image = preprocess_image(image)
    landmarks_dict, faces = detect_faces_and_landmarks(pre_processed_image)

    if not landmarks_dict:
        print("No landmarks detected. Exiting.")
        return
    
    # Add hairline to landmarks and draw them
    add_hairline_to_landmarks(image, landmarks_dict)
    landmarked_image = draw_landmarks(image, landmarks_dict)
    feature_vector = compute_feature_descriptor(landmarks_dict)
    # Display images
    cv2.imshow("Pre-Processed Image", pre_processed_image)
    cv2.imshow("Face Landmarks with Hairline Point", landmarked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(feature_vector)

if __name__ == "__main__":
    main()
