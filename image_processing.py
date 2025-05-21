import cv2 
import numpy as np
import os

def preprocess_image(image_path, target_size=None):
   
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return None

    # Resize if a target size is provided
    if target_size is not None:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # Apply Gaussian Blur to reduce noise
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    return image

def create_difference_image(image1_path, image2_path, output_folder="results"):
   
    
    # Create output directory if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Load and preprocess images
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    if image1 is None or image2 is None:
        print("Error: One or both images could not be loaded. Check the file paths.")
        return

    # Resize to match dimensions
    height, width = image1.shape
    image2 = cv2.resize(image2, (width, height), interpolation=cv2.INTER_AREA)

    # Apply Gaussian Blur to reduce noise
    image1 = cv2.GaussianBlur(image1, (5, 5), 0)
    image2 = cv2.GaussianBlur(image2, (5, 5), 0)

    # Compute absolute pixel-wise difference
    difference = cv2.absdiff(image1, image2)

    # Apply threshold to highlight differences
    _, thresholded_diff = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)

    # Find contours of defects (Fix for OpenCV 3.x and 4.x)
    contour_info = cv2.findContours(thresholded_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contour_info[-2]  # Works for all OpenCV versions

    # Convert the original defective image to color to draw contours
    defect_highlighted = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(defect_highlighted, contours, -1, (0, 0, 255), 2)  # Red contours

    # Apply color mapping for better visibility
    colored_difference = cv2.applyColorMap(difference, cv2.COLORMAP_JET)

    # Display results
    cv2.imshow('Grayscale Difference Image', difference)
    cv2.imshow('Thresholded Difference', thresholded_diff)
    cv2.imshow('Color Mapped Difference', colored_difference)
    cv2.imshow('Defects Highlighted', defect_highlighted)

    # Save output images in the specified folder
    cv2.imwrite(os.path.join(output_folder, "output_difference.jpg"), difference)
    cv2.imwrite(os.path.join(output_folder, "output_thresholded.jpg"), thresholded_diff)
    cv2.imwrite(os.path.join(output_folder, "output_colored.jpg"), colored_difference)
    cv2.imwrite(os.path.join(output_folder, "output_defects.jpg"), defect_highlighted)
    
    print(f"Processed images saved successfully in the '{output_folder}' folder.")

    # Wait for a key event and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    create_difference_image("C:/Users/yoges/Downloads/Final/input/template_pcb.jpg","C:/Users/yoges/Downloads/Final/input/defected_pcb.jpg",output_folder="C:/Users/yoges/Downloads/Final/output")