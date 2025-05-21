import cv2
import numpy as np

def enhance_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to load image.")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization to enhance contrast
    enhanced = cv2.equalizeHist(gray)
    
    # Show the images
    cv2.imshow('Original Image', img)
    cv2.imshow('Grayscale Image', gray)
    cv2.imshow('Enhanced Image', enhanced)
    
    # Save the enhanced image
    cv2.imwrite('grayscale.jpg', gray)
    
    # Wait for a key press and close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
enhance_image('C:/Users/yoges/Downloads/Final/uploads/def.jpg')
