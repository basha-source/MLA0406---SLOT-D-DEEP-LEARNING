import cv2
import numpy as np

# Load image
image = cv2.imread("leaf.png")

if image is None:
    print("Image not found!")
    exit()

# Resize image
image = cv2.resize(image, (400, 400))

# Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Green leaf mask
lower_green = np.array([25, 40, 40])
upper_green = np.array([90, 255, 255])
green_mask = cv2.inRange(hsv, lower_green, upper_green)

# Brown/Yellow disease mask
lower_disease = np.array([10, 50, 50])
upper_disease = np.array([35, 255, 255])
disease_mask = cv2.inRange(hsv, lower_disease, upper_disease)

# Count pixels
green_pixels = cv2.countNonZero(green_mask)
disease_pixels = cv2.countNonZero(disease_mask)

total = green_pixels + disease_pixels

if total > 0:
    disease_percent = (disease_pixels / total) * 100
else:
    disease_percent = 0

# Classification
if disease_percent < 5:
    result = "Healthy Leaf"
elif disease_percent < 20:
    result = "Mild Disease"
else:
    result = "Severely Diseased"

print("Disease Percentage: {:.2f}%".format(disease_percent))
print("Result:", result)

# Display
cv2.imshow("Leaf", image)
cv2.imshow("Disease Mask", disease_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()