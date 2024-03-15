import cv2

def detect_handwritten_text(image_path):

    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    handwritten_text_detected = False

    for contour in contours:

        area = cv2.contourArea(contour)
        

        if area > 2000:
            handwritten_text_detected = True
            break 

    return handwritten_text_detected

# Example usage:
image_path = 'image3.jpg'
result = detect_handwritten_text(image_path)
if result:
    print("Handwritten text detected")
else:
    print("No handwritten text detected")
