import cv2
import numpy as np

def save_image(image, step_name):
    # Generate a unique filename
    filename = f'{step_name}.jpg'
    cv2.imwrite(filename, image)

def detect_rectangles(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding to get a binary image
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    save_image(thresh, "9_thresh")

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    horizontal_rectangles = []
    heights = []

    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the approximated contour has four points (rectangle)
        if len(approx) == 4:
            # Get the bounding box
            x, y, w, h = cv2.boundingRect(approx)

            aspect_ratio = w / float(h)

            # Filter for horizontal rectangles (w should be greater than h)
            if aspect_ratio > 4 and w > 100 and w > h:  # Adjust the condition if necessary for your definition of "horizontal"
                heights.append(h)

                horizontal_rectangles.append(approx)  # Store the rectangle points

                # Draw the rectangle on the image
                cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)  # Green rectangles

    save_image(image, "10_rect")

    res = ''
    median_value = np.median(heights)
    print(f"Median: {median_value}, height: {heights}")
    for height in heights:
        if height > median_value:
            res = res + '1'
        else:
            res = res + '0'
    return res

def decode_bars(image):
    return detect_rectangles(image)

def validate(s):
    # Check length
    if len(s) != 13:
        print(f"Incorrect lenght")
        return False
    
    # Check first three characters
    if not s.startswith('000'):
        print(f"No 3 leading zeros")
        return False
    
    # Include the first three bits in the XOR calculation
    xor_result = 0
    for bit in s[:10]:  # Calculate XOR for bits 1 to 10 (indexes 0 to 9)
        xor_result ^= int(bit)
    
    # Check if the XOR result matches bit 11 (index 10)
    if int(s[10]) != xor_result:
        print(f"Incorrect bit 11 xor")
        return False
    
    # Validate that bit 12 (index 11) is the inverse of bit 11
    if int(s[11]) != (1 - xor_result):
        print(f"Incorrect bit 12 inv xor")
        return False

    if (int(s[12])) != 1:
        print(f"Incorrect bit 13 one")
        return False
    
    return True

def todec(s):
    res = s[3:10]
    return int(res, 2)

# Run the process on the camera feed
camera = cv2.VideoCapture(0)  # Adjust to your camera index if necessary

while True:
    ret, frame = camera.read()
    if not ret:
        break

    save_image(frame, "0_captured")

    # Decode the barcode from the image
    decoded_barcode = decode_bars(frame)
    print(f"Decoded: {decoded_barcode}")
    if decoded_barcode:
        if (validate(decoded_barcode)):
            print(f"Decoded barcode: {decoded_barcode}")
            print(f">> {todec(decoded_barcode)} >>")
    else:
        print("No valid barcode detected.")

camera.release()
