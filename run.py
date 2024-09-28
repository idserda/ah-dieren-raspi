import cv2
import numpy as np

def decode_custom_barcode(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding to get a black and white image (invert colors)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)  # Invert for dark bars on white

    # Find contours of the black areas
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variable to store the best rectangle
    barcode_rect = None
    barcode_area = None

    # Loop through the contours to find the rectangle for the barcode
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the approximated contour has four points (rectangle)
        if len(approx) == 4:
            # Calculate the area of the contour
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter out small areas; adjust as necessary
                barcode_rect = approx
                # Create a mask for the barcode area
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [barcode_rect], -1, (255), thickness=cv2.FILLED)
                # Bitwise AND to extract the barcode area
                barcode_area = cv2.bitwise_and(image, image, mask=mask)

                # Crop the barcode area using the bounding box
                x, y, w, h = cv2.boundingRect(barcode_rect)
                barcode_cropped = barcode_area[y:y+h, x:x+w]

                # Decode the cropped barcode region
                return decode_bars(barcode_cropped)

    return None

def decode_bars(image):
    # Convert cropped image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply binary thresholding to get a black and white image
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Sum along the height to detect vertical bars
    vertical_sum = np.sum(thresh == 255, axis=0)

    # Detect changes in the vertical sum to identify edges
    bar_positions = []
    previous_value = vertical_sum[0]
    for i in range(1, len(vertical_sum)):
        if vertical_sum[i] != previous_value:  # Edge detected
            bar_positions.append(i)
        previous_value = vertical_sum[i]

    # Ensure we have 14 positions (start of each bar and the end of the last)
    if len(bar_positions) != 14:
        print(f"Error: Expected 14 edges but found {len(bar_positions)}")
        return None

    # Calculate the widths of the 13 bars
    bar_widths = [bar_positions[i] - bar_positions[i - 1] for i in range(1, len(bar_positions))]

    # Classify each bar as '0' (small) or '1' (wide)
    barcode_data = ""
    for width in bar_widths:
        if width >= 2:  # Wide bar (1), adjust threshold based on your image resolution
            barcode_data += "1"
        else:  # Small bar (0)
            barcode_data += "0"

    return barcode_data

# Open camera
camera = cv2.VideoCapture(0)

frame_count = 0
while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Decode the barcode from the detected rectangle
    decoded_barcode = decode_custom_barcode(frame)
    if decoded_barcode:
        print(f"Frame {frame_count}: Decoded barcode: {decoded_barcode}")
    else:
        print(f"Frame {frame_count}: No valid barcode detected")

    # Print a log statement for every frame
    frame_count += 1

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
