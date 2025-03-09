import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model('sign_language_model.h5')

# Define class names (Ensure these match your dataset classes)
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'asl-alphabet-test', 'del', 'nothing', 'space']
	
# Start Webcam Capture
cap = cv2.VideoCapture(0)  # Use 0 for default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    print("📷 Webcam initialized. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()  # Capture frame-by-frame
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Resize and normalize frame
        resized_frame = cv2.resize(frame, (64, 64))
        normalized_frame = resized_frame / 255.0
        input_frame = np.expand_dims(normalized_frame, axis=0)

        # Make Prediction
        prediction = model.predict(input_frame)
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = class_names[predicted_class_index]

        # Display prediction on frame
        cv2.putText(frame, predicted_class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Sign Language Detection', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

print("✅ Real-time ASL detection stopped.")

