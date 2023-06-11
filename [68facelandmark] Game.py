import cv2
import dlib

# Load the face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('Headball/68facelandmark/shape_predictor_68_face_landmarks.dat')

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Create a window for displaying the game
window_name = 'Headball Game'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Initialize the ball position and other necessary variables
ball_radius = 20
ball_color = (0, 0, 255)  # Red color
ball_position = (0, 0)  # Initial position of the ball

ball_speed_x = 9
ball_speed_y = 0  
gravity = 0.5

# Initialize the score
score = 0

while True:
    # Capture frame from the camera
    ret, frame = cap.read()

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_detector(gray)

    # Process each detected face
    for rect in faces:
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Check if the ball touches the top of the face
        if ball_position[1] + ball_radius >= y and ball_speed_y > 0:
            ball_speed_y = -ball_speed_y
            score += 1


    # Update the ball position
    ball_speed_y += gravity
    ball_position = (ball_position[0] + ball_speed_x, ball_position[1] + ball_speed_y)

    # Check for collision with screen edges
    if ball_position[0] < 0 or ball_position[0] > frame.shape[1] - ball_radius:
        ball_speed_x = -ball_speed_x
    if ball_position[1] < 0 or ball_position[1] > frame.shape[0] - ball_radius:
        ball_speed_y = -ball_speed_y

    # Convert ball position to integers
    ball_position = (int(ball_position[0]), int(ball_position[1]))

    # Draw the ball on the frame
    cv2.circle(frame, ball_position, ball_radius, ball_color, -1)

    # Display the score on the frame
    cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame with the ball and score
    cv2.imshow(window_name, frame)

    # Check for key press to exit the game loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
