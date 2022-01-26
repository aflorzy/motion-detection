import cv2
import imutils
import datetime

cap = cv2.VideoCapture(0)
width = int(cap.get(3))
height = int(cap.get(4))
# Create bounding box
left = int((width/2)-(width*0.05))
right = int((width/2)+(width*0.05))
top = int((height/2)-(height*0.25))
bottom = int((height/2)+(height*0.1))

firstFrame = None
text = "Unoccupied"
while True:
    # Start camera video capture
    ret, frame = cap.read()
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Add blur/smooth frame
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Initialize firstFrame (background)
    if firstFrame is None:
        firstFrame = gray
        continue

    # Store differences between current frame and background
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilate threshold image to fill in holes
    thresh = cv2.dilate(thresh, None, iterations=2)
    # Find contours on threshhold image
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		        cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # Loop over contours
    for c in contours:
		# if the contour is too small, ignore it
        if cv2.contourArea(c) < 500:
            continue
        # Computer bounding box and draw around contours
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Occupied"

    cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # Original, color feed with boxes
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.imshow("Security Feed", frame)
    # Resize threshold region
    thresh = thresh[top:bottom, left:right]
    cv2.imshow("Thresh", thresh)
    # Cool-looking feed with differences shown by grayscale intensities
    cv2.imshow("Frame Delta", frameDelta)


    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()