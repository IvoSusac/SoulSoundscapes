from flask import Flask, redirect, url_for, session, request, render_template, Response, jsonify
from flask_session import Session
import time
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import cv2
from deepface import DeepFace
import os

app = Flask(__name__)

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global variable to store the detected emotion
detected_emotion = 'neutral'

def generate_frames():
    global detected_emotion
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = rgb_frame[y:y + h, x:x + w]
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            detected_emotion = result[0]['dominant_emotion']

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, detected_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

app = Flask(__name__)
app.secret_key = 'YOUR_SECRET_KEY'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
REDIRECT_URI = 'http://localhost:5000/callback'

sp_oauth = SpotifyOAuth(client_id=CLIENT_ID, 
                        client_secret=CLIENT_SECRET, 
                        redirect_uri=REDIRECT_URI, 
                        scope="user-library-read user-top-read")

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/authorize')
def authorize():
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)

@app.route('/callback')
def callback():
    session.clear()
    code = request.args.get('code')
    token_info = sp_oauth.get_access_token(code)
    session['token_info'] = token_info
    return redirect(url_for('index'))

def get_token():
    token_info = session.get('token_info', None)
    if not token_info:
        return redirect(url_for('login'))
    now = int(time.time())
    is_token_expired = token_info['expires_at'] - now < 60
    if is_token_expired:
        token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
    return token_info

@app.route('/index')
def index():
    token_info = get_token()
    if not token_info:
        return redirect(url_for('login'))
    sp = spotipy.Spotify(auth=token_info['access_token'])
    # Now you can use the Spotify API to get user information or recommendations
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion')
def emotion():
    global detected_emotion
    return jsonify(emotion=detected_emotion)

if __name__ == '__main__':
    app.run(debug=True)