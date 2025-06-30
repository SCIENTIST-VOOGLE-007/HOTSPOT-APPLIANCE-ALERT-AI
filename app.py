import os
import time
import logging
from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, jsonify
from threading import Lock
from datetime import datetime
from werkzeug.utils import secure_filename
import cv2  # Import cv2 globally
from datetime import timedelta

from werkzeug.exceptions import RequestEntityTooLarge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secure random key
app.config['UPLOAD_FOLDER'] = 'static/uploads'
# To something larger like 100MB:
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config.update(
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=30),
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax'
)
# Initialize in-memory "database" for demo purposes
users = {
    'admin': {'password': 'admin123', 'name': 'Admin User', 'alerts': []}
}

# Camera and detection state
camera_lock = Lock()
camera_stream = None
current_analysis = {
    'person_count': 0,
    'appliances': [],
    'alert_required': False,
    'timestamp': ''
}

# Create required directories
for dir_path in ['static/uploads', 'static/results', 'static/uploads/batch', 'static/results/batch']:
    os.makedirs(dir_path, exist_ok=True)

def init_camera():
    global camera_stream
    with camera_lock:
        if camera_stream is None:
            try:
                camera_stream = cv2.VideoCapture(0)
                if not camera_stream.isOpened():
                    logger.error("Could not open camera")
                    return False
                logger.info("Camera initialized successfully")
                return True
            except Exception as e:
                logger.error(f"Camera initialization failed: {str(e)}")
                return False
        return True

def get_frame():
    if camera_stream is None:
        return None
    try:
        ret, frame = camera_stream.read()
        if not ret:
            logger.warning("Failed to capture frame")
            return None
        return frame
    except Exception as e:
        logger.error(f"Frame capture error: {str(e)}")
        return None

def generate_frames():
    while True:
        frame = get_frame()
        if frame is None:
            continue
        try:
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            logger.error(f"Frame encoding error: {str(e)}")
            continue

# Authentication functions
def login_user(username, password):
    user = users.get(username)
    if user and user['password'] == password:
        session['username'] = username
        return True
    return False

def register_user(username, password):
    if username in users:
        return False
    users[username] = {'password': password, 'name': username, 'alerts': []}
    return True

def logout_user():
    session.pop('username', None)

def is_logged_in():
    return 'username' in session

def current_user():
    return session.get('username')

def get_user_by_username(username):
    return users.get(username)

from ultralytics import YOLO

# Load the YOLO model (place this at the global level or in a function to avoid reloading per request)
model = YOLO('yolov8n.pt')  # Path to your model file

def process_image(filepath):
    try:
        img = cv2.imread(filepath)
        if img is None:
            return None, {'error': 'Could not read image file'}
        
        # Perform object detection
        results = model(img)
        
        # Initialize analysis
        analysis = {
            'person_count': 0,
            'appliances': [],
            'alert_required': False,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Process detection results
        for result in results:
            boxes = result.boxes.xyxy  # Bounding box coordinates
            classes = result.boxes.cls  # Class indices
            names = result.names  # Class names
            for box, cls in zip(boxes, classes):
                class_name = names[int(cls)]
                if class_name == 'person':
                    analysis['person_count'] += 1
                elif class_name in ['oven', 'microwave', 'stove', 'toaster']:  # Add relevant appliance classes
                    analysis['appliances'].append(class_name)
                    if class_name in ['oven', 'stove']:  # Trigger alert for potential hotspots
                        analysis['alert_required'] = True
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img, analysis
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        return None, {'error': str(e)}


def process_folder(image_paths):
    processed_images = []
    analyses = []
    try:
        for filepath in image_paths:
            # Process each image using YOLO model
            img = cv2.imread(filepath)
            if img is None:
                continue
                
            # Get actual detection results
            results = model(img)
            filename = os.path.basename(filepath)
            
            # Initialize analysis
            analysis = {
                'person_count': 0,
                'appliances': [],
                'alert_required': False,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Process detection results
            for result in results:
                boxes = result.boxes.xyxy
                classes = result.boxes.cls
                names = result.names
                for box, cls in zip(boxes, classes):
                    class_name = names[int(cls)]
                    if class_name == 'person':
                        analysis['person_count'] += 1
                    elif class_name in ['oven', 'microwave', 'stove', 'toaster']:
                        analysis['appliances'].append(class_name)
                        if class_name in ['oven', 'stove']:
                            analysis['alert_required'] = True
                    
                    # Draw bounding boxes
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            processed_images.append((filename, img))
            analyses.append(analysis)
            
        return processed_images, analyses
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        return [], []
def process_video(filepath):
    try:
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return None, {'error': 'Could not open video file'}
            
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        output_path = os.path.join('static/results', 'processed_' + os.path.basename(filepath))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        analysis = {
            'person_count': 0,
            'appliances': [],
            'alert_required': False,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process each frame
            results = model(frame)
            for result in results:
                boxes = result.boxes.xyxy
                classes = result.boxes.cls
                names = result.names
                for box, cls in zip(boxes, classes):
                    class_name = names[int(cls)]
                    if class_name == 'person':
                        analysis['person_count'] += 1
                    elif class_name in ['oven', 'microwave', 'stove', 'toaster']:
                        analysis['appliances'].append(class_name)
                        if class_name in ['oven', 'stove']:
                            analysis['alert_required'] = True
                    
                    # Draw bounding boxes
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, class_name, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            out.write(frame)
            
        cap.release()
        out.release()
        return output_path, analysis
        
    except Exception as e:
        logger.error(f"Video processing error: {str(e)}")
        return None, {'error': str(e)}
    

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    flash('File is too large (max 100MB)', 'danger')
    return redirect(request.url)

# Routes
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if is_logged_in():
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if login_user(username, password):
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid credentials', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if register_user(username, password):
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        flash('Username already exists', 'danger')
    return render_template('register.html')

@app.route('/logout')
def logout():
    logout_user()
    flash('Logged out successfully', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if not is_logged_in():
        return redirect(url_for('login'))
    user = get_user_by_username(current_user())
    return render_template('dashboard.html', user=user)

@app.route('/webcam')
def webcam_page():
    if not is_logged_in():
        return redirect(url_for('login'))
    return render_template('webcam.html')

@app.route('/video_feed')
def video_feed():
    if not is_logged_in():
        return Response("Unauthorized", status=401)
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection')
def start_detection():
    if not is_logged_in():
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
    if not init_camera():
        return jsonify({'status': 'error', 'message': 'Camera initialization failed'})
    current_analysis.update({
        'person_count': 1,
        'appliances': ['oven', 'microwave'],
        'alert_required': False,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    return jsonify({
        'status': 'success',
        'message': 'Detection started',
        'data': current_analysis
    })

@app.route('/stop_detection')
def stop_detection():
    global camera_stream
    with camera_lock:
        if camera_stream is not None:
            camera_stream.release()
            camera_stream = None
    return jsonify({'status': 'success', 'message': 'Camera released'})

@app.route('/detection_status')
def detection_status():
    return jsonify(current_analysis)

@app.route('/detect/image', methods=['GET', 'POST'])
def detect_single_image():
    print(f"Session at start: {session}")  # Debug session
    if not is_logged_in():
        print("Not logged in, redirecting")  # Debug
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        print(f"Session during POST: {session}")  # Debug session
        if not is_logged_in():
            flash('Session expired, please login again', 'danger')
            return redirect(url_for('login'))
            
        print(f"Files received: {request.files}")  # Debug files
        if 'image' not in request.files:
            flash('No image provided', 'danger')
            return redirect(request.url)
            
        file = request.files['image']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
            
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(f"Saving to: {filepath}")  # Debug path
            file.save(filepath)
            
            # Process the image
            print("Processing image...")  # Debug
            annotated, analysis = process_image(filepath)
            if annotated is None:
                flash('Failed to process image', 'danger')
                return redirect(request.url)
                
            output_path = f'results/processed_{filename}'
            full_output_path = os.path.join('static', output_path)
            print(f"Saving processed image to: {full_output_path}")  # Debug
            cv2.imwrite(full_output_path, annotated)
            
            # Verify the file was saved
            if not os.path.exists(full_output_path):
                print("Failed to save processed image!")  # Debug
                flash('Failed to save processed image', 'danger')
                return redirect(request.url)
            
            print("Rendering result template")  # Debug
            return render_template('result.html',
                                image_path=output_path,
                                analysis=analysis)
            
        except Exception as e:
            print(f"Error: {str(e)}")  # Debug
            logger.error(f"Image processing error: {str(e)}")
            flash('Failed to process image', 'danger')
            return redirect(request.url)
            
    return render_template('detect_images.html')



@app.route('/detect/images', methods=['GET', 'POST'])
def detect_image_directory():
    if not is_logged_in():
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        if 'images' not in request.files:
            flash('No images provided', 'danger')
            return redirect(url_for('dashboard'))
        
        files = request.files.getlist('images')
        if not files or all(file.filename == '' for file in files):
            flash('No selected files', 'danger')
            return redirect(url_for('dashboard'))
        
        try:
            folder_path = 'static/uploads/batch'
            os.makedirs(folder_path, exist_ok=True)
            image_paths = []
            
            for file in files:
                if file and file.filename:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(folder_path, filename)
                    file.save(filepath)
                    image_paths.append(filepath)
            
            # Process images and get results
            processed_images, analyses = process_folder(image_paths)
            
            # Prepare results for template
            results = []
            output_dir = 'static/results/batch'
            os.makedirs(output_dir, exist_ok=True)
            
            for (filename, img), analysis in zip(processed_images, analyses):
                out_path = os.path.join(output_dir, filename)
                cv2.imwrite(out_path, img)
                results.append({
                    'filename': filename,
                    'path': out_path,
                    'analysis': analysis
                })
            
            return render_template('batch_result.html', results=results)
            
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            flash('Failed to process images', 'danger')
            return redirect(url_for('dashboard'))
    
    return render_template('detect_folder.html')

@app.route('/detect/video', methods=['GET', 'POST'])
def detect_video_file():
    if not is_logged_in():
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        if 'video' not in request.files:
            flash('No video file provided', 'danger')
            return redirect(url_for('detect_video_file'))
            
        file = request.files['video']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(url_for('detect_video_file'))
            
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save in chunks to handle large files
            with open(filepath, 'wb') as f:
                chunk_size = 4096
                while True:
                    chunk = file.stream.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
            
            # Process video
            output_path, analysis = process_video(filepath)
            if output_path and 'error' not in analysis:
                return render_template('video_result.html', 
                                    video_path=output_path.replace('static/', ''), 
                                    analysis=analysis)
            flash(f"Failed to process video: {analysis.get('error', 'Unknown error')}", 'danger')
            
        except RequestEntityTooLarge:
            # This will be caught by the error handler above
            raise
        except Exception as e:
            logger.error(f"Video processing error: {str(e)}")
            flash('Failed to process video', 'danger')
            
    return render_template('detect_video.html')

# Utility routes
@app.route('/test_camera')
def test_camera():
    if not init_camera():
        return "Camera not available"
    frame = get_frame()
    if frame is None:
        return "No frame captured"
    cv2.imwrite('static/results/test.jpg', frame)
    return f'<img src="{url_for("static", filename="results/test.jpg")}">'

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404  # Add 404.html

@app.errorhandler(500)
def internal_error(e):
    return render_template('500.html'), 500  # Add 500.html

# Optional: Handle favicon to avoid 404
@app.route('/favicon.ico')
def favicon():
    return '', 404  # This will return a 404 instead of trying to serve a missing file
if __name__ == '__main__':
    try:
        start_time = time.time()
        port = int(os.environ.get('PORT', 5000))
        logger.info(f"Starting application on port {port}")
        app.run(host='0.0.0.0', port=port, debug=True)
        logger.info(f"Application startup completed in {time.time() - start_time} seconds")
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")