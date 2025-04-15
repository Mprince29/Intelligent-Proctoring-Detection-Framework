import cv2
import numpy as np
import time
import logging
import os
from collections import deque

class ObjectDetector:
    """Class for detecting objects in video frames"""
    
    def __init__(self, video_source=None, confidence_threshold=0.5):
        """Initialize the object detector
        
        Args:
            video_source: OpenCV VideoCapture object
            confidence_threshold: Minimum confidence for object detection (0-1)
        """
        self.video_source = video_source
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger("object_detector")
        
        # Object detection history
        self.detected_objects = []
        # Keep track of recent detections to avoid logging duplicates
        self.recent_detections = deque(maxlen=30)  # About 1-2 seconds of history
        
        # Initialize variables for frame processing
        self.frame = None
        self.processed_frame = None
        self.last_processed_time = time.time()
        self.process_every_n_frames = 10  # Process every 10 frames for performance
        self.frame_counter = 0
        
        # Metrics
        self.object_count = {}  # Count of each object type
        self.last_detection_time = None
        self.detection_history = []  # List of tuples (object_name, timestamp)
        
        # Load the COCO class labels
        self.labels_path = os.path.join(os.path.dirname(__file__), "coco_names.txt")
        if not os.path.exists(self.labels_path):
            self._download_coco_names()
        
        self.labels = open(self.labels_path).read().strip().split("\n")
        
        # Load the neural network model - We'll use YOLOv4-tiny for a good balance of speed and accuracy
        self.model_path = os.path.join(os.path.dirname(__file__), "yolov4-tiny.weights")
        self.config_path = os.path.join(os.path.dirname(__file__), "yolov4-tiny.cfg")
        
        if not os.path.exists(self.model_path) or not os.path.exists(self.config_path):
            self._download_model()
        
        try:
            self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.model_path)
            
            # Check if OpenCV DNN CUDA backend is available and use GPU if possible
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.logger.info("Setting CUDA backend for object detection")
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                self.logger.info("CUDA not available, using CPU for object detection")
                # Use OpenCL if available
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
                
            # Get output layer names
            self.layer_names = self.net.getLayerNames()
            try:
                # OpenCV 4.5.4+
                self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            except:
                # Older OpenCV versions
                self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
                
            self.logger.info("Object detector initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize object detector: {e}")
            self.net = None
            self.output_layers = None
    
    def _download_coco_names(self):
        """Download COCO class names file"""
        self.logger.info("Downloading COCO class names...")
        import urllib.request
        
        url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
        try:
            urllib.request.urlretrieve(url, self.labels_path)
            self.logger.info("COCO class names downloaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to download COCO class names: {e}")
            # Fallback to creating a basic list of common objects
            with open(self.labels_path, "w") as f:
                f.write("\n".join([
                    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
                    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
                ]))
                self.logger.info("Created fallback COCO class names file")
    
    def _download_model(self):
        """Download YOLOv4-tiny model files"""
        self.logger.info("Downloading YOLOv4-tiny model files...")
        import urllib.request
        
        # Download the configuration file
        cfg_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
        try:
            urllib.request.urlretrieve(cfg_url, self.config_path)
            self.logger.info("YOLOv4-tiny config downloaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to download YOLOv4-tiny config: {e}")
        
        # Download the weights file (larger file, can take some time)
        weights_url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
        try:
            self.logger.info("Downloading YOLOv4-tiny weights (this may take a while)...")
            urllib.request.urlretrieve(weights_url, self.model_path)
            self.logger.info("YOLOv4-tiny weights downloaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to download YOLOv4-tiny weights: {e}")
            # Try an alternative source
            alt_weights_url = "https://pjreddie.com/media/files/yolov3-tiny.weights"
            try:
                self.logger.info("Trying alternative YOLOv3-tiny weights...")
                self.model_path = self.model_path.replace("yolov4-tiny.weights", "yolov3-tiny.weights")
                self.config_path = self.config_path.replace("yolov4-tiny.cfg", "yolov3-tiny.cfg")
                urllib.request.urlretrieve(alt_weights_url, self.model_path)
                
                # Also get the v3 config
                alt_cfg_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
                urllib.request.urlretrieve(alt_cfg_url, self.config_path)
                self.logger.info("YOLOv3-tiny model files downloaded as fallback")
            except Exception as e2:
                self.logger.error(f"Failed to download alternative model: {e2}")
    
    def process_frame(self, frame=None):
        """Process a frame to detect objects
        
        Args:
            frame: The frame to process. If None, use the internal video source.
            
        Returns:
            The processed frame with object detection visualization
        """
        if self.net is None:
            return frame
        
        if frame is None:
            if self.video_source is None:
                return None
            
            ret, frame = self.video_source.read()
            if not ret:
                return None
        
        self.frame = frame.copy()
        self.frame_counter += 1
        
        # Only process every Nth frame for performance
        if self.frame_counter % self.process_every_n_frames != 0:
            # If we have a previous processed frame, overlay the last detection results
            if self.processed_frame is not None:
                return self.processed_frame
            else:
                return self.frame
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Create a blob from the frame
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        
        # Pass the blob through the network
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        # Initialize lists for detected objects
        class_ids = []
        confidences = []
        boxes = []
        
        # Process the outputs
        for output in outputs:
            for detection in output:
                # Extract the scores (confidence values)
                scores = detection[5:]
                # Find the class with the highest score
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter out weak detections
                if confidence > self.confidence_threshold:
                    # Convert YOLO coords to screen coords
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression to remove duplicate overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)
        
        # Create a copy of the frame to draw on
        result_frame = frame.copy()
        
        # Track new detections in this frame
        current_detections = set()
        current_time = time.time()
        
        # Draw the results and update detection history
        if len(indices) > 0:
            self.last_detection_time = current_time
            
            # Define a list of colors for visualization
            np.random.seed(42)  # For consistent colors
            colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype=np.uint8)
            
            try:
                # OpenCV 4.5.4+
                for i in indices:
                    self._process_detection(i, boxes, class_ids, confidences, colors, result_frame, current_detections, current_time)
            except:
                # Older OpenCV versions
                for i in indices.flatten():
                    self._process_detection(i, boxes, class_ids, confidences, colors, result_frame, current_detections, current_time)
        
        # Add a timestamp to the frame
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
        cv2.putText(result_frame, timestamp, (10, height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        self.processed_frame = result_frame
        self.last_processed_time = current_time
        
        return result_frame
    
    def _process_detection(self, index, boxes, class_ids, confidences, colors, frame, current_detections, current_time):
        """Process a single detection and update tracking data"""
        x, y, w, h = boxes[index]
        class_id = class_ids[index]
        confidence = confidences[index]
        
        # Get the object name
        object_name = self.labels[class_id]
        current_detections.add(object_name)
        
        # Update object count
        self.object_count[object_name] = self.object_count.get(object_name, 0) + 1
        
        # Check if this is a new detection
        is_new = True
        for recent_obj, recent_time in self.recent_detections:
            # If the same object was detected within the last second, don't count as new
            if recent_obj == object_name and current_time - recent_time < 1.0:
                is_new = False
                break
        
        # If new detection, log it
        if is_new:
            detection_entry = {
                "object": object_name,
                "timestamp": current_time,
                "formatted_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time)),
                "confidence": confidence
            }
            self.detected_objects.append(detection_entry)
            self.detection_history.append((object_name, current_time))
            self.logger.info(f"Detected new object: {object_name} at {detection_entry['formatted_time']} (conf: {confidence:.2f})")
        
        # Add to recent detections
        self.recent_detections.append((object_name, current_time))
        
        # Draw bounding box and label on the frame
        color = tuple(map(int, colors[class_id]))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw label background
        text = f"{object_name}: {confidence:.2f}"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x, y - text_size[1] - 10), (x + text_size[0], y), color, -1)
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw text
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    def get_detection_metrics(self):
        """Get metrics about object detections
        
        Returns:
            Dictionary with detection metrics
        """
        current_time = time.time()
        
        # Calculate time since last detection
        time_since_last = None
        if self.last_detection_time:
            time_since_last = current_time - self.last_detection_time
        
        # Get the 5 most recent detections
        recent_detections = []
        for obj, timestamp in self.detection_history[-5:]:
            formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
            recent_detections.append({
                "object": obj,
                "time": formatted_time,
                "seconds_ago": int(current_time - timestamp)
            })
        
        # Get the 5 most common objects
        most_common = sorted(self.object_count.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_objects_detected": sum(self.object_count.values()),
            "unique_objects_detected": len(self.object_count),
            "most_common_objects": most_common,
            "recent_detections": recent_detections,
            "time_since_last_detection": time_since_last,
            "last_processed_time": self.last_processed_time
        }
    
    def get_all_detections(self):
        """Get all detected objects with timestamps
        
        Returns:
            List of all detected objects
        """
        return self.detected_objects
    
    def save_detections_to_file(self, filename="object_detections.csv"):
        """Save all detected objects to a CSV file
        
        Args:
            filename: Name of the CSV file to save to
            
        Returns:
            Boolean indicating success
        """
        try:
            import csv
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Object", "Timestamp", "Formatted Time", "Confidence"])
                
                for detection in self.detected_objects:
                    writer.writerow([
                        detection["object"],
                        detection["timestamp"],
                        detection["formatted_time"],
                        detection["confidence"]
                    ])
            
            self.logger.info(f"Saved {len(self.detected_objects)} object detections to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save object detections: {e}")
            return False