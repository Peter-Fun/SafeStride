import cv2
import torch
import numpy as np
import struct
import tempfile
import os
import threading
import time
import queue
import argparse
import json
import math
import re
import pyttsx3
import googlemaps
import socket
import sounddevice as sd
import torchvision.transforms as transforms
from PIL import Image
from enum import Enum
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict
from collections import deque
from datetime import datetime
from polyline import decode as decode_polyline
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from roboflow import Roboflow

        


API_KEY = "GOOGLE_MAPS_API_KEY"
ROBOFLOW_API_KEY = "ROBOFLOW_API_KEY"
ROBOFLOW_PROJECT_ID = "new-chairs-tables-skhwo"
ROBOFLOW_VERSION = 2

GROUND_CLASS_KEYWORDS = [
    'floor', 'ground', 'road', 'sidewalk', 'pavement', 'path', 'walkway',
    'street', 'asphalt', 'concrete', 'earth', 'soil', 'grass', 'sand',
    'gravel', 'dirt', 'terrain', 'surface', 'platform', 'deck'
]

PERSON_CLASS_KEYWORDS = [
    'person', 'people', 'human', 'man', 'woman', 'child', 'baby',
    'pedestrian', 'walker', 'individual', 'figure', 'body'
]

OBSTACLE_CLASS_KEYWORDS = [
    'wall', 'ceiling', 'sky', 'tree', 'building', 'window', 'door',
    'car', 'vehicle', 'furniture', 'table', 'chair', 'bed', 'sofa',
    'pole', 'column', 'post', 'beam', 'barrier', 'fence', 'sign'
] + PERSON_CLASS_KEYWORDS

class NavigationState(Enum):
    """Navigation system states"""
    INITIALIZING = "initializing"
    GPS_FOLLOWING = "gps_following"
    OBSTACLE_AVOIDANCE = "obstacle_avoidance"
    GROUND_FOLLOWING = "ground_following"
    PATH_BLOCKED = "path_blocked"
    DESTINATION_REACHED = "destination_reached"
    EMERGENCY_STOP = "emergency_stop"

class TurnDirection(Enum):
    """Turn direction indicators"""
    STRAIGHT = "straight"
    SLIGHT_LEFT = "slight_left"
    LEFT = "left"
    SHARP_LEFT = "sharp_left"
    SLIGHT_RIGHT = "slight_right"
    RIGHT = "right"
    SHARP_RIGHT = "sharp_right"
    U_TURN = "u_turn"

@dataclass
class NavigationWaypoint:
    """Navigation waypoint with guidance information"""
    lat: float
    lon: float
    instruction: str
    distance_to_next: float
    heading: float
    turn_direction: TurnDirection
    street_name: str = ""
    reached: bool = False

@dataclass
class SystemConfig:
    """Enhanced system configuration for navigation"""
    # Model settings
    model_type: str = "segformer"
    use_coreml: bool = True
    device: str = "auto"
    
    # Performance settings
    performance_mode: str = "fast"
    disable_ai_inference: bool = False
    
    # Camera settings
    camera_index: int = 1
    input_width: int = 720
    input_height: int = 1280
    model_input_size: int = 512
    
    # Navigation settings
    waypoint_proximity_threshold: float = 5.0
    obstacle_avoidance_distance: float = 2.0
    max_deviation_angle: float = 45.0
    gps_accuracy_threshold: float = 10.0
    
    # Visual navigation settings
    ground_confidence_threshold: float = 0.4
    min_ground_coverage: float = 0.15
    obstacle_detection_threshold: float = 0.3
    
    # Roboflow settings
    roboflow_confidence_threshold: float = 50.0
    
    # Pathfinding settings
    path_length: int = 15
    show_path: bool = True
    overlay_alpha: float = 0.4
    obstacle_margin: int = 20
    
    # Spatial audio settings
    enable_audio: bool = True
    audio_volume: float = 0.7
    beep_frequency: float = 1000
    beep_interval: float = 0.5
    audio_channels: int = 2
    audio_sample_rate: int = 44100
    
    # Audio settings
    tts_enabled: bool = True
    tts_rate: int = 180
    tts_volume: float = 0.9
    distance_announcements: List[float] = field(default_factory=lambda: [100, 50, 20, 10])
    
    # Safety settings
    emergency_stop_time: float = 3.0
    max_path_blocked_time: float = 10.0

@dataclass
class NavigationState_Data:
    """Current navigation state data"""
    current_position: Tuple[float, float] = (0.0, 0.0)
    current_heading: float = 0.0
    target_waypoint: Optional[NavigationWaypoint] = None
    waypoints: List[NavigationWaypoint] = field(default_factory=list)
    current_waypoint_index: int = 0
    distance_to_destination: float = float('inf')
    state: NavigationState = NavigationState.INITIALIZING
    last_gps_update: float = 0.0
    last_heading_update: float = 0.0
    path_blocked_start_time: Optional[float] = None
    emergency_stop_start_time: Optional[float] = None

class LocationData:
    """Thread-safe storage for iPhone location and heading data"""
    def __init__(self):
        self._lock = threading.Lock()
        self._latitude = 0.0
        self._longitude = 0.0
        self._heading = 0.0
        self._last_update = 0.0
        self._is_valid = False

    def update(self, latitude: float, longitude: float, heading: float):
        with self._lock:
            self._latitude = latitude
            self._longitude = longitude
            self._heading = heading
            self._last_update = time.time()
            self._is_valid = True

    def get_data(self):
        with self._lock:
            return (self._latitude, self._longitude, self._heading, self._last_update, self._is_valid)

class LocationServer:
    """TCP server to receive location data from iPhone"""
    def __init__(self, port: int, location_data: LocationData):
        self.port = port
        self.location_data = location_data
        self.running = False
        self.server_socket = None
        self.server_thread = None
        self.client_threads = []
        self.is_healthy = True
        self.last_health_check = time.time()

    def start(self):
        self.running = True
        self.is_healthy = True
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()

    def stop(self):
        self.running = False
        if self.server_socket:
            try: self.server_socket.close()
            except: pass
        for thread in self.client_threads:
            if thread.is_alive():
                thread.join(timeout=0.5)

    def check_health(self):
        """Check if the server thread is still alive and healthy"""
        if not self.running:
            return True
        
        if self.server_thread and not self.server_thread.is_alive():
            self.is_healthy = False
            return False
        
        self.last_health_check = time.time()
        return True

    def restart_if_needed(self):
        """Restart the server if it's unhealthy"""
        if not self.is_healthy and self.running:
            self.stop()
            time.sleep(1)
            self.start()

    def _server_loop(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.settimeout(1.0)
            self.server_socket.bind(('', self.port))
            self.server_socket.listen(5)
            while self.running:
                try:
                    client_socket, client_address = self.server_socket.accept()
                    print("Video client connected from:", client_address)
                    client_thread = threading.Thread(
                        target=self._handle_location_client,
                        args=(client_socket, client_address),
                        daemon=True
                    )
                    client_thread.start()
                    self.client_threads.append(client_thread)
                    self.client_threads = [t for t in self.client_threads if t.is_alive()]
                except socket.timeout:
                    continue
        except Exception as e:
            self.is_healthy = False
        finally:
            if self.server_socket:
                try: self.server_socket.close()
                except: pass

    def _handle_location_client(self, client_socket, client_address):
        try:
            client_socket.settimeout(2.0)
            print("Location client connected from:", client_address)
            while self.running:
                data = client_socket.recv(1024)
                if not data:
                    break
                json_str = data.decode('utf-8')
                location_info = json.loads(json_str)
                latitude = float(location_info['latitude'])
                longitude = float(location_info['longitude'])
                heading = float(location_info['heading'])
                self.location_data.update(latitude, longitude, heading)
                client_socket.send(b"OK")
        except Exception as e:
            pass
        finally:
            try: 
                client_socket.close()
            except: 
                pass

class LocationService:
    """Handle GPS location and compass heading"""
    def __init__(self, location_data: LocationData):
        self.current_location = (0.0, 0.0)
        self.current_heading = 0.0
        self.last_update = 0.0
        self.location_data = location_data
        self.accuracy = float('inf')
        self.heading_history = deque(maxlen=5)
        
    def get_current_location(self) -> Tuple[float, float]:
        """Get current GPS location"""
        lat, lon, heading, last_update, is_valid = self.location_data.get_data()
        return (lat, lon) if is_valid else (0.0, 0.0)
    
    def get_current_heading(self) -> float:
        """Get current compass heading from iPhone magnetometer"""
        lat, lon, heading, last_update, is_valid = self.location_data.get_data()
        return heading if is_valid else 0.0
    
    def _circular_mean(self, angles: List[float]) -> float:
        """Calculate circular mean for compass headings"""
        if not angles:
            return 0.0
        
        x = sum(math.cos(math.radians(angle)) for angle in angles) / len(angles)
        y = sum(math.sin(math.radians(angle)) for angle in angles) / len(angles)
        
        mean_angle = math.degrees(math.atan2(y, x))
        return (mean_angle + 360) % 360
    
    def calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing between two GPS coordinates"""
        delta_lon = math.radians(lon2 - lon1)
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        
        x = math.sin(delta_lon) * math.cos(lat2_rad)
        y = math.cos(lat1_rad) * math.sin(lat2_rad) - (
            math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)
        )
        
        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360
    
    def calculate_turn_angle(self, current_heading: float, target_bearing: float) -> float:
        """Calculate turn angle from current heading to target bearing"""
        diff = target_bearing - current_heading
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return diff
    
    def get_direction_instruction(self, current_heading: float, target_bearing: float) -> str:
        """Get human-readable direction instruction"""
        turn_angle = self.calculate_turn_angle(current_heading, target_bearing)
        
        if abs(turn_angle) <= 15:
            return "continue straight ahead"
        elif 15 < turn_angle <= 45:
            return "turn slightly right"
        elif 45 < turn_angle <= 135:
            return "turn right"
        elif 135 < turn_angle <= 180:
            return "turn sharp right"
        elif -45 <= turn_angle < -15:
            return "turn slightly left"
        elif -135 <= turn_angle < -45:
            return "turn left"
        elif -180 <= turn_angle < -135:
            return "turn sharp left"
        else:
            return "continue straight ahead"
    
    def get_cardinal_direction(self, heading: float) -> str:
        """Convert heading to cardinal direction"""
        directions = [
            "north", "northeast", "east", "southeast",
            "south", "southwest", "west", "northwest"
        ]
        normalized = (heading + 22.5) % 360
        index = int(normalized / 45)
        return directions[index]
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two GPS coordinates in meters"""
        R = 6371000
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_phi/2)**2 + 
             math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c

class TTSService:
    """Text-to-speech service for navigation guidance"""
    def __init__(self, config: SystemConfig):
        self.config = config
        self.engine = None
        self.speech_queue = queue.Queue()
        self.speaking = False
        self.enabled = config.tts_enabled
        self.speech_thread = None
        self.is_healthy = True
        
        if self.enabled:
            self._initialize_tts()
            self._start_speech_thread()
    
    def _initialize_tts(self):
        """Initialize TTS engine"""
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', self.config.tts_rate)
            self.engine.setProperty('volume', self.config.tts_volume)
        except Exception as e:
            self.enabled = False
    
    def _start_speech_thread(self):
        """Start background thread for speech synthesis"""
        self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()

    def check_health(self):
        """Check if the TTS thread is still alive and healthy"""
        if not self.enabled:
            return True
        
        if self.speech_thread and not self.speech_thread.is_alive():
            self.is_healthy = False
            return False
        
        return True

    def restart_if_needed(self):
        """Restart the TTS service if it's unhealthy"""
        if not self.is_healthy and self.enabled:
            self._start_speech_thread()
            self.is_healthy = True
    
    def _speech_worker(self):
        """Background worker for speech synthesis"""
        while True:
            try:
                text, priority = self.speech_queue.get(timeout=1.0)
                if text and self.enabled and self.engine:
                    self.speaking = True
                    self.engine.say(text)
                    self.engine.runAndWait()
                    self.speaking = False
            except queue.Empty:
                continue
            except Exception as e:
                self.speaking = False
    
    def speak(self, text: str, priority: int = 1):
        """Add text to speech queue"""
        if self.enabled and text:
            try:
                if priority > 5:
                    while not self.speech_queue.empty():
                        try:
                            self.speech_queue.get_nowait()
                        except queue.Empty:
                            break
                
                self.speech_queue.put((text, priority), timeout=0.1)
            except queue.Full:
                pass
    
    def speak_direction(self, instruction: str, distance: float = None, current_heading: float = None, target_bearing: float = None):
        """Speak navigation direction with compass-based guidance"""
        direction_parts = []
        if distance is not None:
            if distance < 20:
                direction_parts.append(f"In {int(distance)} meters")
            elif distance < 100:
                direction_parts.append(f"In {int(distance/10)*10} meters")
            else:
                direction_parts.append(f"In {int(distance/100)*100} meters")
        
        if current_heading is not None and target_bearing is not None:
            turn_angle = target_bearing - current_heading
            while turn_angle > 180:
                turn_angle -= 360
            while turn_angle < -180:
                turn_angle += 360
            if abs(turn_angle) <= 15:
                compass_instruction = "continue straight"
            elif 15 < turn_angle <= 45:
                compass_instruction = f"turn slightly right, about {int(turn_angle)} degrees"
            elif 45 < turn_angle <= 135:
                compass_instruction = f"turn right, about {int(turn_angle)} degrees"
            elif 135 < turn_angle:
                compass_instruction = f"turn sharp right, about {int(turn_angle)} degrees"
            elif -45 <= turn_angle < -15:
                compass_instruction = f"turn slightly left, about {int(abs(turn_angle))} degrees"
            elif -135 <= turn_angle < -45:
                compass_instruction = f"turn left, about {int(abs(turn_angle))} degrees"
            elif turn_angle < -135:
                compass_instruction = f"turn sharp left, about {int(abs(turn_angle))} degrees"
            else:
                compass_instruction = "continue straight"
            direction_parts.append(compass_instruction)
        
        direction_parts.append(instruction.lower())
        
        full_instruction = ", ".join(direction_parts)
        self.speak(full_instruction, priority=3)
    
    def speak_heading_check(self, current_heading: float, target_bearing: float):
        """Help user orient to correct heading"""
        turn_angle = target_bearing - current_heading
        while turn_angle > 180:
            turn_angle -= 360
        while turn_angle < -180:
            turn_angle += 360
        
        current_cardinal = self._get_cardinal_direction(current_heading)
        target_cardinal = self._get_cardinal_direction(target_bearing)
        
        if abs(turn_angle) <= 10:
            self.speak(f"Good, you're facing the right direction. Continue {current_cardinal}.", priority=4)
        else:
            if turn_angle > 0:
                self.speak(f"Turn right {int(abs(turn_angle))} degrees. Face more {target_cardinal}.", priority=5)
            else:
                self.speak(f"Turn left {int(abs(turn_angle))} degrees. Face more {target_cardinal}.", priority=5)
    
    def _get_cardinal_direction(self, heading: float) -> str:
        """Convert heading to cardinal direction"""
        directions = [
            "north", "northeast", "east", "southeast",
            "south", "southwest", "west", "northwest"
        ]
        normalized = (heading + 22.5) % 360
        index = int(normalized / 45)
        return directions[index]
    
    def speak_warning(self, warning: str):
        """Speak safety warning with high priority"""
        self.speak(f"Warning: {warning}", priority=8)
    
    def speak_status(self, status: str):
        """Speak general status update"""
        self.speak(status, priority=2)

class SpatialAudioGuide:
    """Spatial audio guidance system with AirPods support"""
    def __init__(self, config: SystemConfig):
        self.config = config
        self.last_beep_time = 0
        self.audio_stream = None
        self.selected_device = None
        self.is_healthy = True
        if self.config.enable_audio:
            self.setup_audio()
    
    def setup_audio(self):
        """Initialize audio system with device detection"""
        try:
            # Query available audio devices
            devices = sd.query_devices()

            # Prioritize AirPods and headphone devices
            preferred_device = None
            for i, dev in enumerate(devices):
                device_name = dev['name'].lower()
                output_channels = dev['max_output_channels']
                
                if (("airpod" in device_name or 
                     "headphone" in device_name or 
                     "bluetooth" in device_name) and 
                    output_channels >= 2):
                    preferred_device = i
            
            # Select device
            if preferred_device is not None:
                self.selected_device = preferred_device
                device_name = devices[preferred_device]['name']
            else:
                self.selected_device = sd.default.device[1]
                device_name = devices[self.selected_device]['name']
            
            # Initialize audio stream
            self.audio_stream = sd.OutputStream(
                device=self.selected_device,
                channels=self.config.audio_channels,
                samplerate=self.config.audio_sample_rate,
                blocksize=1024,
            )
            self.audio_stream.start()
            
            # Play startup confirmation sound
            self._play_startup_sound()
            
        except Exception as e:
            self.audio_stream = None
    
    def _play_startup_sound(self):
        """Play startup sound to confirm audio is working"""
        if self.audio_stream is None:
            return
            
        duration = 0.5
        t = np.linspace(0, duration, int(self.config.audio_sample_rate * duration), False)
        
        # Create ascending sweep + beep
        sweep_duration = int(len(t) * 0.6)
        sweep_t = t[:sweep_duration]
        freq = np.linspace(300, 1200, len(sweep_t))
        sweep = np.sin(2 * np.pi * freq * sweep_t)
        
        beep_t = t[sweep_duration:]
        beep = np.sin(2 * np.pi * 1000 * beep_t)
        
        sound = np.concatenate([sweep, beep])
        envelope = np.concatenate([
            np.linspace(0, 1, len(t)//8),
            np.ones(len(t) - len(t)//4),
            np.linspace(1, 0, len(t)//8)
        ])
        sound = sound * envelope * 0.9
        
        # Create stereo with panning effect
        left_channel = sound.copy()
        right_channel = sound.copy()
        
        pan_effect = np.sin(np.linspace(0, 3*np.pi, len(t)))
        left_gain = 0.5 + 0.5 * pan_effect
        right_gain = 0.5 - 0.5 * pan_effect
        
        left_channel *= left_gain
        right_channel *= right_gain
        
        stereo_sound = np.column_stack((left_channel, right_channel))
        
        try:
            self.audio_stream.write(stereo_sound.astype(np.float32))
            time.sleep(0.1)
        except Exception as e:
            print(f"Audio playback error: {e}")

    def check_health(self):
        """Check audio system health"""
        return self.is_healthy

    def restart_if_needed(self):
        """Restart audio system if needed"""
        if not self.is_healthy and self.config.enable_audio:
            self.setup_audio()
            self.is_healthy = True

    def update_audio_guidance(self, path_points: List[Tuple[int, int]], 
                             frame_width: int, frame_height: int):
        """Generate spatial audio guidance based on path direction"""
        
        if (self.audio_stream is None or 
            not self.config.enable_audio or 
            len(path_points) < 3):
            return
        
        current_time = time.time()
        if current_time - self.last_beep_time < self.config.beep_interval:
            return
        
        # Calculate path direction from first few points
        direction_point = path_points[min(2, len(path_points) - 1)]
        center_x = frame_width / 2
        
        # Convert to direction: -1.0 = left, 0 = center, 1.0 = right
        direction = (direction_point[0] - center_x) / (center_x)
        direction = np.clip(direction, -1.0, 1.0)
        
        # Generate spatial beep
        self._generate_spatial_beep(direction)
        self.last_beep_time = current_time

    def _generate_spatial_beep(self, direction: float):
        """Generate spatially positioned beep"""
        
        if self.audio_stream is None:
            return
        
        try:
            # Generate beep with harmonics for clarity
            duration = 0.15
            sample_rate = self.config.audio_sample_rate
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            
            # Multi-harmonic beep
            freq = self.config.beep_frequency
            fundamental = 0.7 * np.sin(2 * np.pi * freq * t)
            harmonic1 = 0.2 * np.sin(2 * np.pi * freq * 2 * t)
            harmonic2 = 0.1 * np.sin(2 * np.pi * freq * 3 * t)
            beep = fundamental + harmonic1 + harmonic2
            
            # Apply fade in/out
            fade_samples = int(0.02 * sample_rate)
            beep[:fade_samples] *= np.linspace(0, 1, fade_samples)
            beep[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            
            # Apply spatial positioning
            left_vol, right_vol = self._calculate_stereo_volumes(direction)
            
            # Create stereo channels
            left_channel = beep * left_vol
            right_channel = beep * right_vol
            stereo_beep = np.column_stack((left_channel, right_channel))
            
            # Apply volume and play
            volume = self.config.audio_volume
            self.audio_stream.write(stereo_beep.astype(np.float32) * volume)
            
        except Exception as e:
            pass

    def _calculate_stereo_volumes(self, direction: float) -> Tuple[float, float]:
        """Calculate left/right channel volumes for spatial positioning"""
        
        direction = max(-1.0, min(1.0, direction))
        
        # Apply square-root panning law
        if direction < 0:  # Sound to the left
            pan_position = -direction
            left_vol = 1.0
            right_vol = np.cos(pan_position * np.pi / 2)
        else:  # Sound to the right
            pan_position = direction
            left_vol = np.cos(pan_position * np.pi / 2)
            right_vol = 1.0
        
        return left_vol, right_vol

    def test_audio(self) -> bool:
        """Test audio functionality"""
        
        if self.audio_stream is None:
            return False
        
        try:
            duration = 0.5
            sample_rate = self.config.audio_sample_rate
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            
            # Create test sound
            test_tone = 0.8 * np.sin(2 * np.pi * 440 * t)
            
            # Apply envelope
            envelope = np.concatenate([
                np.linspace(0, 1, len(t)//8),
                np.ones(len(t) - len(t)//4),
                np.linspace(1, 0, len(t)//8)
            ])
            test_tone = test_tone * envelope
            
            # Test left channel
            left_test = np.column_stack((test_tone, np.zeros_like(test_tone)))
            self.audio_stream.write(left_test.astype(np.float32))
            
            time.sleep(0.1)
            
            # Test right channel
            right_test = np.column_stack((np.zeros_like(test_tone), test_tone))
            self.audio_stream.write(right_test.astype(np.float32))
            
            return True
            
        except Exception as e:
            return False
    
    def close(self):
        """Close audio resources"""
        if self.audio_stream is not None:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception as e:
                pass
            finally:
                self.audio_stream = None

class GroundDetectionModel:
    """Enhanced ground detection model with obstacle detection and Roboflow integration"""
    def __init__(self, config: SystemConfig):
        self.config = config
        self.device = self._setup_device()
        self.model = None
        self.processor = None
        self.coreml_model = None
        
        self.roboflow_model = None
        self.roboflow_enabled = False
        
        self.ground_class_indices = []
        self.obstacle_class_indices = []
        self.person_class_indices = []
        self.id2label = {}
        self.label2id = {}
        
        try:
            self._load_model()
            self._setup_class_mappings()
            self._load_roboflow_model()
        except Exception as e:
            self.model = None
    
    def _setup_device(self) -> torch.device:
        """Setup compute device"""
        if self.config.device == "auto":
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.config.device)
        return device
    
    def _load_model(self):
        """Load semantic segmentation model"""
        try:
            if self.config.model_type == "segformer":
                self.processor = SegformerImageProcessor.from_pretrained(
                    "nvidia/segformer-b0-finetuned-ade-512-512"
                )
                self.model = SegformerForSemanticSegmentation.from_pretrained(
                    "nvidia/segformer-b0-finetuned-ade-512-512"
                ).to(self.device)
                self.model.eval()
            else:
                self.model = deeplabv3_mobilenet_v3_large(pretrained=True).to(self.device)
                self.model.eval()
                
        except Exception as e:
            pass
    
    def _load_roboflow_model(self):
        """Load Roboflow model for chair/table detection"""
        try:
            rf = Roboflow(api_key=ROBOFLOW_API_KEY)
            project = rf.workspace().project(ROBOFLOW_PROJECT_ID)
            self.roboflow_model = project.version(ROBOFLOW_VERSION).model
            self.roboflow_enabled = True
            
        except ImportError as e:
            self.roboflow_enabled = False
        except Exception as e:
            self.roboflow_enabled = False
    
    def _setup_class_mappings(self):
        """Setup class mappings for ground and obstacle detection"""
        try:
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'id2label'):
                self.id2label = self.model.config.id2label
                self.label2id = self.model.config.label2id
            else:
                self.id2label = {i: f"class_{i}" for i in range(150)}
                self.label2id = {v: k for k, v in self.id2label.items()}
                       
        except Exception as e:
            self.id2label = {i: f"class_{i}" for i in range(21)}
            self.label2id = {v: k for k, v in self.id2label.items()}
    
    def _create_roboflow_mask(self, frame: np.ndarray, predictions: Dict) -> np.ndarray:
        """Create mask from Roboflow chair/table detections"""
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        
        try:
            if 'predictions' in predictions:
                for detection in predictions['predictions']:
                    if detection['confidence'] >= self.config.roboflow_confidence_threshold / 100.0:
                        if 'points' in detection and len(detection['points']) > 2:
                            points = detection['points']
                            pts = []
                            for point in points:
                                x = int(max(0, min(point.get('x', 0), w-1)))
                                y = int(max(0, min(point.get('y', 0), h-1)))
                                pts.append([x, y])
                            
                            if len(pts) >= 3:
                                pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                                temp_mask = np.zeros((h, w), dtype=np.uint8)
                                cv2.fillPoly(temp_mask, [pts], 1)
                                mask = mask | (temp_mask > 0)
                        
                        else:
                            x = int(detection.get('x', 0) - detection.get('width', 0) / 2)
                            y = int(detection.get('y', 0) - detection.get('height', 0) / 2)
                            x2 = int(detection.get('x', 0) + detection.get('width', 0) / 2)
                            y2 = int(detection.get('y', 0) + detection.get('height', 0) / 2)
                            
                            x = max(0, min(x, w - 1))
                            y = max(0, min(y, h - 1))
                            x2 = max(0, min(x2, w - 1))
                            y2 = max(0, min(y2, h - 1))
                            
                            if x2 > x and y2 > y:
                                mask[y:y2, x:x2] = True
                        
        except Exception as e:
            pass
        return mask

    def _run_roboflow_inference_safe(self, frame: np.ndarray) -> Optional[Dict]:
        """Ultra-safe Roboflow inference"""
        if not self.roboflow_enabled or self.roboflow_model is None:
            return None
        temp_path = None
        try:
            try:
                rgb_frame = frame.copy()
                rgb_frame[:, :, 0] = frame[:, :, 2]
                rgb_frame[:, :, 2] = frame[:, :, 0]
                
            except Exception as conv_error:
                rgb_frame = frame
                
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                pil_image = Image.fromarray(rgb_frame)
                
                with threading.Lock():
                    pil_image.save(temp_path, 'JPEG', quality=85, optimize=True)
                
                with threading.Lock():
                    predictions = self.roboflow_model.predict(
                        temp_path, 
                        confidence=self.config.roboflow_confidence_threshold
                    ).json()
                
                if isinstance(predictions, dict) and 'predictions' in predictions:
                    return predictions
                else:
                    return None
            except Exception as e:
                pass
                
        except Exception as e:
            return None
        finally:
            if temp_path:
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except Exception as e:
                    pass
            try:
                del rgb_frame, pil_image
            except:
                pass
    
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """Analyze frame for ground and obstacles"""
        try:
            h, w = frame.shape[:2]
            result = {
                'ground_mask': np.zeros((h, w), dtype=bool),
                'obstacle_mask': np.zeros((h, w), dtype=bool),
                'ground_coverage': 0.0,
                'obstacle_coverage': 0.0,
                'person_coverage': 0.0,
                'path_ground_coverage': 0.0,
                'path_has_obstacles': False,
                'safe_to_proceed': True,
                'roboflow_predictions': None,
                'roboflow_mask': np.zeros((h, w), dtype=bool)
            }
            
            if self.config.disable_ai_inference:
                result['ground_mask'][h//2:, :] = True
                result['ground_coverage'] = 0.5
                result['path_ground_coverage'] = 0.5
                result['safe_to_proceed'] = True
                return result
            
            # Run SegFormer/DeepLabv3 semantic segmentation
            if self.model is not None:
                if self.processor: # SegFormer
                    inputs = self.processor(images=frame, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        logits = outputs.logits
                        
                        upsampled_logits = torch.nn.functional.interpolate(
                            logits, size=(h, w), mode="bilinear", align_corners=False
                        )
                        predicted_segmentation = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
                        
                        del upsampled_logits
                        del logits
                        del outputs

                    for k, v in inputs.items():
                        del v
                    del inputs
                else:  # DeepLabv3
                    transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((self.config.model_input_size, self.config.model_input_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    
                    input_tensor = transform(frame).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        output = self.model(input_tensor)['out']
                        predicted_segmentation = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
                        
                        del output

                    del input_tensor
                    predicted_segmentation = cv2.resize(
                        predicted_segmentation.astype(np.uint8), 
                        (w, h), 
                        interpolation=cv2.INTER_NEAREST
                    )
                
                # Create masks
                ground_mask = np.zeros((h, w), dtype=bool)
                obstacle_mask = np.zeros((h, w), dtype=bool)
                
                for class_id in np.unique(predicted_segmentation):
                    if class_id < len(self.id2label):
                        class_name = self.id2label[class_id].lower()
                        class_pixels = predicted_segmentation == class_id
                        
                        if any(keyword in class_name for keyword in GROUND_CLASS_KEYWORDS):
                            ground_mask |= class_pixels
                        elif any(keyword in class_name for keyword in OBSTACLE_CLASS_KEYWORDS):
                            obstacle_mask |= class_pixels
                result['ground_mask'] = ground_mask
                result['obstacle_mask'] = obstacle_mask
            
            # Run Roboflow chair/table detection
            if self.roboflow_enabled and self.roboflow_model is not None:
                try:
                    roboflow_predictions = self._run_roboflow_inference_safe(frame)
                    result['roboflow_predictions'] = roboflow_predictions
                    
                    if roboflow_predictions:
                        roboflow_mask = self._create_roboflow_mask(frame, roboflow_predictions)
                        result['roboflow_mask'] = roboflow_mask
                        result['obstacle_mask'] = result['obstacle_mask'] | roboflow_mask
                    else:
                        result['roboflow_mask'] = np.zeros((h, w), dtype=bool)
                    
                except Exception as e:
                    result['roboflow_predictions'] = None
                    result['roboflow_mask'] = np.zeros((h, w), dtype=bool)
            else:
                result['roboflow_predictions'] = None
                result['roboflow_mask'] = np.zeros((h, w), dtype=bool)

            # Calculate coverage percentages
            total_pixels = h * w
            result['ground_coverage'] = np.sum(result['ground_mask']) / total_pixels
            result['obstacle_coverage'] = np.sum(result['obstacle_mask']) / total_pixels
            
            # Analyze path ahead
            path_region = result['ground_mask'][int(0.7*h):, :]
            result['path_ground_coverage'] = np.sum(path_region) / path_region.size if path_region.size > 0 else 0
            
            # Detect obstacles in path
            path_obstacle_region = result['obstacle_mask'][int(0.7*h):, :]
            result['path_has_obstacles'] = np.sum(path_obstacle_region) > (path_obstacle_region.size * 0.05)
            
            # Safety assessment
            result['safe_to_proceed'] = (
                result['ground_coverage'] >= self.config.min_ground_coverage and
                result['obstacle_coverage'] < self.config.obstacle_detection_threshold and
                result['path_ground_coverage'] > self.config.min_ground_coverage
            )
            
            del predicted_segmentation
            return result
            
        except Exception as e:
            h, w = frame.shape[:2]
            return {
                'ground_mask': np.ones((h, w), dtype=bool),
                'obstacle_mask': np.zeros((h, w), dtype=bool),
                'ground_coverage': 1.0,
                'obstacle_coverage': 0.0,
                'person_coverage': 0.0,
                'path_ground_coverage': 1.0,
                'path_has_obstacles': False,
                'safe_to_proceed': True,
                'roboflow_predictions': None,
                'roboflow_mask': np.zeros((h, w), dtype=bool)
            }

class PathVisualizer:
    """Tesla FSD-style intelligent pathfinding with advanced obstacle avoidance"""
    def __init__(self, config):
        self.config = config
        
        # Path generation parameters
        self.path_smoothing_factor = 0.6
        self.lookahead_distance = 8
        self.path_width_samples = 21
        self.min_path_width = 30
        self.path_confidence_threshold = 0.3
        
        # Tesla FSD-style visualization parameters
        self.base_path_width = 80
        self.min_path_width_perspective = 12
        self.perspective_strength = 0.8
        self.ar_ground_offset = 3
        self.depth_fade_strength = 0.6
        
        # Path generation strategies
        self.strategies = ['center_biased', 'left_biased', 'right_biased', 'widest_path']

    def generate_path(self, ground_mask: np.ndarray, obstacle_mask: np.ndarray, 
             current_heading: float = None, target_bearing: float = None) -> List[Tuple[int, int]]:
        """Generate optimal path with adaptive length based on obstacles"""
        
        h, w = ground_mask.shape
        
        # Proper safe mask calculation with erosion
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.config.obstacle_margin, self.config.obstacle_margin))
        eroded_obstacles = cv2.dilate(obstacle_mask.astype(np.uint8), kernel, iterations=1)
        safe_mask = ground_mask & ~(eroded_obstacles.astype(bool))
        
        # Calculate adaptive path length based on obstacles ahead
        adaptive_length = self._calculate_adaptive_path_length(safe_mask)
        candidate_paths = []
        
        center_path = self._generate_center_biased_path(safe_mask, adaptive_length)
        if center_path:
            candidate_paths.append(center_path)
        left_path = self._generate_directional_path(safe_mask, bias=-0.3, path_length=adaptive_length)
        if left_path:
            candidate_paths.append(left_path)
        right_path = self._generate_directional_path(safe_mask, bias=0.3, path_length=adaptive_length)
        if right_path:
            candidate_paths.append(right_path)
        wide_path = self._generate_widest_path(safe_mask, adaptive_length)
        if wide_path:
            candidate_paths.append(wide_path)
        
        if not candidate_paths:
            return []
        
        best_path = self._select_best_path(candidate_paths, safe_mask)
        if current_heading is not None and target_bearing is not None:
            best_path = self._apply_gps_bias(best_path, current_heading, target_bearing, w)
        
        smoothed_path = self._smooth_path(best_path, safe_mask)
        return smoothed_path

    def _generate_center_biased_path(self, mask: np.ndarray, path_length: int = None) -> List[Tuple[int, int]]:
        """Generate path biased toward center with adaptive length"""
        h, w = mask.shape
        path_points = []
        
        if path_length is None:
            path_length = self.config.path_length
        
        start_x = w // 2
        start_y = h - 20
        current_x = start_x
        
        segment_height = max(3, h // (path_length * 2))
        
        for i in range(path_length):
            current_y = start_y - (i * segment_height)
            
            if current_y < h * 0.1:
                break
            
            next_x = self._find_best_x_position(mask, current_x, current_y, bias=0.0)
            
            if next_x is None:
                break
            
            steering_force = (next_x - current_x) * 0.7
            max_steering = min(25, w // 15)
            steering_force = np.clip(steering_force, -max_steering, max_steering)
            
            current_x = current_x + steering_force
            current_x = np.clip(current_x, 30, w - 30)
            
            if self._check_path_width(mask, int(current_x), int(current_y)):
                path_points.append((int(current_x), int(current_y)))
            else:
                break
        
        return path_points

    def _generate_directional_path(self, mask: np.ndarray, bias: float, path_length: int = None) -> List[Tuple[int, int]]:
        """Generate path with directional bias and adaptive length"""
        h, w = mask.shape
        path_points = []
        
        if path_length is None:
            path_length = self.config.path_length
        
        start_x = int(w * (0.5 + bias * 0.3))
        start_y = h - 20
        current_x = start_x
        
        segment_height = max(3, h // (path_length * 2))
        
        for i in range(path_length):
            current_y = start_y - (i * segment_height)
            
            if current_y < h * 0.1:
                break
            next_x = self._find_best_x_position(mask, current_x, current_y, bias=bias)
            
            if next_x is None:
                break
            current_x = next_x
            current_x = np.clip(current_x, 30, w - 30)
            
            if self._check_path_width(mask, int(current_x), int(current_y)):
                path_points.append((int(current_x), int(current_y)))
            else:
                break
        return path_points

    def _generate_widest_path(self, mask: np.ndarray, path_length: int = None) -> List[Tuple[int, int]]:
        """Generate path that follows the widest safe areas with adaptive length"""
        h, w = mask.shape
        path_points = []
        
        if path_length is None:
            path_length = self.config.path_length
        
        start_x = w // 2
        start_y = h - 20
        current_x = start_x
        
        segment_height = max(3, h // (path_length * 2))
        for i in range(path_length):
            current_y = start_y - (i * segment_height)
            if current_y < h * 0.1:
                break
        
            best_x, max_width = self._find_widest_position(mask, current_y)
            if best_x is None or max_width < self.min_path_width:
                break
            
            steering = (best_x - current_x) * 0.5
            max_steering = min(30, w // 12)
            steering = np.clip(steering, -max_steering, max_steering)
            current_x = current_x + steering
            current_x = np.clip(current_x, 30, w - 30)
            
            path_points.append((int(current_x), int(current_y)))
        return path_points

    def _find_best_x_position(self, mask: np.ndarray, current_x: float, 
                         current_y: int, bias: float = 0.0) -> Optional[float]:
        """Find best X position considering lookahead and bias"""
        h, w = mask.shape
        
        search_height = min(20, h // 20)
        y_start = max(0, current_y - search_height // 2)
        y_end = min(h, current_y + search_height // 2)
        if y_start >= y_end:
            return None
        
        x_samples = np.linspace(max(0, current_x - w//3), 
                            min(w-1, current_x + w//3), 
                            self.path_width_samples)
        best_x = None
        best_score = -1
        
        for x in x_samples:
            score = self._evaluate_position(mask, int(x), current_y, bias)
            if score > best_score and score > self.path_confidence_threshold:
                best_score = score
                best_x = x
        
        return best_x

    def _find_widest_position(self, mask: np.ndarray, y: int) -> Tuple[Optional[float], float]:
        """Find X position with maximum safe width"""
        h, w = mask.shape
        
        search_height = min(15, h // 25)
        y_start = max(0, y - search_height // 2)
        y_end = min(h, y + search_height // 2)
        if y_start >= y_end:
            return None, 0
        
        safe_strip = mask[y_start:y_end, :]
        safe_row = np.sum(safe_strip, axis=0) / (y_end - y_start)
        safe_threshold = 0.7
        safe_positions = safe_row > safe_threshold
        if not np.any(safe_positions):
            return None, 0
        
        contiguous_regions = []
        in_region = False
        region_start = 0
        
        for i, is_safe in enumerate(safe_positions):
            if is_safe and not in_region:
                region_start = i
                in_region = True
            elif not is_safe and in_region:
                contiguous_regions.append((region_start, i-1))
                in_region = False
        if in_region:
            contiguous_regions.append((region_start, len(safe_positions)-1))
        if not contiguous_regions:
            return None, 0
        
        widest_region = max(contiguous_regions, key=lambda r: r[1] - r[0])
        region_width = widest_region[1] - widest_region[0]
        region_center = (widest_region[0] + widest_region[1]) / 2
        return region_center, region_width

    def _evaluate_position(self, mask: np.ndarray, x: int, y: int, bias: float) -> float:
        """Evaluate how good a position is for the path"""
        h, w = mask.shape
        if x < 0 or x >= w or y < 0 or y >= h:
            return 0.0
        score = 0.0
        
        # Local safety check
        local_radius = 8
        x_start = max(0, x - local_radius)
        x_end = min(w, x + local_radius)
        y_start = max(0, y - local_radius)
        y_end = min(h, y + local_radius)
        
        local_area = mask[y_start:y_end, x_start:x_end]
        local_safety = np.mean(local_area)
        score += local_safety * 0.4
        
        # Lookahead safety check
        lookahead_segments = min(self.lookahead_distance, (y - h//10) // 10)
        lookahead_score = 0.0
        
        for i in range(1, lookahead_segments + 1):
            future_y = y - i * 10
            if future_y < h * 0.1:
                break
            future_area = mask[max(0, future_y-5):min(h, future_y+5),
                            max(0, x-10):min(w, x+10)]
            if future_area.size > 0:
                future_safety = np.mean(future_area)
                weight = 1.0 / (i + 1)
                lookahead_score += future_safety * weight
        if lookahead_segments > 0:
            lookahead_score /= lookahead_segments
        
        score += lookahead_score * 0.3
        
        # Path width bonus
        width_score = self._calculate_width_score(mask, x, y)
        score += width_score * 0.2
        
        # Directional bias
        center_x = w // 2
        bias_factor = 1.0 + bias * ((x - center_x) / (w // 2))
        bias_factor = np.clip(bias_factor, 0.5, 1.5)
        score *= bias_factor
        
        return score

    def _calculate_width_score(self, mask: np.ndarray, x: int, y: int) -> float:
        """Calculate score based on available path width"""
        h, w = mask.shape
        
        left_extent = 0
        right_extent = 0
        
        # Scan left
        for i in range(1, min(x, w//4)):
            if x - i < 0 or not mask[y, x - i]:
                break
            left_extent = i
        
        # Scan right  
        for i in range(1, min(w - x, w//4)):
            if x + i >= w or not mask[y, x + i]:
                break
            right_extent = i
        
        total_width = left_extent + right_extent
        normalized_width = total_width / (w // 4)
        width_score = 1.0 / (1.0 + np.exp(-5 * (normalized_width - 0.3)))
        
        return width_score

    def _check_path_width(self, mask: np.ndarray, x: int, y: int) -> bool:
        """Check if there's enough safe width for the path"""
        h, w = mask.shape
        
        if x < 0 or x >= w or y < 0 or y >= h:
            return False
        
        half_width = self.min_path_width // 2
        x_start = max(0, x - half_width)
        x_end = min(w, x + half_width)
        y_start = max(0, y - 3)
        y_end = min(h, y + 3)
        
        path_area = mask[y_start:y_end, x_start:x_end]
        if path_area.size == 0:
            return False
        
        safety_ratio = np.mean(path_area)
        return safety_ratio > 0.7
    
    def _calculate_adaptive_path_length(self, safe_mask: np.ndarray) -> int:
        """Calculate optimal path length based on obstacle density ahead"""
        h, w = safe_mask.shape
        
        # Start from bottom of frame, scan upward
        start_y = h - 20
        max_possible_length = min(self.config.path_length, start_y // 3)
        
        # Scan forward in segments to find obstacles
        segment_height = max(3, h // (max_possible_length * 2))
        safe_segments = 0
        
        for i in range(max_possible_length):
            segment_y = start_y - (i * segment_height)
            if segment_y < h * 0.1:
                break
                
            # Check central corridor safety
            corridor_width = min(80, w // 4)
            x_start = max(0, w//2 - corridor_width//2)
            x_end = min(w, w//2 + corridor_width//2)
            y_start = max(0, segment_y - segment_height//2)
            y_end = min(h, segment_y + segment_height//2)
            if y_start >= y_end or x_start >= x_end:
                break
                
            segment_area = safe_mask[y_start:y_end, x_start:x_end]
            safety_ratio = np.mean(segment_area) if segment_area.size > 0 else 0
            
            if safety_ratio > 0.6:
                safe_segments += 1
            else:
                break
        
        # Adaptive length: minimum 3, maximum config.path_length
        adaptive_length = max(3, min(safe_segments + 2, self.config.path_length))
        return adaptive_length

    def _apply_gps_bias(self, path_points: List[Tuple[int, int]], current_heading: float, 
                   target_bearing: float, frame_width: int) -> List[Tuple[int, int]]:
        """Apply dynamic bias toward GPS destination bearing - FIXED VERSION"""
        
        if len(path_points) < 2:
            return path_points
        
        # Calculate heading error
        heading_error = self._normalize_angle(target_bearing - current_heading)
        error_magnitude = abs(heading_error)
        
        # Only apply significant bias if error is > 20 degrees
        if error_magnitude < 20:
            return path_points
        
        # Calculate total vertical distance of the path
        bottom_y = path_points[0][1]
        top_y = min([p[1] for p in path_points])
        total_y_distance = max(1, bottom_y - top_y)
        fov_degrees = 90
        normalized_error = np.clip(heading_error / fov_degrees, -0.8, 0.8)
        max_curve_pixels = frame_width * 0.25
        curve_strength = normalized_error * max_curve_pixels
        bias_strength = 0.15 + 0.25 * (1 / (1 + np.exp(-0.03 * (error_magnitude - 45))))
        bias_strength = min(0.4, bias_strength)
        biased_path = []
        start_x = frame_width // 2
        
        biased_path.append((start_x, path_points[0][1]))
        for i in range(1, len(path_points)):
            x, y = path_points[i]
            y_progress = min(1.0, max(0.0, (bottom_y - y) / total_y_distance))
            curve_factor = y_progress * y_progress * (3 - 2 * y_progress)

            desired_x = start_x + curve_strength * curve_factor
            blended_x = (1 - bias_strength) * x + bias_strength * desired_x
            blended_x = np.clip(blended_x, frame_width * 0.1, frame_width * 0.9)
            biased_path.append((int(blended_x), y))
        return biased_path

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-180, 180] range"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def _select_best_path(self, candidate_paths: List[List[Tuple[int, int]]], 
                     mask: np.ndarray) -> List[Tuple[int, int]]:
        """Select the best path from candidates"""
        if not candidate_paths:
            return []
        if len(candidate_paths) == 1:
            return candidate_paths[0]
        
        best_path = None
        best_score = -1
        for path in candidate_paths:
            score = self._score_path(path, mask)
            if score > best_score:
                best_score = score
                best_path = path
        return best_path if best_path else candidate_paths[0]

    def _score_path(self, path: List[Tuple[int, int]], mask: np.ndarray) -> float:
        """Score a complete path based on safety, length, and smoothness"""
        if not path:
            return 0.0
        h, w = mask.shape
        score = 0.0
        
        length_score = len(path) / self.config.path_length
        score += length_score * 0.3
        safety_scores = []
        for x, y in path:
            if 0 <= x < w and 0 <= y < h:
                area_size = 8
                x_start = max(0, x - area_size)
                x_end = min(w, x + area_size)
                y_start = max(0, y - area_size)
                y_end = min(h, y + area_size)
                area_safety = np.mean(mask[y_start:y_end, x_start:x_end])
                safety_scores.append(area_safety)
        if safety_scores:
            avg_safety = np.mean(safety_scores)
            score += avg_safety * 0.4
        
        if len(path) > 2:
            turn_penalties = []
            for i in range(1, len(path) - 1):
                prev_x, prev_y = path[i-1]
                curr_x, curr_y = path[i]
                next_x, next_y = path[i+1]

                v1 = (curr_x - prev_x, curr_y - prev_y)
                v2 = (next_x - curr_x, next_y - curr_y)
                turn_magnitude = abs(v1[0] - v2[0]) / max(1, w // 20)
                turn_penalties.append(min(1.0, turn_magnitude))
            if turn_penalties:
                smoothness = 1.0 - np.mean(turn_penalties)
                score += smoothness * 0.2
        if len(path) > 1:
            forward_progress = (path[0][1] - path[-1][1]) / h
            score += forward_progress * 0.1
        return score

    def _smooth_path(self, path: List[Tuple[int, int]], mask: np.ndarray) -> List[Tuple[int, int]]:
        """Smooth path for Tesla-like curves"""
        if len(path) < 3:
            return path
        
        smoothed = [path[0]]
        for i in range(1, len(path) - 1):
            prev_x, prev_y = path[i-1]
            curr_x, curr_y = path[i]
            next_x, next_y = path[i+1]
            
            smooth_x = (prev_x * 0.15 + curr_x * 0.7 + next_x * 0.15)
            smooth_y = curr_y
            smooth_x = int(smooth_x)
            if (0 <= smooth_x < mask.shape[1] and 0 <= smooth_y < mask.shape[0]):
                margin = 5
                x_start = max(0, smooth_x - margin)
                x_end = min(mask.shape[1], smooth_x + margin)
                y_start = max(0, smooth_y - margin)
                y_end = min(mask.shape[0], smooth_y + margin)
                
                area_safety = np.mean(mask[y_start:y_end, x_start:x_end])
                if area_safety > 0.6:
                    smoothed.append((smooth_x, smooth_y))
                else:
                    smoothed.append((curr_x, curr_y))
            else:
                smoothed.append((curr_x, curr_y))
        if len(path) > 1:
            smoothed.append(path[-1]) 
        return smoothed

    def draw_path(self, frame: np.ndarray, path_points: List[Tuple[int, int]]) -> np.ndarray:
        """Draw Tesla FSD-style path with enhanced AR visuals"""
        if len(path_points) < 2:
            return frame
        
        result = frame.copy()
        h, w = frame.shape[:2]
        
        path_overlay = np.zeros_like(frame, dtype=np.float32)
        self._draw_ar_ground_path(path_overlay, path_points, h, w)
        self._draw_path_edges(path_overlay, path_points, h, w)
        self._draw_center_line(path_overlay, path_points, h, w)
        self._blend_path_overlay(result, path_overlay)
        self._draw_direction_indicators(result, path_points, h, w)
        return result

    def _draw_ar_ground_path(self, overlay: np.ndarray, path_points: List[Tuple[int, int]], 
                            frame_height: int, frame_width: int):
        """Draw main AR ground-projected path surface"""
        
        tesla_blue = (255, 120, 0)
        for i in range(len(path_points) - 1):
            start_point = path_points[i]
            end_point = path_points[i + 1]
            
            start_width = self._calculate_perspective_width(start_point[1], frame_height)
            end_width = self._calculate_perspective_width(end_point[1], frame_height)
            segment_alpha = self._calculate_depth_alpha(
                (start_point[1] + end_point[1]) // 2, frame_height
            )
            path_polygon = self._create_path_segment_polygon(
                start_point, end_point, start_width, end_width
            )
            if path_polygon is not None:
                segment_color = tuple(int(c * segment_alpha) for c in tesla_blue)
                cv2.fillPoly(overlay, [path_polygon], segment_color)

    def _calculate_perspective_width(self, y_position: int, frame_height: int) -> float:
        """Calculate path width based on perspective"""
        normalized_y = (frame_height - y_position) / frame_height
        normalized_y = max(0, min(1, normalized_y))
        perspective_factor = normalized_y ** (1.0 + self.perspective_strength)
        
        width_range = self.base_path_width - self.min_path_width_perspective
        current_width = self.min_path_width_perspective + (width_range * perspective_factor)
        return current_width

    def _calculate_depth_alpha(self, y_position: int, frame_height: int) -> float:
        """Calculate transparency based on depth"""
        normalized_y = (frame_height - y_position) / frame_height
        normalized_y = max(0, min(1, normalized_y))
        base_alpha = 0.9
        fade_alpha = 0.3
        alpha_range = base_alpha - fade_alpha
        
        depth_factor = normalized_y ** self.depth_fade_strength
        return fade_alpha + (alpha_range * depth_factor)

    def _create_path_segment_polygon(self, start_point: Tuple[int, int], end_point: Tuple[int, int],
                                   start_width: float, end_width: float) -> np.ndarray:
        """Create trapezoid polygon for path segment"""
        start_x, start_y = start_point
        end_x, end_y = end_point
        
        dx = end_x - start_x
        dy = end_y - start_y
        length = np.sqrt(dx*dx + dy*dy)
        if length == 0:
            return None
        
        dx_norm = dx / length
        dy_norm = dy / length
        perp_x = -dy_norm
        perp_y = dx_norm
    
        start_half_width = start_width / 2
        end_half_width = end_width / 2
        start_left = (int(start_x + perp_x * start_half_width), 
                     int(start_y + perp_y * start_half_width))
        start_right = (int(start_x - perp_x * start_half_width), 
                      int(start_y - perp_y * start_half_width))
        end_left = (int(end_x + perp_x * end_half_width), 
                   int(end_y + perp_y * end_half_width))
        end_right = (int(end_x - perp_x * end_half_width), 
                    int(end_y - perp_y * end_half_width))
        
        polygon = np.array([start_left, start_right, end_right, end_left], dtype=np.int32)
        return polygon

    def _draw_path_edges(self, overlay: np.ndarray, path_points: List[Tuple[int, int]], 
                        frame_height: int, frame_width: int):
        """Draw Tesla-style path edge lines"""
        
        edge_points_left = []
        edge_points_right = []
        for i, (x, y) in enumerate(path_points):
            width = self._calculate_perspective_width(y, frame_height)
            half_width = width / 2
            
            if i < len(path_points) - 1:
                next_x, next_y = path_points[i + 1]
                dx, dy = next_x - x, next_y - y
            elif i > 0:
                prev_x, prev_y = path_points[i - 1]
                dx, dy = x - prev_x, y - prev_y
            else:
                dx, dy = 0, -1

            length = np.sqrt(dx*dx + dy*dy)
            if length > 0:
                dx, dy = dx / length, dy / length
                perp_x, perp_y = -dy, dx
                left_point = (int(x + perp_x * half_width), int(y + perp_y * half_width))
                right_point = (int(x - perp_x * half_width), int(y - perp_y * half_width))
                edge_points_left.append(left_point)
                edge_points_right.append(right_point)

        light_blue = (200, 200, 100)
        for i in range(len(edge_points_left) - 1):
            alpha = self._calculate_depth_alpha(edge_points_left[i][1], frame_height)
            thickness = max(1, int(3 * alpha))
            edge_color = tuple(int(c * alpha) for c in light_blue)
            
            cv2.line(overlay, edge_points_left[i], edge_points_left[i+1], edge_color, thickness)
            cv2.line(overlay, edge_points_right[i], edge_points_right[i+1], edge_color, thickness)

    def _draw_center_line(self, overlay: np.ndarray, path_points: List[Tuple[int, int]], 
                         frame_height: int, frame_width: int):
        """Draw Tesla-style dashed center line"""
        
        dash_length = 15
        gap_length = 10
        current_distance = 0
        in_dash = True
        white = (255, 255, 255)
        
        for i in range(len(path_points) - 1):
            start_point = path_points[i]
            end_point = path_points[i + 1]
            segment_length = np.sqrt(
                (end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2
            )
            if segment_length == 0:
                continue
            
            steps = max(1, int(segment_length))
            for step in range(steps):
                t = step / steps
                current_x = int(start_point[0] + (end_point[0] - start_point[0]) * t)
                current_y = int(start_point[1] + (end_point[1] - start_point[1]) * t)
                if in_dash:
                    alpha = self._calculate_depth_alpha(current_y, frame_height)
                    thickness = max(1, int(2 * alpha))
                    center_color = tuple(int(c * alpha) for c in white)
                    if step < steps - 1:
                        next_t = (step + 1) / steps
                        next_x = int(start_point[0] + (end_point[0] - start_point[0]) * next_t)
                        next_y = int(start_point[1] + (end_point[1] - start_point[1]) * next_t)
                        cv2.line(overlay, (current_x, current_y), (next_x, next_y), center_color, thickness)
                current_distance += 1
                pattern_length = dash_length if in_dash else gap_length
                if current_distance >= pattern_length:
                    current_distance = 0
                    in_dash = not in_dash

    def _blend_path_overlay(self, frame: np.ndarray, overlay: np.ndarray):
        """Blend path overlay with frame"""
        overlay_normalized = overlay / 255.0
        frame_normalized = frame.astype(np.float32) / 255.0
        
        overlay_gray = cv2.cvtColor(overlay_normalized, cv2.COLOR_BGR2GRAY)
        alpha_mask = np.clip(overlay_gray, 0, 1)
        alpha_mask = np.stack([alpha_mask] * 3, axis=2)
        blended = frame_normalized * (1 - alpha_mask * 0.8) + overlay_normalized * alpha_mask
        
        glow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        glow_mask = cv2.dilate((overlay_gray > 0.1).astype(np.uint8), glow_kernel)
        glow_effect = cv2.GaussianBlur(glow_mask.astype(np.float32), (9, 9), 2) * 0.1
        glow_effect = np.stack([glow_effect] * 3, axis=2)
        
        blended = blended + glow_effect * np.array([1.0, 0.8, 0.4])
        result = np.clip(blended * 255, 0, 255).astype(np.uint8)
        np.copyto(frame, result)

    def _draw_direction_indicators(self, frame: np.ndarray, path_points: List[Tuple[int, int]],
                                 frame_height: int, frame_width: int):
        """Draw direction indicators and terminus"""
        if len(path_points) < 4:
            return
        arrow_spacing = max(2, len(path_points) // 4)
        
        for i in range(0, len(path_points) - 2, arrow_spacing):
            if i + 2 < len(path_points):
                curr_point = path_points[i]
                next_point = path_points[i + 2]
                
                alpha = self._calculate_depth_alpha(curr_point[1], frame_height)
                scale = alpha
                arrow_length = int(12 * scale)
                if arrow_length < 3:
                    continue
                
                dx = next_point[0] - curr_point[0]
                dy = next_point[1] - curr_point[1]
                length = np.sqrt(dx*dx + dy*dy)
                if length > 0:
                    dx, dy = dx / length, dy / length
                    arrow_color = (int(255 * alpha), int(180 * alpha), int(50 * alpha))
                    
                    tip_x = int(curr_point[0] + dx * arrow_length)
                    tip_y = int(curr_point[1] + dy * arrow_length)
                    thickness = max(1, int(2 * alpha))
                    cv2.arrowedLine(frame, curr_point, (tip_x, tip_y), arrow_color, thickness)
        if path_points:
            x, y = path_points[-1]
            alpha = self._calculate_depth_alpha(y, frame_height)
            scale = alpha
            
            circle_radius = int(8 * scale)
            inner_radius = int(4 * scale)
            if circle_radius > 2:
                outer_color = (int(100 * alpha), int(255 * alpha), int(100 * alpha))
                inner_color = (int(150 * alpha), int(255 * alpha), int(150 * alpha))
                cv2.circle(frame, (x, y), circle_radius, outer_color, max(1, int(2 * alpha)))
                if inner_radius > 0:
                    cv2.circle(frame, (x, y), inner_radius, inner_color, -1)
    
                font_scale = 0.3 + 0.3 * alpha
                if font_scale > 0.2:
                    text_color = (int(200 * alpha), int(255 * alpha), int(200 * alpha))
                    cv2.putText(frame, "PATH", (x - int(15 * scale), y - int(15 * scale)), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

class iPhoneVideoServer:
    """TCP server to receive video frames from iPhone"""
    def __init__(self, port: int, frame_queue: queue.Queue):
        self.port = port
        self.frame_queue = frame_queue
        self.running = False
        self.server_socket = None
        self.server_thread = None
        self.client_threads = []
        self.is_healthy = True
        
    def start(self):
        """Start the video server in background thread"""
        self.running = True
        self.is_healthy = True
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()

    def check_health(self):
        """Check if the video server thread is still alive and healthy"""
        if not self.running:
            return True
        
        if self.server_thread and not self.server_thread.is_alive():
            self.is_healthy = False
            return False
        
        return True

    def restart_if_needed(self):
        """Restart the video server if it's unhealthy"""
        if not self.is_healthy and self.running:
            self.stop()
            time.sleep(1)
            self.start()
        
    def stop(self):
        """Stop the video server"""
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        for thread in self.client_threads:
            if thread.is_alive():
                thread.join(timeout=0.5)
        
    def _server_loop(self):
        """Main server loop"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.settimeout(1.0)
            self.server_socket.bind(('', self.port))
            self.server_socket.listen(5)
            while self.running:
                try:
                    client_socket, client_address = self.server_socket.accept()
                    client_thread = threading.Thread(
                        target=self._handle_video_client, 
                        args=(client_socket, client_address),
                        daemon=True
                    )
                    client_thread.start()
                    self.client_threads.append(client_thread)
                    
                    self.client_threads = [t for t in self.client_threads if t.is_alive()]   
                except Exception as e:
                    pass  
        except Exception as e:
            self.is_healthy = False
        finally:
            if self.server_socket:
                try:
                    self.server_socket.close()
                except:
                    pass
                
    def _handle_video_client(self, client_socket, client_address):
        """Handle video client connection"""
        try:
            client_socket.settimeout(5.0)
            while self.running:
                try:
                    header_data = self._recv_all(client_socket, 4)
                    if not header_data or len(header_data) != 4:
                        break
                    frame_length = struct.unpack('!I', header_data)[0]
                    
                    frame_data = self._recv_all(client_socket, frame_length)
                    if not frame_data or len(frame_data) != frame_length:
                        break
                    
                    frame = self._decode_frame_safe(frame_data)
                    if frame is not None:
                        try:
                            if self.frame_queue.full():
                                # Remove oldest frame to maintain real-time processing
                                try:
                                    self.frame_queue.get_nowait()
                                except queue.Empty:
                                    pass
                            self.frame_queue.put_nowait(frame)
                        except queue.Full:
                            pass
                    client_socket.send(b"OK")
                        
                except socket.timeout:
                    continue
                except Exception as e:
                    break
                    
        except Exception as e:
            pass
        finally:
            try:
                client_socket.close()
            except:
                pass
    
    def _recv_all(self, sock, length):
        """Receive exact amount of data"""
        data = b''
        while len(data) < length:
            chunk = sock.recv(length - len(data))
            if not chunk:
                return None
            data += chunk
        return data
    
    def _decode_frame_safe(self, frame_data: bytes) -> Optional[np.ndarray]:
        """TLS-safe frame decoding to prevent OpenCV crashes"""
        if not hasattr(self.__class__, '_decode_lock'):
            self.__class__._decode_lock = threading.RLock()
        try:
            frame_np = np.frombuffer(frame_data, dtype=np.uint8)
            with threading.Lock():
                frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                return self._decode_frame_fallback(frame_data)
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                return self._decode_frame_fallback(frame_data)
            if frame.shape[0] < 100 or frame.shape[1] < 100:
                return self._decode_frame_fallback(frame_data)
            return frame
        except Exception as e:
            return None
        finally:
            try:
                del frame_np
            except:
                pass
    
    def _decode_frame_fallback(self, frame_data: bytes) -> Optional[np.ndarray]:
        """Enhanced fallback decoding that completely avoids OpenCV TLS issues"""
        temp_path = None
        try:
            if not frame_data or len(frame_data) < 100:
                return None
            try:
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    temp_path = temp_file.name
                    temp_file.write(frame_data)
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
            except Exception as temp_error:
                return None
            if not os.path.exists(temp_path):
                return None
            file_size = os.path.getsize(temp_path)
            if file_size != len(frame_data):
                return None
            try:
                with Image.open(temp_path) as pil_image:
                    pil_image.verify()
            except Exception as verify_error:
                return None
            
            try:
                with Image.open(temp_path) as pil_image:
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    frame_array = np.array(pil_image)
                    if frame_array is None or frame_array.size == 0:
                        return None

                    try:
                        bgr_frame = frame_array.copy()
                        bgr_frame[:, :, 0] = frame_array[:, :, 2]
                        bgr_frame[:, :, 2] = frame_array[:, :, 0]
                        frame = bgr_frame
                    except Exception as conv_error:
                        frame = frame_array
                    
                    if frame is not None and frame.size > 0 and len(frame.shape) == 3:
                        return frame
                    else:
                        return None    
            except Exception as pil_convert_error:
                return None
        except Exception as fallback_error:
            return None
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as cleanup_error:
                    pass

class iPhoneCameraStreamer:
    """Handle video frames from iPhone app instead of built-in camera"""
    def __init__(self, port=6001, max_queue_size=3):
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.video_server = iPhoneVideoServer(port, self.frame_queue)
        self.running = False
    
    def start(self):
        self.video_server.start()
        self.running = True
        return True
    
    def get_frame(self):
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def stop(self):
        self.running = False
        self.video_server.stop()

class RouteService:
    """Google Maps route planning and management"""
    def __init__(self, api_key: str):
        self.gmaps = googlemaps.Client(key=api_key)
        self.current_route = None
        self.waypoints = []
    
    def find_destination(self, query: str, current_location: Tuple[float, float]) -> Optional[Dict]:
        """Find destination using Google Places API"""
        try:
            places_result = self.gmaps.places_nearby(
                location=current_location,
                keyword=query,
                radius=5000,
                open_now=True
            )
            
            if places_result.get("results"):
                nearest = places_result["results"][0]
                destination = {
                    'name': nearest["name"],
                    'address': nearest.get("vicinity", ""),
                    'location': (
                        nearest["geometry"]["location"]["lat"],
                        nearest["geometry"]["location"]["lng"]
                    ),
                    'place_id': nearest.get("place_id", "")
                }
                return destination
            
        except Exception as e:
            pass
        
        return None
    
    def plan_route(self, start: Tuple[float, float], end: Tuple[float, float]) -> List[NavigationWaypoint]:
        """Plan walking route using Google Directions API"""
        try:
            directions_result = self.gmaps.directions(
                origin=start,
                destination=end,
                mode="walking",
                departure_time=datetime.now()
            )
            
            if not directions_result:
                return []
            route = directions_result[0]
            legs = route["legs"]
            waypoints = []
            for leg in legs:
                steps = leg["steps"]
                for i, step in enumerate(steps):
                    start_loc = step["start_location"]
                    end_loc = step["end_location"]
                    instruction = re.sub(r'<[^>]+>', '', step["html_instructions"])
                    heading = self._calculate_heading(
                        start_loc["lat"], start_loc["lng"],
                        end_loc["lat"], end_loc["lng"]
                    )
                    
                    turn_direction = self._determine_turn_direction(instruction, heading)
                    street_name = self._extract_street_name(instruction)
                    polyline_points = decode_polyline(step["polyline"]["points"])
                    num_points = len(polyline_points)
                    for j, (lat, lon) in enumerate(polyline_points):
                        if j < num_points - 1:
                            next_lat, next_lon = polyline_points[j + 1]
                            distance_to_next = self._calculate_distance(lat, lon, next_lat, next_lon)
                            heading = self._calculate_heading(lat, lon, next_lat, next_lon)
                        else:
                            distance_to_next = 0.0
                            heading = self._calculate_heading(lat, lon, end_loc["lat"], end_loc["lng"])
                        
                        if j == num_points - 1:
                            turn_dir = turn_direction
                        else:
                            turn_dir = TurnDirection.STRAIGHT
                        waypoint = NavigationWaypoint(
                            lat=lat,
                            lon=lon,
                            instruction=instruction,
                            distance_to_next=distance_to_next,
                            heading=heading,
                            turn_direction=turn_dir,
                            street_name=street_name
                        )
                        waypoints.append(waypoint)
            self.waypoints = waypoints
            return waypoints
        except Exception as e:
            return []
    
    def _calculate_heading(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate heading between two points"""
        delta_lon = math.radians(lon2 - lon1)
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        x = math.sin(delta_lon) * math.cos(lat2_rad)
        y = math.cos(lat1_rad) * math.sin(lat2_rad) - (
            math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)
        )
        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two lat/lon points in meters"""
        R = 6371000
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = (math.sin(delta_phi/2)**2 +
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
    
    def _determine_turn_direction(self, instruction: str, heading: float) -> TurnDirection:
        """Determine turn direction from instruction text"""
        instruction_lower = instruction.lower()
        if "u-turn" in instruction_lower or "u turn" in instruction_lower:
            return TurnDirection.U_TURN
        elif "sharp left" in instruction_lower:
            return TurnDirection.SHARP_LEFT
        elif "slight left" in instruction_lower or "bear left" in instruction_lower:
            return TurnDirection.SLIGHT_LEFT
        elif "left" in instruction_lower or "turn left" in instruction_lower:
            return TurnDirection.LEFT
        elif "sharp right" in instruction_lower:
            return TurnDirection.SHARP_RIGHT
        elif "slight right" in instruction_lower or "bear right" in instruction_lower:
            return TurnDirection.SLIGHT_RIGHT
        elif "right" in instruction_lower or "turn right" in instruction_lower:
            return TurnDirection.RIGHT
        else:
            return TurnDirection.STRAIGHT
    
    def _extract_street_name(self, instruction: str) -> str:
        """Extract street name from instruction"""
        patterns = [
            r'onto ([^,\n]+)',
            r'on ([^,\n]+)',
            r'toward ([^,\n]+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, instruction, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""

class NavigationSystem:
    """Main navigation system coordinator with Tesla FSD pathfinding"""
    def __init__(self, config: SystemConfig, api_key: str):
        self.config = config
        self.location_data = LocationData()
        self.location_server = LocationServer(port=6000, location_data=self.location_data)
        self.location_server.start()
        self.location_service = LocationService(self.location_data)
        self.tts_service = TTSService(config)
        self.ground_model = GroundDetectionModel(config)
        self.route_service = RouteService(api_key)
        
        self.path_visualizer = PathVisualizer(config)
        self.audio_guide = SpatialAudioGuide(config)
        self.current_path_points = []
        
        self.nav_state = NavigationState_Data()
        
        self.camera = iPhoneCameraStreamer(port=6001, max_queue_size=3)
        self.camera.start()
        self.last_frame_time = time.time()
        self.fps = 0.0
        
        self.last_health_check = time.time()
        self.health_check_interval = 5.0
    
    def check_system_health(self):
        """Check health of all background services and restart if needed"""
        current_time = time.time()
        if current_time - self.last_health_check < self.health_check_interval:
            return
        
        self.last_health_check = current_time
        
        try:
            if not self.location_server.check_health():
                self.location_server.restart_if_needed()
            if hasattr(self.camera, 'video_server'):
                if not self.camera.video_server.check_health():
                    self.camera.video_server.restart_if_needed()
            if not self.tts_service.check_health():
                self.tts_service.restart_if_needed()
            if hasattr(self, 'audio_guide') and not self.audio_guide.check_health():
                self.audio_guide.restart_if_needed() 
        except Exception as e:
            pass
    
    def start_navigation(self, destination_query: str):
        """Start navigation to destination"""
        self.tts_service.speak("Starting navigation system", priority=5)
        current_location = self.location_service.get_current_location()
        print(current_location)
        current_heading = self.location_service.get_current_heading()
        
        self.nav_state.current_position = current_location
        self.nav_state.current_heading = current_heading
        
        destination = self.route_service.find_destination(destination_query, current_location)
        if not destination:
            self.tts_service.speak("Could not find destination. Please try again.", priority=7)
            return False
        self.tts_service.speak(f"Destination found: {destination['name']}. Planning route.", priority=5)
        
        waypoints = self.route_service.plan_route(current_location, destination['location'])
        if not waypoints:
            self.tts_service.speak("Could not plan route. Please try again.", priority=7)
            return False
        
        self.nav_state.waypoints = waypoints
        self.nav_state.current_waypoint_index = 0
        self.nav_state.target_waypoint = waypoints[0] if waypoints else None
        self.nav_state.state = NavigationState.GPS_FOLLOWING

        print("\nWaypoints:")
        for idx, wp in enumerate(waypoints):
            print(f"  {idx+1}: lat={wp.lat:.6f}, lon={wp.lon:.6f}, heading={wp.heading:.1f}°, "
                f"distance={wp.distance_to_next:.1f}m, turn={wp.turn_direction.value}, "
                f"instruction='{wp.instruction}'")
        
        total_distance = sum(wp.distance_to_next for wp in waypoints)
        self.nav_state.distance_to_destination = total_distance
        self.tts_service.speak(
            f"Route planned. Total distance: {int(total_distance)} meters. Starting navigation.",
            priority=5
        )
        
        if self.nav_state.target_waypoint:
            time.sleep(1)
            target_bearing = self.location_service.calculate_bearing(
                current_location[0], current_location[1],
                self.nav_state.target_waypoint.lat, self.nav_state.target_waypoint.lon
            )
            cardinal_direction = self.location_service.get_cardinal_direction(target_bearing)
            self.tts_service.speak(f"Initial direction: face {cardinal_direction}.", priority=6)
            time.sleep(2)
            current_heading = self.location_service.get_current_heading()
            self.tts_service.speak_heading_check(current_heading, target_bearing)
        return True
    
    def update_navigation(self):
        """Main navigation update loop with enhanced pathfinding"""
        self.check_system_health()
        if self.nav_state.state == NavigationState.INITIALIZING:
            return
        
        current_location = self.location_service.get_current_location()
        current_heading = self.location_service.get_current_heading()
        
        self.nav_state.current_position = current_location
        self.nav_state.current_heading = current_heading
        self.nav_state.last_gps_update = time.time()
        
        visual_analysis = None
        frame = self.camera.get_frame()
        if frame is not None:
            visual_analysis = self.ground_model.analyze_frame(frame)

            if visual_analysis and self.nav_state.target_waypoint:
                current_lat, current_lon = self.nav_state.current_position
                target_bearing = self.location_service.calculate_bearing(
                    current_lat, current_lon,
                    self.nav_state.target_waypoint.lat, 
                    self.nav_state.target_waypoint.lon
                )
                
                self.current_path_points = self.path_visualizer.generate_path(
                    visual_analysis['ground_mask'],
                    visual_analysis['obstacle_mask'],
                    current_heading=self.nav_state.current_heading,
                    target_bearing=target_bearing
                )
                
                self.audio_guide.update_audio_guidance(
                    self.current_path_points,
                    self.config.input_width,
                    self.config.input_height
                )
        if self.nav_state.state == NavigationState.GPS_FOLLOWING:
            self._handle_gps_following(visual_analysis, current_heading)
        elif self.nav_state.state == NavigationState.OBSTACLE_AVOIDANCE:
            self._handle_obstacle_avoidance(visual_analysis, current_heading)
        elif self.nav_state.state == NavigationState.GROUND_FOLLOWING:
            self._handle_ground_following(visual_analysis, current_heading)
        elif self.nav_state.state == NavigationState.PATH_BLOCKED:
            self._handle_path_blocked(visual_analysis, current_heading)
        
        self._check_waypoint_completion()
        current_time = time.time()
        self.fps = 1.0 / (current_time - self.last_frame_time)
        self.last_frame_time = current_time
        
        return frame, visual_analysis
    
    def _handle_gps_following(self, visual_analysis: Optional[Dict], current_heading: float):
        """Handle GPS-based navigation with compass guidance"""
        if not self.nav_state.target_waypoint:
            return
        current_lat, current_lon = self.nav_state.current_position
        target_lat, target_lon = self.nav_state.target_waypoint.lat, self.nav_state.target_waypoint.lon
        distance_to_target = self.location_service.calculate_distance(
            current_lat, current_lon, target_lat, target_lon
        )
        bearing_to_target = self.location_service.calculate_bearing(
            current_lat, current_lon, target_lat, target_lon
        )
        
        turn_angle = self.location_service.calculate_turn_angle(current_heading, bearing_to_target)
        # Check if user is facing the wrong direction
        if abs(turn_angle) > 45:
            if not hasattr(self, '_last_heading_check') or time.time() - self._last_heading_check > 3:
                self.tts_service.speak_heading_check(current_heading, bearing_to_target)
                self._last_heading_check = time.time()

        if visual_analysis:
            if not visual_analysis['safe_to_proceed']:
                if visual_analysis['path_has_obstacles']:
                    self.nav_state.state = NavigationState.OBSTACLE_AVOIDANCE
                    self.tts_service.speak_warning("Obstacle detected. Switching to visual navigation.")
                    return
                elif visual_analysis['path_ground_coverage'] < self.config.min_ground_coverage:
                    self.nav_state.state = NavigationState.PATH_BLOCKED
                    self.nav_state.path_blocked_start_time = time.time()
                    self.tts_service.speak_warning("Path unclear. Please wait.")
                    return

        for announce_distance in self.config.distance_announcements:
            if (distance_to_target <= announce_distance and 
                distance_to_target > announce_distance - 5):
                
                instruction = self.nav_state.target_waypoint.instruction
                self.tts_service.speak_direction(
                    instruction, 
                    distance_to_target, 
                    current_heading, 
                    bearing_to_target
                )
                break
    
    def _handle_obstacle_avoidance(self, visual_analysis: Optional[Dict], current_heading: float):
        """Handle obstacle avoidance mode"""
        if not visual_analysis:
            self.nav_state.state = NavigationState.GPS_FOLLOWING
            return
        if visual_analysis['safe_to_proceed']:
            self.nav_state.state = NavigationState.GPS_FOLLOWING
            self.tts_service.speak_status("Path clear. Resuming GPS navigation.")
            if self.nav_state.target_waypoint:
                current_lat, current_lon = self.nav_state.current_position
                target_lat, target_lon = self.nav_state.target_waypoint.lat, self.nav_state.target_waypoint.lon
                bearing_to_target = self.location_service.calculate_bearing(
                    current_lat, current_lon, target_lat, target_lon
                )
                self.tts_service.speak_heading_check(current_heading, bearing_to_target)
        else:
            if visual_analysis['person_coverage'] > 0.1:
                self.tts_service.speak_warning("People detected. Please wait or find alternate path.")
            elif visual_analysis['path_has_obstacles']:
                self.tts_service.speak_warning("Obstacles in path. Looking for safe route.")
    
    def _handle_ground_following(self, visual_analysis: Optional[Dict], current_heading: float):
        """Handle ground-following navigation mode"""
        if not visual_analysis:
            self.nav_state.state = NavigationState.GPS_FOLLOWING
            return
        if visual_analysis['ground_coverage'] < self.config.min_ground_coverage:
            self.nav_state.state = NavigationState.PATH_BLOCKED
            self.nav_state.path_blocked_start_time = time.time()
            self.tts_service.speak_warning("Ground surface unclear.")
        elif visual_analysis['safe_to_proceed']:
            self.nav_state.state = NavigationState.GPS_FOLLOWING
            if self.nav_state.target_waypoint:
                current_lat, current_lon = self.nav_state.current_position
                target_lat, target_lon = self.nav_state.target_waypoint.lat, self.nav_state.target_waypoint.lon
                bearing_to_target = self.location_service.calculate_bearing(
                    current_lat, current_lon, target_lat, target_lon
                )
                self.tts_service.speak_heading_check(current_heading, bearing_to_target)
    
    def _handle_path_blocked(self, visual_analysis: Optional[Dict], current_heading: float):
        """Handle blocked path situation"""
        current_time = time.time()
        if self.nav_state.path_blocked_start_time is None:
            self.nav_state.path_blocked_start_time = current_time
        blocked_duration = current_time - self.nav_state.path_blocked_start_time
        
        if visual_analysis and visual_analysis['safe_to_proceed']:
            self.nav_state.state = NavigationState.GPS_FOLLOWING
            self.nav_state.path_blocked_start_time = None
            self.tts_service.speak_status("Path is clear. Continuing navigation.")
            if self.nav_state.target_waypoint:
                current_lat, current_lon = self.nav_state.current_position
                target_lat, target_lon = self.nav_state.target_waypoint.lat, self.nav_state.target_waypoint.lon
                bearing_to_target = self.location_service.calculate_bearing(
                    current_lat, current_lon, target_lat, target_lon
                )
                self.tts_service.speak_heading_check(current_heading, bearing_to_target)
        elif blocked_duration > self.config.max_path_blocked_time:
            self.nav_state.state = NavigationState.EMERGENCY_STOP
            self.nav_state.emergency_stop_start_time = current_time
            self.tts_service.speak("Path has been blocked for a while. Please ask for assistance or find an alternate route.", priority=9)
    
    def _check_waypoint_completion(self):
        """Check if current waypoint has been reached"""
        if not self.nav_state.target_waypoint:
            return
        current_lat, current_lon = self.nav_state.current_position
        target_lat, target_lon = self.nav_state.target_waypoint.lat, self.nav_state.target_waypoint.lon
        distance = self.location_service.calculate_distance(
            current_lat, current_lon, target_lat, target_lon
        )
        
        if distance <= self.config.waypoint_proximity_threshold:
            self.nav_state.target_waypoint.reached = True
            self.nav_state.current_waypoint_index += 1
            if self.nav_state.current_waypoint_index >= len(self.nav_state.waypoints):
                self.nav_state.state = NavigationState.DESTINATION_REACHED
                self.tts_service.speak("Destination reached! Navigation complete.", priority=9)
            else:
                self.nav_state.target_waypoint = self.nav_state.waypoints[self.nav_state.current_waypoint_index]
                remaining = len(self.nav_state.waypoints) - self.nav_state.current_waypoint_index
                self.tts_service.speak(f"Waypoint reached. {remaining} turns remaining.", priority=4)
    
    def create_visualization(self, frame: np.ndarray, visual_analysis: Dict) -> np.ndarray:
        """Create comprehensive visualization with Tesla FSD pathfinding"""
        if frame is None:
            return np.zeros((self.config.input_height, self.config.input_width, 3), dtype=np.uint8)
        
        result = frame.copy()
        if 'ground_mask' in visual_analysis:
            ground_overlay = np.zeros_like(result)
            ground_overlay[visual_analysis['ground_mask']] = (0, 255, 0)
            result = cv2.addWeighted(result, 0.7, ground_overlay, 0.3, 0)
        if 'obstacle_mask' in visual_analysis:
            obstacle_overlay = np.zeros_like(result)
            obstacle_overlay[visual_analysis['obstacle_mask']] = (0, 0, 255)
            result = cv2.addWeighted(result, 0.8, obstacle_overlay, 0.2, 0)

        if hasattr(self, 'current_path_points') and self.current_path_points and self.config.show_path:
            result = self.path_visualizer.draw_path(result, self.current_path_points)
        if ('roboflow_predictions' in visual_analysis and 
            visual_analysis['roboflow_predictions'] is not None):
            try:
                result = self._draw_roboflow_detections(result, visual_analysis['roboflow_predictions'])
            except Exception as e:
                pass
        self._add_info_overlay(result, visual_analysis)
        return result
    
    def _draw_roboflow_detections(self, frame: np.ndarray, predictions: Dict) -> np.ndarray:
        """Draw Roboflow chair/table detections on frame"""
        try:
            if not predictions or 'predictions' not in predictions:
                return frame

            colors = {
                'chair': (0, 200, 200),
                'table': (200, 0, 200),
            }
            h, w = frame.shape[:2]
            for detection in predictions['predictions']:
                try:
                    confidence = detection.get('confidence', 0)
                    class_name = detection.get('class', '').lower()
                    if confidence * 100 >= self.config.roboflow_confidence_threshold:
                        color = colors.get(class_name, (128, 128, 128))
                        x = int(detection.get('x', 0) - detection.get('width', 0) / 2)
                        y = int(detection.get('y', 0) - detection.get('height', 0) / 2)
                        x2 = int(detection.get('x', 0) + detection.get('width', 0) / 2)
                        y2 = int(detection.get('y', 0) + detection.get('height', 0) / 2)
                        
                        x = max(0, min(x, w - 1))
                        y = max(0, min(y, h - 1))
                        x2 = max(0, min(x2, w - 1))
                        y2 = max(0, min(y2, h - 1))
                        cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
                        if 'points' in detection and len(detection['points']) > 2:
                            points = detection['points']
                            pts = []
                            for point in points:
                                px = int(max(0, min(point.get('x', 0), w-1)))
                                py = int(max(0, min(point.get('y', 0), h-1)))
                                pts.append([px, py])
                            if len(pts) >= 3:
                                pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                                cv2.polylines(frame, [pts], True, color, 2)
                        label = f"{class_name}: {confidence:.2f}"
                        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        label_x = x
                        label_y = y + text_height + 10
                        if label_x + text_width > w:
                            label_x = w - text_width - 5
                        
                        cv2.rectangle(
                            frame, 
                            (label_x, label_y - text_height - 10), 
                            (label_x + text_width, label_y), 
                            color, 
                            -1
                        )
                        cv2.putText(
                            frame, 
                            label, 
                            (label_x, label_y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, 
                            (255, 255, 255), 
                            2
                        )
                except Exception as e:
                    continue
        except Exception as e:
            pass
        return frame
    
    def _add_info_overlay(self, frame: np.ndarray, visual_analysis: Dict):
        """Add information overlay to frame"""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 255)
        
        state_text = f"State: {self.nav_state.state.value.upper()}"
        cv2.putText(frame, state_text, (10, 30), font, 0.7, color, 2)

        if self.nav_state.target_waypoint:
            waypoint_text = f"Waypoint: {self.nav_state.current_waypoint_index + 1}/{len(self.nav_state.waypoints)}"
            cv2.putText(frame, waypoint_text, (10, 60), font, 0.6, color, 2)

        lat, lon, heading, last_update, is_valid = self.location_data.get_data()
        if is_valid and (time.time() - last_update) < 5.0:
            heading_text = f"Heading: {heading:.1f}°"
            cv2.putText(frame, heading_text, (10, 90), font, 0.6, color, 2)
            gps_text = f"Lat: {lat:.6f}  Lon: {lon:.6f}"
            cv2.putText(frame, gps_text, (10, h - 20), font, 0.5, (255,255,0), 2)
        else:
            cv2.putText(frame, "No GPS/Heading", (10, h - 20), font, 0.5, (0,0,255), 2)

        if self.nav_state.target_waypoint:
            current_lat, current_lon = self.nav_state.current_position
            target_bearing = self.location_service.calculate_bearing(
                current_lat, current_lon,
                self.nav_state.target_waypoint.lat, self.nav_state.target_waypoint.lon
            )
            bearing_text = f"Target: {target_bearing:.0f}°"
            cv2.putText(frame, bearing_text, (10, 120), font, 0.6, color, 2)

            turn_angle = self.location_service.calculate_turn_angle(
                self.nav_state.current_heading, target_bearing
            )
            turn_text = f"Turn: {turn_angle:+.0f}°"
            turn_color = (0, 255, 0) if abs(turn_angle) <= 15 else (0, 165, 255)
            cv2.putText(frame, turn_text, (10, 150), font, 0.6, turn_color, 2)
        
        if visual_analysis:
            ground_pct = visual_analysis.get('ground_coverage', 0) * 100
            ground_text = f"Ground: {ground_pct:.1f}%"
            cv2.putText(frame, ground_text, (10, 180), font, 0.6, color, 2)
            safe_text = f"Safe: {'YES' if visual_analysis.get('safe_to_proceed', False) else 'NO'}"
            safe_color = (0, 255, 0) if visual_analysis.get('safe_to_proceed', False) else (0, 0, 255)
            cv2.putText(frame, safe_text, (10, 210), font, 0.6, safe_color, 2)
        
        audio_text = f"Audio: {'ON' if self.config.enable_audio else 'OFF'}"
        cv2.putText(frame, audio_text, (10, 240), font, 0.6, color, 2)
        
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (w - 120, 30), font, 0.6, color, 2)
        
        status_color = (0, 255, 0) if self.nav_state.state == NavigationState.GPS_FOLLOWING else (0, 165, 255)
        cv2.circle(frame, (w - 30, 30), 10, status_color, -1)
        
        center_x, center_y = w - 30, 80
        arrow_length = 20
        north_x = int(center_x + arrow_length * math.sin(math.radians(self.nav_state.current_heading)))
        north_y = int(center_y - arrow_length * math.cos(math.radians(self.nav_state.current_heading)))
        cv2.arrowedLine(frame, (center_x, center_y), (north_x, north_y), (255, 255, 0), 2, tipLength=0.3)
        cv2.putText(frame, "N", (center_x - 5, center_y + 35), font, 0.4, (255, 255, 0), 1)
    
    def stop_navigation(self):
        """Stop navigation and cleanup"""
        self.nav_state.state = NavigationState.INITIALIZING
        self.tts_service.speak("Navigation stopped.", priority=5)
        if self.camera:
            self.camera.stop()
        if self.location_server:
            self.location_server.stop()
        if hasattr(self, 'audio_guide'):
            self.audio_guide.close()
        cv2.destroyAllWindows()

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="Intelligent Pedestrian Navigation System with Tesla FSD Pathfinding")
    parser.add_argument("--destination", type=str, help="Destination to navigate to")
    parser.add_argument("--api-key", type=str, default=API_KEY, help="Google Maps API key")
    parser.add_argument("--model", choices=["segformer", "deeplabv3"], default="segformer", help="Vision model")
    parser.add_argument("--camera", type=int, default=1, help="Camera index")
    parser.add_argument("--no-tts", action="store_true", help="Disable text-to-speech")
    parser.add_argument("--roboflow-confidence", type=float, default=50.0, help="Roboflow confidence threshold (0-100)")
    parser.add_argument("--disable-roboflow", action="store_true", help="Disable Roboflow chair/table detection")
    parser.add_argument("--no-audio", action="store_true", help="Disable spatial audio")
    parser.add_argument("--audio-volume", type=float, default=0.7, help="Audio volume (0.0-1.0)")
    args = parser.parse_args()
    
    config = SystemConfig(
        model_type=args.model,
        camera_index=args.camera,
        tts_enabled=not args.no_tts,
        roboflow_confidence_threshold=args.roboflow_confidence,
        enable_audio=not args.no_audio,
        audio_volume=args.audio_volume
    )
    
    nav_system = NavigationSystem(config, args.api_key)
    
    if args.disable_roboflow:
        nav_system.ground_model.roboflow_enabled = False
            

    time.sleep(3)
    
    print("\nTesla FSD Pedestrian Navigation System")
    print("=" * 60)
    print("Commands:")
    print("  Enter destination to start navigation")
    print("  'q' to quit")
    print("  's' to save current frame")
    print("  'h' for heading check (compass direction)")
    print("  'r' to toggle Roboflow chair/table detection")
    print("  'a' to test spatial audio")
    print("  'v' to toggle spatial audio on/off")
    print("  'p' to toggle path visualization")
    print("  '+/-' to adjust audio volume")
    print("=" * 60)
    
    try:
        while True:
            try:
                if nav_system.nav_state.state == NavigationState.INITIALIZING:
                    if args.destination:
                        destination = args.destination
                        args.destination = None
                    else:
                        destination = input("\nEnter destination (or 'q' to quit): ").strip()
                    if destination.lower() == 'q':
                        break
                    if destination:
                        success = nav_system.start_navigation(destination)
                        if not success:
                            continue

                frame, visual_analysis = nav_system.update_navigation()
                if frame is not None and visual_analysis is not None:
                    display_frame = nav_system.create_visualization(frame, visual_analysis)
                    cv2.imshow('Tesla FSD Pedestrian Navigation', display_frame)
                else:
                    continue
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and frame is not None:
                    filename = f"navigation_frame_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Frame saved as {filename}")
                elif key == ord('h'):
                    if nav_system.nav_state.target_waypoint:
                        current_heading = nav_system.location_service.get_current_heading()
                        current_lat, current_lon = nav_system.nav_state.current_position
                        target_bearing = nav_system.location_service.calculate_bearing(
                            current_lat, current_lon,
                            nav_system.nav_state.target_waypoint.lat, 
                            nav_system.nav_state.target_waypoint.lon
                        )
                        nav_system.tts_service.speak_heading_check(current_heading, target_bearing)
                elif key == ord('r'):
                    nav_system.ground_model.roboflow_enabled = not nav_system.ground_model.roboflow_enabled
                    status = "ENABLED" if nav_system.ground_model.roboflow_enabled else "DISABLED"
                    print(f"Roboflow chair/table detection: {status}")
                    nav_system.tts_service.speak(f"Chair and table detection {status.lower()}")
                elif key == ord('a'):
                    print("Testing spatial audio system...")
                    if hasattr(nav_system, 'audio_guide'):
                        nav_system.audio_guide.test_audio()
                    else:
                        print("Audio system not available")
                elif key == ord('v'):
                    if hasattr(nav_system, 'audio_guide'):
                        nav_system.config.enable_audio = not nav_system.config.enable_audio
                        status = "enabled" if nav_system.config.enable_audio else "disabled"
                        print(f"Spatial audio {status}")
                        nav_system.tts_service.speak(f"Spatial audio {status}")
                elif key == ord('p'):
                    nav_system.config.show_path = not nav_system.config.show_path
                    status = "enabled" if nav_system.config.show_path else "disabled"
                    print(f"Tesla FSD path visualization {status}")
                elif key == ord('+') or key == ord('='):
                    if hasattr(nav_system, 'audio_guide'):
                        nav_system.config.audio_volume = min(1.0, nav_system.config.audio_volume + 0.1)
                        print(f"Audio volume: {nav_system.config.audio_volume:.1f}")
                elif key == ord('-'):
                    if hasattr(nav_system, 'audio_guide'):
                        nav_system.config.audio_volume = max(0.0, nav_system.config.audio_volume - 0.1)
                        print(f"Audio volume: {nav_system.config.audio_volume:.1f}")
                
                if nav_system.nav_state.state == NavigationState.DESTINATION_REACHED:
                    input("\nDestination reached! Press Enter to start new navigation...")
                    nav_system.nav_state.state = NavigationState.INITIALIZING
                    
            except Exception as e:
                print(f"An error occurred: {e}")
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        nav_system.stop_navigation()
        print("Tesla FSD Navigation system stopped. Goodbye!")

if __name__ == "__main__":
    main()
