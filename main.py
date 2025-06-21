"""
행동 패턴 분석 시스템 메인 모듈

모든 모듈을 통합하여 전체 파이프라인을 실행합니다.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
import cv2
import threading
import signal  # 시그널 모듈 추가
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# 모듈 임포트
from modules.video_input.video_loader import create_video_loader
from modules.object_detection.detector import create_detector
from modules.pose_estimation.pose_estimator import create_pose_estimator
from modules.action_recognition.action_recognizer import create_action_recognizer
from modules.visualization.visualizer import Visualizer
from utils.s3_utils import download_model
from modules.alert.alert import AlertManager
from config import (
    SYSTEM_CONFIG, ALERT_CONFIG, VIDEO_CONFIG, VISUALIZATION_CONFIG,
    DETECTION_CONFIG, POSE_CONFIG, ACTION_CONFIG, DANGER_ZONE_CONFIG,
    toggle_gpu_mode, USE_GPU
)

# 로깅 설정
logging.basicConfig(
    level=getattr(logging, SYSTEM_CONFIG['log_level']),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(SYSTEM_CONFIG['log_file']),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class BehaviorAnalysisSystem:
    """행동 패턴 분석 시스템 통합 클래스"""
    
    def __init__(self):
        """시스템 초기화"""
        self.video_loader = None
        self.detector = None
        self.pose_estimator = None
        self.action_recognizer = None
        self.visualizer = None
        self.alert_manager = None
        
        # 실행 상태 플래그 추가
        self.running = True
        
        # 성능 최적화 설정
        self.frame_resize_factor = SYSTEM_CONFIG.get('frame_resize_factor', 1.0)
        
        # 모든 프레임 분석을 위해 skip_frames 강제 설정
        VIDEO_CONFIG['skip_frames'] = 1
        
        # 하드웨어 가속 초기화
        self._initialize_hardware_acceleration()
        
    def _initialize_hardware_acceleration(self):
        """하드웨어 가속 및 성능 최적화 초기화"""
        try:
            # CUDA 사용 가능 여부 확인
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                # CUDA 정보 출력
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "없음"
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9 if gpu_count > 0 else 0
                
                logger.info(f"CUDA 가속 초기화: 사용 가능한 GPU {gpu_count}개, {gpu_name}, {gpu_mem:.2f}GB")
                
                # cuDNN 설정
                if SYSTEM_CONFIG.get('cudnn_benchmark', False):
                    cudnn.benchmark = True
                    logger.info("cuDNN 벤치마크 모드 활성화 (반복 작업 최적화)")
                
                if SYSTEM_CONFIG.get('cuda_stream_per_device', False):
                    logger.info("CUDA 스트림 최적화 활성화")
                
                # 메모리 관리 전략 설정
                memory_management = SYSTEM_CONFIG.get('memory_management', 'balanced')
                if memory_management == 'aggressive':
                    # 메모리 적극적 확보 (성능 향상, 다른 프로그램과 충돌 가능성)
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, 'memory_stats'):
                        logger.debug(f"CUDA 메모리 통계: {torch.cuda.memory_stats()}")
                elif memory_management == 'conservative':
                    # 보수적 메모리 사용 (안정성 우선)
                    gpu_fraction = SYSTEM_CONFIG.get('gpu_memory_fraction', 0.7)
                    if gpu_fraction < 1.0:
                        try:
                            # GPU 메모리 제한
                            import GPUtil
                            device_id = 0  # 첫 번째 GPU
                            gpu = GPUtil.getGPUs()[device_id]
                            total_mem = gpu.memoryTotal
                            max_mem = int(total_mem * gpu_fraction)
                            logger.info(f"GPU 메모리 제한: {max_mem}MB ({gpu_fraction*100:.0f}% 사용)")
                        except ImportError:
                            logger.warning("GPUtil 패키지가 설치되지 않아 GPU 메모리 제한을 적용할 수 없습니다.")
                        except Exception as e:
                            logger.warning(f"GPU 메모리 제한 적용 실패: {e}")
                
                # PyTorch 최적화 설정
                if SYSTEM_CONFIG.get('use_half_precision', True):
                    torch.set_float32_matmul_precision('medium')  # 또는 'high', 'highest'
                    logger.info("PyTorch FP32 행렬 곱셈 정밀도 설정: medium")
                    
                # TensorRT 지원 확인
                try:
                    import tensorrt
                    logger.info(f"TensorRT 지원 확인: 버전 {tensorrt.__version__}")
                except ImportError:
                    logger.warning("TensorRT 패키지가 설치되지 않았습니다. TensorRT 최적화를 사용할 수 없습니다.")
                    
                # OpenCV CUDA 지원 확인
                try:
                    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                        logger.info(f"OpenCV CUDA 백엔드 사용 가능: 장치 {cv2.cuda.getCudaEnabledDeviceCount()}개")
                    else:
                        logger.warning("OpenCV CUDA 백엔드를 사용할 수 없습니다.")
                except Exception as e:
                    logger.warning(f"OpenCV CUDA 지원 확인 실패: {e}")
                
            else:
                logger.warning("CUDA를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
        except Exception as e:
            logger.error(f"하드웨어 가속 초기화 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
    def initialize(self):
        """
        모든 모듈 초기화
        
        Returns:
            bool: 초기화 성공 여부
        """
        try:
            logger.info("행동 분석 시스템 초기화 중...")
            
            # GPU 사용 설정 확인
            if torch.cuda.is_available() and USE_GPU:
                logger.info(f"CUDA 가능: {torch.cuda.is_available()}, 장치: {torch.cuda.get_device_name(0)}")
                # CUDA 메모리 관리 설정
                torch.cuda.set_per_process_memory_fraction(SYSTEM_CONFIG['gpu_memory_fraction'])
                # cuDNN 벤치마크 모드 설정
                torch.backends.cudnn.benchmark = SYSTEM_CONFIG['cudnn_benchmark']
                logger.info("GPU 가속 활성화됨")
            else:
                logger.info("CPU 모드로 실행 중")
            
            # 필요한 디렉토리 생성
            os.makedirs('models', exist_ok=True)
            os.makedirs('data', exist_ok=True)
            os.makedirs('results', exist_ok=True)
            os.makedirs('logs', exist_ok=True)
            
            # 알림 관리자 초기화
            self.alert_manager = AlertManager()
            if not self.alert_manager.initialize():
                logger.error("알림 관리자 초기화 실패")
                return False
            
            # 모델 다운로드
            self._download_models()
            
            # 비디오 로더 생성
            self.video_loader = create_video_loader()
            if not self.video_loader:
                logger.error("비디오 로더 생성 실패")
                return False
            
            # 객체 검출기 생성
            self.detector = create_detector()
            if not self.detector:
                logger.error("객체 검출기 생성 실패")
                return False
            
            # 자세 추정기 생성
            self.pose_estimator = create_pose_estimator()
            if not self.pose_estimator:
                logger.error("자세 추정기 생성 실패")
                return False
            
            # 행동 인식기 생성
            self.action_recognizer = create_action_recognizer()
            if not self.action_recognizer:
                logger.error("행동 인식기 생성 실패")
                return False
            
            # 시각화 모듈 생성
            self.visualizer = Visualizer()
            
            # 모델 로드
            logger.info("모델 로드 중...")
            
            if not self.detector.load_model():
                logger.error("객체 검출 모델 로드 실패")
                return False
            
            if not self.pose_estimator.load_model():
                logger.error("자세 추정 모델 로드 실패")
                return False
            
            if not self.action_recognizer.load_model():
                logger.error("행동 인식 모델 로드 실패")
                return False
            
            logger.info("행동 인식 모델 로드 완료.")
            
            logger.info("모든 모델 로드 완료. 더미 추론을 통해 GPU 초기화를 진행합니다...")

            # 더미 추론을 위한 작은 검은색 이미지 생성 (예: 1x1 또는 작은 크기)
            # 모델의 실제 입력 크기와 유사하게 하되, 내용은 중요하지 않음
            # detector와 pose_estimator의 예상 입력 크기를 알아야 할 수 있음 (config에서 가져오거나 기본값 사용)
            dummy_img_size = (640, 480) # 일반적인 크기, 필요시 조절
            try:
                # DETECTOR_CONFIG나 POSE_CONFIG에서 imgsz를 가져오려고 시도
                # detector용
                detector_imgsz = DETECTION_CONFIG.get('imgsz', [640, 640]) # 리스트 또는 단일 값일 수 있음
                if isinstance(detector_imgsz, list):
                    dummy_img_h, dummy_img_w = detector_imgsz[0], detector_imgsz[1]
                else:
                    dummy_img_h, dummy_img_w = detector_imgsz, detector_imgsz
                dummy_input_detector = np.zeros((dummy_img_h, dummy_img_w, 3), dtype=np.uint8)
                logger.info(f"Detector 더미 추론 실행 (입력 크기: {dummy_input_detector.shape[:2]})...")
                _ = self.detector.detect(dummy_input_detector) # 첫 추론 강제 실행
                logger.info("Detector 더미 추론 완료.")
            except Exception as e:
                logger.warning(f"Detector 더미 추론 중 오류 발생 (무시하고 진행): {e}")

            try:
                # pose_estimator용
                pose_imgsz = POSE_CONFIG.get('imgsz', 640) # 리스트 또는 단일 값일 수 있음
                if isinstance(pose_imgsz, list):
                    dummy_img_h, dummy_img_w = pose_imgsz[0], pose_imgsz[1]
                else:
                    dummy_img_h, dummy_img_w = pose_imgsz, pose_imgsz
                dummy_input_pose = np.zeros((dummy_img_h, dummy_img_w, 3), dtype=np.uint8)
                logger.info(f"Pose Estimator 더미 추론 실행 (입력 크기: {dummy_input_pose.shape[:2]})...")
                _ = self.pose_estimator.estimate_pose(dummy_input_pose) # 첫 추론 강제 실행
                logger.info("Pose Estimator 더미 추론 완료.")
            except Exception as e:
                logger.warning(f"Pose Estimator 더미 추론 중 오류 발생 (무시하고 진행): {e}")
            
            # 행동 인식기는 현재 규칙 기반이므로 별도 더미 추론 불필요할 수 있으나,
            # 만약 딥러닝 기반 행동 인식 모델을 사용하고 유사한 지연 로딩이 있다면 추가 필요
            if hasattr(self.action_recognizer, 'recognize') and not ACTION_CONFIG['model_type'] in ['rule', 'simple']:
                try:
                    # 행동 인식기는 자세 추정 결과를 입력으로 받을 수 있으므로, 더미 자세 데이터 생성
                    # 예시: 17개의 키포인트, 각 키포인트는 (x, y, confidence)
                    dummy_pose_data = {'keypoints': [[0,0,0]]*17, 'connections': [], 'bbox': [0,0,1,1], 'confidence': 0, 'class_id': 0, 'object_id':0}
                    logger.info(f"Action Recognizer 더미 추론 실행...")
                    _ = self.action_recognizer.recognize(dummy_pose_data)
                    logger.info("Action Recognizer 더미 추론 완료.")
                except Exception as e:
                    logger.warning(f"Action Recognizer 더미 추론 중 오류 발생 (무시하고 진행): {e}")

            logger.info("모든 모델 GPU 초기화(더미 추론) 완료.")

            logger.info("시스템 초기화 완료")
            return True
        except Exception as e:
            logger.error(f"시스템 초기화 중 오류 발생: {e}")
            return False
        
    def _download_models(self):
        """모델 파일 존재 여부 확인 및 필요 시 YOLO 모델 자동 다운로드"""
        # 다운로드할 모델 목록
        missing_models = []
        
        # 모델 파일 이름 추출
        detection_model_name = os.path.basename(DETECTION_CONFIG['model_path'])
        pose_model_name = os.path.basename(POSE_CONFIG['model_path'])
        action_model_name = os.path.basename(ACTION_CONFIG['model_path'])
        
        # 모델 경로를 'models' 폴더로 고정
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        
        # 모델 경로 재설정
        detection_model_path = os.path.join(models_dir, detection_model_name)
        pose_model_path = os.path.join(models_dir, pose_model_name)
        action_model_path = os.path.join(models_dir, action_model_name)
        
        # config 경로 재설정 (전역 변수 수정)
        DETECTION_CONFIG['model_path'] = detection_model_path
        POSE_CONFIG['model_path'] = pose_model_path
        ACTION_CONFIG['model_path'] = action_model_path
        
        logger.info(f"객체 검출 모델 경로: {detection_model_path}")
        logger.info(f"자세 추정 모델 경로: {pose_model_path}")
        logger.info(f"행동 인식 모델 경로: {action_model_path}")
        
        # YOLO 객체 탐지 모델 확인 및 다운로드
        if not os.path.exists(detection_model_path):
            model_size = DETECTION_CONFIG['model_size']
            logger.info(f"객체 검출 모델 파일이 없습니다: {detection_model_name}. 자동으로 다운로드합니다.")
            
            try:
                # Ultralytics YOLO 모델 다운로드
                import torch
                
                # 모델 다운로드 경로 설정
                if 'yolo11' in detection_model_name.lower():
                    model_url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11{model_size}.pt"
                elif 'yolov8' in detection_model_name.lower():
                    model_url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8{model_size}.pt"
                elif 'yolov5' in detection_model_name.lower():
                    model_url = f"https://github.com/ultralytics/assets/releases/download/v7.0/yolov5{model_size}.pt"
                else:
                    model_url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8{model_size}.pt"
                
                logger.info(f"YOLO 객체 탐지 모델 다운로드 중: {model_url}")
                
                # 모델 다운로드
                import urllib.request
                urllib.request.urlretrieve(model_url, detection_model_path)
                
                logger.info(f"YOLO 객체 탐지 모델 다운로드 완료: {detection_model_path}")
            except Exception as e:
                logger.error(f"YOLO 객체 탐지 모델 다운로드 실패: {e}")
                missing_models.append(detection_model_name)
            
        # YOLO 자세 추정 모델 확인 및 다운로드
        if not os.path.exists(pose_model_path):
            model_size = POSE_CONFIG['model_size']
            pose_model_type = POSE_CONFIG['model_type'].lower()
            
            # Mediapipe는 내장 모델이므로 다운로드 불필요
            if pose_model_type == 'mediapipe':
                logger.info("Mediapipe는 내장 모델을 사용합니다.")
            # YOLO 계열 모델 다운로드 (yolo, yolov8, yolov11)
            elif pose_model_type in ['yolo', 'yolov8', 'yolov11']:
                logger.info(f"자세 추정 모델 파일이 없습니다: {pose_model_name}. 자동으로 다운로드합니다.")
                
                try:
                    # 모델 다운로드 경로 설정
                    if pose_model_type == 'yolov11' or 'yolo11' in pose_model_name.lower():
                        model_url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11{model_size}-pose.pt"
                    elif pose_model_type == 'yolov8' or 'yolov8' in pose_model_name.lower():
                        model_url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8{model_size}-pose.pt"
                    else:  # 'yolo'
                        model_url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8{model_size}-pose.pt"
                    
                    logger.info(f"YOLO 자세 추정 모델 다운로드 중: {model_url}")
                    
                    # 모델 다운로드
                    import urllib.request
                    urllib.request.urlretrieve(model_url, pose_model_path)
                    
                    logger.info(f"YOLO 자세 추정 모델 다운로드 완료: {pose_model_path}")
                except Exception as e:
                    logger.error(f"YOLO 자세 추정 모델 다운로드 실패: {e}")
                    missing_models.append(pose_model_name)
            else:
                missing_models.append(pose_model_name)
                logger.warning(f"지원하지 않는 자세 추정 모델 유형입니다: {pose_model_type}")
                
        # 행동 인식 모델 (규칙 기반이나 simple 인식기는 모델 불필요)
        if ACTION_CONFIG['model_type'] not in ['rule', 'simple'] and not os.path.exists(action_model_path):
            missing_models.append(action_model_name)
            logger.warning(f"행동 인식 모델 파일이 없습니다: {action_model_name}")
        
        # 사용자에게 모델 파일 설치 안내
        if missing_models:
            logger.error(f"다음 모델 파일을 models 폴더에 설치해야 합니다: {', '.join(missing_models)}")
            logger.error("프로그램이 정상적으로 작동하지 않을 수 있습니다.")
        else:
            logger.info("모든 필요한 모델 파일이 존재합니다.")
        
    def process_video(self):
        """비디오 처리 메인 루프"""
        # 멀티스레딩 설정
        enable_threading = SYSTEM_CONFIG.get('enable_threading', False)
        
        # 비디오 소스 열기
        if not self.video_loader.open():
            logger.error("비디오 소스를 열 수 없습니다.")
            return
            
        frame_size = self.video_loader.frame_size
        fps = self.video_loader.fps
        
        # 프레임 크기 축소 적용 (설정된 경우)
        if self.frame_resize_factor != 1.0:
            new_width = int(frame_size[0] * self.frame_resize_factor)
            new_height = int(frame_size[1] * self.frame_resize_factor)
            logger.info(f"프레임 크기 조정: {frame_size} -> ({new_width}, {new_height})")
        
        # 비디오 저장 설정
        self.visualizer.setup_video_writer(frame_size, fps)
        
        logger.info(f"비디오 처리 시작, 크기: {frame_size}, FPS: {fps}")
        logger.info(f"현재 설정 - skip_frames: {VIDEO_CONFIG['skip_frames']}, performance_mode: {VIDEO_CONFIG['performance_mode']}")
        logger.info(f"탐지 설정 - confidence_threshold: {DETECTION_CONFIG['confidence_threshold']}, model_size: {DETECTION_CONFIG['model_size']}")
        
        frame_count = 0
        process_time = time.time()
        
        # FPS 조절을 위한 변수 추가
        target_fps = SYSTEM_CONFIG.get('target_fps', 30)  # 목표 FPS (기본값 30)
        frame_time = 1.0 / max(1, target_fps)  # 프레임당 시간 (초)
        next_frame_time = time.time()
        
        # 디버깅을 위한 변수 추가
        detection_count = 0  # 성공적인 객체 탐지 수
        action_count = 0     # 성공적인 행동 인식 수
        
        # 결과 저장용 변수
        last_results = {
            'frame': None,
            'detections': [],
            'poses': [],
            'actions': [],
            'danger_violations': []
        }
        
        # 스레드 관리
        analysis_thread = None
        analysis_result = None
        
        # 분석 함수 (스레드로 실행)
        def analyze_frame(frame, current_frame_count):
            # 프레임 크기 조정 (성능 향상용)
            if self.frame_resize_factor != 1.0:
                new_width = int(frame.shape[1] * self.frame_resize_factor)
                new_height = int(frame.shape[0] * self.frame_resize_factor)
                
                # 크기 변경된 프레임으로 분석
                resized_frame = cv2.resize(frame, (new_width, new_height))
                
                # 객체(사람) 검출
                detections = self.detector.detect(resized_frame)
                
                # 크기 조정 비율에 맞게 바운딩 박스 좌표 조정
                for i in range(len(detections)):
                    # x1, y1, x2, y2, confidence, class_id 형식
                    detections[i][0] = detections[i][0] / self.frame_resize_factor  # x1
                    detections[i][1] = detections[i][1] / self.frame_resize_factor  # y1
                    detections[i][2] = detections[i][2] / self.frame_resize_factor  # x2
                    detections[i][3] = detections[i][3] / self.frame_resize_factor  # y2
            else:
                # 원본 크기로 분석
                detections = self.detector.detect(frame)
                
            # 검출된 객체가 없으면 빈 결과 반환
            if len(detections) == 0:
                return {
                    'detections': [],
                    'poses': [],
                    'actions': [],
                    'danger_violations': []
                }
                
            # 각 검출 객체에 대해 자세 추정 및 행동 인식 수행
            poses = []
            actions = []
            danger_violations = []
            
            for idx, detection in enumerate(detections):
                # 바운딩 박스 추출
                bbox = detection[:4]
                
                # 신뢰도가 임계값보다 낮으면 건너뛰기
                confidence = detection[4]
                class_id = int(detection[5]) if len(detection) > 5 else 0
                
                if confidence < DETECTION_CONFIG.get('confidence_threshold', 0.5):
                    continue
                
                # 자세 추정
                pose = self.pose_estimator.estimate_pose(frame, bbox)
                if pose is None:
                    pose = {'keypoints': [], 'connections': [], 'bbox': bbox, 'confidence': 0, 'class_id': class_id}
                else:
                    # 클래스 ID 추가
                    pose['class_id'] = class_id
                    # 객체 ID 추가 (추적 ID 대신 검출 인덱스 사용)
                    pose['object_id'] = idx
                    
                # 행동 인식 (객체 ID 전달)
                action = self.action_recognizer.recognize(pose)
                
                # 객체 클래스 정보 추가
                action['class_id'] = class_id
                # 객체 ID 추가
                action['object_id'] = idx
                
                # 위험 구역 침범 감지
                danger_violation = self.action_recognizer.detect_danger_zone_violation(pose)
                
                poses.append(pose)
                actions.append(action)
                danger_violations.append(danger_violation)
                
                # 알림 처리
                if ALERT_CONFIG['enabled']:
                    self.alert_manager.handle_alert(action, danger_violation, frame)
            
            return {
                'detections': detections,
                'poses': poses,
                'actions': actions,
                'danger_violations': danger_violations
            }
        
        # 프레임 처리 루프
        self.running = True  # 실행 상태 플래그 초기화
        while self.running:  # running 대신 self.running 사용
            loop_start_time = time.time()
            
            # 프레임 읽기
            ret, frame = self.video_loader.read_frame()
            if not ret:
                break
                
            frame_count += 1
            
            # 분석할 프레임 선택 (프레임 건너뛰기로 성능 최적화)
            analyze_current_frame = frame_count % VIDEO_CONFIG['skip_frames'] == 0
            
            if analyze_current_frame:
                if enable_threading and analysis_thread is None:
                    # 이전 분석 결과가 있으면 사용
                    if analysis_result is not None:
                        last_results.update(analysis_result)
                        analysis_result = None
                    
                    # 새 프레임 분석 시작 (스레드로) - frame_count 전달
                    analysis_thread = threading.Thread(
                        target=lambda: setattr(self, '_temp_result', analyze_frame(frame.copy(), frame_count))
                    )
                    analysis_thread.start()
                elif not enable_threading:
                    # 스레딩 없이 직접 분석 - frame_count 전달
                    analysis_result = analyze_frame(frame, frame_count)
                    last_results.update(analysis_result)
                    
                    # 디버깅용: 탐지 및 행동 개수 확인
                    if len(analysis_result.get('detections', [])) > 0:
                        detection_count += 1
                    if len(analysis_result.get('actions', [])) > 0:
                        action_count += 1
            
            # 스레드가 완료되었는지 확인
            if enable_threading and analysis_thread is not None:
                if not analysis_thread.is_alive():
                    # 결과 가져오기
                    if hasattr(self, '_temp_result'):
                        analysis_result = self._temp_result
                        delattr(self, '_temp_result')
                    
                    analysis_thread = None
            
            # 마지막 분석 결과로 시각화
            last_results['frame'] = frame
            
            # 시각화 전 데이터 검증
            detections = last_results.get('detections', [])
            poses = last_results.get('poses', [])
            actions = last_results.get('actions', [])
            danger_violations = last_results.get('danger_violations', [])
            
            # detections 데이터 검증 및 필터링
            validated_detections = []
            for det in detections:
                # 올바른 형식의 detection인지 확인 (x1, y1, x2, y2, confidence, class_id)
                if det is None or not isinstance(det, (list, tuple)) or len(det) < 4:
                    continue
                    
                # 바운딩 박스 좌표가 정수형인지 확인
                try:
                    # 데이터 복사 및 변환
                    validated_det = det.copy() if isinstance(det, list) else list(det)
                    
                    # 좌표값만 정수로 변환 (나머지는 원래 형식 유지)
                    for i in range(4):
                        if i < len(validated_det):
                            validated_det[i] = int(validated_det[i])
                            
                    validated_detections.append(validated_det)
                except (TypeError, ValueError) as e:
                    logger.warning(f"검출 데이터 변환 중 오류: {e}, 데이터: {det}")
                    continue
            
            # 데이터 검증 결과 로깅 - 빈도 줄이고 로그 내용 최소화
            if frame_count % 100 == 0:  # 30에서 100으로 변경
                # 로그 메시지 삭제
                pass
            
            # 유효한 탐지 데이터가 없는 경우 원본 detections 사용
            if len(validated_detections) == 0 and isinstance(detections, np.ndarray) and detections.size > 0:
                validated_detections = detections.tolist() if isinstance(detections, np.ndarray) else detections
            
            # VISUALIZATION_CONFIG에서 텍스트 표시 설정 확인 및 강제 활성화
            if 'show_labels' in VISUALIZATION_CONFIG:
                VISUALIZATION_CONFIG['show_labels'] = True
            if 'show_confidence' in VISUALIZATION_CONFIG:    
                VISUALIZATION_CONFIG['show_confidence'] = True
            if 'show_action_label' in VISUALIZATION_CONFIG:
                VISUALIZATION_CONFIG['show_action_label'] = True
            if 'show_bbox' in VISUALIZATION_CONFIG:
                VISUALIZATION_CONFIG['show_bbox'] = True
            
            # Visualizer 인스턴스의 설정도 직접 업데이트
            self.visualizer.show_bbox = True
            self.visualizer.show_action_label = True
            self.visualizer.show_skeleton = VISUALIZATION_CONFIG.get('show_skeleton', True)
            
            # 시각화 및 키 입력 처리
            output_frame, key = self.visualizer.visualize_frame(
                frame, 
                validated_detections,  # 검증된 detections 사용 
                poses, 
                actions,
                danger_violations
            )
            
            # ESC 키로 종료
            if key == 27:  # ESC 키
                logger.info("사용자에 의해 분석이 중단되었습니다.")
                self.running = False  # running 대신 self.running 사용
                break
            
            # 중간 과정 로깅
            if frame_count % 100 == 0:
                elapsed = time.time() - process_time
                fps_actual = 100 / max(0.001, elapsed)
                # 로그 메시지 삭제
                process_time = time.time()
                
            # FPS 조절을 위한 지연 시간 계산 및 적용
            elapsed_loop_time = time.time() - loop_start_time
            if elapsed_loop_time < frame_time:
                # 프레임 처리가 목표 FPS보다 빠르면 지연 시간 추가
                sleep_time = frame_time - elapsed_loop_time
                time.sleep(sleep_time)
            
            # 다음 프레임 처리 시작 시간 계산
            next_frame_time = max(next_frame_time + frame_time, time.time())
                
            # 사전 훈련된 행동 인식 모델 처리 (기존 자세 추정 및 행동 인식 루프 이후)
            use_pretrained_action = ACTION_CONFIG['model_type'] == 'pretrained'
            if use_pretrained_action and len(validated_detections) > 0:
                # 사전 훈련된 모델로 행동 인식 수행
                pretrained_actions = self.action_recognizer.process_frame(frame, validated_detections)
                
                # 결과 업데이트 (기존 actions 대체)
                if pretrained_actions:
                    # 인덱스 기반 매핑 (pretrained_actions의 object_id와 actions의 인덱스 일치)
                    for i, act in enumerate(pretrained_actions):
                        if i < len(actions):
                            actions[i] = act  # 기존 actions 업데이트
                        else:
                            actions.append(act)  # 새 action 추가
                
        # 자원 해제
        self.video_loader.release()
        self.visualizer.release()
        
        if enable_threading and analysis_thread is not None and analysis_thread.is_alive():
            analysis_thread.join(timeout=1.0)
        
        logger.info(f"비디오 처리 완료, 총 프레임: {frame_count}")
        

def parse_arguments():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description='행동 패턴 분석 시스템')
    
    parser.add_argument('--input', type=str, help='입력 비디오 파일 또는 카메라 ID')
    parser.add_argument('--output', type=str, help='출력 비디오 파일')
    parser.add_argument('--display', action='store_true', help='화면에 비디오 표시')
    parser.add_argument('--model_type', type=str, choices=['yolo', 'mediapipe'], help='자세 추정 모델 유형')
    parser.add_argument('--action_recognizer', type=str, choices=['rule', 'lstm', 'hybrid', 'pretrained', 'simple'], help='행동 인식기 유형')
    parser.add_argument('--use_gpu', action='store_true', help='GPU 사용 (config.py의 설정 덮어쓰기)')
    parser.add_argument('--use_cpu', action='store_true', help='CPU 사용 (config.py의 설정 덮어쓰기)')
    parser.add_argument('--type', type=str, choices=['file', 'camera', 'rtsp', 'image_folder'], help='입력 소스 유형')
    parser.add_argument('--image_pattern', type=str, help='이미지 폴더 사용 시 이미지 파일 패턴 (예: *.jpg)')
    
    return parser.parse_args()


def update_config_from_args(args):
    """
    명령줄 인수로부터 설정 업데이트
    
    Args:
        args: 명령줄 인수
    """
    # GPU/CPU 모드 설정
    if hasattr(args, 'use_gpu') and args.use_gpu:
        # GPU 모드 활성화
        toggle_gpu_mode(True)
        logger.info("명령줄 인수 --use_gpu로 GPU 모드가 활성화되었습니다.")
    elif hasattr(args, 'use_cpu') and args.use_cpu:
        # CPU 모드 활성화
        toggle_gpu_mode(False)
        logger.info("명령줄 인수 --use_cpu로 CPU 모드가 활성화되었습니다.")
    else:
        # 인수가 없으면 config.py의 기본 설정 사용
        logger.info(f"GPU/CPU 관련 명령줄 인수가 없으므로 기본 설정 사용: GPU 모드 = {'활성화' if USE_GPU else '비활성화'}")
    
    # 입력 소스 유형 설정
    if hasattr(args, 'type') and args.type:
        VIDEO_CONFIG['source_type'] = args.type
        logger.info(f"입력 소스 유형을 {args.type}로 설정했습니다.")
    
    # 이미지 패턴 설정
    if hasattr(args, 'image_pattern') and args.image_pattern:
        VIDEO_CONFIG['image_pattern'] = args.image_pattern
        logger.info(f"이미지 파일 패턴을 {args.image_pattern}로 설정했습니다.")
    
    # 비디오 입력 설정 업데이트
    if args.input:
        VIDEO_CONFIG['source_path'] = args.input
        
        # 입력이 이미지 폴더인 경우
        if hasattr(args, 'type') and args.type == 'image_folder':
            VIDEO_CONFIG['image_folder'] = args.input
            logger.info(f"이미지 폴더 경로를 {args.input}로 설정했습니다.")
        # 입력이 RTSP 스트림인 경우
        elif hasattr(args, 'type') and args.type == 'rtsp':
            VIDEO_CONFIG['rtsp_url'] = args.input
            logger.info(f"RTSP URL을 {args.input}로 설정했습니다.")
        # 입력이 카메라인 경우
        elif hasattr(args, 'type') and args.type == 'camera':
            try:
                VIDEO_CONFIG['camera_id'] = int(args.input)
                logger.info(f"카메라 ID를 {args.input}로 설정했습니다.")
            except ValueError:
                logger.warning(f"카메라 ID는 정수여야 합니다. 기본값 0을 사용합니다.")
                VIDEO_CONFIG['camera_id'] = 0
        # 입력이 파일인 경우 (기본)
        else:
            # 비디오 파일 크기 확인하여 업데이트
            try:
                video = cv2.VideoCapture(args.input)
                if video.isOpened():
                    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    # config에서 자동 업데이트가 활성화되어 있는 경우에만 업데이트
                    if VIDEO_CONFIG.get('override_from_source', True):
                        VIDEO_CONFIG['frame_width'] = width
                        VIDEO_CONFIG['frame_height'] = height
                    
                    video.release()
                    logger.info(f"비디오 파일 크기를 감지: {width}x{height}")
            except Exception as e:
                logger.error(f"비디오 파일 크기 확인 중 오류: {e}")
            
    # 출력 파일 경로 설정
    if args.output:
        VISUALIZATION_CONFIG['output_path'] = args.output
        VISUALIZATION_CONFIG['save_output'] = True
        logger.info(f"출력 파일 경로를 {args.output}로 설정했습니다.")
        
    # 실시간 표시 여부 설정
    if args.display:
        VISUALIZATION_CONFIG['display'] = True
        logger.info("실시간 표시 모드가 활성화되었습니다.")
        
    # 객체 검출기 설정
    if hasattr(args, 'model_type') and args.model_type:
        DETECTION_CONFIG['model_type'] = args.model_type
        
    # 행동 인식기 설정
    if hasattr(args, 'action_recognizer') and args.action_recognizer:
        ACTION_CONFIG['model_type'] = args.action_recognizer
    
    # 프레임 크기 조정 설정 - 속성이 있는지 먼저 확인
    if hasattr(args, 'frame_resize_factor') and args.frame_resize_factor:
        SYSTEM_CONFIG['frame_resize_factor'] = args.frame_resize_factor
        logger.info(f"프레임 크기 조정 비율을 {args.frame_resize_factor}로 설정했습니다.")
    
    logger.debug(f"현재 비디오 프레임 크기: {VIDEO_CONFIG['frame_width']}x{VIDEO_CONFIG['frame_height']}")
    logger.debug(f"위험 구역 설정 이미지 크기: {DANGER_ZONE_CONFIG.get('original_frame_width')}x{DANGER_ZONE_CONFIG.get('original_frame_height')}")
    
    # 스케일링 설정 정보 로깅
    if VIDEO_CONFIG.get('scale_coordinates', True):
        scaling_ratio_w = VIDEO_CONFIG['frame_width'] / DANGER_ZONE_CONFIG.get('original_frame_width', VIDEO_CONFIG['frame_width'])
        scaling_ratio_h = VIDEO_CONFIG['frame_height'] / DANGER_ZONE_CONFIG.get('original_frame_height', VIDEO_CONFIG['frame_height'])
        logger.debug(f"좌표 자동 스케일링: 활성화, 비율 - 가로: {scaling_ratio_w:.4f}, 세로: {scaling_ratio_h:.4f}")
    else:
        logger.debug("좌표 자동 스케일링: 비활성화")


def main():
    """메인 함수"""
    # 명령줄 인수 파싱
    args = parse_arguments()
    
    # 설정 업데이트
    update_config_from_args(args)
    
    # 시스템 초기화
    system = BehaviorAnalysisSystem()
    if not system.initialize():
        logger.error("시스템 초기화 실패")
        return
    
    # Ctrl+C 핸들러 설정
    def signal_handler(sig, frame):
        logger.info("Ctrl+C가 감지되었습니다. 프로그램을 종료합니다...")
        system.running = False  # 시스템 종료 플래그 설정
        
    # SIGINT 신호 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
        
    # 비디오 처리
    system.process_video()
    
    logger.info("프로그램 종료")
    

if __name__ == "__main__":
    main() 