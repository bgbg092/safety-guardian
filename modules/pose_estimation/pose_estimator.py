"""
자세 추정(Pose Estimation) 모듈

다양한 자세 추정 모델을 사용하여 사람의 관절 위치를 추정하는 기능을 제공합니다.
"""

import os
import sys
import cv2
import numpy as np
import logging
import torch
from abc import ABC, abstractmethod

# 상위 디렉토리 추가해서 config.py 접근 가능하게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import POSE_CONFIG

logger = logging.getLogger(__name__)

class PoseEstimator(ABC):
    """자세 추정기 추상 클래스"""
    
    @abstractmethod
    def load_model(self):
        """모델 로드"""
        pass
        
    @abstractmethod
    def estimate_pose(self, frame, bbox=None):
        """
        이미지에서 자세 추정
        
        Args:
            frame (numpy.ndarray): 입력 이미지
            bbox (list, optional): 바운딩 박스 [x1, y1, x2, y2]
            
        Returns:
            list: 추정된 관절 위치 정보
        """
        pass


class MediapipePoseEstimator(PoseEstimator):
    """Mediapipe를 사용한 자세 추정기"""
    
    def __init__(self, min_confidence=0.3, tracking=True):
        """
        Args:
            min_confidence (float): 최소 신뢰도 임계값
            tracking (bool): 연속 프레임 간 추적 활성화 여부
        """
        self.min_confidence = min_confidence or POSE_CONFIG['min_confidence']
        self.tracking = tracking if tracking is not None else POSE_CONFIG['tracking']
        self.model = None
        
    def load_model(self):
        """
        Mediapipe Pose 모델 로드
        
        Returns:
            bool: 모델 로드 성공 여부
        """
        try:
            import mediapipe as mp
            
            self.mp = mp
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            # Mediapipe Pose 모델 초기화
            self.model = self.mp_pose.Pose(
                static_image_mode=not self.tracking,  # False=비디오 스트림 최적화
                model_complexity=1,  # 0, 1, 2 중 선택 (높을수록 정확도 증가, 속도 감소)
                smooth_landmarks=True,  # 랜드마크 스무딩
                enable_segmentation=False,  # 세그멘테이션 비활성화
                min_detection_confidence=self.min_confidence,
                min_tracking_confidence=self.min_confidence if self.tracking else 0.0
            )
            
            logger.info("Mediapipe Pose 모델 로드 성공")
            return True
            
        except Exception as e:
            logger.error(f"Mediapipe Pose 모델 로드 실패: {e}")
            return False
            
    def estimate_pose(self, frame, bbox=None):
        """
        이미지에서 자세 추정
        
        Args:
            frame (numpy.ndarray): 입력 이미지 (BGR 포맷, OpenCV)
            bbox (list, optional): 바운딩 박스 [x1, y1, x2, y2]
            
        Returns:
            dict: 추정된 관절 위치 정보
                {
                    'keypoints': 관절 좌표 (x, y, 신뢰도),
                    'connections': 연결 정보 (관절 간 연결 목록),
                    'bbox': 사용된 바운딩 박스 [x1, y1, x2, y2],
                    'confidence': 전체 추정 신뢰도
                }
        """
        if self.model is None:
            logger.error("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")
            return None
            
        try:
            # 바운딩 박스가 제공된 경우 해당 영역만 잘라내기
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox[:4])
                # 바운딩 박스가 이미지 경계를 벗어나지 않도록 보정
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                if x1 >= x2 or y1 >= y2:
                    logger.warning("유효하지 않은 바운딩 박스")
                    return None
                    
                roi = frame[y1:y2, x1:x2].copy()
            else:
                roi = frame.copy()
                x1, y1 = 0, 0
                
            # RGB로 변환 (Mediapipe는 RGB 입력 예상)
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            # 자세 추정 실행
            results = self.model.process(rgb_roi)
            
            if not results.pose_landmarks:
                logger.debug("자세 추정 결과 없음")
                return None
                
            # 관절 추출
            landmarks = results.pose_landmarks.landmark
            
            # 관절 좌표 변환 및 신뢰도 추출
            h, w, _ = roi.shape
            keypoints = []
            
            # Mediapipe Pose는 33개의 랜드마크를 제공
            for idx, landmark in enumerate(landmarks):
                # Mediapipe 좌표계를 픽셀 좌표계로 변환 (상대좌표 → 절대좌표)
                x = landmark.x * w + x1  # 원본 이미지에 맞게 오프셋 적용
                y = landmark.y * h + y1
                visibility = landmark.visibility  # 신뢰도
                
                keypoints.append((x, y, visibility))
                
            # Mediapipe의 키포인트 연결 정보를 사용
            connections = [(connection[0], connection[1]) 
                          for connection in self.mp_pose.POSE_CONNECTIONS]
                
            # 전체 신뢰도 (visibility의 평균)
            confidence = np.mean([kp[2] for kp in keypoints])
            
            return {
                'keypoints': keypoints,
                'connections': connections,
                'bbox': [x1, y1, x2, y2] if bbox is not None else [0, 0, frame.shape[1], frame.shape[0]],
                'confidence': float(confidence)
            }
            
        except Exception as e:
            logger.error(f"자세 추정 중 예외 발생: {e}")
            return None
            
    def draw_pose(self, frame, pose_result, color=(0, 255, 0), thickness=2):
        """
        추정된 자세를 이미지에 그리기
        
        Args:
            frame (numpy.ndarray): 입력 이미지
            pose_result (dict): estimate_pose()의 반환값
            color (tuple): 관절 색상 (B,G,R)
            thickness (int): 선 두께
            
        Returns:
            numpy.ndarray: 자세가 그려진 이미지
        """
        if pose_result is None:
            return frame
            
        keypoints = pose_result['keypoints']
        connections = pose_result['connections']
        
        # 관절 그리기
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > self.min_confidence:
                cv2.circle(frame, (int(x), int(y)), thickness*2, color, -1)
                
        # 관절 연결선 그리기
        for start_idx, end_idx in connections:
            start_x, start_y, start_conf = keypoints[start_idx]
            end_x, end_y, end_conf = keypoints[end_idx]
            
            if start_conf > self.min_confidence and end_conf > self.min_confidence:
                cv2.line(frame, (int(start_x), int(start_y)), 
                         (int(end_x), int(end_y)), color, thickness)
                         
        return frame


# 다른 자세 추정 모델 구현은 필요에 따라 추가 가능 (OpenPose, HRNet 등)

class YOLOPoseEstimator(PoseEstimator):
    """YOLO 계열(YOLOv8, YOLOv11 등)을 사용한 자세 추정기"""
    
    def __init__(self, min_confidence=0.3):
        """
        Args:
            min_confidence (float): 최소 신뢰도 임계값
        """
        self.model_path = POSE_CONFIG['model_path']
        self.model_type = POSE_CONFIG['model_type'].lower()
        self.model_size = POSE_CONFIG['model_size']
        self.min_confidence = min_confidence or POSE_CONFIG['min_confidence']
        self.device = POSE_CONFIG.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        # 하드웨어 가속 설정
        self.use_cuda = 'cuda' in self.device
        # POSE_CONFIG에 use_tensorrt, use_half_precision이 없을 수 있으므로 get으로 기본값 False 설정
        self.use_tensorrt = POSE_CONFIG.get('use_tensorrt', False) if self.use_cuda else False
        self.use_half_precision = POSE_CONFIG.get('use_half_precision', False) if self.use_cuda else False
        self.imgsz = POSE_CONFIG.get('imgsz', 640) # 모델 입력 크기, 기본 640

    def load_model(self):
        """
        YOLO Pose 모델 로드
        
        Returns:
            bool: 모델 로드 성공 여부
        """
        try:
            from ultralytics import YOLO
            
            # 1. 경로 설정
            pt_path = self.model_path
            engine_path = pt_path.replace('.pt', '.engine')
            onnx_path = pt_path.replace('.pt', '.onnx')

            # 1-a. TensorRT 엔진 우선 로드
            if self.use_tensorrt and os.path.exists(engine_path):
                try:
                    logger.info(f"TensorRT Pose 모델 우선 로드 시도: {engine_path}")
                    self.model = YOLO(engine_path)
                    logger.info("TensorRT Pose 모델 로드 성공 (우선).")
                    return True
                except Exception as e:
                    logger.warning(f"TensorRT Pose 모델 로드 실패, 다른 형식으로 계속 진행: {e}")

            # 2. 최적화된 ONNX(있으면) 로드
            if os.path.exists(onnx_path):
                try:
                    logger.info(f"ONNX Pose 모델 로드 시도: {onnx_path}")
                    self.model = YOLO(onnx_path)
                    logger.info("ONNX Pose 모델 로드 성공.")
                    return True
                except Exception as e:
                    logger.warning(f"ONNX Pose 모델 로드 실패, .pt 로드로 계속 진행: {e}")

            # 3. PyTorch(.pt) 모델 로드 (기존 로직 유지)
            # YOLO pose 모델 로드
            normalized_model_type = self.model_type.lower().replace('v', '')
            model_name = f"{normalized_model_type}{self.model_size}-pose"
            
            # 모델 디렉토리 경로 설정
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            # 모델 파일 경로
            model_file = os.path.join(models_dir, f"{model_name}.pt")
            
            # 모델 파일이 존재하는지 확인
            if not os.path.exists(model_file):
                # 원래 작업 디렉토리 저장
                original_cwd = os.getcwd()
                try:
                    # 모델 디렉토리로 이동
                    os.chdir(models_dir)
                    # 모델 다운로드 (models 디렉토리에서 진행)
                    logger.info(f"모델 파일을 다운로드합니다: {model_name}.pt")
                    self.model = YOLO(f"{model_name}.pt") # 파일명으로 다운로드 시도
                    
                    # 다운로드된 파일을 올바른 위치로 이동 (필요한 경우)
                    # YOLO 라이브러리는 종종 현재 작업 디렉토리에 모델을 다운로드합니다.
                    downloaded_model_in_cwd = os.path.join(os.getcwd(), f"{model_name}.pt")
                    if os.path.exists(downloaded_model_in_cwd) and not os.path.samefile(downloaded_model_in_cwd, model_file):
                        import shutil
                        shutil.move(downloaded_model_in_cwd, model_file)
                        logger.info(f"다운로드된 모델 파일을 {model_file}로 이동했습니다.")
                    elif not os.path.exists(model_file):
                        logger.error(f"모델 파일 다운로드에 실패했거나 찾을 수 없습니다: {model_file}")
                        # self.model.export(format='pt', path=model_file) # YOLO 객체에서 직접 저장 시도 (필요시)
                        # return False # 실패 처리

                except Exception as e:
                    logger.error(f"모델 다운로드 중 오류 발생: {e}")
                    return False
                finally:
                    # 원래 작업 디렉토리로 복귀
                    os.chdir(original_cwd)
            else:
                logger.info(f"기존 모델 파일을 사용합니다: {model_file}")
            
            # 기본 PT 모델 로드
            self.model = YOLO(model_file)
            logger.info(f"{model_name}.pt 모델 로드 완료.")
            is_tensorrt_model_loaded = False # TensorRT 로드 성공 플래그

            # TensorRT 변환 및 로드
            if self.use_tensorrt and self.use_cuda:
                engine_file_name = f"{model_name}.engine"
                tensorrt_path = os.path.join(models_dir, engine_file_name)
                logger.info(f"TensorRT 모델 확인: {tensorrt_path}")

                if not os.path.exists(tensorrt_path):
                    logger.info(f"TensorRT 엔진 파일을 찾을 수 없습니다. {model_file}에서 변환을 시도합니다...")
                    original_cwd = os.getcwd()
                    try:
                        os.chdir(models_dir) # 모델 디렉토리에서 export 실행
                        pt_model_for_export = YOLO(os.path.basename(model_file)) # 현재 디렉토리의 pt 파일로 YOLO 객체 다시 생성
                        pt_model_for_export.export(format="engine", 
                                                 half=self.use_half_precision, 
                                                 imgsz=self.imgsz, 
                                                 device=self.device)
                        
                        # export 후 생성된 파일명 확인 (라이브러리 버전에 따라 다를 수 있음)
                        # 일반적인 경우: <model_name>.engine
                        exported_engine_in_cwd = os.path.join(os.getcwd(), engine_file_name)
                        
                        if os.path.exists(exported_engine_in_cwd) and not os.path.samefile(exported_engine_in_cwd, tensorrt_path):
                            import shutil
                            shutil.move(exported_engine_in_cwd, tensorrt_path)
                            logger.info(f"생성된 TensorRT 엔진을 {tensorrt_path}로 이동했습니다.")
                        elif not os.path.exists(tensorrt_path):
                             logger.error(f"TensorRT 엔진 파일 생성에 실패했거나 찾을 수 없습니다: {tensorrt_path}")
                        else:
                            logger.info(f"TensorRT 엔진이 {tensorrt_path}에 성공적으로 생성되었습니다.")
                            
                    except Exception as e:
                        logger.error(f"TensorRT 변환 중 오류 발생: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                    finally:
                        os.chdir(original_cwd)
                
                if os.path.exists(tensorrt_path):
                    try:
                        self.model = YOLO(tensorrt_path) # TensorRT 모델 로드
                        logger.info(f"TensorRT 모델 로드 성공: {tensorrt_path}")
                        is_tensorrt_model_loaded = True # 성공 시 플래그 설정
                    except Exception as e:
                        logger.error(f"TensorRT 모델 로드 실패. 기본 PT 모델을 사용합니다. 오류: {e}")
                        self.model = YOLO(model_file) # 실패 시 PT 모델로 복귀
                else:
                    logger.warning("TensorRT 엔진 파일을 사용할 수 없습니다. 기본 PT 모델을 사용합니다.")
                    self.model = YOLO(model_file) # TensorRT 파일이 없으면 PT 모델 사용
            
            # (선택적) ONNX 변환 및 로드 로직 (필요시 여기에 추가)
            # if self.use_onnx and self.use_cuda: ...

            # 최종 모델 디바이스 설정 및 FP16 적용 (TensorRT가 아닌 경우)
            if not is_tensorrt_model_loaded: # TensorRT 모델이 로드되지 않은 경우에만 실행
                self.model.to(self.device)
                if self.use_half_precision and self.use_cuda:
                    # TensorRT를 사용하지 않을 때만 직접 half() 호출
                    # 이미 TensorRT 모델을 로드했거나, PyTorch 모델을 FP16으로 로드한 경우 중복 적용 방지
                    if hasattr(self.model, 'model') and hasattr(self.model.model, 'half'):
                        self.model.model.half()
                    elif hasattr(self.model, 'half'): # YOLO 객체 자체에 half 메소드가 있을 경우
                        self.model.half()
                    logger.info(f"모델에 FP16 (half precision) 적용됨 (Device: {self.device})")

            logger.info("YOLO Pose 모델 초기화 완료.")
            return True
            
        except Exception as e:
            logger.error(f"YOLO Pose 모델 로드 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    def estimate_pose(self, frame, bbox=None):
        """
        이미지에서 자세 추정
        
        Args:
            frame (numpy.ndarray): 입력 이미지 (BGR 포맷, OpenCV)
            bbox (list, optional): 바운딩 박스 [x1, y1, x2, y2]
            
        Returns:
            dict: 추정된 관절 위치 정보
        """
        if self.model is None:
            logger.error("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")
            return None
            
        try:
            # 바운딩 박스 처리
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox[:4])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                roi = frame[y1:y2, x1:x2].copy()
            else:
                roi = frame.copy()
                x1, y1 = 0, 0
                
            # YOLO pose estimation 실행
            results = self.model(roi, imgsz=self.imgsz, conf=self.min_confidence, task='pose', verbose=False)
            
            if len(results) == 0 or not results[0].keypoints:
                logger.debug("자세 추정 결과 없음")
                return None
                
            # 키포인트 추출 (YOLO는 17개의 키포인트 제공) [[3]](#__3)
            keypoints_data = results[0].keypoints.data[0]
            keypoints = []
            
            # 키포인트 좌표 변환
            for kp in keypoints_data:
                x = float(kp[0]) + x1  # 원본 이미지 좌표로 변환
                y = float(kp[1]) + y1
                conf = float(kp[2])  # 신뢰도
                keypoints.append((x, y, conf))
                
            # COCO 데이터셋 키포인트 연결 정보 [[4]](#__4)
            connections = [
                (5, 7), (7, 9), (6, 8), (8, 10),  # 팔
                (5, 6), (5, 11), (6, 12),         # 어깨
                (11, 13), (13, 15), (12, 14), (14, 16),  # 다리
                (11, 12),  # 엉덩이
                (0, 1), (0, 2), (1, 3), (2, 4)    # 얼굴
            ]
            
            # 전체 신뢰도 계산
            confidence = float(np.mean([kp[2] for kp in keypoints]))
            
            return {
                'keypoints': keypoints,
                'connections': connections,
                'bbox': [x1, y1, x2, y2] if bbox is not None else [0, 0, frame.shape[1], frame.shape[0]],
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"자세 추정 중 예외 발생: {e}")
            return None
            
    def draw_pose(self, frame, pose_result, color=(0, 255, 0), thickness=2):
        """
        추정된 자세를 이미지에 그리기
        
        Args:
            frame (numpy.ndarray): 입력 이미지
            pose_result (dict): estimate_pose()의 반환값
            color (tuple): 관절 색상 (B,G,R)
            thickness (int): 선 두께
            
        Returns:
            numpy.ndarray: 자세가 그려진 이미지
        """
        if pose_result is None:
            return frame
            
        keypoints = pose_result['keypoints']
        connections = pose_result['connections']
        
        # 관절 그리기
        for x, y, conf in keypoints:
            if conf > self.min_confidence:
                cv2.circle(frame, (int(x), int(y)), thickness*2, color, -1)
                
        # 관절 연결선 그리기
        for start_idx, end_idx in connections:
            start_x, start_y, start_conf = keypoints[start_idx]
            end_x, end_y, end_conf = keypoints[end_idx]
            
            if start_conf > self.min_confidence and end_conf > self.min_confidence:
                cv2.line(frame, (int(start_x), int(start_y)), 
                         (int(end_x), int(end_y)), color, thickness)
                         
        return frame



def create_pose_estimator():
    """
    설정에 따라 적절한 자세 추정기를 생성합니다.
    
    Returns:
        PoseEstimator: 생성된 자세 추정기
    """
    model_type = POSE_CONFIG['model_type'].lower()
    
    if model_type == 'mediapipe':
        estimator = MediapipePoseEstimator(
            min_confidence=POSE_CONFIG['min_confidence'],
            tracking=POSE_CONFIG['tracking']
        )
    elif model_type in ['yolo', 'yolov8', 'yolov11']:  # YOLO 계열 모델들 지원
        estimator = YOLOPoseEstimator(
            min_confidence=POSE_CONFIG['min_confidence']
        )
    else:
        logger.error(f"지원하지 않는 자세 추정 모델 유형: {model_type}")
        return None
        
    return estimator
