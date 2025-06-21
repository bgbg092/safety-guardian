"""
행동 인식(Action Recognition) 모듈

자세 추정 결과를 바탕으로 사람의 행동을 분류하는 기능을 제공합니다.
"""

import os
import sys
import numpy as np
import logging
import torch
import cv2
import time
import math
from collections import deque
from abc import ABC, abstractmethod

# 상위 디렉토리 추가해서 config.py 접근 가능하게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import ACTION_CONFIG, DANGER_ZONE_CONFIG, VIDEO_CONFIG

logger = logging.getLogger(__name__)

# onnxruntime 모듈 임포트 - 명시적으로 ort 별칭 사용
try:
    import onnxruntime as ort
    logger.info("onnxruntime 모듈 임포트 성공")
except ImportError:
    logger.warning("onnxruntime 모듈을 가져올 수 없습니다. pip install onnxruntime로 설치하세요.")
    ort = None

class ActionRecognizer(ABC):
    """행동 인식기 추상 클래스"""
    
    @abstractmethod
    def load_model(self):
        """모델 로드"""
        pass
        
    @abstractmethod
    def recognize(self, pose_data):
        """
        자세 추정 결과를 바탕으로 행동 분류
        
        Args:
            pose_data: 자세 추정 결과
            
        Returns:
            dict: 분류된 행동 정보 {'action': 행동 라벨, 'confidence': 신뢰도}
        """
        pass
        
    def detect_danger_zone_violation(self, pose_data):
        """
        위험 구역 침범 여부 감지
        
        Args:
            pose_data (dict): 자세 추정 결과
                
        Returns:
            dict: 위험 구역 침범 정보 {'violated': True/False, 'zone_index': 침범한 구역 인덱스}
        """
        if not DANGER_ZONE_CONFIG['enabled'] or pose_data is None:
            return {'violated': False, 'zone_index': -1}
            
        keypoints = pose_data['keypoints']
        zones = DANGER_ZONE_CONFIG['zones']
        
        # 키포인트 수에 따라 모델 타입 판별 (Mediapipe:33개, YOLO:17개)
        is_yolo_model = len(keypoints) < 20
        
        # 모니터링할 키포인트 인덱스 설정
        monitored_indices = []
        if 'ankles' in DANGER_ZONE_CONFIG['monitored_keypoints']:
            if is_yolo_model:
                monitored_indices.extend([15, 16])  # YOLO: 왼쪽/오른쪽 발목
            else:
                monitored_indices.extend([27, 28])  # Mediapipe: 왼쪽/오른쪽 발목
                
        if 'hips' in DANGER_ZONE_CONFIG['monitored_keypoints']:
            if is_yolo_model:
                monitored_indices.extend([11, 12])  # YOLO: 왼쪽/오른쪽 엉덩이
            else:
                monitored_indices.extend([23, 24])  # Mediapipe: 왼쪽/오른쪽 엉덩이
        
        # 스케일링 비율 계산 (현재 프레임 크기와 원본 설정 이미지 크기 비교)
        if VIDEO_CONFIG.get('scale_coordinates', True):
            # 현재 프레임 크기
            frame_width = VIDEO_CONFIG['frame_width']
            frame_height = VIDEO_CONFIG['frame_height']
            
            # 위험 구역 설정 시 사용한 원본 이미지 크기
            original_width = DANGER_ZONE_CONFIG.get('original_frame_width', frame_width)
            original_height = DANGER_ZONE_CONFIG.get('original_frame_height', frame_height)
            
            # 비율 계산
            width_ratio = frame_width / original_width
            height_ratio = frame_height / original_height
            
            logger.debug(f"위험 구역 감지 - 스케일링 비율: 가로 {width_ratio:.2f}, 세로 {height_ratio:.2f}")
        else:
            # 스케일링 비활성화된 경우 비율 1.0으로 설정
            width_ratio = 1.0
            height_ratio = 1.0
        
        # 각 위험 구역에 대해 검사
        for i, zone_data in enumerate(zones):
            # zone_data가 딕셔너리인지 확인하고 'coordinates' 키 사용
            if isinstance(zone_data, dict):
                zone_coords = zone_data.get('coordinates')
            else:
                # 이전 버전 호환성을 위해 리스트 형식도 지원 (또는 경고 로깅)
                zone_coords = zone_data
                logger.warning(f"Zone data for index {i} is not a dictionary, using as coordinates directly. Consider updating config.")

            if not zone_coords: # 좌표가 없는 경우 건너뛰기
                logger.warning(f"Zone {i} has no coordinates.")
                continue

            if isinstance(zone_coords, list):
                if len(zone_coords) == 4 and not isinstance(zone_coords[0], (list, tuple)):  # 직사각형 [x1, y1, x2, y2]
                    # 스케일링 적용
                    x1 = int(zone_coords[0] * width_ratio)
                    y1 = int(zone_coords[1] * height_ratio)
                    x2 = int(zone_coords[2] * width_ratio)
                    y2 = int(zone_coords[3] * height_ratio)
                    
                    # 모니터링할 키포인트가 직사각형 영역 내에 있는지 확인
                    for idx in monitored_indices:
                        if idx < len(keypoints) and len(keypoints[idx]) >= 3:
                            x, y, conf = keypoints[idx]
                            if conf > 0.3:  # 신뢰도가 충분히 높은 경우만 검사
                                if x1 <= x <= x2 and y1 <= y <= y2:
                                    return {'violated': True, 'zone_index': i}
                elif len(zone_coords) > 2:  # 폴리곤 (점 목록 형식)
                    # 스케일링 적용된 폴리곤 포인트
                    scaled_points = []
                    for point in zone_coords:
                        # 튜플이나 리스트인 경우 (x,y) 형식으로 처리
                        if isinstance(point, (tuple, list)) and len(point) >= 2:
                            scaled_x = int(point[0] * width_ratio)
                            scaled_y = int(point[1] * height_ratio)
                            scaled_points.append((scaled_x, scaled_y))
                        else:
                            # 잘못된 좌표 형식
                            logger.warning(f"잘못된 좌표 형식: {point}, 무시합니다.")
                            continue
                    
                    # 점이 충분하지 않으면 건너뜀
                    if len(scaled_points) < 3:
                        logger.warning(f"위험 구역 {i+1}의 점이 3개 미만입니다: {len(scaled_points)}개. 건너뜁니다.")
                        continue
                        
                    # 폴리곤을 numpy 배열로 변환
                    poly_points = np.array(scaled_points, np.int32)
                    
                    # 모니터링할 키포인트가 폴리곤 내에 있는지 확인
                    for idx in monitored_indices:
                        if idx < len(keypoints) and len(keypoints[idx]) >= 3:
                            x, y, conf = keypoints[idx]
                            if conf > 0.3:  # 신뢰도가 충분히 높은 경우만 검사
                                # cv2.pointPolygonTest 함수: 점이 폴리곤 내부이면 양수, 외부이면 음수, 경계선 위면 0 반환
                                if cv2.pointPolygonTest(poly_points, (int(x), int(y)), False) >= 0:
                                    return {'violated': True, 'zone_index': i}
            else:
                logger.warning(f"지원되지 않는 위험 구역 형식: {zone_data}. 직사각형/폴리곤 형식만 지원됩니다.")
        
        # 위험 구역 침범 없음
        return {'violated': False, 'zone_index': -1}


class RuleBasedActionRecognizer(ActionRecognizer):
    """규칙 기반 행동 인식기"""
    
    def __init__(self, falling_threshold=None, stillness_frames=None):
        """
        Args:
            falling_threshold (float): 넘어짐 감지를 위한 높이 비율 임계값
            stillness_frames (int): 정지 상태 감지를 위한 프레임 수
        """
        self.falling_threshold = (falling_threshold or 
                                 ACTION_CONFIG['rule_config']['falling_threshold'])
        self.stillness_frames = (stillness_frames or 
                                ACTION_CONFIG['rule_config']['stillness_frames'])
        
        # 새로운 설정값 추가
        self.standing_min_height_ratio = ACTION_CONFIG['rule_config'].get('standing_min_height_ratio', 1.5)
        self.min_keypoint_confidence = ACTION_CONFIG['rule_config'].get('min_keypoint_confidence', 0.3)
        self.detection_consistency_frames = ACTION_CONFIG['rule_config'].get('detection_consistency_frames', 5)
        
        self.classes = ACTION_CONFIG['classes']
        self.prev_keypoints = None
        self.still_count = 0
        self.keypoint_history = deque(maxlen=30)  # 최근 30프레임의 키포인트 저장
        
        # 정적 객체 탐지 개선을 위한 변수들
        self.detection_history = []  # 최근 감지 이력
        self.last_valid_keypoints = None  # 마지막으로 신뢰도가 충분한 키포인트
        self.last_action = None  # 마지막 감지된 행동
        
    def load_model(self):
        """
        규칙 기반 행동 인식은 별도의 모델 로드가 필요 없음
        
        Returns:
            bool: 항상 True 반환
        """
        logger.info("규칙 기반 행동 인식기 초기화 완료")
        return True
        
    def recognize(self, pose_data):
        """
        자세 추정 결과를 바탕으로 행동 분류
        
        Args:
            pose_data (dict): 자세 추정 결과
                {
                    'keypoints': [(x1, y1, c1), (x2, y2, c2), ...],
                    'connections': [(idx1, idx2), ...],
                    'bbox': [x1, y1, x2, y2],
                    'confidence': 전체 신뢰도
                }
            
        Returns:
            dict: 분류된 행동 정보 {'action': 행동 라벨, 'confidence': 신뢰도}
        """
        # 입력 데이터가 없는 경우 처리
        if pose_data is None:
            # 이전 감지 결과가 있으면 사용 (정적 객체 탐지 개선)
            if self.last_action is not None and self.last_valid_keypoints is not None:
                self.detection_history.append(self.last_action)
                # 일정 수 이상 동일한 행동이 감지되면 해당 행동 반환
                if len(self.detection_history) >= self.detection_consistency_frames:
                    most_common_action = max(set(self.detection_history), key=self.detection_history.count)
                    return {'action': most_common_action, 'confidence': 0.6}
            return {'action': 'unknown', 'confidence': 0.0}
            
        keypoints = pose_data['keypoints']
        confidence = pose_data['confidence']
        
        # 키포인트 히스토리 업데이트
        self.keypoint_history.append(keypoints)
        
        # 신뢰도가 높은 키포인트면 저장 (정적 객체 탐지 개선)
        if confidence >= self.min_keypoint_confidence:
            self.last_valid_keypoints = keypoints
        elif self.last_valid_keypoints is not None:
            # 신뢰도가 낮으면 마지막으로 신뢰도가 높았던 키포인트 사용
            keypoints = self.last_valid_keypoints
            
        # 자세 추정 품질이 너무 낮으면 이전 결과 활용
        if confidence < self.min_keypoint_confidence and self.last_action is not None:
            self.detection_history.append(self.last_action)
            if len(self.detection_history) > self.detection_consistency_frames:
                self.detection_history.pop(0)
            
            # 일정 수 이상 동일한 행동이 감지되면 해당 행동 반환
            if len(self.detection_history) >= self.detection_consistency_frames:
                most_common_action = max(set(self.detection_history), key=self.detection_history.count)
                return {'action': most_common_action, 'confidence': 0.5}
                
        # 뻗은 자세 감지 (넘어짐)
        is_falling = self._detect_falling(keypoints)
        if is_falling:
            action_result = {'action': 'falling', 'confidence': 0.8}
            self.last_action = 'falling'
            self.detection_history.append('falling')
            return action_result
            
        # 움직임 여부 검사 (정지 상태)
        is_still = self._detect_stillness(keypoints)
        
        # 서 있는 자세 감지 (새로 추가)
        is_standing = self._detect_standing(keypoints)
        
        # 정지 상태에서 서 있는 자세인지 앉은 자세인지 구분
        if is_still:
            if is_standing:
                action_result = {'action': 'standing', 'confidence': 0.7}
                self.last_action = 'standing'
                self.detection_history.append('standing')
                return action_result
            else:
                # 하체가 보이지 않는 경우를 걸러내기 위해 앉은 자세 추가 검증 실행
                if self._verify_sitting_posture(pose_data):
                    action_result = {'action': 'sitting', 'confidence': 0.7}
                    self.last_action = 'sitting'
                    self.detection_history.append('sitting')
                    return action_result
                else:
                    # 검증 실패 시 기본(normal)로 분류
                    action_result = {'action': 'normal', 'confidence': 0.5}
                    self.last_action = 'normal'
                    self.detection_history.append('normal')
                    return action_result
        # 서 있는 자세이지만 정지 상태가 아니면 (움직이는 서 있는 자세)
        elif is_standing:
            # 걷기/뛰기 감지
            movement_type = self._detect_movement(self.keypoint_history)
            if movement_type == 'walking':
                action_result = {'action': 'walking', 'confidence': 0.7}
                self.last_action = 'walking'
                self.detection_history.append('walking')
                return action_result
            elif movement_type == 'fighting' and 'fighting' in self.classes:
                action_result = {'action': 'fighting', 'confidence': 0.7}
                self.last_action = 'fighting'
                self.detection_history.append('fighting')
                return action_result
            else:
                action_result = {'action': 'standing', 'confidence': 0.7}
                self.last_action = 'standing'
                self.detection_history.append('standing')
                return action_result
            
        # 걷기/뛰기 감지
        movement_type = self._detect_movement(self.keypoint_history)
        if movement_type == 'walking':
            action_result = {'action': 'walking', 'confidence': 0.7}
            self.last_action = 'walking'
            self.detection_history.append('walking')
            return action_result
        elif movement_type == 'fighting' and 'fighting' in self.classes:
            action_result = {'action': 'fighting', 'confidence': 0.7}
            self.last_action = 'fighting'
            self.detection_history.append('fighting')
            return action_result
            
        # 기본값
        action_result = {'action': 'normal', 'confidence': 0.5}
        self.last_action = 'normal'
        self.detection_history.append('normal')
        
        # 감지 이력 크기 제한
        if len(self.detection_history) > self.detection_consistency_frames:
            self.detection_history.pop(0)
            
        return action_result
        
    def _detect_falling(self, keypoints):
        """
        넘어짐 감지 (누워있거나 쓰러진 자세)
        
        Args:
            keypoints (list): 자세 추정 키포인트
            
        Returns:
            bool: 넘어짐 여부
        """
        # 키포인트가 없거나 충분하지 않은 경우 조용히 false 반환
        if not keypoints or len(keypoints) < 5:
            return False
            
        # 키포인트 수에 따라 모델 타입 판별 (Mediapipe:33개, YOLO:17개)
        is_yolo_model = len(keypoints) < 20
        
        if is_yolo_model:
            # YOLO 키포인트 인덱스 (COCO 형식)
            NOSE = 0
            LEFT_SHOULDER = 5
            RIGHT_SHOULDER = 6
            LEFT_HIP = 11
            RIGHT_HIP = 12
            LEFT_ANKLE = 15
            RIGHT_ANKLE = 16
        else:
            # Mediapipe 키포인트 인덱스
            NOSE = 0
            LEFT_SHOULDER = 11
            RIGHT_SHOULDER = 12
            LEFT_HIP = 23
            RIGHT_HIP = 24
            LEFT_ANKLE = 27
            RIGHT_ANKLE = 28
        
        # 필요한 키포인트 인덱스 중 최댓값
        max_required_index = max(NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE)
        
        # 키포인트 개수가 필요한 인덱스보다 적으면 조용히 false 반환
        if len(keypoints) <= max_required_index:
            return False
        
        # 관련 키포인트 추출
        try:
            nose = np.array(keypoints[NOSE][:2])
            left_shoulder = np.array(keypoints[LEFT_SHOULDER][:2])
            right_shoulder = np.array(keypoints[RIGHT_SHOULDER][:2])
            left_hip = np.array(keypoints[LEFT_HIP][:2])
            right_hip = np.array(keypoints[RIGHT_HIP][:2])
            left_ankle = np.array(keypoints[LEFT_ANKLE][:2])
            right_ankle = np.array(keypoints[RIGHT_ANKLE][:2])
            
            # 키포인트 신뢰도가 낮으면 계산 안함
            if (keypoints[NOSE][2] < 0.5 or keypoints[LEFT_SHOULDER][2] < 0.5 or
                keypoints[RIGHT_SHOULDER][2] < 0.5 or keypoints[LEFT_HIP][2] < 0.5 or
                keypoints[RIGHT_HIP][2] < 0.5):
                return False
                
            # 상체의 수직 방향 계산
            shoulders = (left_shoulder + right_shoulder) / 2
            hips = (left_hip + right_hip) / 2
            vertical_vector = shoulders - hips
            
            # 수직 벡터와 수평선 사이의 각도 계산
            horizontal = np.array([1, 0])
            vertical_norm = np.linalg.norm(vertical_vector)
            
            if vertical_norm > 0:
                cos_angle = np.dot(vertical_vector, horizontal) / (vertical_norm * 1.0)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
                
                # 각도가 특정 임계값보다 작으면 (수평에 가까우면) 넘어짐으로 판단
                return abs(angle - 90) > 45
                
            return False
        except (IndexError, TypeError) as e:
            logger.debug(f"넘어짐 감지 중 오류: {e}")  # WARNING에서 DEBUG로 변경
            return False
            
    def _detect_stillness(self, keypoints):
        """
        정지 상태 감지 (움직임이 거의 없는 상태)
        
        Args:
            keypoints (list): 자세 추정 키포인트
            
        Returns:
            bool: 정지 상태 여부
        """
        # 이전 키포인트가 없으면 정지 상태가 아님
        if self.prev_keypoints is None:
            self.prev_keypoints = keypoints
            return False
            
        # 주요 키포인트의 움직임 계산
        motion = 0
        count = 0
        
        for i, (curr_kp, prev_kp) in enumerate(zip(keypoints, self.prev_keypoints)):
            # 신뢰도가 낮은 키포인트는 무시
            if curr_kp[2] < 0.5 or prev_kp[2] < 0.5:
                continue
                
            # 키포인트 간 거리 계산
            dist = np.sqrt((curr_kp[0] - prev_kp[0]) ** 2 + (curr_kp[1] - prev_kp[1]) ** 2)
            motion += dist
            count += 1
            
        # 평균 움직임 계산
        avg_motion = motion / max(count, 1)
        
        # 움직임이 임계값보다 작으면 정지 카운트 증가
        if avg_motion < 3.0:  # 픽셀 단위 임계값
            self.still_count += 1
        else:
            self.still_count = 0
            
        self.prev_keypoints = keypoints
        
        # 연속된 프레임에서 정지 상태가 감지되면 참 반환
        return self.still_count >= self.stillness_frames
        
    def _detect_movement(self, keypoint_history):
        """
        움직임 패턴 감지 (걷기/뛰기/싸움)
        
        Args:
            keypoint_history (deque): 최근 프레임의 키포인트 히스토리
            
        Returns:
            str: 움직임 유형 ('walking', 'normal' 등 config의 classes에 정의된 행동)
        """
        if len(keypoint_history) < 2:
            return 'normal'
            
        curr_keypoints = keypoint_history[-1]
        prev_keypoints = keypoint_history[-2]

        # 필수 키포인트 인덱스 및 신뢰도 확인
        LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
        LEFT_KNEE, RIGHT_KNEE = 13, 14
        LEFT_ANKLE, RIGHT_ANKLE = 15, 16
        LEFT_WRIST, RIGHT_WRIST = 9, 10 # 싸움 감지를 위해 추가

        required_indices = [
            LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_KNEE, RIGHT_KNEE, 
            LEFT_ANKLE, RIGHT_ANKLE, LEFT_WRIST, RIGHT_WRIST
        ]

        if len(curr_keypoints) <= max(required_indices) or len(prev_keypoints) <= max(required_indices):
            return 'normal'

        if any(curr_keypoints[i][2] < self.min_keypoint_confidence for i in required_indices):
            return 'normal'

        # --- 싸움 감지 로직 (기존 로직 유지) ---
        if 'fighting' in self.classes:
            wrist_motion = []
            for i in range(1, len(keypoint_history)):
                prev_kps = keypoint_history[i-1]
                curr_kps = keypoint_history[i]
                if len(prev_kps) > RIGHT_WRIST and len(curr_kps) > RIGHT_WRIST:
                    try:
                        left_wrist_prev = np.array(prev_kps[LEFT_WRIST][:2])
                        left_wrist_curr = np.array(curr_kps[LEFT_WRIST][:2])
                        right_wrist_prev = np.array(prev_kps[RIGHT_WRIST][:2])
                        right_wrist_curr = np.array(curr_kps[RIGHT_WRIST][:2])
                        
                        left_wrist_dist = np.linalg.norm(left_wrist_curr - left_wrist_prev)
                        right_wrist_dist = np.linalg.norm(right_wrist_curr - right_wrist_prev)
                        wrist_motion.append(max(left_wrist_dist, right_wrist_dist))
                    except (IndexError, TypeError):
                        pass
            
            if wrist_motion:
                avg_wrist_motion = np.mean(wrist_motion)
                max_wrist_motion = np.max(wrist_motion)
                if max_wrist_motion > 30 and avg_wrist_motion > 15:
                    return 'fighting'

        # --- 걷기 감지 로직 (개선된 로직 적용) ---
        if 'walking' in self.classes:
            # 어깨 너비 기준으로 동적 임계값 설정
            lshoulder = np.array(curr_keypoints[LEFT_SHOULDER][:2])
            rshoulder = np.array(curr_keypoints[RIGHT_SHOULDER][:2])
            shoulder_width = np.linalg.norm(lshoulder - rshoulder)
            
            if shoulder_width < 1:
                shoulder_width = 30 # 기본 너비

            movement_threshold = shoulder_width * 0.15

            # 무릎 또는 발목의 움직임 감지
            prev_lknee = np.array(prev_keypoints[LEFT_KNEE][:2])
            prev_rknee = np.array(prev_keypoints[RIGHT_KNEE][:2])
            curr_lknee = np.array(curr_keypoints[LEFT_KNEE][:2])
            curr_rknee = np.array(curr_keypoints[RIGHT_KNEE][:2])
            
            prev_lankle = np.array(prev_keypoints[LEFT_ANKLE][:2])
            prev_rankle = np.array(prev_keypoints[RIGHT_ANKLE][:2])
            curr_lankle = np.array(curr_keypoints[LEFT_ANKLE][:2])
            curr_rankle = np.array(curr_keypoints[RIGHT_ANKLE][:2])

            lknee_dist = np.linalg.norm(curr_lknee - prev_lknee)
            rknee_dist = np.linalg.norm(curr_rknee - prev_rknee)
            lankle_dist = np.linalg.norm(curr_lankle - prev_lankle)
            rankle_dist = np.linalg.norm(curr_rankle - prev_rankle)
            
            leg_moved = (lknee_dist > movement_threshold or rknee_dist > movement_threshold or
                         lankle_dist > movement_threshold or rankle_dist > movement_threshold)

            if leg_moved:
                return 'walking'

        # 기본값
        default_action = 'normal' if 'normal' in self.classes else self.classes[0]
        return default_action

    def detect_danger_zone_violation(self, pose_data):
        """
        위험 구역 침범 여부 감지 (부모 클래스의 메서드 사용)
        
        Args:
            pose_data (dict): 자세 추정 결과
                
        Returns:
            dict: 위험 구역 침범 정보 {'violated': True/False, 'zone_index': 침범한 구역 인덱스}
        """
        return super().detect_danger_zone_violation(pose_data)

    def _detect_standing(self, keypoints):
        """
        서 있는 자세 감지
        
        Args:
            keypoints (list): 자세 추정 키포인트
            
        Returns:
            bool: 서 있는 자세 여부
        """
        # 키포인트가 없거나 충분하지 않은 경우 False 반환
        if not keypoints or len(keypoints) < 17:  # COCO 형식 키포인트(17개)
            return False
            
        # 키포인트 수에 따라 모델 타입 판별 (Mediapipe:33개, YOLO:17개)
        is_yolo_model = len(keypoints) < 20
        
        if is_yolo_model:
            # YOLO 키포인트 인덱스 (COCO 형식)
            LEFT_SHOULDER = 5
            RIGHT_SHOULDER = 6
            LEFT_HIP = 11
            RIGHT_HIP = 12
            LEFT_ANKLE = 15
            RIGHT_ANKLE = 16
        else:
            # Mediapipe 키포인트 인덱스
            LEFT_SHOULDER = 11
            RIGHT_SHOULDER = 12
            LEFT_HIP = 23
            RIGHT_HIP = 24
            LEFT_ANKLE = 27
            RIGHT_ANKLE = 28
            
        # 신뢰도 체크 (최소 신뢰도 낮춤)
        if (keypoints[LEFT_SHOULDER][2] < self.min_keypoint_confidence or
            keypoints[RIGHT_SHOULDER][2] < self.min_keypoint_confidence or
            keypoints[LEFT_HIP][2] < self.min_keypoint_confidence or
            keypoints[RIGHT_HIP][2] < self.min_keypoint_confidence or
            keypoints[LEFT_ANKLE][2] < self.min_keypoint_confidence or
            keypoints[RIGHT_ANKLE][2] < self.min_keypoint_confidence):
            return False
            
        try:
            # 어깨, 엉덩이, 발목 키포인트 추출
            shoulders = np.array([(keypoints[LEFT_SHOULDER][0] + keypoints[RIGHT_SHOULDER][0])/2,
                               (keypoints[LEFT_SHOULDER][1] + keypoints[RIGHT_SHOULDER][1])/2])
            hips = np.array([(keypoints[LEFT_HIP][0] + keypoints[RIGHT_HIP][0])/2,
                         (keypoints[LEFT_HIP][1] + keypoints[RIGHT_HIP][1])/2])
            ankles = np.array([(keypoints[LEFT_ANKLE][0] + keypoints[RIGHT_ANKLE][0])/2,
                           (keypoints[LEFT_ANKLE][1] + keypoints[RIGHT_ANKLE][1])/2])
            
            # 1. 신장 계산 (어깨-발목 수직 거리)
            height = abs(ankles[1] - shoulders[1])
            
            # 2. 너비 계산 (좌우 어깨 또는 엉덩이 중 넓은 것)
            shoulder_width = abs(keypoints[LEFT_SHOULDER][0] - keypoints[RIGHT_SHOULDER][0])
            hip_width = abs(keypoints[LEFT_HIP][0] - keypoints[RIGHT_HIP][0])
            width = max(shoulder_width, hip_width)
            
            # 3. 높이 대 너비 비율 계산
            height_width_ratio = height / max(width, 1)  # 0으로 나누기 방지
            
            # 4. 수직 정렬 확인 (어깨-엉덩이-발목이 수직에 가까운지)
            shoulders_to_ankles_vector = ankles - shoulders
            vertical_vector = np.array([0, 1])  # 수직 아래 방향
            
            # 벡터 정규화
            shoulders_to_ankles_norm = np.linalg.norm(shoulders_to_ankles_vector)
            if shoulders_to_ankles_norm > 0:
                normalized_vector = shoulders_to_ankles_vector / shoulders_to_ankles_norm
                # 수직선과의 유사도 (내적)
                vertical_alignment = np.dot(normalized_vector, vertical_vector)
                
                # --- 무릎 각도(hip-knee-ankle)가 160° 이상인지 추가 확인 ---
                # 모델 종류에 따라 무릎 인덱스가 다름 (YOLO: 13,14 / Mediapipe: 25,26)
                LEFT_KNEE = 13 if is_yolo_model else 25
                RIGHT_KNEE = 14 if is_yolo_model else 26

                def _calc_angle(p1, p2, p3):
                    """세 점(p1-p2-p3) 사이 각도를 도(deg) 단위로 반환"""
                    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]])
                    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]])
                    len1 = np.linalg.norm(v1)
                    len2 = np.linalg.norm(v2)
                    if len1 == 0 or len2 == 0:
                        return 0.0
                    cos_val = np.clip(np.dot(v1, v2) / (len1 * len2), -1.0, 1.0)
                    return np.degrees(np.arccos(cos_val))

                # 신뢰도 체크가 이미 끝났으므로 바로 각도 계산
                left_knee_angle = _calc_angle(keypoints[LEFT_HIP], keypoints[LEFT_KNEE], keypoints[LEFT_ANKLE])
                right_knee_angle = _calc_angle(keypoints[RIGHT_HIP], keypoints[RIGHT_KNEE], keypoints[RIGHT_ANKLE])

                # 서 있는 자세 판단: 
                # 1) 높이/너비 비율이 기준보다 크고 (세로로 길어야 함)
                # 2) 수직 정렬이 0.8보다 크면 (거의 수직으로 서 있어야 함)
                # 3) 어깨가 엉덩이보다 위에 있어야 함
                # 4) 양쪽 무릎 각도가 160° 이상 (거의 펴져 있어야 함)
                return (height_width_ratio > self.standing_min_height_ratio and
                       vertical_alignment > 0.8 and
                       shoulders[1] < hips[1] and
                       left_knee_angle > 160 and right_knee_angle > 160)
                       
        except (IndexError, ZeroDivisionError, ValueError) as e:
            logger.debug(f"서 있는 자세 감지 중 오류: {e}")
            return False
            
        return False

    def _verify_sitting_posture(self, pose):
        """
        무릎 각도·키포인트 위치를 이용해 실제 앉은 자세인지 추가 검증합니다.

        Args:
            pose (dict): 자세 추정 결과

        Returns:
            bool: 앉은 자세로 판단되면 True
        """
        keypoints = pose.get('keypoints', []) if pose else []
        if not keypoints or len(keypoints) < 17:
            return False

        # 요구 신뢰도 (조금 완화 가능)
        if any(keypoints[idx][2] < 0.5 for idx in [11,12,13,14,15,16]):
            return False

        # 무릎·엉덩이 y 차이
        left_knee_hip = abs(keypoints[13][1] - keypoints[11][1])
        right_knee_hip = abs(keypoints[14][1] - keypoints[12][1])

        # 무릎 각도 계산 함수
        def _calc_angle(p1, p2, p3):
            v1 = [p1[0]-p2[0], p1[1]-p2[1]]
            v2 = [p3[0]-p2[0], p3[1]-p2[1]]
            len1 = np.hypot(*v1)
            len2 = np.hypot(*v2)
            if len1==0 or len2==0:
                return np.pi
            cos = np.clip((v1[0]*v2[0]+v1[1]*v2[1])/(len1*len2), -1.0, 1.0)
            return np.arccos(cos)

        left_angle = _calc_angle(keypoints[11], keypoints[13], keypoints[15])
        right_angle = _calc_angle(keypoints[12], keypoints[14], keypoints[16])

        # 발목이 무릎보다 앞으로 나왔는지
        left_forward = keypoints[15][0] > keypoints[13][0] + 10
        right_forward = keypoints[16][0] > keypoints[14][0] + 10

        return (((left_knee_hip < 20 or right_knee_hip < 20) and
                 (left_angle < 1.8 or right_angle < 1.8)) and
                (left_forward or right_forward))


class LSTMActionRecognizer(ActionRecognizer):
    """LSTM 기반 행동 인식기"""
    
    def __init__(self, model_path=None, sequence_length=30, threshold=0.7):
        """
        Args:
            model_path (str): 모델 파일 경로 (None이면 config에서 가져옴)
            sequence_length (int): 입력 시퀀스 길이
            threshold (float): 분류 임계값
        """
        self.model_path = model_path or ACTION_CONFIG['model_path']
        self.sequence_length = sequence_length or ACTION_CONFIG['sequence_length']
        self.threshold = threshold or ACTION_CONFIG['threshold']
        self.classes = ACTION_CONFIG['classes']
        self.model = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 시퀀스 저장소
        self.pose_sequence = deque(maxlen=self.sequence_length)
        
    def load_model(self):
        """
        LSTM 모델 로드
        
        Returns:
            bool: 모델 로드 성공 여부
        """
        model_type = ACTION_CONFIG['model_type']
        
        # rule 기반 모델은 별도의 모델 파일 불필요
        if model_type.lower() == 'rule':
            logger.info("규칙 기반 행동 인식기 초기화 완료")
            return True
            
        # hybrid 모델도 rule 기능은 기본 내장
        if model_type.lower() == 'hybrid':
            # LSTM 모델 파일 체크
            if not os.path.exists(self.model_path):
                logger.warning(f"LSTM 모델 파일이 존재하지 않음: {self.model_path}, 규칙 기반 행동 인식으로 대체합니다.")
                # 모델 타입을 rule로 변경하여 기본 행동 인식은 가능하게 함
                self.model_type = 'rule'
                return True
                
            try:
                # ONNX 런타임 모델 로드
                self.model = ort.InferenceSession(self.model_path)
                logger.info(f"LSTM 행동 인식 모델 로드 성공: {self.model_path}")
                return True
            except Exception as e:
                logger.error(f"LSTM 모델 로드 실패: {e}")
                logger.warning("규칙 기반 행동 인식으로 대체합니다.")
                # 모델 타입을 rule로 변경하여 기본 행동 인식은 가능하게 함
                self.model_type = 'rule'
                return True
        
        # 기타 모델 처리
        try:
            # 현재는 추가 모델 지원 없음
            logger.warning(f"지원하지 않는 모델 유형: {model_type}, 규칙 기반 행동 인식으로 대체합니다.")
            self.model_type = 'rule'
            return True
        except Exception as e:
            logger.error(f"모델 로드 중 예외 발생: {e}")
            self.model_type = 'rule'  # 기본값으로 fallback
            return True  # 에러가 있더라도 rule 기반으로 계속 진행
        
    def _normalize_keypoints(self, keypoints):
        """
        키포인트 정규화
        
        Args:
            keypoints (list): 키포인트 리스트 [(x, y, conf), ...]
            
        Returns:
            numpy.ndarray: 정규화된 키포인트 특징 벡터
        """
        # 키포인트 좌표만 추출 (신뢰도 제외)
        kp_coords = np.array([list(kp[:2]) for kp in keypoints])
        
        if len(kp_coords) == 0:
            return np.zeros((len(keypoints), 2))
            
        # 중심 이동 및 스케일 정규화
        mean_coord = np.mean(kp_coords, axis=0)
        kp_centered = kp_coords - mean_coord
        
        # 스케일 정규화를 위한 최대 거리 계산
        max_dist = np.max(np.linalg.norm(kp_centered, axis=1))
        if max_dist > 0:
            kp_scaled = kp_centered / max_dist
        else:
            kp_scaled = kp_centered
            
        return kp_scaled
        
    def recognize(self, pose_data):
        """
        자세 추정 결과를 바탕으로 행동 분류
        
        Args:
            pose_data (dict): 자세 추정 결과
                {
                    'keypoints': [(x1, y1, c1), (x2, y2, c2), ...],
                    'connections': [(idx1, idx2), ...],
                    'bbox': [x1, y1, x2, y2],
                    'confidence': 전체 신뢰도
                }
            
        Returns:
            dict: 분류된 행동 정보 {'action': 행동 라벨, 'confidence': 신뢰도}
        """
        if pose_data is None or self.model is None:
            return {'action': 'unknown', 'confidence': 0.0}
            
        keypoints = pose_data['keypoints']
        
        # 키포인트 정규화 및 시퀀스 업데이트
        normalized_keypoints = self._normalize_keypoints(keypoints)
        
        # 피처 벡터 생성 (2D 좌표를 1D로 변환)
        feature_vector = normalized_keypoints.flatten()
        
        # 시퀀스 업데이트
        self.pose_sequence.append(feature_vector)
        
        # 시퀀스가 아직 충분히 쌓이지 않았다면
        if len(self.pose_sequence) < self.sequence_length:
            return {'action': 'unknown', 'confidence': 0.0}
            
        try:
            # 시퀀스 데이터를 모델 입력 형태로 변환
            sequence_data = np.array(list(self.pose_sequence))
            sequence_tensor = torch.FloatTensor(sequence_data).unsqueeze(0).to(self.device)
            
            # 모델 추론
            with torch.no_grad():
                outputs = self.model.run(None, {self.model.get_inputs()[0].name: sequence_tensor})
                probs = torch.nn.functional.softmax(torch.from_numpy(outputs[0]), dim=1)
                max_prob, predicted = torch.max(probs, 1)
                
                predicted_idx = predicted.item()
                confidence = max_prob.item()
                
                # 임계값보다 낮은 신뢰도는 unknown으로 처리
                if confidence < self.threshold:
                    return {'action': 'unknown', 'confidence': confidence}
                    
                # 결과 반환
                action = self.classes[predicted_idx] if predicted_idx < len(self.classes) else 'unknown'
                return {'action': action, 'confidence': confidence}
                
        except Exception as e:
            logger.error(f"행동 인식 중 예외 발생: {e}")
            return {'action': 'unknown', 'confidence': 0.0}

    def detect_danger_zone_violation(self, pose_data):
        """
        위험 구역 침범 여부 감지 (부모 클래스의 메서드 사용)
        
        Args:
            pose_data (dict): 자세 추정 결과
                
        Returns:
            dict: 위험 구역 침범 정보 {'violated': True/False, 'zone_index': 침범한 구역 인덱스}
        """
        return super().detect_danger_zone_violation(pose_data)


class HybridActionRecognizer(ActionRecognizer):
    """LSTM과 규칙 기반을 혼합한 행동 인식기"""
    
    def __init__(self, model_path=None):
        """
        초기화
        
        Args:
            model_path (str): LSTM 모델 파일 경로
        """
        self.model_path = model_path or ACTION_CONFIG['model_path']
        self.classes = ACTION_CONFIG['classes']
        self.threshold = ACTION_CONFIG['threshold']
        self.sequence_length = ACTION_CONFIG['sequence_length']
        self.batch_processing = ACTION_CONFIG.get('batch_processing', False)
        self.feature_type = ACTION_CONFIG.get('feature_type', 'position')
        self.temporal_window = ACTION_CONFIG.get('temporal_window', 15)
        self.use_gpu = ACTION_CONFIG.get('use_gpu', False)
        
        # LSTM 모델용 추론 세션
        self.session = None
        
        # 규칙 기반 인식기
        self.rule_recognizer = RuleBasedActionRecognizer()
        
        # 키포인트 시퀀스 저장용 버퍼
        self.pose_buffer = deque(maxlen=self.sequence_length)
        
        # 최근 n개 행동 결과 저장용 버퍼
        self.action_buffer = deque(maxlen=10)
        
        # 이전 키포인트 (속도 계산용)
        self.prev_keypoints = None
        self.prev_time = time.time()
        
    def load_model(self):
        """
        LSTM 모델 로드
        
        Returns:
            bool: 모델 로드 성공 여부
        """
        model_type = ACTION_CONFIG['model_type']
        
        # rule 기반 모델은 별도의 모델 파일 불필요
        if model_type.lower() == 'rule':
            logger.info("규칙 기반 행동 인식기 초기화 완료")
            return True
            
        # hybrid 모델도 rule 기능은 기본 내장
        if model_type.lower() == 'hybrid':
            # LSTM 모델 파일 체크
            if not os.path.exists(self.model_path):
                logger.warning(f"LSTM 모델 파일이 존재하지 않음: {self.model_path}, 규칙 기반 행동 인식으로 대체합니다.")
                # 모델 타입을 rule로 변경하여 기본 행동 인식은 가능하게 함
                self.model_type = 'rule'
                return True
                
            try:
                # ONNX 런타임 모델 로드
                self.session = ort.InferenceSession(self.model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider'])
                logger.info(f"LSTM 행동 인식 모델 로드 성공: {self.model_path}")
                
                # 규칙 기반 인식기 초기화
                self.rule_recognizer.load_model()
                
                return True
            except Exception as e:
                logger.error(f"LSTM 모델 로드 실패: {e}")
                logger.warning("규칙 기반 행동 인식으로 대체합니다.")
                # 모델 타입을 rule로 변경하여 기본 행동 인식은 가능하게 함
                self.model_type = 'rule'
                return True
        
        # 기타 모델 처리
        try:
            # 현재는 추가 모델 지원 없음
            logger.warning(f"지원하지 않는 모델 유형: {model_type}, 규칙 기반 행동 인식으로 대체합니다.")
            self.model_type = 'rule'
            return True
        except Exception as e:
            logger.error(f"모델 로드 중 예외 발생: {e}")
            self.model_type = 'rule'  # 기본값으로 fallback
            return True  # 에러가 있더라도 rule 기반으로 계속 진행
        
    def extract_features(self, pose):
        """
        자세에서 특징 추출 - train_coco_lstm_custom.py와 동일한 방식으로 구현
        
        Args:
            pose (dict): 자세 추정 결과
            
        Returns:
            numpy.ndarray: 추출된 특징 벡터 (102차원)
        """
        # 기본 위치 특징 (x, y, confidence) * 17
        keypoints = pose.get('keypoints', [])
        if not keypoints:
            return np.zeros(102, dtype=np.float32)  # 출력 특징 수는 102
        
        # 1. 키포인트 위치 특징 (x, y, confidence) * 17 = 51
        position_features = np.array(keypoints, dtype=np.float32).flatten()
        
        # 2. 각도 특징 (몸통 각도, 팔다리 각도 등) = 10개
        angle_features = np.zeros(10, dtype=np.float32)
        
        # 각도 계산 (두 벡터의 각도)
        def calculate_angle(p1, p2, p3):
            if p1[2] < 0.3 or p2[2] < 0.3 or p3[2] < 0.3:
                return 0.0
                
            vec1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            vec2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            vec1 = vec1 / norm1
            vec2 = vec2 / norm2
            
            # 내적으로 각도 계산 (라디안)
            dot = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
            angle = np.arccos(dot)
            
            return angle
        
        # COCO 키포인트 인덱스: [코(0), 왼쪽눈(1), 오른쪽눈(2), 왼쪽귀(3), 오른쪽귀(4), 
        # 왼쪽어깨(5), 오른쪽어깨(6), 왼쪽팔꿈치(7), 오른쪽팔꿈치(8), 왼쪽손목(9), 오른쪽손목(10),
        # 왼쪽엉덩이(11), 오른쪽엉덩이(12), 왼쪽무릎(13), 오른쪽무릎(14), 왼쪽발목(15), 오른쪽발목(16)]
        
        # 각도 계산을 위한 관절 그룹 (train_coco_lstm_custom.py와 동일)
        joint_groups = [
            (5, 7, 9),    # 왼쪽 어깨-팔꿈치-손목
            (6, 8, 10),   # 오른쪽 어깨-팔꿈치-손목
            (11, 13, 15), # 왼쪽 엉덩이-무릎-발목
            (12, 14, 16), # 오른쪽 엉덩이-무릎-발목
            (5, 11, 13),  # 왼쪽 어깨-엉덩이-무릎
            (6, 12, 14),  # 오른쪽 어깨-엉덩이-무릎
            (0, 5, 6),    # 코-왼쪽어깨-오른쪽어깨
            (0, 5, 11),   # 코-왼쪽어깨-왼쪽엉덩이
            (0, 6, 12),   # 코-오른쪽어깨-오른쪽엉덩이
            (5, 6, 0)     # 왼쪽어깨-오른쪽어깨-코
        ]
        
        # 각도 계산
        if len(keypoints) >= 17:
            for j, (p1, p2, p3) in enumerate(joint_groups):
                if (p1 < len(keypoints) and p2 < len(keypoints) and p3 < len(keypoints) and 
                    keypoints[p1][2] > 0.1 and keypoints[p2][2] > 0.1 and keypoints[p3][2] > 0.1):
                    # 관절 위치
                    x1, y1 = keypoints[p1][0], keypoints[p1][1]
                    x2, y2 = keypoints[p2][0], keypoints[p2][1]
                    x3, y3 = keypoints[p3][0], keypoints[p3][1]
                    
                    # 벡터 계산
                    v1 = [x1 - x2, y1 - y2]
                    v2 = [x3 - x2, y3 - y2]
                    
                    # 벡터 크기
                    len_v1 = np.sqrt(v1[0]**2 + v1[1]**2)
                    len_v2 = np.sqrt(v2[0]**2 + v2[1]**2)
                    
                    if len_v1 > 0 and len_v2 > 0:
                        # 내적 계산
                        dot_product = v1[0]*v2[0] + v1[1]*v2[1]
                        
                        # 각도 계산 (라디안)
                        angle = np.arccos(np.clip(dot_product / (len_v1 * len_v2), -1.0, 1.0))
                        angle_features[j] = angle
        
        # 3. 속도 특징 (키포인트 이동 속도) = 20개 (train_coco_lstm_custom.py와 일치)
        # 주요 관절 선택 (11개: 코, 어깨들, 팔꿈치들, 엉덩이들, 무릎들, 발목들)
        selected_joints = [0, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16]
        velocities_selected = np.zeros(20, dtype=np.float32)  # 10개 관절 x,y (20개 값)
        
        if self.prev_keypoints is not None and len(self.prev_keypoints) == len(keypoints):
            # 현재 시간
            current_time = time.time()
            dt = max(0.001, current_time - self.prev_time)
            
            # 키포인트 위치만 추출 (x, y)
            curr_positions = np.array([[kp[0], kp[1]] for kp in keypoints])
            prev_positions = np.array([[kp[0], kp[1]] for kp in self.prev_keypoints])
            
            # 속도 계산 (현재 위치 - 이전 위치) / 시간
            velocities = (curr_positions - prev_positions) / dt
            
            # 선택된 관절의 속도만 추출
            idx = 0
            for joint_idx in selected_joints:
                if joint_idx < len(keypoints) and joint_idx < len(self.prev_keypoints):
                    if idx < 10 and 2*idx+1 < 20:  # 최대 10개 관절, 20개 값
                        if keypoints[joint_idx][2] > 0.3 and self.prev_keypoints[joint_idx][2] > 0.3:
                            velocities_selected[2*idx] = velocities[joint_idx][0]  # x 속도
                            velocities_selected[2*idx+1] = velocities[joint_idx][1]  # y 속도
                idx += 1
                
        # 이전 키포인트 업데이트
        self.prev_keypoints = keypoints.copy()
        self.prev_time = time.time()
        
        # 기본 특징 결합 (51 + 10 + 20 = 81)
        features = np.concatenate([position_features, angle_features, velocities_selected])
        
        # 추가 파생 특징 계산 (총 102개 차원이 되도록)
        extra_features = np.zeros(102 - 81, dtype=np.float32)
        
        # 키포인트 쌍 간의 거리 (상위 10개 중요 관계 - 21개의 추가 특징)
        important_pairs = [
            (0, 5),    # 코-왼쪽어깨
            (0, 6),    # 코-오른쪽어깨
            (5, 6),    # 왼쪽어깨-오른쪽어깨
            (5, 7),    # 왼쪽어깨-왼쪽팔꿈치
            (6, 8),    # 오른쪽어깨-오른쪽팔꿈치
            (5, 11),   # 왼쪽어깨-왼쪽엉덩이
            (6, 12),   # 오른쪽어깨-오른쪽엉덩이
            (11, 12),  # 왼쪽엉덩이-오른쪽엉덩이
            (11, 13),  # 왼쪽엉덩이-왼쪽무릎
            (12, 14),  # 오른쪽엉덩이-오른쪽무릎
            (13, 15),  # 왼쪽무릎-왼쪽발목
            (14, 16),  # 오른쪽무릎-오른쪽발목
        ]
        
        for i, (p1, p2) in enumerate(important_pairs):
            if i < len(extra_features) and p1 < len(keypoints) and p2 < len(keypoints):
                if keypoints[p1][2] > 0.3 and keypoints[p2][2] > 0.3:
                    dx = keypoints[p1][0] - keypoints[p2][0]
                    dy = keypoints[p1][1] - keypoints[p2][1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    extra_features[i] = distance
        
        # 최종 특징 벡터 (102차원)
        final_features = np.zeros(102, dtype=np.float32)
        final_features[:len(features)] = features
        final_features[len(features):len(features)+len(extra_features)] = extra_features
        
        return final_features
            
    def recognize(self, pose):
        """
        행동 인식 (LSTM + 규칙)
        
        Args:
            pose (dict): 자세 추정 결과
            
        Returns:
            dict: 행동 인식 결과 {'action': 행동 라벨, 'confidence': 신뢰도}
        """
        # LSTM 모델이 로드되지 않았으면 규칙 기반으로 인식
        if self.session is None:
            return self.rule_recognizer.recognize(pose)
            
        # 특징 추출
        features = self.extract_features(pose)
        
        # 버퍼에 추가
        self.pose_buffer.append(features)
        
        # 버퍼에 충분한 프레임이 쌓이지 않았으면 규칙 기반으로 인식
        if len(self.pose_buffer) < self.sequence_length:
            return self.rule_recognizer.recognize(pose)
            
        try:
            # 시퀀스 데이터 생성
            sequence = np.array(list(self.pose_buffer), dtype=np.float32)
            
            # 값이 NaN이면 0으로 대체
            sequence = np.nan_to_num(sequence, nan=0.0)
            
            # 입력 형태 조정 (batch_size, sequence_length, feature_dim)
            input_data = sequence.reshape(1, self.sequence_length, -1)
            
            # 입력 이름 가져오기
            input_name = self.session.get_inputs()[0].name
            
            # 추론 실행
            output = self.session.run(None, {input_name: input_data})[0]
            
            # 결과 처리 (softmax 적용)
            exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
            probabilities = exp_output / np.sum(exp_output, axis=1, keepdims=True)
            
            # 예측 결과
            class_idx = np.argmax(probabilities[0])
            confidence = probabilities[0][class_idx]
            
            # 규칙 기반 결과도 얻기
            rule_result = self.rule_recognizer.recognize(pose)
            
            # 기본 행동은 normal로 시작
            final_action = 'normal'
            final_confidence = 0.6  # normal에 대한 기본 신뢰도 (정상 상태 기본값)
            
            # LSTM 모델 예측 행동
            lstm_action = self.classes[class_idx]
            
            # fighting 감지에 대한 더 보수적인 접근 - 임계값 대폭 상향
            high_confidence_threshold = self.threshold + 0.18  # 0.98로 상향(기존 0.8 + 0.18)
            
            # fighting에 대한 확률 순위 확인 - classes에 있는 경우에만
            fighting_idx = self.classes.index('fighting') if 'fighting' in self.classes else -1
            fighting_probability = probabilities[0][fighting_idx] if fighting_idx >= 0 else 0
            
            # 행동별 인식 조건 (행동 유형에 따라 다른 임계값 적용)
            if 'fighting' in self.classes and lstm_action == 'fighting' and confidence > high_confidence_threshold:
                # fighting은 매우 높은 임계값 필요 + 움직임 확인 + 추가 검증
                if self._has_significant_movement(pose) and self._verify_fighting_posture(pose):
                    # 추가 검증: fighting 감지는 연속된 프레임에서만 인식하도록 함
                    # 이전 5개 프레임 중 3개 이상이 fighting이어야 함 (강화된 조건)
                    if 'action_buffer' in self.__dict__ and len(self.action_buffer) >= 5:
                        fighting_count = self.action_buffer.count('fighting')
                        if fighting_count >= 3:  # 이전 5개 프레임 중 3개 이상이 fighting (기존 1개에서 3개로 강화)
                            final_action = 'fighting'
                            final_confidence = confidence
                        else:
                            # fighting 가능성은 있지만 연속성이 부족하면 other로 분류
                            final_action = 'other'
                            final_confidence = 0.6
                    else:
                        # 버퍼가 충분하지 않으면 other로 분류
                        final_action = 'other'
                        final_confidence = 0.6
                else:
                    # 자세 검증 실패하면 other로 분류
                    final_action = 'other'
                    final_confidence = 0.6
            elif lstm_action == 'sitting' and confidence > high_confidence_threshold:  # sitting도 더 높은 임계값 적용
                # sitting은 실제로 정지 상태인지 확인 + 추가 자세 검증
                if self._is_stationary(pose) and self._verify_sitting_posture(pose):
                    final_action = 'sitting'
                    final_confidence = confidence
                else:
                    final_action = 'other'
                    final_confidence = 0.6
            elif lstm_action == 'walking' and confidence > self.threshold:
                # walking은 실제로 이동 중인지 확인
                if self._has_walking_movement(pose):
                    final_action = 'walking'
                    final_confidence = confidence
                else:
                    final_action = 'other'
                    final_confidence = 0.6
            elif lstm_action == 'normal' and confidence > (self.threshold - 0.3):
                # normal은 더 낮은 임계값 적용 (더 쉽게 normal로 분류)
                final_action = 'normal'
                final_confidence = confidence
            else:
                # 모든 조건을 만족하지 않으면 'other'로 분류
                final_action = 'other'
                final_confidence = 0.6
                
            # fighting 감지 확률이 0.5 이상이면 추가 검증 - fighting이 classes에 있는 경우에만
            if 'fighting' in self.classes and fighting_probability > 0.5 and final_action != 'fighting':
                # fighting 가능성은 있지만 확실하지 않을 때 other로 분류
                final_action = 'other'
                final_confidence = max(final_confidence, fighting_probability)
                
            # 결과를 버퍼에 추가
            # self.action_buffer.append(final_action)
            
            # # 안정적인 결과를 위해 최근 n개 행동 중 가장 빈번한 행동 선택
            # if len(self.action_buffer) >= 5:
            #     from collections import Counter
            #     action_counts = Counter(self.action_buffer)
                
            #     # 우선순위 고려: 뚜렷하게 많은 경우만 적용
            #     most_common = action_counts.most_common()
                
            #     # normal이 충분히 많으면 normal 선택 (normal에 가중치 부여, 더 강화)
            #     normal_count = action_counts.get('normal', 0)
            #     total_count = len(self.action_buffer)
                
            #     if normal_count >= total_count * 0.5:  # 50% 이상이 normal이면 (기존 40%에서 상향)
            #         most_common_action = 'normal'
            #     # fighting이 감지되면 매우 엄격한 검증 (연속해서 나타나야 함) - fighting이 classes에 있는 경우에만
            #     elif 'fighting' in self.classes and action_counts.get('fighting', 0) > 0:
            #         # 최소 70% 이상이 fighting일 때만 fighting으로 분류
            #         if action_counts.get('fighting', 0) >= total_count * 0.7:
            #             most_common_action = 'fighting'
            #         else:
            #             most_common_action = 'other'  # fighting 의심은 있지만 확실하지 않으면 other로 분류
            #     elif len(most_common) > 0 and most_common[0][1] > total_count * 0.7:  # 특정 행동이 70% 이상 (기존 60%에서 상향)
            #         most_common_action = most_common[0][0]
            #     else:
            #         # 뚜렷한 행동이 없으면 other 선택
            #         most_common_action = 'other'
                
            #     # 최종 결과
            #     return {
            #         'action': most_common_action,
            #         'confidence': final_confidence
            #     }
            
            # # 최종 결과
            # return {
            #     'action': final_action,
            #     'confidence': final_confidence
            # }
            
        except Exception as e:
            logger.error(f"LSTM 행동 인식 중 오류 발생: {e}")
            # 오류 발생 시 other로 대체
            return {'action': 'other', 'confidence': 0.5}
            
        # 명시적 반환 추가 (주석 처리된 코드 대신)
        return {
            'action': final_action,
            'confidence': final_confidence
        }

        
    def _has_significant_movement(self, pose):
        """
        큰 움직임이 있는지 확인 (싸움 행동 검증용)
        
        Args:
            pose (dict): 자세 추정 결과
            
        Returns:
            bool: 큰 움직임이 있는지 여부
        """
        if self.prev_keypoints is None or len(self.prev_keypoints) == 0:
            return False
            
        keypoints = pose.get('keypoints', [])
        if not keypoints or len(keypoints) < 10:
            return False
            
        # 주요 관절 (손목, 팔꿈치, 어깨)의 움직임 확인 - 어깨도 추가해서 상체 전체 움직임 확인
        key_joints = [5, 6, 7, 8, 9, 10]  # 어깨, 팔꿈치, 손목
        
        # 움직임 계산
        joint_motions = []
        valid_counts = 0
        
        for idx in key_joints:
            if idx < len(keypoints) and idx < len(self.prev_keypoints):
                # 신뢰도 체크 - 더 높은 임계값 요구 (0.7 -> 0.8)
                if keypoints[idx][2] > 0.8 and self.prev_keypoints[idx][2] > 0.8:
                    dx = keypoints[idx][0] - self.prev_keypoints[idx][0]
                    dy = keypoints[idx][1] - self.prev_keypoints[idx][1]
                    motion = np.sqrt(dx*dx + dy*dy)
                    joint_motions.append(motion)
                    valid_counts += 1
                    
        # 충분한 관절이 감지되지 않으면 움직임 없음으로 판단
        if valid_counts < 4:  # 최소 4개 이상의 관절이 감지되어야 함
            return False
            
        # 평균 및 최대 움직임 계산
        avg_motion = np.mean(joint_motions)
        max_motion = np.max(joint_motions)
        
        # 움직임 패턴 분석 - 매우 엄격한 조건
        # 1. 평균 움직임이 35 이상 (기존 30보다 더 높게)
        # 2. 최대 움직임이 50 이상 (매우 빠른 움직임)
        # 3. 움직임의 표준편차가 큼 (불규칙한 움직임, 싸움의 특징)
        
        fast_avg_motion = avg_motion > 35.0
        very_fast_max_motion = max_motion > 50.0
        motion_std = np.std(joint_motions)
        irregular_motion = motion_std > 10.0
        
        # 위 세 가지 조건 중 최소 두 가지를 만족해야 함
        conditions_met = sum([fast_avg_motion, very_fast_max_motion, irregular_motion])
        
        return conditions_met >= 2
        
    def _verify_fighting_posture(self, pose):
        """
        싸움 자세인지 추가 검증 (팔 위치, 몸통 자세 등)
        
        Args:
            pose (dict): 자세 추정 결과
            
        Returns:
            bool: 싸움 자세 여부
        """
        keypoints = pose.get('keypoints', [])
        if not keypoints or len(keypoints) < 17:
            return False
            
        # 팔이 앞으로 뻗어 있거나 높이 올라가 있는지 확인
        # 코(0), 왼쪽 어깨(5), 오른쪽 어깨(6), 왼쪽 팔꿈치(7), 오른쪽 팔꿈치(8), 왼쪽 손목(9), 오른쪽 손목(10)
        
        # 신뢰도 체크 - 더 높은 신뢰도 요구 (0.7 -> 0.8)
        confidence_check = (
            keypoints[0][2] > 0.8 and keypoints[5][2] > 0.8 and keypoints[6][2] > 0.8 and
            keypoints[7][2] > 0.8 and keypoints[8][2] > 0.8 and
            keypoints[9][2] > 0.8 and keypoints[10][2] > 0.8
        )
        
        if not confidence_check:
            return False
            
        # 손목이 어깨보다 높은 위치에 있는지 확인 - 더 엄격한 조건으로 변경 (30 -> 40)
        left_arm_raised = keypoints[9][1] < keypoints[5][1] - 40  # 왼손목 y좌표가 왼어깨보다 훨씬 더 높게 있어야 함
        right_arm_raised = keypoints[10][1] < keypoints[6][1] - 40  # 오른손목 y좌표가 오른어깨보다 훨씬 더 높게 있어야 함
        
        # 손목이 몸 앞으로 뻗어 있는지 확인 - 더 정확한 방법으로 개선
        left_arm_extended = self._is_arm_extended(keypoints[5], keypoints[7], keypoints[9])
        right_arm_extended = self._is_arm_extended(keypoints[6], keypoints[8], keypoints[10])
        
        # 양팔 중 하나라도 올라가 있거나 뻗어 있으면서, 움직임이 빠른 경우
        # 더욱 엄격한 조건: 최소 세 가지 이상의 조건을 충족해야 함
        both_arms_raised = left_arm_raised and right_arm_raised
        both_arms_extended = left_arm_extended and right_arm_extended
        one_raised_one_extended = (left_arm_raised and right_arm_extended) or (right_arm_raised and left_arm_extended)
        
        # 팔 자세 기본 검증
        arm_posture_check = both_arms_raised or both_arms_extended or one_raised_one_extended
        
        if not arm_posture_check:
            return False
            
        # 추가 검증: 팔 움직임의 일관성과 지속성 확인
        # 현재 프레임의 손목 위치 추가
        current_left_wrist = np.array([keypoints[9][0], keypoints[9][1]])
        current_right_wrist = np.array([keypoints[10][0], keypoints[10][1]])
        
        # 이전 위치와 현재 위치 비교
        if self.prev_keypoints is not None and len(self.prev_keypoints) >= 11:
            if (len(self.prev_keypoints[9]) >= 2 and self.prev_keypoints[9][2] > 0.7 and 
                len(self.prev_keypoints[10]) >= 2 and self.prev_keypoints[10][2] > 0.7):
                
                prev_left_wrist = np.array([self.prev_keypoints[9][0], self.prev_keypoints[9][1]])
                prev_right_wrist = np.array([self.prev_keypoints[10][0], self.prev_keypoints[10][1]])
                
                # 움직임 벡터 계산
                left_motion_vector = current_left_wrist - prev_left_wrist
                right_motion_vector = current_right_wrist - prev_right_wrist
                
                # 움직임 크기 계산
                left_motion_magnitude = np.linalg.norm(left_motion_vector)
                right_motion_magnitude = np.linalg.norm(right_motion_vector)
                
                # 매우 빠른 움직임이 있어야 함 (30 -> 40)
                fast_motion = left_motion_magnitude > 40 or right_motion_magnitude > 40
                
                # 마지막 검증: 다음 모든 조건을 만족해야 함
                # 1. 팔 자세가 싸움과 유사함
                # 2. 빠른 움직임이 있음
                # 3. 상체가 앞으로 기울어져 있음 (공격적 자세)
                
                # 상체 기울기 확인 (어깨와 엉덩이 위치로 판단)
                if len(keypoints) > 12 and keypoints[11][2] > 0.7 and keypoints[12][2] > 0.7:
                    shoulders_center = np.array([(keypoints[5][0] + keypoints[6][0])/2, 
                                             (keypoints[5][1] + keypoints[6][1])/2])
                    hips_center = np.array([(keypoints[11][0] + keypoints[12][0])/2, 
                                         (keypoints[11][1] + keypoints[12][1])/2])
                    
                    # 상체가 앞으로 기울어졌는지 확인 (x축 차이)
                    torso_lean_forward = abs(shoulders_center[0] - hips_center[0]) > 30
                    
                    # 최종 판단: 모든 조건 충족 필요
                    return arm_posture_check and fast_motion and torso_lean_forward
                
                # 엉덩이 정보가 없으면 팔 자세와 빠른 움직임만으로 판단
                return arm_posture_check and fast_motion
        
        # 이전 프레임이 없는 경우에는 더 보수적으로 판단 (false 반환)
        return False
        
    def _is_arm_extended(self, shoulder, elbow, wrist):
        """
        팔이 뻗어 있는지 확인
        
        Args:
            shoulder: 어깨 키포인트 [x, y, confidence]
            elbow: 팔꿈치 키포인트 [x, y, confidence] 
            wrist: 손목 키포인트 [x, y, confidence]
            
        Returns:
            bool: 팔이 뻗어 있는지 여부
        """
        # 어깨-팔꿈치 벡터와 팔꿈치-손목 벡터 계산
        v1 = [elbow[0] - shoulder[0], elbow[1] - shoulder[1]]
        v2 = [wrist[0] - elbow[0], wrist[1] - elbow[1]]
        
        # 벡터 크기
        len_v1 = np.sqrt(v1[0]**2 + v1[1]**2)
        len_v2 = np.sqrt(v2[0]**2 + v2[1]**2)
        
        if len_v1 < 20 or len_v2 < 20:  # 팔이 충분히 길게 뻗어있어야 함
            return False
            
        # 각도 계산 (내적)
        dot_product = v1[0]*v2[0] + v1[1]*v2[1]
        cos_angle = dot_product / (len_v1 * len_v2)
        
        # 각도가 거의 일직선이면 (각도 > 160도) 팔이 뻗어 있는 것으로 판단 
        # cos(160도) = -0.94 (이전 cos(150도) = -0.866보다 더 엄격)
        return cos_angle < -0.94
        
    def _is_stationary(self, pose):
        """
        자세가 정지 상태인지 확인 (sitting 행동 검증용)
        
        Args:
            pose (dict): 자세 추정 결과
            
        Returns:
            bool: 정지 상태 여부
        """
        if self.prev_keypoints is None:
            return False
            
        keypoints = pose.get('keypoints', [])
        if not keypoints:
            return False
            
        # 전체 키포인트의 움직임 확인
        total_motion = 0
        count = 0
        
        for i, (curr_kp, prev_kp) in enumerate(zip(keypoints, self.prev_keypoints)):
            if curr_kp[2] > 0.5 and prev_kp[2] > 0.5:
                dx = curr_kp[0] - prev_kp[0]
                dy = curr_kp[1] - prev_kp[1]
                motion = np.sqrt(dx*dx + dy*dy)
                total_motion += motion
                count += 1
                
        if count > 0:
            avg_motion = total_motion / count
            return avg_motion < 1.5  # 더 작은 움직임만 허용 (매우 정적인 상태)
            
        return False
        
    def _verify_sitting_posture(self, pose):
        """
        무릎 각도·키포인트 위치를 이용해 실제 앉은 자세인지 추가 검증합니다.

        Args:
            pose (dict): 자세 추정 결과

        Returns:
            bool: 앉은 자세로 판단되면 True
        """
        keypoints = pose.get('keypoints', []) if pose else []
        if not keypoints or len(keypoints) < 17:
            return False

        # 요구 신뢰도 (조금 완화 가능)
        if any(keypoints[idx][2] < 0.5 for idx in [11,12,13,14,15,16]):
            return False

        # 무릎·엉덩이 y 차이
        left_knee_hip = abs(keypoints[13][1] - keypoints[11][1])
        right_knee_hip = abs(keypoints[14][1] - keypoints[12][1])

        # 무릎 각도 계산 함수
        def _calc_angle(p1, p2, p3):
            v1 = [p1[0]-p2[0], p1[1]-p2[1]]
            v2 = [p3[0]-p2[0], p3[1]-p2[1]]
            len1 = np.hypot(*v1)
            len2 = np.hypot(*v2)
            if len1==0 or len2==0:
                return np.pi
            cos = np.clip((v1[0]*v2[0]+v1[1]*v2[1])/(len1*len2), -1.0, 1.0)
            return np.arccos(cos)

        left_angle = _calc_angle(keypoints[11], keypoints[13], keypoints[15])
        right_angle = _calc_angle(keypoints[12], keypoints[14], keypoints[16])

        # 발목이 무릎보다 앞으로 나왔는지
        left_forward = keypoints[15][0] > keypoints[13][0] + 10
        right_forward = keypoints[16][0] > keypoints[14][0] + 10

        return (((left_knee_hip < 20 or right_knee_hip < 20) and
                 (left_angle < 1.8 or right_angle < 1.8)) and
                (left_forward or right_forward))
        
    def _has_walking_movement(self, pose):
        """
        걷는 움직임인지 확인 (walking 행동 검증용)
        
        Args:
            pose (dict): 자세 추정 결과
            
        Returns:
            bool: 걷는 움직임 여부
        """
        if self.prev_keypoints is None:
            return False
            
        keypoints = pose.get('keypoints', [])
        if not keypoints:
            return False
            
        # 다리 관절 (무릎, 발목)의 움직임 확인
        leg_joints = [13, 14, 15, 16]  # 무릎, 발목
        
        total_motion = 0
        count = 0
        
        for idx in leg_joints:
            if idx < len(keypoints) and idx < len(self.prev_keypoints):
                if keypoints[idx][2] > 0.5 and self.prev_keypoints[idx][2] > 0.5:
                    dx = keypoints[idx][0] - self.prev_keypoints[idx][0]
                    dy = keypoints[idx][1] - self.prev_keypoints[idx][1]
                    motion = np.sqrt(dx*dx + dy*dy)
                    total_motion += motion
                    count += 1
                    
        if count > 0:
            avg_motion = total_motion / count
            return avg_motion > 3.0 and avg_motion < 30.0  # 적절한 걷기 속도 범위
            
        return False

    def detect_danger_zone_violation(self, pose):
        """
        위험 구역 침범 감지 (규칙 기반 방식 그대로 사용)
        
        Args:
            pose (dict): 자세 추정 결과
            
        Returns:
            dict: 위험 구역 침범 결과 {'violated': 침범 여부, 'zone_index': 구역 인덱스}
        """
        return self.rule_recognizer.detect_danger_zone_violation(pose)

    def _detect_standing(self, keypoints):
        """
        서 있는 자세 감지
        
        Args:
            keypoints (list): 자세 추정 키포인트
            
        Returns:
            bool: 서 있는 자세 여부
        """
        # 키포인트가 없거나 충분하지 않은 경우 False 반환
        if not keypoints or len(keypoints) < 17:  # COCO 형식 키포인트(17개)
            return False
            
        # 키포인트 수에 따라 모델 타입 판별 (Mediapipe:33개, YOLO:17개)
        is_yolo_model = len(keypoints) < 20
        
        if is_yolo_model:
            # YOLO 키포인트 인덱스 (COCO 형식)
            LEFT_SHOULDER = 5
            RIGHT_SHOULDER = 6
            LEFT_HIP = 11
            RIGHT_HIP = 12
            LEFT_ANKLE = 15
            RIGHT_ANKLE = 16
        else:
            # Mediapipe 키포인트 인덱스
            LEFT_SHOULDER = 11
            RIGHT_SHOULDER = 12
            LEFT_HIP = 23
            RIGHT_HIP = 24
            LEFT_ANKLE = 27
            RIGHT_ANKLE = 28
            
        # 신뢰도 체크 (최소 신뢰도 낮춤)
        if (keypoints[LEFT_SHOULDER][2] < self.min_keypoint_confidence or
            keypoints[RIGHT_SHOULDER][2] < self.min_keypoint_confidence or
            keypoints[LEFT_HIP][2] < self.min_keypoint_confidence or
            keypoints[RIGHT_HIP][2] < self.min_keypoint_confidence or
            keypoints[LEFT_ANKLE][2] < self.min_keypoint_confidence or
            keypoints[RIGHT_ANKLE][2] < self.min_keypoint_confidence):
            return False
            
        try:
            # 어깨, 엉덩이, 발목 키포인트 추출
            shoulders = np.array([(keypoints[LEFT_SHOULDER][0] + keypoints[RIGHT_SHOULDER][0])/2,
                               (keypoints[LEFT_SHOULDER][1] + keypoints[RIGHT_SHOULDER][1])/2])
            hips = np.array([(keypoints[LEFT_HIP][0] + keypoints[RIGHT_HIP][0])/2,
                         (keypoints[LEFT_HIP][1] + keypoints[RIGHT_HIP][1])/2])
            ankles = np.array([(keypoints[LEFT_ANKLE][0] + keypoints[RIGHT_ANKLE][0])/2,
                           (keypoints[LEFT_ANKLE][1] + keypoints[RIGHT_ANKLE][1])/2])
            
            # 1. 신장 계산 (어깨-발목 수직 거리)
            height = abs(ankles[1] - shoulders[1])
            
            # 2. 너비 계산 (좌우 어깨 또는 엉덩이 중 넓은 것)
            shoulder_width = abs(keypoints[LEFT_SHOULDER][0] - keypoints[RIGHT_SHOULDER][0])
            hip_width = abs(keypoints[LEFT_HIP][0] - keypoints[RIGHT_HIP][0])
            width = max(shoulder_width, hip_width)
            
            # 3. 높이 대 너비 비율 계산
            height_width_ratio = height / max(width, 1)  # 0으로 나누기 방지
            
            # 4. 수직 정렬 확인 (어깨-엉덩이-발목이 수직에 가까운지)
            shoulders_to_ankles_vector = ankles - shoulders
            vertical_vector = np.array([0, 1])  # 수직 아래 방향
            
            # 벡터 정규화
            shoulders_to_ankles_norm = np.linalg.norm(shoulders_to_ankles_vector)
            if shoulders_to_ankles_norm > 0:
                normalized_vector = shoulders_to_ankles_vector / shoulders_to_ankles_norm
                # 수직선과의 유사도 (내적)
                vertical_alignment = np.dot(normalized_vector, vertical_vector)
                
                # 서 있는 자세 판단: 
                # 1) 높이/너비 비율이 기준보다 크고 (세로로 길어야 함)
                # 2) 수직 정렬이 0.8보다 크면 (거의 수직으로 서 있어야 함)
                # 3) 어깨가 엉덩이보다 위에 있어야 함
                return (height_width_ratio > self.standing_min_height_ratio and
                       vertical_alignment > 0.8 and
                       shoulders[1] < hips[1])
                       
        except (IndexError, ZeroDivisionError, ValueError) as e:
            logger.debug(f"서 있는 자세 감지 중 오류: {e}")
            return False
            
        return False


# SimpleActionRecognizer 클래스 추가
class SimpleActionRecognizer(ActionRecognizer):
    """간단한 규칙 기반 행동 인식기 (객체 ID별로 행동 상태 관리)"""
    
    def __init__(self):
        """초기화"""
        # 객체별 행동 상태를 저장할 딕셔너리
        self.object_states = {}
        
        # 지원하는 행동 클래스 목록
        self.classes = ACTION_CONFIG['classes']
        
        # 기본 상태 값 정의 (모든 행동 클래스에 대한 상태 추가)
        self.default_state = {
            "is_normal": True,
            "is_sitting": False,
            "is_standing": False,
            "is_walking": False,
            "is_other": False,
            "is_falling": False  # falling은 별도로 관리
        }
        
        # 추가 설정값
        self.standing_min_height_ratio = ACTION_CONFIG['rule_config'].get('standing_min_height_ratio', 1.5)
        self.min_keypoint_confidence = ACTION_CONFIG['rule_config'].get('min_keypoint_confidence', 0.3)
        self.stillness_frames = ACTION_CONFIG['rule_config'].get('stillness_frames', 20)
        
        # 이전 프레임 정보 저장용
        self.prev_keypoints = {}  # 객체별 이전 키포인트
        self.still_count = {}     # 객체별 정지 카운트
        self.keypoint_history = {}  # 객체별 키포인트 히스토리
    
    def load_model(self):
        """
        간단한 행동 인식기는 별도의 모델이 필요 없음
        
        Returns:
            bool: 항상 True 반환
        """
        logger.info("간단한 행동 인식기 초기화 완료")
        return True
    
    def recognize(self, pose_data):
        """
        자세 추정 결과를 바탕으로 행동 분류
        
        Args:
            pose_data (dict): 자세 추정 결과
                {
                    'keypoints': [(x1, y1, c1), (x2, y2, c2), ...],
                    'connections': [(idx1, idx2), ...],
                    'bbox': [x1, y1, x2, y2],
                    'confidence': 전체 신뢰도,
                    'object_id': 객체 ID (추적 ID)
                }
            
        Returns:
            dict: 분류된 행동 정보
        """
        # 입력 데이터가 없는 경우 처리
        if pose_data is None:
            return {"action": "unknown", "confidence": 0.0}
        
        # 객체 ID 추출 (없으면 기본값 0 사용)
        object_id = pose_data.get('object_id', 0)
        
        # 해당 객체의 상태가 없으면 초기화
        if object_id not in self.object_states:
            self.object_states[object_id] = self.default_state.copy()
            self.still_count[object_id] = 0
            self.keypoint_history[object_id] = []
        
        # 자세 데이터에서 키포인트 추출
        keypoints = pose_data.get('keypoints', [])
        confidence = pose_data.get('confidence', 0.0)
        
        # 키포인트가 없으면 기존 상태 반환
        if not keypoints:
            return self._get_action_from_state(object_id)
        
        # 키포인트 히스토리 업데이트
        if len(self.keypoint_history.get(object_id, [])) >= 30:
            self.keypoint_history[object_id].pop(0)
        self.keypoint_history[object_id].append(keypoints)
        
        # 1. 넘어짐 감지 (가장 우선순위가 높은 행동)
        if self._detect_falling(keypoints, object_id):
            self._reset_all_states(object_id)
            self.object_states[object_id]["is_falling"] = True
            return {"action": "falling", "confidence": 0.9}
        
        # 2. 정지 상태 감지
        is_still = self._detect_stillness(keypoints, object_id)
        
        # 3. 서 있음 감지
        is_standing = self._detect_standing(keypoints, object_id)
        
        # 4. 앉아 있음 감지
        is_sitting = self._detect_sitting(keypoints, object_id)
        
        # 5. 걷기 감지
        is_walking = self._detect_walking(keypoints, object_id)
        
        # 행동 상태 업데이트
        if is_still:
            if is_sitting:
                self._reset_all_states(object_id)
                self.object_states[object_id]["is_sitting"] = True
            elif is_standing:
                self._reset_all_states(object_id)
                self.object_states[object_id]["is_standing"] = True
            else:
                self._reset_all_states(object_id)
                self.object_states[object_id]["is_other"] = True
        elif is_walking:
            self._reset_all_states(object_id)
            self.object_states[object_id]["is_walking"] = True
        elif is_standing:
            self._reset_all_states(object_id)
            self.object_states[object_id]["is_standing"] = True
        else:
            # 감지된 행동이 없으면 normal로 설정
            self._reset_all_states(object_id)
            self.object_states[object_id]["is_normal"] = True
        
        # 이전 키포인트 저장
        self.prev_keypoints[object_id] = keypoints
        
        # 최종 행동 반환
        return self._get_action_from_state(object_id)
    
    def _reset_all_states(self, object_id):
        """모든 상태를 False로 리셋"""
        for key in self.object_states[object_id]:
            self.object_states[object_id][key] = False
    
    def _get_action_from_state(self, object_id):
        """상태에서 행동 추출"""
        state = self.object_states[object_id]
        
        if state.get("is_falling", False):
            return {"action": "falling", "confidence": 0.9}
        elif state.get("is_sitting", False):
            return {"action": "sitting", "confidence": 0.8}
        elif state.get("is_walking", False):
            return {"action": "walking", "confidence": 0.8}
        elif state.get("is_standing", False):
            return {"action": "standing", "confidence": 0.8}
        elif state.get("is_other", False):
            return {"action": "other", "confidence": 0.7}
        elif state.get("is_normal", False):
            return {"action": "normal", "confidence": 0.7}
        else:
            return {"action": "normal", "confidence": 0.5}
    
    def _detect_falling(self, keypoints, object_id, conf_threshold=0.5):
        """
        넘어짐 감지
        
        Args:
            keypoints (list): 자세 추정 키포인트
            object_id: 객체 ID
            conf_threshold (float): 신뢰도 임계값
            
        Returns:
            bool: 넘어짐 여부
        """
        # 키포인트가 없거나 충분하지 않은 경우 False 반환
        if not keypoints or len(keypoints) < 17:  # COCO 형식 키포인트(17개)
            return False
        
        # 주요 키포인트 추출
        nose = keypoints[0] if len(keypoints) > 0 else None
        left_ankle = keypoints[15] if len(keypoints) > 15 else None
        right_ankle = keypoints[16] if len(keypoints) > 16 else None
        left_knee = keypoints[13] if len(keypoints) > 13 else None
        right_knee = keypoints[14] if len(keypoints) > 14 else None
        
        # 키포인트가 없으면 False 반환
        if not all([nose, left_ankle or right_ankle, left_knee or right_knee]):
            return False
        
        # 신뢰도 확인
        nose_conf = nose[2] if len(nose) > 2 else 0
        left_ankle_conf = left_ankle[2] if left_ankle and len(left_ankle) > 2 else 0
        right_ankle_conf = right_ankle[2] if right_ankle and len(right_ankle) > 2 else 0
        left_knee_conf = left_knee[2] if left_knee and len(left_knee) > 2 else 0
        right_knee_conf = right_knee[2] if right_knee and len(right_knee) > 2 else 0
        
        # 좌표 추출
        nose_y = nose[1] if len(nose) > 1 else 0
        left_ankle_y = left_ankle[1] if left_ankle and len(left_ankle) > 1 else 0
        right_ankle_y = right_ankle[1] if right_ankle and len(right_ankle) > 1 else 0
        left_knee_y = left_knee[1] if left_knee and len(left_knee) > 1 else 0
        right_knee_y = right_knee[1] if right_knee and len(right_knee) > 1 else 0
        
        # 넘어짐 감지 로직: 코(nose)가 발목(ankle)이나 무릎(knee)보다 아래에 있으면 넘어진 것으로 판단
        if (nose_conf > conf_threshold and 
            ((left_ankle_conf > conf_threshold) or (right_ankle_conf > conf_threshold)) and
            ((left_knee_conf > conf_threshold) or (right_knee_conf > conf_threshold))):
            
            # 발목과 무릎의 평균 y 좌표 계산
            ankle_y = (left_ankle_y + right_ankle_y) / 2 if left_ankle_conf > conf_threshold and right_ankle_conf > conf_threshold else (left_ankle_y if left_ankle_conf > conf_threshold else right_ankle_y)
            knee_y = (left_knee_y + right_knee_y) / 2 if left_knee_conf > conf_threshold and right_knee_conf > conf_threshold else (left_knee_y if left_knee_conf > conf_threshold else right_knee_y)
            
            # 코가 발목이나 무릎보다 아래에 있으면 넘어진 것으로 판단
            if nose_y > ankle_y and nose_y > knee_y:
                return True
                
        return False
    
    def _detect_stillness(self, keypoints, object_id, threshold=3.0):
        """
        정지 상태 감지
        
        Args:
            keypoints (list): 자세 추정 키포인트
            object_id: 객체 ID
            threshold (float): 움직임 임계값
            
        Returns:
            bool: 정지 상태 여부
        """
        # 이전 키포인트가 없으면 정지 상태가 아님
        if object_id not in self.prev_keypoints:
            return False
            
        prev_keypoints = self.prev_keypoints[object_id]
        
        # 주요 키포인트의 움직임 계산
        motion = 0
        count = 0
        
        for i, (curr_kp, prev_kp) in enumerate(zip(keypoints, prev_keypoints)):
            # 신뢰도가 낮은 키포인트는 무시
            if curr_kp[2] < 0.5 or prev_kp[2] < 0.5:
                continue
                
            # 키포인트 간 거리 계산
            dist = np.sqrt((curr_kp[0] - prev_kp[0]) ** 2 + (curr_kp[1] - prev_kp[1]) ** 2)
            motion += dist
            count += 1
            
        # 평균 움직임 계산
        avg_motion = motion / max(count, 1)
        
        # 움직임이 임계값보다 작으면 정지 카운트 증가
        if avg_motion < threshold:
            self.still_count[object_id] = self.still_count.get(object_id, 0) + 1
        else:
            self.still_count[object_id] = 0
            
        # 연속된 프레임에서 정지 상태가 감지되면 참 반환
        return self.still_count[object_id] >= self.stillness_frames
    
    def _detect_standing(self, keypoints, object_id, conf_threshold=0.5):
        """
        서 있음 감지
        
        Args:
            keypoints (list): 자세 추정 키포인트
            object_id: 객체 ID
            conf_threshold (float): 신뢰도 임계값
            
        Returns:
            bool: 서 있음 여부
        """
        # 키포인트가 없거나 충분하지 않은 경우 False 반환
        if not keypoints or len(keypoints) < 17:  # COCO 형식 키포인트(17개)
            return False
            
        # 키포인트 인덱스 (COCO 형식)
        LEFT_SHOULDER = 5
        RIGHT_SHOULDER = 6
        LEFT_HIP = 11
        RIGHT_HIP = 12
        LEFT_ANKLE = 15
        RIGHT_ANKLE = 16
        
        # 신뢰도 체크
        if (keypoints[LEFT_SHOULDER][2] < self.min_keypoint_confidence or
            keypoints[RIGHT_SHOULDER][2] < self.min_keypoint_confidence or
            keypoints[LEFT_HIP][2] < self.min_keypoint_confidence or
            keypoints[RIGHT_HIP][2] < self.min_keypoint_confidence or
            keypoints[LEFT_ANKLE][2] < self.min_keypoint_confidence or
            keypoints[RIGHT_ANKLE][2] < self.min_keypoint_confidence):
            return False
            
        try:
            # 어깨, 엉덩이, 발목 키포인트 추출
            shoulders = np.array([(keypoints[LEFT_SHOULDER][0] + keypoints[RIGHT_SHOULDER][0])/2,
                               (keypoints[LEFT_SHOULDER][1] + keypoints[RIGHT_SHOULDER][1])/2])
            hips = np.array([(keypoints[LEFT_HIP][0] + keypoints[RIGHT_HIP][0])/2,
                         (keypoints[LEFT_HIP][1] + keypoints[RIGHT_HIP][1])/2])
            ankles = np.array([(keypoints[LEFT_ANKLE][0] + keypoints[RIGHT_ANKLE][0])/2,
                           (keypoints[LEFT_ANKLE][1] + keypoints[RIGHT_ANKLE][1])/2])
            
            # 1. 신장 계산 (어깨-발목 수직 거리)
            height = abs(ankles[1] - shoulders[1])
            
            # 2. 너비 계산 (좌우 어깨 또는 엉덩이 중 넓은 것)
            shoulder_width = abs(keypoints[LEFT_SHOULDER][0] - keypoints[RIGHT_SHOULDER][0])
            hip_width = abs(keypoints[LEFT_HIP][0] - keypoints[RIGHT_HIP][0])
            width = max(shoulder_width, hip_width)
            
            # 3. 높이 대 너비 비율 계산
            height_width_ratio = height / max(width, 1)  # 0으로 나누기 방지
            
            # 4. 수직 정렬 확인 (어깨-엉덩이-발목이 수직에 가까운지)
            shoulders_to_ankles_vector = ankles - shoulders
            vertical_vector = np.array([0, 1])  # 수직 아래 방향
            
            # 벡터 정규화
            shoulders_to_ankles_norm = np.linalg.norm(shoulders_to_ankles_vector)
            if shoulders_to_ankles_norm > 0:
                normalized_vector = shoulders_to_ankles_vector / shoulders_to_ankles_norm
                # 수직선과의 유사도 (내적)
                vertical_alignment = np.dot(normalized_vector, vertical_vector)
                
                # 서 있는 자세 판단: 
                # 1) 높이/너비 비율이 기준보다 크고 (세로로 길어야 함)
                # 2) 수직 정렬이 0.8보다 크면 (거의 수직으로 서 있어야 함)
                # 3) 어깨가 엉덩이보다 위에 있어야 함
                return (height_width_ratio > self.standing_min_height_ratio and
                       vertical_alignment > 0.8 and
                       shoulders[1] < hips[1])
                       
        except (IndexError, ZeroDivisionError, ValueError) as e:
            logger.debug(f"서 있는 자세 감지 중 오류: {e}")
            return False
            
        return False
    
    def _detect_sitting(self, keypoints, object_id, conf_threshold=0.5):
        """
        앉아있는 자세 감지
        
        Args:
            keypoints (list): 자세 추정 키포인트
            object_id: 객체 ID
            conf_threshold (float): 신뢰도 임계값
            
        Returns:
            bool: 앉아있는 자세 여부
        """
        # 키포인트가 없거나 충분하지 않은 경우 False 반환
        if not keypoints or len(keypoints) < 17:  # COCO 형식 키포인트(17개)
            return False
            
        # 키포인트 인덱스 (COCO 형식)
        LEFT_HIP = 11
        RIGHT_HIP = 12
        LEFT_KNEE = 13
        RIGHT_KNEE = 14
        LEFT_ANKLE = 15
        RIGHT_ANKLE = 16
        
        # 신뢰도 체크
        if (keypoints[LEFT_HIP][2] < conf_threshold or
            keypoints[RIGHT_HIP][2] < conf_threshold or
            keypoints[LEFT_KNEE][2] < conf_threshold or
            keypoints[RIGHT_KNEE][2] < conf_threshold or
            keypoints[LEFT_ANKLE][2] < conf_threshold or
            keypoints[RIGHT_ANKLE][2] < conf_threshold):
            return False
            
        try:
            # 무릎과 엉덩이의 y좌표 차이 계산 (앉은 자세는 차이가 작음)
            left_knee_hip_diff = abs(keypoints[LEFT_KNEE][1] - keypoints[LEFT_HIP][1])
            right_knee_hip_diff = abs(keypoints[RIGHT_KNEE][1] - keypoints[RIGHT_HIP][1])
            
            # 무릎 각도 계산 (엉덩이-무릎-발목)
            # 벡터 계산 (왼쪽)
            left_v1 = [keypoints[LEFT_HIP][0] - keypoints[LEFT_KNEE][0], 
                      keypoints[LEFT_HIP][1] - keypoints[LEFT_KNEE][1]]
            left_v2 = [keypoints[LEFT_ANKLE][0] - keypoints[LEFT_KNEE][0], 
                      keypoints[LEFT_ANKLE][1] - keypoints[LEFT_KNEE][1]]
            
            # 벡터 계산 (오른쪽)
            right_v1 = [keypoints[RIGHT_HIP][0] - keypoints[RIGHT_KNEE][0], 
                       keypoints[RIGHT_HIP][1] - keypoints[RIGHT_KNEE][1]]
            right_v2 = [keypoints[RIGHT_ANKLE][0] - keypoints[RIGHT_KNEE][0], 
                       keypoints[RIGHT_ANKLE][1] - keypoints[RIGHT_KNEE][1]]
            
            # 벡터 크기
            left_len_v1 = np.sqrt(left_v1[0]**2 + left_v1[1]**2)
            left_len_v2 = np.sqrt(left_v2[0]**2 + left_v2[1]**2)
            right_len_v1 = np.sqrt(right_v1[0]**2 + right_v1[1]**2)
            right_len_v2 = np.sqrt(right_v2[0]**2 + right_v2[1]**2)
            
            # 무릎 각도 계산 (내적)
            left_dot_product = left_v1[0]*left_v2[0] + left_v1[1]*left_v2[1]
            right_dot_product = right_v1[0]*right_v2[0] + right_v1[1]*right_v2[1]
            
            left_cos_angle = left_dot_product / (left_len_v1 * left_len_v2) if left_len_v1 > 0 and left_len_v2 > 0 else 0
            right_cos_angle = right_dot_product / (right_len_v1 * right_len_v2) if right_len_v1 > 0 and right_len_v2 > 0 else 0
            
            left_angle = np.arccos(np.clip(left_cos_angle, -1.0, 1.0))
            right_angle = np.arccos(np.clip(right_cos_angle, -1.0, 1.0))
            
            # 앉은 자세는 무릎 각도가 좁음 (약 90도 근처, 라디안으로 약 1.5)
            is_knee_bent = (left_angle < 1.8 or right_angle < 1.8)
            
            # 발목이 무릎보다 앞에 있는지 확인 (앉은 자세일 때 일반적)
            left_ankle_forward = keypoints[LEFT_ANKLE][0] > keypoints[LEFT_KNEE][0] + 10
            right_ankle_forward = keypoints[RIGHT_ANKLE][0] > keypoints[RIGHT_KNEE][0] + 10
            
            # 다음 조건을 모두 만족하면 앉은 자세로 판단:
            # 1. 엉덩이와 무릎의 y좌표 차이가 작고 (수직 거리가 가까움)
            # 2. 무릎이 구부러져 있고 (각도가 작음)
            # 3. 발목이 무릎보다 앞에 있음
            return (((left_knee_hip_diff < 20 or right_knee_hip_diff < 20) and is_knee_bent) and
                    (left_ankle_forward or right_ankle_forward))
                
        except (IndexError, ZeroDivisionError, ValueError) as e:
            logger.debug(f"앉은 자세 감지 중 오류: {e}")
            return False
            
        return False
    
    def _detect_walking(self, keypoints, object_id):
        """
        걷는 움직임 감지
        
        Args:
            keypoints (list): 자세 추정 키포인트
            object_id: 객체 ID
            
        Returns:
            bool: 걷는 움직임 여부
        """
        # 1. 앉아있으면 걷는 것이 아님
        if self._detect_sitting(keypoints, object_id):
            return False

        # 2. 필수 키포인트 인덱스 및 신뢰도 확인
        # COCO 키포인트 인덱스
        LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
        LEFT_KNEE, RIGHT_KNEE = 13, 14
        LEFT_ANKLE, RIGHT_ANKLE = 15, 16

        required_indices = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]
        
        if len(keypoints) <= max(required_indices):
            return False
            
        if any(keypoints[i][2] < self.min_keypoint_confidence for i in required_indices):
            return False

        # 3. 이전 프레임 데이터 확인
        if object_id not in self.prev_keypoints or self.prev_keypoints[object_id] is None:
            return False
            
        prev_keypoints = self.prev_keypoints[object_id]
        if len(prev_keypoints) <= max(required_indices):
            return False

        # 4. 어깨 너비 기준으로 동적 임계값 설정
        lshoulder = np.array(keypoints[LEFT_SHOULDER][:2])
        rshoulder = np.array(keypoints[RIGHT_SHOULDER][:2])
        shoulder_width = np.linalg.norm(lshoulder - rshoulder)
        
        # 어깨 너비가 0에 가까우면 기본값 사용
        if shoulder_width < 1:
            shoulder_width = 30 # 기본 너비

        # 움직임 임계값을 어깨 너비의 15%로 설정
        movement_threshold = shoulder_width * 0.15

        # 5. 무릎 또는 발목의 움직임 감지
        prev_lknee = np.array(prev_keypoints[LEFT_KNEE][:2])
        prev_rknee = np.array(prev_keypoints[RIGHT_KNEE][:2])
        curr_lknee = np.array(keypoints[LEFT_KNEE][:2])
        curr_rknee = np.array(keypoints[RIGHT_KNEE][:2])
        
        prev_lankle = np.array(prev_keypoints[LEFT_ANKLE][:2])
        prev_rankle = np.array(prev_keypoints[RIGHT_ANKLE][:2])
        curr_lankle = np.array(keypoints[LEFT_ANKLE][:2])
        curr_rankle = np.array(keypoints[RIGHT_ANKLE][:2])

        lknee_dist = np.linalg.norm(curr_lknee - prev_lknee)
        rknee_dist = np.linalg.norm(curr_rknee - prev_rknee)
        lankle_dist = np.linalg.norm(curr_lankle - prev_lankle)
        rankle_dist = np.linalg.norm(curr_rankle - prev_rankle)
        
        leg_moved = (lknee_dist > movement_threshold or rknee_dist > movement_threshold or
                     lankle_dist > movement_threshold or rankle_dist > movement_threshold)

        return leg_moved
    
    def get_actions(self, object_id=None):
        """
        모든 객체의 행동 상태 반환 또는 특정 객체의 행동 상태 반환
        
        Args:
            object_id: 객체 ID (지정하면 해당 객체의 상태만 반환)
            
        Returns:
            dict: 행동 상태
        """
        if object_id is not None:
            # 특정 객체의 상태만 반환
            if object_id in self.object_states:
                return self.object_states[object_id]
            else:
                return self.default_state.copy()
        
        # 모든 객체의 상태 반환
        return self.object_states


# 파일 중간에 새로운 PretrainedActionRecognizer 클래스 추가
# SimpleActionRecognizer 클래스 뒤에 추가

class PretrainedActionRecognizer(ActionRecognizer):
    """사전 학습된 외부 모델을 사용하는 행동 인식기"""
    
    def __init__(self, model_path=None):
        """
        Args:
            model_path (str, optional): 사전 학습된 모델 경로. None인 경우 기본 모델 사용
        """
        super().__init__()
        
        self.model_path = model_path or ACTION_CONFIG.get('pretrained_model_path', 'models/pretrained_action_recognition.pt')
        
        # 마스터 설정에 따라 디바이스 설정 (USE_GPU는 config.py에서 임포트됨)
        # ACTION_CONFIG의 use_gpu는 항상 USE_GPU와 동일하게 설정됨
        if not ACTION_CONFIG.get('use_gpu', False):
            self.device = torch.device('cpu')
            logger.info("마스터 설정에 따라 CPU 사용")
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if self.device.type == 'cuda':
                logger.info("마스터 설정에 따라 GPU 사용")
            else:
                logger.info("GPU를 사용하도록 설정되었으나 사용할 수 없어 CPU 사용")
        
        self.model = None
        self.transform = None
        self.input_size = ACTION_CONFIG.get('pretrained_input_size', (112, 112))
        self.clip_len = ACTION_CONFIG.get('pretrained_clip_len', 16)
        self.class_names = ACTION_CONFIG.get('pretrained_classes', ACTION_CONFIG['classes'])
        
        # 객체 ID별 프레임 버퍼 저장을 위한 딕셔너리
        self.frame_buffers = {}
        self.last_results = {}
        
        logger.info(f"사전 학습된 행동 인식 모델 초기화: {self.model_path}, 장치: {self.device}")
    
    def load_model(self):
        """
        사전 학습된 모델 로드
        
        Returns:
            bool: 모델 로드 성공 여부
        """
        try:
            import torchvision
            
            # 장치 설정 확인
            logger.info(f"사전 학습된 행동 인식 모델 로드 중, 사용 장치: {self.device}")
            
            # 모델 로드 시도 (여러 방법 시도)
            try:
                # 최신 버전 TorchVision API 사용
                self.model = torchvision.models.video.r3d_18(weights="KINETICS400_V1")
                logger.info("R3D-18 모델 로드 성공 (weights=KINETICS400_V1)")
            except Exception as modern_api_error:
                # 이전 버전 TorchVision API 사용
                logger.warning(f"최신 API 로드 실패: {modern_api_error}, 이전 API 시도...")
                try:
                    self.model = torchvision.models.video.r3d_18(pretrained=True)
                    logger.info("R3D-18 모델 로드 성공 (pretrained=True)")
                except Exception as legacy_api_error:
                    # 모델만 로드 (가중치 없이)
                    logger.warning(f"사전 학습 모델 로드 실패: {legacy_api_error}, 기본 모델 사용")
                    self.model = torchvision.models.video.r3d_18()
                    logger.info("R3D-18 모델 구조만 로드 (가중치 없음)")
            
            # CPU에서 모델 구성 (디바이스 일관성 유지를 위해)
            self.model = self.model.cpu()
            
            # 모델의 마지막 FC 레이어를 원하는 클래스 수에 맞게 변경
            num_classes = len(self.class_names)
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
            
            # 사용자 지정 모델 경로가 있으면 해당 가중치 로드 (파인튜닝된 모델)
            if os.path.exists(self.model_path):
                logger.info(f"사전 학습된 가중치 로드 중: {self.model_path}")
                try:
                    # CPU에서 로드 후 나중에 디바이스로 이동
                    state_dict = torch.load(self.model_path, map_location='cpu')
                    self.model.load_state_dict(state_dict)
                    logger.info("사용자 지정 가중치 로드 성공")
                except Exception as load_error:
                    logger.error(f"가중치 로드 실패: {load_error}")
            
            # 입력 전처리 변환 설정 (CPU에서)
            self.transform = torch.nn.Sequential(
                torchvision.transforms.Normalize(
                    mean=[0.43216, 0.394666, 0.37645],
                    std=[0.22803, 0.22145, 0.216989]
                )
            )
            
            # 모델을 평가 모드로 설정
            self.model.eval()
            
            # 모델과 변환을 원하는 디바이스로 이동
            logger.info(f"모델을 {self.device}로 이동 중...")
            self.model = self.model.to(self.device)
            self.transform = self.transform.to(self.device)
            
            logger.info(f"사전 학습된 행동 인식 모델 설정 완료. 클래스: {self.class_names}, 장치: {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"사전 학습된 행동 인식 모델 로드 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 모델과 변환을 None으로 설정하여 오류 방지
            self.model = None
            self.transform = None
            return False
    
    def recognize(self, pose_data):
        """
        자세 추정 결과를 바탕으로 행동 분류
        
        Args:
            pose_data (dict): 자세 추정 결과
                
        Returns:
            dict: 분류된 행동 정보 {'action': 행동 라벨, 'confidence': 신뢰도}
        """
        if pose_data is None:
            return {'action': 'unknown', 'confidence': 0.0}
        
        # 객체 ID 추출
        object_id = pose_data.get('object_id', 0)
        
        # 모델이 로드되지 않았으면 unknown 반환
        if self.model is None:
            return {'action': 'unknown', 'confidence': 0.0}
        
        # 해당 객체의 프레임 버퍼가 없으면 초기화
        if object_id not in self.frame_buffers:
            self.frame_buffers[object_id] = []
        
        # 키포인트 좌표를 사용하여 해당 부분 이미지 생성
        try:
            # 입력 프레임이 없으면 최근 결과 반환
            if object_id in self.last_results:
                last_result = self.last_results[object_id]
                return last_result
            
            # 충분한 프레임이 없으면 unknown 반환
            if len(self.frame_buffers[object_id]) < self.clip_len:
                return {'action': 'unknown', 'confidence': 0.0}
                
            # 프레임 버퍼에서 필요한 만큼의 프레임 가져오기
            frames = self.frame_buffers[object_id][-self.clip_len:]
            
            # [clip_len, channel, height, width] -> [batch, channel, clip_len, height, width]
            # 모든 프레임이 같은 디바이스에 있는지 확인
            frames_on_cpu = [f.cpu() if f.device.type != 'cpu' else f for f in frames]
            input_frames = torch.stack(frames_on_cpu).permute(1, 0, 2, 3).unsqueeze(0)
            
            # 입력 텐서를 모델과 같은 디바이스로 이동
            input_frames = input_frames.to(self.device)
            
            # 모델에 입력하여 예측
            with torch.no_grad():
                outputs = self.model(input_frames)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
                # 최대 확률 클래스 및 신뢰도 추출
                confidence, class_idx = torch.max(probabilities, dim=0)
                
                # 결과 저장
                action_class = self.class_names[min(class_idx.item(), len(self.class_names)-1)]
                result = {
                    'action': action_class,
                    'confidence': confidence.item(),
                    'object_id': object_id
                }
                
                # 최근 결과 저장
                self.last_results[object_id] = result
                
                return result
                
        except Exception as e:
            logger.warning(f"행동 인식 중 오류 발생: {e}")
            return {'action': 'unknown', 'confidence': 0.0}
    
    def add_frame(self, frame, object_id):
        """
        프레임 추가 (외부에서 호출)
        
        Args:
            frame (numpy.ndarray): 입력 프레임 (RGB)
            object_id: 객체 ID
        """
        if frame is None or self.model is None:
            return
            
        if object_id not in self.frame_buffers:
            self.frame_buffers[object_id] = []
        
        # 프레임 전처리
        try:
            # RGB -> 텐서 변환 (CPU에서 수행)
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            
            # 크기 조정 (CPU에서 수행)
            frame_tensor = torch.nn.functional.interpolate(
                frame_tensor.unsqueeze(0), 
                size=self.input_size, 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
            # 정규화 - transform은 GPU에 있을 수 있으므로 CPU에 복제하여 사용
            if hasattr(self, 'transform') and self.transform is not None:
                try:
                    # 모델이 CPU 모드면 정규화도 CPU에서 실행
                    if self.device.type == 'cpu':
                        frame_tensor = self.transform(frame_tensor)
                    else:
                        # GPU 모드면 정규화를 GPU에서 실행 후 다시 CPU로 가져옴
                        frame_tensor = self.transform(frame_tensor.to(self.device)).cpu()
                except Exception as norm_error:
                    logger.warning(f"정규화 오류, 기본 정규화 적용: {norm_error}")
                    # 기본 정규화 적용 (평균 0.5, 표준편차 0.5)
                    frame_tensor = (frame_tensor - 0.5) / 0.5
            
            # 버퍼에 추가 (항상 CPU 텐서로 저장)
            self.frame_buffers[object_id].append(frame_tensor.cpu())
            
            # 버퍼 크기 제한
            max_buffer_size = self.clip_len * 2
            if len(self.frame_buffers[object_id]) > max_buffer_size:
                self.frame_buffers[object_id] = self.frame_buffers[object_id][-max_buffer_size:]
                
        except Exception as e:
            logger.warning(f"프레임 추가 중 오류 발생: {e}")
    
    def process_frame(self, frame, bboxes, object_ids):
        """
        프레임 처리 및 행동 인식 수행
        
        Args:
            frame (numpy.ndarray): 입력 프레임 (BGR 형식)
            bboxes (list): 바운딩 박스 좌표 리스트 [(x1, y1, x2, y2), ...]
            object_ids (list): 객체 ID 리스트
            
        Returns:
            list: 각 객체별 행동 인식 결과 리스트
        """
        if frame is None or len(bboxes) == 0 or self.model is None:
            return []
        
        # BGR -> RGB 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = []
        
        # 각 객체별로 프레임 처리
        for i, (bbox, object_id) in enumerate(zip(bboxes, object_ids)):
            x1, y1, x2, y2 = bbox
            
            try:
                # 좌표가 올바른지 확인하고 보정
                x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(frame.shape[1], x2)), int(min(frame.shape[0], y2))
                
                # 바운딩 박스가 너무 작으면 건너뛰기
                if x2 <= x1 or y2 <= y1 or x2 - x1 < 10 or y2 - y1 < 10:
                    results.append({'action': 'unknown', 'confidence': 0.0, 'object_id': object_id})
                    continue
                
                # 객체 영역 추출
                person_frame = rgb_frame[y1:y2, x1:x2]
                
                # 영역이 너무 작으면 건너뛰기
                if person_frame.shape[0] < 10 or person_frame.shape[1] < 10:
                    results.append({'action': 'unknown', 'confidence': 0.0, 'object_id': object_id})
                    continue
                    
                # 프레임 추가
                self.add_frame(person_frame, object_id)
                
                # 행동 인식 결과 가져오기
                pose_data = {'object_id': object_id}
                action_result = self.recognize(pose_data)
                
                # 결과 저장
                results.append(action_result)
                
            except Exception as e:
                logger.warning(f"프레임 처리 중 오류 발생: {e} (object_id: {object_id})")
                results.append({'action': 'unknown', 'confidence': 0.0, 'object_id': object_id})
        
        return results
    
    def detect_danger_zone_violation(self, pose_data):
        """
        위험 구역 침범 여부 감지 (부모 클래스의 구현 사용)
        """
        return super().detect_danger_zone_violation(pose_data)


# create_action_recognizer 함수 수정
def create_action_recognizer():
    """
    설정에 따라 적절한 행동 인식기를 생성합니다.
    
    Returns:
        ActionRecognizer: 생성된 행동 인식기
    """
    model_type = ACTION_CONFIG['model_type']
    
    if model_type == 'rule':
        # 규칙 기반 행동 인식기 사용
        return RuleBasedActionRecognizer()
    elif model_type == 'lstm':
        return LSTMActionRecognizer(ACTION_CONFIG['model_path'])
    elif model_type == 'hybrid':
        return HybridActionRecognizer(ACTION_CONFIG['model_path'])
    elif model_type == 'pretrained':
        # 사전 학습된 모델 사용
        return PretrainedActionRecognizer(ACTION_CONFIG.get('pretrained_model_path', None))
    elif model_type == 'simple':
        # 간단한 행동 인식기 사용 (모델 파일 필요 없음)
        return SimpleActionRecognizer()
    else:
        logger.error(f"지원하지 않는 행동 인식기 유형: {model_type}")
        # 기본값으로 SimpleActionRecognizer 사용
        return SimpleActionRecognizer()