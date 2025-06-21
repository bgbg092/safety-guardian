"""
행동 패턴 분석 시스템 설정 파일

이 파일은 모든 모듈에서 사용되는 파라미터를 중앙 집중식으로 관리합니다.
각 모듈별 설정을 쉽게 변경할 수 있도록 구성되어 있습니다.
"""
import os

# 하드웨어 가속 마스터 설정 (True: GPU 사용, False: CPU 사용)
USE_GPU = True

# AWS S3 설정
AWS_CONFIG = {
    'aws_access_key_id': os.environ.get("AWS_ACCESS_KEY_ID"),
    'aws_secret_access_key': os.environ.get("AWS_SECRET_ACCESS_KEY"),
    'region_name': "ap-northeast-2",  # 서울 리전
    'bucket_name': "dklab.vision"
}

# 비디오 입력 설정
VIDEO_CONFIG = {
    'source_type': 'camera',  # 'file'에서 'camera'로 변경
    'source_path': 'data/sample_video.mp4',
    'camera_id': 0,
    'rtsp_url': 'rtsp://example.com/stream',
    'image_folder': 'data/frames',  # 이미지 프레임이 저장된 폴더 경로
    'image_pattern': '*.jpg',       # 이미지 파일 패턴 (*.jpg, *.png 등)
    'fps': 30,                      # 이미지 시퀀스를 비디오로 처리할 때의 FPS -> 10에서 30으로 변경
    'frame_width': 1920,             # 기본 프레임 너비 -> 640에서 1920으로 변경
    'frame_height': 1080,            # 기본 프레임 높이 -> 480에서 1080으로 변경
    'skip_frames': 3,               # 모든 프레임 분석을 위해 1로 변경 (원래 4) -> 2에서 3으로 변경
    'scale_coordinates': True,      # 좌표 자동 스케일링 활성화 여부
    'override_from_source': True,   # 소스에서 감지된 크기로 config 자동 업데이트 여부 -> False에서 True로 변경
    'resize_display': True,         # 화면 표시 크기 조절 가능 여부
    'initial_window_width': 1280,   # 초기 창 너비
    'initial_window_height': 720,   # 초기 창 높이
    'performance_mode': 'high_speed',  # 성능 모드 (high_speed, balanced, high_accuracy) - 높은 정확도로 변경
    'camera_width': 640,            # 카메라 캡처 너비
    'camera_height': 480,           # 카메라 캡처 높이
    'camera_fps': 30                # 카메라 캡처 FPS
}

# YOLO 객체 검출 설정
DETECTION_CONFIG = {
    'model_type': 'yolov11',  # 'yolov11', 'yolov8' (v 포함 유지)
    'model_size': 'm',  # 'n', 's', 'm', 'l', 'x' - 작은 모델로 유지 -> 'm'에서 's'로 변경
    'model_path': 'vision/models/yolo11m.pt',  # 오타(vison) 수정
    'confidence_threshold': 0.35,  # 다중 객체 탐지를 위해 임계값 완화 (0.6 → 0.35)
    'nms_threshold': 0.45,
    'device': 'cuda:0' if USE_GPU else 'cpu',  # 마스터 설정에 따라 자동 결정
    'classes': [0],  # 사람만 감지
    # 하드웨어 가속 옵션
    'use_tensorrt': USE_GPU,         # USE_GPU 설정에 따라 결정
    'use_opencv_cuda': USE_GPU,      # USE_GPU 설정에 따라 결정
    'use_half_precision': USE_GPU,   # USE_GPU 설정에 따라 결정
    'detector_type': 'yolo',      # 감지기 유형
    # COCO 데이터셋 클래스 ID:
    # 0: person, 1: bicycle, 2: car, 3: motorcycle, 4: airplane, 
    # 5: bus, 6: train, 7: truck, 8: boat, 9: traffic light,
    # 10: fire hydrant, 11: stop sign, 12: parking meter, 13: bench, 14: bird,
    # 15: cat, 16: dog, 17: horse, 18: sheep, 19: cow,
    # 20: elephant, 21: bear, 22: zebra, 23: giraffe, 24: backpack,
    # 25: umbrella, 26: handbag, 27: tie, 28: suitcase, 29: frisbee,
    # 30: skis, 31: snowboard, 32: sports ball, 33: kite, 34: baseball bat,
    # 35: baseball glove, 36: skateboard, 37: surfboard, 38: tennis racket, 39: bottle,
    # 40: wine glass, 41: cup, 42: fork, 43: knife, 44: spoon,
    # 45: bowl, 46: banana, 47: apple, 48: sandwich, 49: orange,
    # 50: broccoli, 51: carrot, 52: hot dog, 53: pizza, 54: donut,
    # 55: cake, 56: chair, 57: couch, 58: potted plant, 59: bed,
    # 60: dining table, 61: toilet, 62: tv, 63: laptop, 64: mouse,
    # 65: remote, 66: keyboard, 67: cell phone, 68: microwave, 69: oven,
    # 70: toaster, 71: sink, 72: refrigerator, 73: book, 74: clock,
    # 75: vase, 76: scissors, 77: teddy bear, 78: hair dryer, 79: toothbrush
    # 
    # 예시:
    # classes: [0]        - 사람만 감지
    # classes: [2, 5, 7]  - 자동차, 버스, 트럭만 감지
    # classes: [0, 15, 16] - 사람, 고양이, 개만 감지
}

# 자세 추정 설정
POSE_CONFIG = {
    'model_type': 'yolov11',  # 'yolo11'에서 'yolov11'로 복원 (v 포함)
    'model_size': 'm',  # 'n'에서 's'로 변경 (yolo11s-pose.pt 사용 시) -> 'm'에서 's'로 변경
    'model_path': 'vision/models/yolo11m-pose.pt',  # 실제 모델 경로로 수정 -> yolo11m-pose.pt에서 yolo11s-pose.pt로 변경
    'device': 'cuda:0' if USE_GPU else 'cpu',
    'min_confidence': 0.3, # 필요에 따라 조절
    'tracking': True,
    # 하드웨어 가속 옵션 추가
    'use_tensorrt': USE_GPU,  # USE_GPU 마스터 설정 따름s
    'use_half_precision': USE_GPU,  # USE_GPU 마스터 설정 따름
    'imgsz': 416,             # 모델 변환 시 사용할 입력 이미지 크기 -> 640에서 416으로 변경
}

# 행동 인식 설정
ACTION_CONFIG = {
    'model_type': 'simple',  # 'rule', 'hybrid', 'lstm', 'pretrained', 'simple'
    'model_path': 'models/coco_lstm_custom_best.onnx',  # COCO 키포인트 기반 LSTM 모델
    'sequence_length': 30,  # COCO 키포인트 시퀀스 길이
    'classes': ['normal', 'sitting', 'standing', 'walking', 'other'],  # 기본 클래스
    'threshold': 0.8,  # 높은 임계값으로 설정하여 더 확실한 행동만 감지
    'batch_processing': True,  # 배치 처리 활성화
    'feature_type': 'combined',  # combined:위치+각도+속도 특징 사용 'position': 위치만
    'temporal_window': 15,  # 시간 윈도우 크기
    'use_gpu': USE_GPU,  # 마스터 설정에 따라 자동 결정
    'rule_config': {
        'falling_threshold': 0.6,  # 넘어짐 감지를 위한 높이 비율 임계값(더 엄격하게)
        'stillness_frames': 5,    # 정지 상태 감지를 위한 프레임 수(더 오래 정지해야 sitting으로 판단)
        'standing_min_height_ratio': 1.0,  # standing 감지를 위한 최소 높이 비율 (신장/너비)
        'min_keypoint_confidence': 0.1,  # 키포인트 신뢰도 최소값 (낮춰서 정적 객체도 탐지)
        'detection_consistency_frames': 5  # 정적 객체 감지를 위한 최소 연속 프레임 수
    },
    # 사전 학습된 모델 관련 설정
    'pretrained_model_path': 'models/action_recognition_kinetics.onnx',  # ONNX 모델로 변경
    'pretrained_input_size': (112, 112),  # 입력 이미지 크기
    'pretrained_clip_len': 16,  # 클립 길이 (프레임 수)
    'pretrained_classes': [  # 우리 시스템에 맞게 클래스 목록 간소화
        'normal', 'sitting', 'standing', 'walking', 'falling'
    ],
    'download_pretrained': False,  # 사전 학습된 모델 자동 다운로드 비활성화
    'pretrained_device': 'cuda' if USE_GPU else 'cpu',  # 마스터 설정에 따라 자동 결정
    
    # ONNX 관련 설정 추가
    'use_onnx': True,  # ONNX 런타임 사용 여부
    'onnx_execution_provider': 'CUDAExecutionProvider' if USE_GPU else 'CPUExecutionProvider',  # 마스터 설정에 따라 자동 결정
    'onnx_optimization_level': 99  # ONNX 최적화 레벨 (0-99)
}

# 시각화 설정
VISUALIZATION_CONFIG = {
    'show_bbox': True,                  # 바운딩 박스 표시 여부 -> False로 변경
    'show_skeleton': True,              # 골격 표시 여부 -> False로 변경
    'show_action_label': True,          # 행동 라벨 표시 여부 -> False로 변경
    'show_labels': False,                # 라벨 표시 여부 (추가) -> False로 변경
    'show_confidence': False,            # 신뢰도 표시 여부 (추가) -> False로 변경
    'show_danger_zones': True,          # 위험 구역 표시 여부 (추가) -> False로 변경
    'font_size': 1.0,                   # 글자 크기
    'line_thickness': 2,                # 선 두께
    'save_output': True,                # 출력 저장 여부
    'output_path': 'results/output.mp4', # 출력 파일 경로
    'display': True,                    # 실시간 표시 여부 (FPS 확인을 위해 True 유지, 필요시 False로)
    'window_name': 'Behavior Analysis',  # 창 이름
    'resizable': True,                  # 창 크기 조절 가능 여부
    'show_fps': True,                   # FPS 표시 여부 (로그로도 확인 가능하면 False 고려)
    'show_controls_help': False,         # 컨트롤 도움말 표시 여부 -> False로 변경
    'zoom_factor': 1.0,                 # 초기 확대/축소 비율
    'min_zoom': 0.5,                    # 최소 확대/축소 비율
    'max_zoom': 3.0                     # 최대 확대/축소 비율
}

# 알림 설정
ALERT_CONFIG = {
    'enabled': True,
    'alert_actions': ['falling'],  # 'fighting' 제거 - 알림 발생시킬 행동 목록
    'min_consecutive_frames': 2,  # 연속으로 몇 프레임 이상 감지되어야 알림을 발생시킬지
    'cooldown_seconds': 60,  # 공통 쿨다운 시간(초) - 이전 버전과의 호환성을 위해 유지 (사용하지 않음)
    'method_cooldowns': {      # 각 알림 방법별 쿨다운 시간(초)
        'console': 30,         # 콘솔 알림은 30초마다
        'file': 60,            # 파일 알림은 60초마다
        'email': 300,          # 이메일 알림은 5분마다
        'api': 120,            # API 알림은 2분마다
        'image': 60,           # 이미지 저장은 60초마다
        'sound': 30,           # 소리 알림은 30초마다
        'kakao': 300           # 카카오 메시지 알림은 5분마다
    },
    'alert_methods': ['console', 'sound', 'file'],  # 'console', 'file', 'email', 'api', 'sound', 'kakao'
    'alert_file': 'results/alerts.log',
    'email_config': {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'sender_email': 'ahhm1004@gmail.com',
        'receiver_email': 'ahhm1004@gmail.com',
        'password': 'pvyl xgha hwru cysx',
        'message_template': {
            'subject': '행동 패턴 분석 시스템 경고',
            'body': '안녕하세요,\n\n위험 감지 분석 시스템에서 다음과 같은 경고가 발생했습니다:\n\n{message}\n\n시간: {timestamp}\n위치: {location}\n신뢰도: {confidence}\n\n이 메일은 자동으로 생성된 알림입니다.\n행동 패턴 분석 시스템'
        }
    },
    'api_config': {
        'url': 'https://example.com/api/alerts',
        'headers': {'Content-Type': 'application/json'},
        'auth_token': 'your-auth-token'
    },
    'sound_config': {
        'volume': 80,  # 소리 크기 (1-100)
        'sound_file': 'alert_sounds/alert.wav',  # 알림음 파일 경로 (없을 경우 기본음 사용)
        'default_sound': True,  # 기본음 사용 여부 (sound_file이 없거나 읽기 실패 시)
        'duration': 1,  # 알림음 재생 시간(초)
        'repeat': 2     # 반복 횟수
    },
    'kakao_config': {
        'api_key': 'your-kakao-api-key',
        'template_id': 'your-template-id',
        'receiver_ids': ['receiver-id-1', 'receiver-id-2'],
        'message_template': {
            'text': '행동 패턴 분석 시스템 경고: {message}\n시간: {timestamp}\n위치: {location}\n신뢰도: {confidence}',
            'button_text': '시스템 확인하기',
            'button_url': 'https://your-monitoring-system.com'
        }
    }
}

# 시스템 일반 설정
SYSTEM_CONFIG = {
    'log_level': 'INFO',  # 'DEBUG'에서 'INFO'로 변경하여 디버그 메시지 숨기기
    'log_file': 'logs/system.log',
    'save_interval': 100,
    'batch_size': 8,
    'num_workers': 8,
    'enable_threading': True,
    'analysis_thread_count': 6,
    'gpu_memory_fraction': 0.8 if USE_GPU else 0.0,  # 마스터 설정에 따라 자동 결정
    'enable_performance_logging': True,
    'frame_resize_factor': 1.0, # 다중 객체 탐지를 위해 원본 해상도 유지 (0.4 → 1.0)
    'skip_frames': 1,
    'target_fps': 30,
    # 하드웨어 가속 설정
    'cuda_stream_per_device': USE_GPU,  # 마스터 설정에 따라 자동 결정
    'cudnn_benchmark': USE_GPU,         # 마스터 설정에 따라 자동 결정
    'optimize_cuda_calls': USE_GPU,     # 마스터 설정에 따라 자동 결정
    'memory_management': 'balanced' if USE_GPU else 'conservative'  # 마스터 설정에 따라 자동 결정
}

# DANGER_ZONE_CONFIG_START
# 위험 구역 설정
DANGER_ZONE_CONFIG = {
    'enabled': True,
    'zones': [
        {"name": '빌라1', "coordinates": [(436, 244), (211, 500), (284, 502), (468, 254), (438, 242)]},
        {"name": '빌라2', "coordinates": [(697, 323), (597, 502), (680, 504), (722, 424), (703, 417), (743, 342), (698, 325)]},
    ],
    'original_frame_width': 904,    # 위험 구역 설정 시 사용한 원본 이미지 너비
    'original_frame_height': 510,   # 위험 구역 설정 시 사용한 원본 이미지 높이
    'monitored_keypoints': ['ankles', 'hips'],  # 감시할 신체 부위 (발목, 허리)
    'alert_message': 'DANGER ZONE ALERT',  # 영문으로 변경하여 인코딩 문제 해결
    'alert_color': (0, 0, 255),  # 빨간색 (BGR)
    'zone_color': (0, 0, 255),   # 빨간색 (BGR)
    'zone_opacity': 0.3          # 투명도 (0~1)
}
# DANGER_ZONE_CONFIG_END

# 하드웨어 설정 로깅
import logging
logger = logging.getLogger(__name__)
if USE_GPU:
    logger.info("하드웨어 설정: GPU 모드 활성화")
else:
    logger.info("하드웨어 설정: CPU 모드 활성화")

def toggle_gpu_mode(enable_gpu=None):
    """
    GPU 모드를 전환하는 함수
    
    Args:
        enable_gpu (bool, optional): GPU 활성화 여부. None이면 현재 설정의 반대로 전환
        
    Returns:
        bool: 변경된 GPU 모드 상태
    """
    global USE_GPU, DETECTION_CONFIG, POSE_CONFIG, ACTION_CONFIG, SYSTEM_CONFIG
    
    # 파라미터가 없으면 현재 모드의 반대로 전환
    if enable_gpu is None:
        enable_gpu = not USE_GPU
    
    # 이미 같은 모드라면 변경하지 않음
    if USE_GPU == enable_gpu:
        return USE_GPU
    
    # USE_GPU 설정 변경
    USE_GPU = enable_gpu
    
    # 관련 설정 업데이트
    DETECTION_CONFIG['device'] = 'cuda:0' if USE_GPU else 'cpu'
    DETECTION_CONFIG['use_tensorrt'] = USE_GPU
    DETECTION_CONFIG['use_opencv_cuda'] = USE_GPU
    DETECTION_CONFIG['use_half_precision'] = USE_GPU
    
    POSE_CONFIG['device'] = 'cuda:0' if USE_GPU else 'cpu'
    
    ACTION_CONFIG['use_gpu'] = USE_GPU
    
    SYSTEM_CONFIG['gpu_memory_fraction'] = 0.8 if USE_GPU else 0.0
    SYSTEM_CONFIG['cuda_stream_per_device'] = USE_GPU
    SYSTEM_CONFIG['cudnn_benchmark'] = USE_GPU
    SYSTEM_CONFIG['optimize_cuda_calls'] = USE_GPU
    SYSTEM_CONFIG['memory_management'] = 'balanced' if USE_GPU else 'conservative'
    
    # 로그 출력
    if USE_GPU:
        logger.info("하드웨어 설정 변경: GPU 모드 활성화")
    else:
        logger.info("하드웨어 설정 변경: CPU 모드 활성화")
    
    return USE_GPU

