"""
모델 다운로드 유틸리티

LSTM 기반 행동 인식 모델을 다운로드하는 스크립트입니다.
"""

import os
import sys
import requests
import logging
import torch
import numpy as np
import onnx
from onnxruntime import InferenceSession

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

def create_simple_lstm_model():
    """
    간단한 LSTM 기반 행동 인식 모델 생성
    
    Returns:
        torch.nn.Module: 생성된 모델
    """
    try:
        import torch
        import torch.nn as nn

        # 입력 특성 수: 키포인트 좌표 (x, y, confidence) * 17개 키포인트 = 51
        # 추가 특성 (각도, 속도 등) = 20
        input_size = 71
        
        # 은닉층 크기
        hidden_size = 128
        
        # 행동 클래스 수 (walking, sitting, falling, fighting, normal, running, waving, crouching, jumping, eating)
        num_classes = 10
        
        # 시퀀스 길이 (프레임 수)
        seq_length = 20
        
        # LSTM 모델 정의
        class ActionLSTM(nn.Module):
            def __init__(self):
                super(ActionLSTM, self).__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2, dropout=0.2)
                self.fc1 = nn.Linear(hidden_size, 64)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.3)
                self.fc2 = nn.Linear(64, num_classes)
                
            def forward(self, x):
                # x 형태: (batch_size, seq_length, input_size)
                lstm_out, _ = self.lstm(x)
                # 마지막 시간 단계의 출력만 사용
                last_out = lstm_out[:, -1, :]
                x = self.fc1(last_out)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.fc2(x)
                return x
                
        # 모델 생성
        model = ActionLSTM()
        logger.info("LSTM 모델 생성 완료")
        return model
        
    except Exception as e:
        logger.error(f"모델 생성 중 오류 발생: {e}")
        return None

def convert_to_onnx(model, output_path):
    """
    PyTorch 모델을 ONNX 형식으로 변환
    
    Args:
        model (torch.nn.Module): 변환할 PyTorch 모델
        output_path (str): 출력 ONNX 파일 경로
        
    Returns:
        bool: 변환 성공 여부
    """
    try:
        import torch
        
        # 모델을 평가 모드로 설정
        model.eval()
        
        # 더미 입력 생성 (배치 크기 1, 시퀀스 길이 20, 입력 크기 71)
        dummy_input = torch.randn(1, 20, 71)
        
        # ONNX 내보내기
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        # ONNX 모델 확인
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        logger.info(f"ONNX 모델 변환 완료: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"ONNX 변환 중 오류 발생: {e}")
        return False

def test_onnx_model(model_path):
    """
    ONNX 모델 테스트
    
    Args:
        model_path (str): ONNX 모델 파일 경로
    
    Returns:
        bool: 테스트 성공 여부
    """
    try:
        # ONNX 런타임 세션 생성
        session = InferenceSession(model_path)
        
        # 입력 이름 가져오기
        input_name = session.get_inputs()[0].name
        
        # 더미 입력 생성
        dummy_input = np.random.randn(1, 20, 71).astype(np.float32)
        
        # 추론 실행
        result = session.run(None, {input_name: dummy_input})
        
        # 결과 검증
        if result[0].shape == (1, 10):  # 배치 크기 1, 클래스 수 10
            logger.info("ONNX 모델 테스트 성공")
            return True
        else:
            logger.error(f"모델 출력 형태가 예상과 다름: {result[0].shape}")
            return False
        
    except Exception as e:
        logger.error(f"ONNX 모델 테스트 중 오류 발생: {e}")
        return False

def download_lstm_model():
    """
    LSTM 행동 인식 모델 다운로드 또는 생성
    
    Returns:
        str: 다운로드된 모델 경로
    """
    # 모델 경로 설정
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'lstm_action.onnx')
    
    # 모델 파일이 이미 있는지 확인
    if os.path.exists(model_path):
        logger.info(f"LSTM 모델이 이미 존재합니다: {model_path}")
        return model_path
    
    logger.info("LSTM 행동 인식 모델 생성 시작...")
    
    # 간단한 LSTM 모델 생성
    model = create_simple_lstm_model()
    if model is None:
        logger.error("LSTM 모델 생성 실패")
        return None
    
    # ONNX로 변환
    if not convert_to_onnx(model, model_path):
        logger.error("ONNX 변환 실패")
        return None
    
    # ONNX 모델 테스트
    if not test_onnx_model(model_path):
        logger.error("ONNX 모델 테스트 실패")
        return None
    
    logger.info(f"LSTM 행동 인식 모델 준비 완료: {model_path}")
    return model_path

if __name__ == "__main__":
    download_lstm_model() 