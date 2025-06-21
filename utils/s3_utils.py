"""
S3 버킷 관련 유틸리티 함수들

데이터셋과 모델 파일을 S3에서 다운로드하고 업로드하는 함수들을 제공합니다.
"""

import os
import boto3
from botocore.exceptions import ClientError
import logging
from pathlib import Path
import sys

# 상위 디렉토리 추가해서 config.py 접근 가능하게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AWS_CONFIG

logger = logging.getLogger(__name__)

def get_s3_client():
    """
    AWS S3 클라이언트를 생성합니다.
    
    Returns:
        boto3.client: S3 클라이언트 객체
    """
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_CONFIG['aws_access_key_id'],
            aws_secret_access_key=AWS_CONFIG['aws_secret_access_key'],
            region_name=AWS_CONFIG['region_name']
        )
        return s3_client
    except Exception as e:
        logger.error(f"S3 클라이언트 생성 실패: {e}")
        return None

def download_file_from_s3(s3_path, local_path):
    """
    S3에서 파일을 다운로드합니다.
    
    Args:
        s3_path (str): S3 내 파일 경로
        local_path (str): 로컬 저장 경로
        
    Returns:
        bool: 다운로드 성공 여부
    """
    s3_client = get_s3_client()
    if not s3_client:
        return False
        
    try:
        # 로컬 디렉토리가 없다면 생성
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        logger.info(f"S3에서 다운로드 중: {s3_path} -> {local_path}")
        s3_client.download_file(
            AWS_CONFIG['bucket_name'],
            s3_path,
            local_path
        )
        logger.info(f"다운로드 완료: {local_path}")
        return True
    except ClientError as e:
        logger.error(f"S3 다운로드 실패: {e}")
        return False

def upload_file_to_s3(local_path, s3_path):
    """
    로컬 파일을 S3에 업로드합니다.
    
    Args:
        local_path (str): 업로드할 로컬 파일 경로
        s3_path (str): S3 내 저장 경로
        
    Returns:
        bool: 업로드 성공 여부
    """
    s3_client = get_s3_client()
    if not s3_client:
        return False
        
    try:
        logger.info(f"S3에 업로드 중: {local_path} -> {s3_path}")
        s3_client.upload_file(
            local_path,
            AWS_CONFIG['bucket_name'],
            s3_path
        )
        logger.info(f"업로드 완료: {s3_path}")
        return True
    except ClientError as e:
        logger.error(f"S3 업로드 실패: {e}")
        return False

def list_s3_files(prefix=""):
    """
    S3 버킷 내 파일 목록을 가져옵니다.
    
    Args:
        prefix (str): 검색할 디렉토리 접두사
        
    Returns:
        list: 파일 경로 목록
    """
    s3_client = get_s3_client()
    if not s3_client:
        return []
        
    try:
        response = s3_client.list_objects_v2(
            Bucket=AWS_CONFIG['bucket_name'],
            Prefix=prefix
        )
        
        if 'Contents' not in response:
            return []
            
        return [item['Key'] for item in response['Contents']]
    except ClientError as e:
        logger.error(f"S3 목록 조회 실패: {e}")
        return []

def download_dataset(dataset_name, target_dir="data"):
    """
    S3에서 데이터셋을 다운로드합니다.
    
    Args:
        dataset_name (str): 데이터셋 이름 (S3 내 디렉토리 이름)
        target_dir (str): 로컬 저장 디렉토리
        
    Returns:
        bool: 다운로드 성공 여부
    """
    s3_files = list_s3_files(f"datasets/{dataset_name}/")
    if not s3_files:
        logger.error(f"데이터셋을 찾을 수 없음: {dataset_name}")
        return False
        
    success = True
    for s3_path in s3_files:
        # S3 경로에서 데이터셋 이름 이후 부분만 추출
        relative_path = s3_path.split(f"datasets/{dataset_name}/", 1)[1]
        local_path = os.path.join(target_dir, dataset_name, relative_path)
        
        if not download_file_from_s3(s3_path, local_path):
            success = False
            
    return success

def download_model(model_name, target_dir="models"):
    """
    S3에서 모델을 다운로드합니다.
    
    Args:
        model_name (str): 모델 파일 이름
        target_dir (str): 로컬 저장 디렉토리
        
    Returns:
        bool: 다운로드 성공 여부
    """
    s3_path = f"models/{model_name}"
    local_path = os.path.join(target_dir, model_name)
    
    # 이미 로컬에 파일이 존재하는지 확인
    if os.path.exists(local_path):
        logger.info(f"모델 파일이 이미 존재합니다: {local_path}")
        return True
    
    # S3에서 다운로드 시도
    success = download_file_from_s3(s3_path, local_path)
    
    # 다운로드 실패 시 로컬에서 모델 생성 시도 (YOLO11인 경우)
    if not success and "yolov11" in model_name.lower():
        logger.info(f"S3 다운로드 실패, 로컬에서 {model_name} 모델 생성 시도")
        try:
            # 모델 크기 추출 ('yolo11s.pt' -> 's')
            model_size = model_name.lower().replace("yolov11", "").replace(".pt", "")
            if not model_size:
                model_size = "s"  # 기본 크기
                
            # 모델 생성기 임포트 및 모델 생성
            from modules.object_detection.model_generator import create_yolo_model
            generation_success = create_yolo_model("yolov11", model_size, local_path)
            
            if generation_success:
                logger.info(f"모델 생성 성공: {local_path}")
                return True
            else:
                logger.error(f"모델 생성 실패: {local_path}")
                return False
        except Exception as e:
            logger.error(f"모델 생성 중 오류 발생: {e}")
            return False
    
    return success 