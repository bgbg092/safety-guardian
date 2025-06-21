"""
비디오 입력(Video Input) 모듈

다양한 소스(파일, 카메라, RTSP 등)에서 비디오 프레임을 읽어오는 기능을 제공합니다.
"""

import os
import sys
import cv2
import glob
import time
import logging
import threading
import queue
from pathlib import Path
from abc import ABC, abstractmethod

# 상위 디렉토리 추가해서 config.py 접근 가능하게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import VIDEO_CONFIG, SYSTEM_CONFIG

logger = logging.getLogger(__name__)

class VideoLoader(ABC):
    """비디오 로더 추상 클래스"""
    
    def __init__(self):
        self.frame_size = (
            VIDEO_CONFIG.get('frame_width', 640),
            VIDEO_CONFIG.get('frame_height', 480)
        )
        self.fps = VIDEO_CONFIG.get('fps', 30)
        self.total_frames = 0
        self.current_frame = 0
        
        # 버퍼링 설정
        self.use_buffer = SYSTEM_CONFIG.get('enable_threading', False)
        self.buffer_size = 30
        self.frame_buffer = queue.Queue(maxsize=self.buffer_size)
        self.buffer_thread = None
        self.buffer_running = False
        
    @abstractmethod
    def open(self):
        """
        비디오 소스 열기
        
        Returns:
            bool: 성공 여부
        """
        pass
        
    @abstractmethod
    def _read_next_frame(self):
        """
        다음 프레임 읽기 (내부용)
        
        Returns:
            tuple: (성공 여부, 프레임 이미지)
        """
        pass
        
    def read_frame(self):
        """
        다음 프레임 읽기
        
        Returns:
            tuple: (성공 여부, 프레임 이미지)
        """
        if self.use_buffer:
            # 버퍼링 스레드가 이미 종료되었고 큐도 비어있으면, 더 이상 읽을 프레임 없음
            if not self.buffer_running and self.frame_buffer.empty():
                # logger.debug("Buffer and thread stopped, no more frames.")
                return False, None

            try:
                # 버퍼에서 프레임 가져오기 (블로킹, 타임아웃 1초)
                # 버퍼링 스레드가 종료 시 (False, None)을 큐에 넣으므로, 이를 통해 종료 감지
                ret, frame = self.frame_buffer.get(block=True, timeout=1.0)
                if ret: # 유효한 프레임 (True, frame_data)
                    self.current_frame += 1
                # (False, None)이 들어오면 ret이 False이므로 current_frame 증가 안 함
                return ret, frame
            except queue.Empty:
                # 1초 타임아웃 발생 (정상적인 종료가 아닌 예외적 상황 간주)
                logger.warning("Frame buffer get timed out. Assuming no more frames or buffering issue.")
                self.buffer_running = False # 문제가 발생했을 수 있으므로 버퍼링 중단 플래그 설정
                return False, None
        else:
            # 버퍼 미사용 시 직접 읽기
            ret, frame = self._read_next_frame()
            if ret: # 유효한 프레임일 때만 카운트
                self.current_frame += 1
            return ret, frame
    
    def _buffer_frames(self):
        """프레임 버퍼링 스레드 함수"""
        try:
            while self.buffer_running:
                if not self.frame_buffer.full():
                    ret, frame = self._read_next_frame()
                    if ret:
                        self.frame_buffer.put((ret, frame))
                    else:
                        # 더 이상 읽을 프레임이 없으면 버퍼링 중단
                        self.buffer_running = False # 루프 종료 조건
                        break 
                else:
                    # 버퍼가 가득 차면 아주 짧게 대기
                    time.sleep(0.001) # 0.01에서 0.001로 변경
        except Exception as e:
            logger.error(f"Buffering thread encountered an error: {e}")
            self.buffer_running = False # 예외 발생 시 스레드 중지
        finally:
            # 루프가 정상적으로 끝나거나(소스 끝), buffer_running이 False로 설정되거나, 예외 발생 시
            # 항상 종료 신호(sentinel)를 큐에 넣도록 보장
            # 큐가 가득 차서 put이 블록될 수 있으므로, try-except 또는 non-blocking put 고려 가능
            # 하지만 maxsize만큼 대기하므로, 일반적으로는 문제 없음.
            # 정말 확실하게 하려면, 큐가 비워질 때까지 기다리거나, 별도 플래그 사용.
            # 여기서는 기존 로직처럼 큐에 공간이 있으면 넣는 것으로 유지.
            try:
                self.frame_buffer.put((False, None), block=False) # non-blocking으로 시도
                logger.info("Buffering thread finished and put sentinel.")
            except queue.Full:
                logger.warning("Buffering thread finished, but queue was full. Sentinel not added. Main thread might time out.")
            except Exception as e_finally: # 혹시 모를 put에서의 다른 예외
                logger.error(f"Error putting sentinel in finally block: {e_finally}")

    def _start_buffering(self):
        """프레임 버퍼링 시작"""
        if self.use_buffer and not self.buffer_running:
            self.buffer_running = True
            self.buffer_thread = threading.Thread(target=self._buffer_frames)
            self.buffer_thread.daemon = True
            self.buffer_thread.start()
            logger.info("프레임 버퍼링을 시작합니다.")
        
    @abstractmethod
    def release(self):
        """자원 해제"""
        # 버퍼링 중단
        if self.buffer_running:
            self.buffer_running = False
            if self.buffer_thread:
                self.buffer_thread.join(timeout=1.0)
            
            # 버퍼 비우기
            while not self.frame_buffer.empty():
                try:
                    self.frame_buffer.get_nowait()
                except queue.Empty:
                    break

class FileVideoLoader(VideoLoader):
    """파일 기반 비디오 로더"""
    
    def __init__(self, file_path):
        """
        Args:
            file_path (str): 비디오 파일 경로
        """
        super().__init__()
        self.file_path = file_path
        self.cap = None
        
    def open(self):
        """
        비디오 파일 열기
        
        Returns:
            bool: 성공 여부
        """
        try:
            self.cap = cv2.VideoCapture(self.file_path)
            
            if not self.cap.isOpened():
                logger.error(f"비디오 파일을 열 수 없음: {self.file_path}")
                return False
                
            # 비디오 속성 설정
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_size = (width, height)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"비디오 파일 열기 성공: {self.file_path}")
            logger.info(f"FPS: {self.fps}, 크기: {self.frame_size}, 총 프레임: {self.total_frames}")
            
            # 버퍼링 시작
            self._start_buffering()
            
            return True
            
        except Exception as e:
            logger.error(f"비디오 파일 열기 실패: {e}")
            return False
            
    def _read_next_frame(self):
        """
        다음 프레임 읽기
        
        Returns:
            tuple: (성공 여부, 프레임 이미지)
        """
        if self.cap is None:
            return False, None
            
        ret, frame = self.cap.read()
        return ret, frame
        
    def release(self):
        """자원 해제"""
        super().release()
        if self.cap:
            self.cap.release()
            self.cap = None

class CameraVideoLoader(VideoLoader):
    """카메라 기반 비디오 로더"""
    
    def __init__(self, camera_id=0):
        """
        Args:
            camera_id (int): 카메라 ID
        """
        super().__init__()
        self.camera_id = camera_id
        self.cap = None
        
        # 소스 타입 및 관련 속성 설정
        self.source_type = VIDEO_CONFIG.get('source_type', 'camera')
        self.source_path = VIDEO_CONFIG.get('source_path', '')
        self.rtsp_url = VIDEO_CONFIG.get('rtsp_url', '')
        self.image_folder = VIDEO_CONFIG.get('image_folder', '')
        self.image_pattern = VIDEO_CONFIG.get('image_pattern', '*.jpg')
        
        # 프레임 크기와 FPS 설정
        self._frame_width = VIDEO_CONFIG.get('frame_width', 640)
        self._frame_height = VIDEO_CONFIG.get('frame_height', 480)
        self._fps = VIDEO_CONFIG.get('fps', 30)
        
    def open(self):
        """
        비디오 소스 열기
        
        Returns:
            bool: 비디오 소스 열기 성공 여부
        """
        try:
            # 소스 타입에 따라 다른 방식으로 열기
            if self.source_type == 'file':
                # 비디오 파일
                self.cap = cv2.VideoCapture(self.source_path)
                if not self.cap.isOpened():
                    logger.error(f"비디오 파일을 열 수 없습니다: {self.source_path}")
                    return False
                    
                # 비디오 정보 가져오기
                self._fps = self.cap.get(cv2.CAP_PROP_FPS)
                self._frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self._frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                logger.info(f"비디오 파일 열기 성공: {self.source_path}")
                logger.info(f"FPS: {self._fps}, 크기: ({self._frame_width}, {self._frame_height})")
                
            elif self.source_type == 'camera':
                # 카메라
                self.cap = cv2.VideoCapture(self.camera_id)
                if not self.cap.isOpened():
                    logger.error(f"카메라를 열 수 없습니다: {self.camera_id}")
                    return False
                
                # 명시적 카메라 속성 설정 (캡처 크기, FPS)
                if 'camera_width' in VIDEO_CONFIG and 'camera_height' in VIDEO_CONFIG:
                    width = VIDEO_CONFIG['camera_width']
                    height = VIDEO_CONFIG['camera_height']
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    logger.info(f"카메라 해상도 설정: {width}x{height}")
                
                if 'camera_fps' in VIDEO_CONFIG:
                    fps = VIDEO_CONFIG['camera_fps']
                    self.cap.set(cv2.CAP_PROP_FPS, fps)
                    logger.info(f"카메라 FPS 설정: {fps}")
                
                # 프레임 버퍼 초기화 및 최적화
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 최소화
                
                # 속성 확인 (설정 후)
                self._fps = self.cap.get(cv2.CAP_PROP_FPS)
                self._frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self._frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # 몇 개의 프레임을 버려서 카메라 센서가 안정화되도록 함
                for _ in range(5):
                    self.cap.read()
                    
                logger.info(f"카메라 열기 성공: {self.camera_id}")
                logger.info(f"FPS: {self._fps}, 크기: ({self._frame_width}, {self._frame_height})")
                logger.info(f"프레임 버퍼링을 시작합니다.")
                
            elif self.source_type == 'rtsp':
                # RTSP 스트림
                self.cap = cv2.VideoCapture(self.rtsp_url)
                if not self.cap.isOpened():
                    logger.error(f"RTSP 스트림을 열 수 없습니다: {self.rtsp_url}")
                    return False
                    
                # RTSP 스트림 최적화 설정
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # 비디오 정보 가져오기
                self._fps = self.cap.get(cv2.CAP_PROP_FPS)
                self._frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self._frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                logger.info(f"RTSP 스트림 열기 성공: {self.rtsp_url}")
                logger.info(f"FPS: {self._fps}, 크기: ({self._frame_width}, {self._frame_height})")
                
            elif self.source_type == 'image_folder':
                # 이미지 폴더
                import glob
                
                # 이미지 파일 목록 가져오기
                self.image_files = sorted(glob.glob(os.path.join(self.image_folder, self.image_pattern)))
                if not self.image_files:
                    logger.error(f"이미지 파일을 찾을 수 없습니다: {self.image_folder}/{self.image_pattern}")
                    return False
                    
                # 첫 번째 이미지로 프레임 크기 설정
                first_image = cv2.imread(self.image_files[0])
                if first_image is None:
                    logger.error(f"이미지 파일을 읽을 수 없습니다: {self.image_files[0]}")
                    return False
                    
                self._frame_height, self._frame_width = first_image.shape[:2]
                self._fps = VIDEO_CONFIG['fps']  # Config에서 FPS 값 가져오기
                self.current_image_index = 0
                
                logger.info(f"이미지 폴더 열기 성공: {self.image_folder}, 총 {len(self.image_files)}개 이미지")
                logger.info(f"FPS: {self._fps}, 크기: ({self._frame_width}, {self._frame_height})")
                
            else:
                logger.error(f"지원하지 않는 소스 타입: {self.source_type}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"비디오 소스 열기 중 예외 발생: {e}")
            return False
            
    def _read_next_frame(self):
        """
        다음 프레임 읽기
        
        Returns:
            tuple: (성공 여부, 프레임 이미지)
        """
        if self.cap is None:
            return False, None
            
        ret, frame = self.cap.read()
        return ret, frame
        
    def release(self):
        """자원 해제"""
        super().release()
        if self.cap:
            self.cap.release()
            self.cap = None

class RTSPVideoLoader(VideoLoader):
    """RTSP 스트림 기반 비디오 로더"""
    
    def __init__(self, rtsp_url):
        """
        Args:
            rtsp_url (str): RTSP 스트림 URL
        """
        super().__init__()
        self.rtsp_url = rtsp_url
        self.cap = None
        
    def open(self):
        """
        RTSP 스트림 열기
        
        Returns:
            bool: 성공 여부
        """
        try:
            # RTSP 최적화 옵션 설정
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
            
            # GStreamer 파이프라인 설정 (옵션)
            use_gstreamer = False
            if use_gstreamer:
                gst_str = (
                    f"rtspsrc location={self.rtsp_url} latency=0 ! "
                    "rtph264depay ! h264parse ! avdec_h264 ! "
                    "videoconvert ! appsink"
                )
                self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
            else:
                self.cap = cv2.VideoCapture(self.rtsp_url)
                # 버퍼 크기 최소화 (지연 감소)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not self.cap.isOpened():
                logger.error(f"RTSP 스트림을 열 수 없음: {self.rtsp_url}")
                return False
                
            # 실제 적용된 해상도 확인
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_size = (width, height)
            
            # 프레임 레이트 확인
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30  # 기본값
                
            logger.info(f"RTSP 스트림 열기 성공: {self.rtsp_url}")
            logger.info(f"FPS: {self.fps}, 크기: {self.frame_size}")
            
            # 버퍼링 시작
            self._start_buffering()
            
            return True
            
        except Exception as e:
            logger.error(f"RTSP 스트림 열기 실패: {e}")
            return False
            
    def _read_next_frame(self):
        """
        다음 프레임 읽기
        
        Returns:
            tuple: (성공 여부, 프레임 이미지)
        """
        if self.cap is None:
            return False, None
            
        # RTSP 스트림에서 최신 프레임을 가져오기 위해 오래된 프레임 건너뛰기
        # (실시간성 향상을 위한 옵션)
        drop_old_frames = False
        if drop_old_frames:
            for _ in range(3):  # 이전 프레임 3개 건너뛰기
                self.cap.grab()
            
        ret, frame = self.cap.read()
        return ret, frame
        
    def release(self):
        """자원 해제"""
        super().release()
        if self.cap:
            self.cap.release()
            self.cap = None

class ImageFolderVideoLoader(VideoLoader):
    """이미지 폴더 기반 비디오 로더"""
    
    def __init__(self, folder_path, pattern="*.jpg"):
        """
        Args:
            folder_path (str): 이미지 폴더 경로
            pattern (str): 이미지 파일 패턴
        """
        super().__init__()
        self.folder_path = folder_path
        self.pattern = pattern
        self.image_files = []
        self.current_index = 0
        
    def open(self):
        """
        이미지 폴더 열기
        
        Returns:
            bool: 성공 여부
        """
        try:
            # 이미지 파일 목록 가져오기
            self.image_files = sorted(glob.glob(os.path.join(self.folder_path, self.pattern)))
            
            if not self.image_files:
                logger.error(f"이미지 파일을 찾을 수 없음: {self.folder_path}/{self.pattern}")
                return False
                
            # 첫 번째 이미지로 프레임 크기 설정
            sample_image = cv2.imread(self.image_files[0])
            if sample_image is None:
                logger.error(f"이미지 파일을 읽을 수 없음: {self.image_files[0]}")
                return False
                
            height, width = sample_image.shape[:2]
            self.frame_size = (width, height)
            self.total_frames = len(self.image_files)
            
            logger.info(f"이미지 폴더 열기 성공: {self.folder_path}")
            logger.info(f"이미지 개수: {self.total_frames}, 크기: {self.frame_size}")
            
            # 버퍼링 시작
            self._start_buffering()
            
            return True
            
        except Exception as e:
            logger.error(f"이미지 폴더 열기 실패: {e}")
            return False
            
    def _read_next_frame(self):
        """
        다음 이미지 읽기
        
        Returns:
            tuple: (성공 여부, 이미지)
        """
        if not self.image_files or self.current_index >= len(self.image_files):
            return False, None
            
        image_path = self.image_files[self.current_index]
        image = cv2.imread(image_path)
        
        if image is None:
            logger.warning(f"이미지 파일을 읽을 수 없음: {image_path}")
            self.current_index += 1
            return self._read_next_frame()
            
        self.current_index += 1
        return True, image
        
    def release(self):
        """자원 해제"""
        super().release()
        self.image_files = []
        self.current_index = 0

def create_video_loader():
    """
    설정에 따라 적절한 비디오 로더를 생성합니다.
    
    Returns:
        VideoLoader: 생성된 비디오 로더
    """
    source_type = VIDEO_CONFIG['source_type']
    
    if source_type == 'file':
        return FileVideoLoader(VIDEO_CONFIG['source_path'])
    elif source_type == 'camera':
        camera_loader = CameraVideoLoader(VIDEO_CONFIG['camera_id'])
        # 소스 타입을 명시적으로 camera로 설정
        camera_loader.source_type = 'camera'
        return camera_loader
    elif source_type == 'rtsp':
        return RTSPVideoLoader(VIDEO_CONFIG['rtsp_url'])
    elif source_type == 'image_folder':
        return ImageFolderVideoLoader(
            VIDEO_CONFIG['image_folder'],
            VIDEO_CONFIG['image_pattern']
        )
    else:
        logger.error(f"지원하지 않는 비디오 소스 유형: {source_type}")
        return None 