"""
시각화(Visualization) 모듈

시스템의 분석 결과를 시각화하는 기능을 제공합니다.
"""

import os
import sys
import cv2
import numpy as np
import logging
import time
from pathlib import Path
import threading
from datetime import datetime
from PIL import ImageFont, ImageDraw, Image

# 상위 디렉토리 추가해서 config.py 접근 가능하게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import VISUALIZATION_CONFIG, DANGER_ZONE_CONFIG, VIDEO_CONFIG, DETECTION_CONFIG

logger = logging.getLogger(__name__)

class Visualizer:
    """분석 결과 시각화 클래스"""
    
    def __init__(self, display=True, save_video=False, window_name='Behavior Analysis'):
        """
        Visualizer 초기화
        
        Args:
            display (bool): 화면 표시 여부
            save_video (bool): 비디오 저장 여부
            window_name (str): 화면 창 이름
        """
        self.display = display
        self.save_video = save_video
        self.window_name = window_name
        
        # 시각화 옵션 설정
        self.show_bbox = VISUALIZATION_CONFIG.get('show_bbox', True)
        self.show_skeleton = VISUALIZATION_CONFIG.get('show_skeleton', True)
        self.show_action_label = VISUALIZATION_CONFIG.get('show_action_label', True)
        self.show_labels = VISUALIZATION_CONFIG.get('show_labels', True)
        self.show_confidence = VISUALIZATION_CONFIG.get('show_confidence', False)
        self.show_fps = VISUALIZATION_CONFIG.get('show_fps', True)
        self.show_timestamp = VISUALIZATION_CONFIG.get('show_timestamp', True)
        self.show_danger_zones = VISUALIZATION_CONFIG.get('show_danger_zones', True)
        self.show_object_id = VISUALIZATION_CONFIG.get('show_object_id', True)
        
        # 클래스 이름 설정
        self.class_names = DETECTION_CONFIG.get('class_names', ['person'])
        
        # 위험 구역 유형 설정
        self.danger_zone_types = DANGER_ZONE_CONFIG.get('zone_types', {})
        
        # 비디오 저장 관련 변수
        self.video_writer = None
        self.video_output_path = None
        self.frame_width = 640
        self.frame_height = 480
        self.fps = 30.0
        self.codec_name = None
        self.frame_size_warning_shown = False  # 프레임 크기 경고 표시 여부
        
        # FPS 및 성능 측정 변수
        self.frame_count = 0
        self.current_fps = 0
        self.last_fps_time = 0
        self.processing_times = []
        self.max_processing_times = 30
        self.avg_frame_time = 0
        
        # 창 생성 (display가 True인 경우)
        if self.display:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # 기본 설정으로 True로 설정
        self.show_bbox = True
        self.show_skeleton = True
        self.show_action_label = True
        
        # 나머지 설정 로드
        self.font_size = VISUALIZATION_CONFIG.get('font_size', 1.0)
        self.line_thickness = VISUALIZATION_CONFIG.get('line_thickness', 2)
        self.save_output = VISUALIZATION_CONFIG.get('save_output', True)
        self.output_path = VISUALIZATION_CONFIG.get('output_path', 'results/output.mp4')
        self.display = VISUALIZATION_CONFIG.get('display', True)
        
        # 추가 시각화 설정
        self.resizable = VISUALIZATION_CONFIG.get('resizable', True)
        self.show_controls_help = VISUALIZATION_CONFIG.get('show_controls_help', True)
        
        # 확대/축소 관련 변수
        self.zoom_factor = VISUALIZATION_CONFIG.get('zoom_factor', 1.0)
        self.min_zoom = VISUALIZATION_CONFIG.get('min_zoom', 0.5)
        self.max_zoom = VISUALIZATION_CONFIG.get('max_zoom', 3.0)
        self.window_width = VIDEO_CONFIG.get('initial_window_width', 1280)
        self.window_height = VIDEO_CONFIG.get('initial_window_height', 720)
        
        # 드래그 관련 변수
        self.dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.pan_x = 0
        self.pan_y = 0
        
        # Pillow 폰트 로드
        self.pil_font = None
        self.pil_font_path = "malgun.ttf" # 기본 폰트 경로, 필요시 전체 경로로 수정
        # 예: self.pil_font_path = "C:/Windows/Fonts/malgun.ttf"
        # VISUALIZATION_CONFIG 에서 font_size 를 가져와서 Pillow 폰트 크기 설정
        # self.font_size 는 배율이므로, 기본 픽셀 크기를 정해야 함. 예: 20px
        default_pil_pixel_size = 20 
        self.pil_font_actual_pixel_size = int(default_pil_pixel_size * self.font_size)
        try:
            self.pil_font = ImageFont.truetype(self.pil_font_path, self.pil_font_actual_pixel_size)
            logger.info(f"Pillow 폰트 로드 성공: {self.pil_font_path}, 크기: {self.pil_font_actual_pixel_size}px")
        except IOError:
            logger.error(f"Pillow 폰트 파일을 찾거나 열 수 없습니다: {self.pil_font_path}. 일부 텍스트가 깨질 수 있습니다.")
        except Exception as e:
            logger.error(f"Pillow 폰트 로드 중 예기치 않은 오류 발생 ({self.pil_font_path}): {e}. 일부 텍스트가 깨질 수 있습니다.")
        
        # 행동별 색상 매핑
        self.action_colors = {
            'normal': (0, 255, 0),     # 녹색
            'walking': (0, 200, 0),    # 짙은 녹색
            'sitting': (255, 255, 0),  # 청록색
            'falling': (0, 0, 255),    # 빨간색
            'fighting': (0, 0, 255),   # 빨간색
            'other': (150, 150, 150),  # 회색
            'unknown': (200, 200, 200) # 밝은 회색
        }
        
        # COCO 클래스 ID별 색상 매핑 (80개 클래스)
        self.class_colors = {}
        for i in range(80):
            # HSV 색상 공간에서 균등하게 분포된 색상 생성
            h = i / 80 * 360
            s = 0.9
            v = 0.9
            # HSV -> BGR 변환
            h = h/2
            c = v * s
            x = c * (1 - abs((h/60) % 2 - 1))
            m = v - c
            
            if h < 60:
                r, g, b = c, x, 0
            elif h < 120:
                r, g, b = x, c, 0
            elif h < 180:
                r, g, b = 0, c, x
            elif h < 240:
                r, g, b = 0, x, c
            elif h < 300:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
                
            r, g, b = (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))
            # OpenCV는 BGR 형식 사용
            self.class_colors[i] = (b, g, r)
        
        # 창 설정 초기화
        if self.display:
            self._init_window()
            
        # 초기화 정보 로깅
        logger.info(f"시각화 창 초기화 완료: {self.window_width}x{self.window_height}")
        logger.info(f"시각화 설정 - 바운딩 박스: {self.show_bbox}, 골격: {self.show_skeleton}, 행동 라벨: {self.show_action_label}")
        
    def _init_window(self):
        """디스플레이 창 초기화"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL if self.resizable else cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)
        
        # 마우스 콜백 설정
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        logger.info(f"시각화 창 초기화 완료: {self.window_width}x{self.window_height}")
        
    def _mouse_callback(self, event, x, y, flags, param):
        """마우스 이벤트 처리 (확대/축소, 드래그)"""
        if event == cv2.EVENT_MOUSEWHEEL:
            # 마우스 휠 이벤트 (Windows)
            if flags > 0:  # 휠 업 (확대)
                self._zoom_in(x, y)
            else:  # 휠 다운 (축소)
                self._zoom_out(x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # 왼쪽 버튼 누름 (드래그 시작)
            self.dragging = True
            self.drag_start_x = x
            self.drag_start_y = y
        elif event == cv2.EVENT_LBUTTONUP:
            # 왼쪽 버튼 뗌 (드래그 끝)
            self.dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            # 마우스 이동 (드래그 중)
            dx = x - self.drag_start_x
            dy = y - self.drag_start_y
            # 부호 변경: 드래그한 방향으로 화면 이동
            self.pan_x -= dx  # 부호 변경
            self.pan_y -= dy  # 부호 변경
            self.drag_start_x = x
            self.drag_start_y = y
            
    def _zoom_in(self, x, y):
        """확대"""
        if self.zoom_factor < self.max_zoom:
            self.zoom_factor *= 1.1
            if self.zoom_factor > self.max_zoom:
                self.zoom_factor = self.max_zoom
                
    def _zoom_out(self, x, y):
        """축소"""
        if self.zoom_factor > self.min_zoom:
            self.zoom_factor *= 0.9
            if self.zoom_factor < self.min_zoom:
                self.zoom_factor = self.min_zoom
    
    def _handle_keyboard_input(self):
        """키보드 입력 처리"""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):  # 리셋
            self.zoom_factor = 1.0
            self.pan_x = 0
            self.pan_y = 0
        elif key == ord('+') or key == ord('='):  # 확대
            self._zoom_in(self.window_width//2, self.window_height//2)
        elif key == ord('-') or key == ord('_'):  # 축소
            self._zoom_out(self.window_width//2, self.window_height//2)
        elif key == ord('h'):  # 도움말 토글
            self.show_controls_help = not self.show_controls_help
        elif key == ord('z'):  # 위험 구역 표시 토글
            self.show_danger_zones = not self.show_danger_zones
            logger.info(f"위험 구역 표시: {'켜짐' if self.show_danger_zones else '꺼짐'}")
        
        return key
        
    def setup_video_writer(self, frame_size, fps):
        """
        비디오 저장을 위한 설정
        
        Args:
            frame_size (tuple): 프레임 크기 (width, height)
            fps (float): 초당 프레임 수
        """
        # 저장 관련 설정 로드
        self.save_video = VISUALIZATION_CONFIG.get('save_output', True)
        self.output_path = VISUALIZATION_CONFIG.get('output_path', 'results/output.mp4')
        
        # 파일 확장자가 명시되지 않은 경우 .mp4를 기본값으로 추가
        if not os.path.splitext(self.output_path)[1]:
            self.output_path += '.mp4'
            
        # 저장이 비활성화된 경우 종료
        if not self.save_video:
            logger.info("비디오 저장 기능이 비활성화되어 있습니다.")
            return
            
        # 출력 디렉토리 생성
        output_dir = os.path.dirname(self.output_path)
        if output_dir:  # 디렉토리가 빈 문자열이 아닌 경우에만 실행
            try:
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"출력 디렉토리 생성 완료: {output_dir}")
            except Exception as e:
                logger.error(f"출력 디렉토리 생성 실패: {e}")
                return
        
        # 이전 비디오 작성기가 있으면 해제
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        # 프레임 크기 유효성 검사
        if not isinstance(frame_size, tuple) or len(frame_size) != 2:
            logger.error(f"잘못된 프레임 크기: {frame_size}, 기본값 (640, 480) 사용")
            frame_size = (640, 480)
        
        # 프레임 크기가 0이 아닌지 확인
        if frame_size[0] <= 0 or frame_size[1] <= 0:
            logger.error(f"잘못된 프레임 크기: {frame_size}, 기본값 (640, 480) 사용")
            frame_size = (640, 480)
            
        # 프레임 크기를 멤버 변수로 저장
        self.frame_width, self.frame_height = frame_size
        
        # FPS 유효성 검사
        if fps <= 0:
            logger.error(f"잘못된 FPS: {fps}, 기본값 30 사용")
            fps = 30
        
        # FPS를 멤버 변수로 저장
        self.fps = fps
        
        # MP4 출력을 원하는 경우 MP4 코덱을 먼저 시도
        if self.output_path.lower().endswith('.mp4'):
            codecs_to_try = [
                ('mp4v', '.mp4'),   # 가장 일반적인 MP4 코덱
                ('avc1', '.mp4'),   # H.264 AVC 코덱(일부 시스템에서 지원)
                ('H264', '.mp4'),   # H.264 코덱(일부 시스템에서 지원)
                ('XVID', '.avi'),   # 실패하면 AVI로 대체
                ('MJPG', '.avi')    # 실패하면 Motion JPEG으로 대체
            ]
        else:
            # 그 외의 경우 일반적인 순서로 시도
            codecs_to_try = [
                ('XVID', '.avi'),
                ('MJPG', '.avi'),
                ('mp4v', '.mp4'),
                ('H264', '.mp4'),
                ('avc1', '.mp4')
            ]
        
        # 경고 플래그 초기화
        self.frame_size_warning_shown = False
        
        for codec_name, ext in codecs_to_try:
            try:
                # 코덱에 맞는 확장자 사용
                if not self.output_path.lower().endswith(ext):
                    output_path = os.path.splitext(self.output_path)[0] + ext
                else:
                    output_path = self.output_path
                
                # 코덱 설정
                fourcc = cv2.VideoWriter_fourcc(*codec_name)
                
                # 임시 비디오 작성기 생성 및 테스트
                temp_writer = cv2.VideoWriter(
                    output_path, fourcc, fps, frame_size
                )
                
                # 비디오 작성기가 정상적으로 열렸는지 확인
                if temp_writer.isOpened():
                    self.video_writer = temp_writer
                    self.output_path = output_path
                    self.codec_name = codec_name
                    
                    # 출력 파일 경로만 간단히 로깅하고 다른 세부 정보는 제외
                    logger.info(f"비디오 저장 설정 완료: {self.output_path}")
                    return
                else:
                    temp_writer.release()
            except Exception as e:
                pass
        
        # 모든 코덱이 실패한 경우
        logger.error("모든 코덱이 실패했습니다. 비디오 저장 기능을 비활성화합니다.")
        self.save_video = False
        self.video_writer = None
        
    def _draw_text_pil(self, frame, text, org_top_left, color_bgr):
        """Pillow를 사용하여 프레임에 텍스트를 그립니다. org_top_left는 텍스트의 좌상단 좌표입니다."""
        if not self.pil_font:
            # Pillow 폰트 로드 실패 시, cv2.putText로 대체 (한글 깨질 수 있음)
            # cv2.putText의 org는 보통 좌하단이므로, 대략적인 위치 조정을 위해 y좌표에 폰트 크기만큼 더해줄 수 있으나,
            # 여기서는 편의상 org_top_left를 그대로 사용하고, font_scale을 작게 조절합니다.
            # 정확한 위치를 위해서는 cv2.getTextSize로 높이를 계산해야 합니다.
            cv2.putText(frame, text, org_top_left, cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5 * self.font_size, color_bgr, self.line_thickness, cv2.LINE_AA)
            return frame
        try:
            # OpenCV 프레임(BGR)을 Pillow 이미지(RGB)로 변환
            cv_image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv_image_rgb)
            draw = ImageDraw.Draw(pil_image)

            # Pillow 색상은 (R, G, B)
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
            
            # Pillow의 draw.text는 xy 좌표를 텍스트의 좌상단으로 사용
            draw.text(org_top_left, text, font=self.pil_font, fill=color_rgb)

            # Pillow 이미지(RGB)를 OpenCV 프레임(BGR)으로 다시 변환
            frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return frame_bgr
        except Exception as e:
            logger.error(f"Pillow로 텍스트 그리기 실패 ('{text[:20]}...'): {e}. cv2.putText 대체 시도.")
            # Pillow 실패 시 cv2.putText로 마지막 시도
            try:
                 cv2.putText(frame, text, org_top_left, cv2.FONT_HERSHEY_SIMPLEX, 
                             0.5 * self.font_size, color_bgr, self.line_thickness, cv2.LINE_AA)
            except Exception as e2:
                logger.error(f"cv2.putText 대체마저 실패: {e2}")
            return frame
        
    def release(self):
        """자원 해제"""
        try:
            if self.video_writer:
                # VideoWriter가 열린 상태인지 확인
                if hasattr(self.video_writer, 'isOpened') and self.video_writer.isOpened():
                    # 비디오 작성기 해제
                    self.video_writer.release()
                    
                    # 파일이 실제로 생성되었는지 확인
                    if os.path.exists(self.output_path):
                        file_size = os.path.getsize(self.output_path)
                        
                        # 파일 크기가 너무 작으면 경고
                        if file_size < 10000:  # 10KB 이하면 의심
                            logger.warning(f"생성된 비디오 파일이 너무 작습니다 ({file_size} 바이트). 비디오가 제대로 저장되지 않았을 수 있습니다.")
                        else:
                            logger.info(f"비디오 저장 완료: {self.output_path}")
                    else:
                        logger.error(f"비디오 파일이 생성되지 않았습니다: {self.output_path}")
                elif self.video_writer is not None:
                    self.video_writer = None
            
            # 창 닫기
            if self.display:
                try:
                    cv2.destroyWindow(self.window_name)
                except Exception as e:
                    logger.error(f"디스플레이 창 닫기 중 오류: {e}")
        except Exception as e:
            logger.error(f"리소스 해제 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
    def visualize_frame(self, frame, detections, poses, actions, danger_violations=None):
        """
        프레임에 탐지 결과, 자세 추정, 행동 인식, 위험 구역 정보를 시각화
        
        Args:
            frame (numpy.ndarray): 원본 비디오 프레임
            detections (list): 객체 검출 결과 리스트 [(x1, y1, x2, y2, confidence, class_id), ...]
            poses (list): 자세 추정 결과 리스트 [pose_data, ...]
            actions (list): 행동 인식 결과 리스트 [{'action': action_label, 'confidence': confidence}, ...]
            danger_violations (list, optional): 위험 구역 침범 정보 리스트 [{'violated': True/False, 'zone_index': zone_index}, ...]
            
        Returns:
            tuple: (시각화된 프레임, 키 입력값)
        """
        # FPS 계산 - 이동 평균 적용
        current_time = time.time()
        if not hasattr(self, 'last_fps_time') or self.last_fps_time == 0:
            self.last_fps_time = current_time
            self.frame_times = []
            self.current_fps = 0
        else:
            time_diff = current_time - self.last_fps_time
            self.last_fps_time = current_time
            
            if time_diff > 0:
                # 이동 평균을 위해 최근 10개 프레임의 시간을 저장
                if not hasattr(self, 'frame_times'):
                    self.frame_times = []
                
                self.frame_times.append(time_diff)
                # 최대 10개 시간만 유지
                if len(self.frame_times) > 10:
                    self.frame_times.pop(0)
                
                # 평균 시간으로 FPS 계산
                avg_time = sum(self.frame_times) / len(self.frame_times)
                self.current_fps = 1.0 / avg_time
        
        # 원본 프레임 복사
        visualized_frame = frame.copy()
        
        # 위험 구역 표시 설정을 config에서 가져옴
        self.show_danger_zones = VISUALIZATION_CONFIG.get('show_danger_zones', True)
        
        # 위험 구역 그리기
        if self.show_danger_zones:
            visualized_frame = self._draw_danger_zones(visualized_frame)
        
        # 바운딩 박스에 객체 ID를 매핑하기 위한 딕셔너리
        object_id_to_index = {}
        
        # 객체 정보 추출 및 표시
        for i, detection in enumerate(detections):
            if len(detection) < 4:
                continue
                
            # 객체 ID 추출 (해당 객체의 index 사용)
            object_id = i
            object_id_to_index[object_id] = i
                
            # 바운딩 박스 추출
            x1, y1, x2, y2 = [int(coord) for coord in detection[:4]]
            
            # 신뢰도와 클래스 정보 추출
            confidence = detection[4] if len(detection) > 4 else 0.0
            class_id = int(detection[5]) if len(detection) > 5 else 0
            
            # 클래스 라벨
            class_label = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
            
            # 행동 정보 가져오기 (해당 객체의 index 사용)
            action_info = actions[i] if i < len(actions) else None
            
            # 행동 라벨
            action_label = ""
            action_confidence = 0.0
            if action_info:
                action_label = action_info.get('action', '')
                action_confidence = action_info.get('confidence', 0.0)
            
            # 위험 구역 침범 정보
            danger_info = danger_violations[i] if danger_violations and i < len(danger_violations) else None
            is_violated = danger_info and danger_info.get('violated', False)
            
            # 바운딩 박스 색상 결정
            color = self._get_bbox_color(action_label, is_violated)
            
            # 바운딩 박스 그리기
            if self.show_bbox:
                cv2.rectangle(visualized_frame, (x1, y1), (x2, y2), color, 2)
            
            # 객체 ID 표시
            if self.show_object_id:
                text = f"ID: {object_id}"
                cv2.putText(visualized_frame, text, (x1 + 5, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6 * self.font_size, (255, 255, 255),
                            self.line_thickness, cv2.LINE_AA)
            
            # 클래스 라벨 표시
            if self.show_labels:
                text = f"{class_label}"
                if self.show_confidence: # confidence는 숫자
                    text += f" {confidence:.2f}"
                # class_label이 한글일 수 있으므로 Pillow 사용 고려 (현재 설정은 영어 'person')
                # 우선 cv2.putText 유지, 필요시 Pillow로 변경
                cv2.putText(visualized_frame, text, (x1 + 5, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7 * self.font_size, (255, 255, 255),
                            self.line_thickness, cv2.LINE_AA)
            
            # 신뢰도 표시는 숫자이므로 cv2.putText 유지
            # if self.show_confidence:
            #     text = f"{confidence:.2f}"
            #     cv2.putText(visualized_frame, text, (x1, y1 + 15),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.font_size, (255, 255, 255),
            #                 self.line_thickness, cv2.LINE_AA)
            
            # 행동 라벨 표시 (한글 가능성 높음)
            if self.show_action_label and action_label:
                text = f"{action_label} ({action_confidence:.2f})"
                # Pillow 사용: org는 (x, y_top_left)
                # cv2.putText의 (x1 + 5, y1 + 30)은 y가 baseline이므로 조정 필요.
                # 여기서는 y1 + 30을 top으로 간주하고, Pillow가 알아서 그리도록 함.
                # 실제로는 Pillow 폰트 높이를 고려하여 y좌표를 정확히 계산하는 것이 좋음.
                # 예: text_y_pil = (y1 + 30) - self.pil_font_actual_pixel_size 
                # 여기서는 (y1+30)을 top으로 사용, 필요시 조정
                pil_text_org = (x1 + 5, y1 + 10) # 위치 미세 조정 (기존 y1+30 보다 위로)
                visualized_frame = self._draw_text_pil(visualized_frame, text, pil_text_org, (255, 255, 255))
        
        # 자세 추정 결과 표시
        if self.show_skeleton:
            for i, pose in enumerate(poses):
                # 키포인트와 연결선 정보 추출
                keypoints = pose.get('keypoints', [])
                connections = pose.get('connections', [])
                
                # 객체 ID 가져오기
                object_id = pose.get('object_id', i)
                
                # 행동 정보 가져오기 (해당 객체의 ID 사용)
                action_info = None
                for action in actions:
                    if action.get('object_id', -1) == object_id:
                        action_info = action
                        break
                
                # 행동 라벨
                action_label = ""
                if action_info:
                    action_label = action_info.get('action', '')
                
                # 위험 구역 침범 정보
                danger_info = None
                if danger_violations and i < len(danger_violations):
                    danger_info = danger_violations[i]
                is_violated = danger_info and danger_info.get('violated', False)
                
                # 골격 색상 결정
                skeleton_color = self._get_skeleton_color(action_label, is_violated)
                
                # 키포인트 및 연결선 그리기
                self._draw_keypoints(visualized_frame, keypoints, connections, skeleton_color)
        
        # 위험 구역 경고 메시지 표시
        for i, danger_info in enumerate(danger_violations or []):
            if danger_info and danger_info.get('violated', False):
                zone_index = danger_info.get('zone_index', -1)
                if zone_index >= 0:
                    zone_type = self.danger_zone_types.get(zone_index, "위험")
                    if i < len(detections):
                        detection = detections[i]
                        x1, y1 = detection[:2]
                        # 하드코딩된 메시지 대신 DANGER_ZONE_CONFIG의 alert_message 사용
                        warning_text = f"! {DANGER_ZONE_CONFIG['alert_message']} ({zone_index+1}) !"
                        # Pillow 사용
                        # cv2.putText의 (int(x1), int(y1) - 30)은 y가 baseline이므로 조정 필요
                        # (int(y1) - 30)을 top으로 간주
                        pil_text_org = (int(x1), int(y1) - 30 - self.pil_font_actual_pixel_size // 2) # 위치 미세 조정
                        visualized_frame = self._draw_text_pil(visualized_frame, warning_text, pil_text_org, DANGER_ZONE_CONFIG['alert_color'])
        
        # UI 요소 표시
        self._draw_ui_elements(visualized_frame)
        
        # 확대/축소 및 이동 변환 적용
        if self.zoom_factor != 1.0 or self.pan_x != 0 or self.pan_y != 0:
            # 프레임 크기 가져오기
            height, width = visualized_frame.shape[:2]
            
            # 확대/축소 및 이동을 위한 변환 행렬 생성
            # 중심을 기준으로 확대/축소
            M = np.float32([
                [self.zoom_factor, 0, -self.pan_x + width * (1 - self.zoom_factor) / 2],
                [0, self.zoom_factor, -self.pan_y + height * (1 - self.zoom_factor) / 2]
            ])
            
            # 변환 적용
            transformed_frame = cv2.warpAffine(
                visualized_frame, M, (width, height),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
            )
            
            # 변환된 프레임 사용
            display_frame = transformed_frame
        else:
            display_frame = visualized_frame
        
        # 화면 출력
        if self.display:
            cv2.imshow(self.window_name, display_frame)
        
        # 비디오 저장
        if self.save_video and self.video_writer:
            try:
                # 원본 시각화 프레임 사용 (display_frame이 아닌 visualized_frame)
                if visualized_frame is not None and visualized_frame.size > 0:
                    # 프레임 크기 가져오기
                    frame_height, frame_width = visualized_frame.shape[:2]
                    
                    # VideoWriter의 예상 크기 가져오기 
                    expected_width = int(self.video_writer.get(cv2.CAP_PROP_FRAME_WIDTH))
                    expected_height = int(self.video_writer.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    # 예상 크기가 유효하지 않으면(0,0) 멤버 변수에 저장된 값 사용
                    if expected_width <= 0 or expected_height <= 0:
                        # 경고 메시지 완전히 제거
                        # if not self.frame_size_warning_shown:
                        #     logger.warning("VideoWriter에서 유효한 프레임 크기를 가져올 수 없습니다. 저장된 프레임 크기를 사용합니다.")
                        #     self.frame_size_warning_shown = True
                        # 멤버 변수의 프레임 크기 사용
                        expected_width = self.frame_width
                        expected_height = self.frame_height
                    
                    # 사이즈가 다르면 리사이즈 (예상 크기가 유효한 경우에만)
                    if frame_width != expected_width or frame_height != expected_height:
                        # 리사이즈 로그 제거
                        # logger.warning(f"프레임 크기 불일치: 현재 {frame_width}x{frame_height}, 예상 {expected_width}x{expected_height}. 리사이즈 수행 중...")
                        try:
                            resized_frame = cv2.resize(visualized_frame, (expected_width, expected_height))
                        except cv2.error as e:
                            # 에러 메시지를 최소화
                            # logger.error(f"프레임 리사이즈 실패: {e}")
                            # 원본 프레임 사용
                            resized_frame = visualized_frame
                    else:
                        # 사이즈가 같으면 그대로 사용
                        resized_frame = visualized_frame
                    
                    # 비디오 저장 수행 및 로깅
                    self.video_writer.write(resized_frame)
                    
                    # 프레임 카운트만 증가시키고 로그는 출력하지 않음
                    self.frame_count += 1
                else:
                    # 경고 메시지도 제거
                    # logger.warning("유효하지 않은 프레임이라 저장하지 않습니다.")
                    pass
            except Exception as e:
                logger.error(f"비디오 프레임 저장 중 오류 발생: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
        # 키 입력 처리 (1ms 대기)
        key = self._handle_keyboard_input()  # 키보드 입력 처리 함수 호출
        
        # 결과 반환
        return visualized_frame, key
        
    def _draw_result(self, frame, detection, pose, action, color_override=None):
        """
        개별 검출 결과 그리기
        
        Args:
            frame (numpy.ndarray): 입력 이미지
            detection (list): 검출된 객체 정보
            pose (dict): 자세 추정 결과
            action (dict): 행동 분류 결과
            color_override (tuple, optional): 색상 오버라이드 (위험 구역 침범 시)
        
        Returns:
            numpy.ndarray: 시각화된 이미지
        """
        # 입력 검사
        if frame is None:
            return None
            
        # 행동 정보가 없는 경우
        if action is None or 'action' not in action or 'confidence' not in action:
            action_label = 'unknown'
            confidence = 0.0
        else:
            action_label = action['action']
            confidence = action['confidence']
        
        # 클래스 ID 가져오기
        class_id = 0
        if detection is not None and len(detection) > 5:
            class_id = int(detection[5])
        elif 'class_id' in action:
            class_id = action['class_id']
        elif pose is not None and 'class_id' in pose:
            class_id = pose['class_id']
        
        # COCO 클래스 이름 매핑 (일부만 정의)
        coco_names = {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 
            5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light", 
            10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 
            15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow", 
            20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack", 
            25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee", 
            30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat", 
            35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle", 
            40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 
            45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 
            50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 
            55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 
            60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 
            65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven", 
            70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock", 
            75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush"
        }
        
        # 클래스 이름 가져오기
        class_name = coco_names.get(class_id, f"class_{class_id}")
        
        # 행동에 따른 색상 선택
        # 1. 색상 오버라이드가 있으면 사용
        if color_override is not None:
            color = color_override
        # 2. 사람(class_id=0)이면 행동에 따른 색상 사용
        elif class_id == 0:
            color = self.action_colors.get(action_label, (200, 200, 200))
        # 3. 그 외 객체는 클래스 ID에 따른 색상 사용
        else:
            color = self.class_colors.get(class_id, (0, 140, 255))  # 기본 색상: 주황색
        
        # 바운딩 박스 그리기
        if self.show_bbox and detection is not None and len(detection) >= 4:
            try:
                x1, y1, x2, y2 = map(int, detection[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.line_thickness)
            except (ValueError, TypeError) as e:
                logger.warning(f"바운딩 박스 그리기 실패: {e}, 데이터: {detection}")
            
        # 골격 그리기
        if self.show_skeleton and pose is not None:
            self._draw_skeleton(frame, pose, color)
            
        # 행동 라벨 그리기
        if self.show_action_label:
            # 기본 위치 설정
            label_position = (50, 50)
            
            try:
                if detection is not None and len(detection) >= 2:
                    x1, y1 = map(int, detection[:2])
                    label_position = (x1, y1 - 10)
                elif pose is not None and 'keypoints' in pose and pose['keypoints'] and len(pose['keypoints']) > 0:
                    # 머리(0번 키포인트)를 기준으로 라벨 위치 설정
                    if len(pose['keypoints'][0]) >= 2:
                        nose_x, nose_y = map(int, pose['keypoints'][0][:2])
                        label_position = (nose_x, nose_y - 20)
            except (ValueError, TypeError, IndexError) as e:
                logger.warning(f"라벨 위치 설정 실패: {e}")
                
            # 행동 라벨 텍스트 생성 및 그리기 (클래스 ID 포함)
            if class_id == 0:  # 사람인 경우
                label_text = f"{class_name}: {action_label} ({confidence:.2f})"
            else:  # 그 외 객체인 경우
                label_text = f"{class_name} ({confidence:.2f})"
                
            font_scale = max(0.7, self.font_size)  # 최소 0.7 크기 보장
            thickness = max(2, self.line_thickness)  # 최소 2 두께 보장
            
            # 텍스트 그리기 (읽기 쉽도록 검은색 외곽선 추가)
            # 배경 사각형 그리기
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            cv2.rectangle(
                frame, 
                (label_position[0], label_position[1] - text_height - 5),
                (label_position[0] + text_width, label_position[1] + 5),
                (0, 0, 0), -1  # 검은색 배경
            )
            
            # 텍스트 그리기
            cv2.putText(
                frame, label_text, label_position,
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),  # 흰색 텍스트
                thickness
            )
            
        return frame
        
    def _draw_skeleton(self, frame, pose, color):
        """
        골격 그리기
        
        Args:
            frame (numpy.ndarray): 입력 이미지
            pose (dict): 자세 추정 결과
            color (tuple): 색상 (B, G, R)
            
        Returns:
            numpy.ndarray: 골격이 그려진 이미지
        """
        # pose가 None이거나 필요한 키가 없는 경우 처리
        if pose is None or 'keypoints' not in pose or 'connections' not in pose:
            return frame
            
        keypoints = pose['keypoints']
        connections = pose['connections']
        
        # 키포인트나 연결이 비어있는 경우 처리
        if not keypoints or not connections:
            return frame
            
        # 키포인트 그리기
        for i, kp in enumerate(keypoints):
            if len(kp) >= 3:  # 키포인트가 x, y, conf 형식인지 확인
                x, y, conf = kp
                if conf > 0.3:  # 최소 신뢰도 이상인 키포인트만 표시
                    cv2.circle(frame, (int(x), int(y)), 
                              max(1, self.line_thickness), color, -1)
                
        # 연결선 그리기
        for conn in connections:
            if len(conn) >= 2:  # 연결이 start_idx, end_idx 형식인지 확인
                start_idx, end_idx = conn
                
                # 인덱스가 유효한지 확인
                if 0 <= start_idx < len(keypoints) and 0 <= end_idx < len(keypoints):
                    # 키포인트 형식이 올바른지 확인
                    if len(keypoints[start_idx]) >= 3 and len(keypoints[end_idx]) >= 3:
                        start_x, start_y, start_conf = keypoints[start_idx]
                        end_x, end_y, end_conf = keypoints[end_idx]
                        
                        if start_conf > 0.3 and end_conf > 0.3:
                            cv2.line(frame, (int(start_x), int(start_y)), 
                                    (int(end_x), int(end_y)), color, self.line_thickness)
                
        return frame
        
    def _draw_frame_info(self, frame):
        """
        프레임 정보 그리기
        
        Args:
            frame (numpy.ndarray): 입력 이미지
            
        Returns:
            numpy.ndarray: 정보가 그려진 이미지
        """
        # 현재 시간 표시
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            frame, timestamp, (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        
        # 프레임 번호 표시
        cv2.putText(
            frame, f"Frame: {self.frame_count}", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        
        # FPS 표시
        if self.show_fps:
            fps_text = f"FPS: {self.current_fps:.1f}"
            cv2.putText(
                frame, fps_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            
        # 확대/축소 비율 표시
        zoom_text = f"Zoom: {self.zoom_factor:.1f}x"
        cv2.putText(
            frame, zoom_text, (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        
        # 컨트롤 도움말 표시
        if self.show_controls_help:
            help_texts = [
                "Controls:",
                "+/- : Zoom in/out",
                "Mouse wheel: Zoom in/out",
                "Mouse drag: Pan image",
                "R: Reset view",
                "H: Toggle help",
                "Z: Toggle danger zones"
            ]
            
            y_pos = 120
            for text in help_texts:
                cv2.putText(
                    frame, text, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
                )
                y_pos += 20
        
        return frame

    def _get_bbox_color(self, action_label, is_violated=False):
        """
        행동 라벨과 위험 침범 여부에 따라 바운딩 박스 색상 결정
        
        Args:
            action_label (str): 행동 라벨
            is_violated (bool): 위험 구역 침범 여부
            
        Returns:
            tuple: (B, G, R) 색상 튜플
        """
        # 위험 구역 침범 시 빨간색
        if is_violated:
            return (0, 0, 255)  # 빨간색
        
        # 행동에 따른 색상
        if action_label == 'falling':
            return (0, 0, 255)  # 빨간색
        elif action_label == 'fighting':
            return (0, 0, 192)  # 다크 레드
        elif action_label == 'standing':
            return (0, 255, 0)  # 초록색
        elif action_label == 'sitting':
            return (255, 165, 0)  # 주황색
        elif action_label == 'walking':
            return (255, 0, 0)  # 파란색
        else:
            return (255, 255, 255)  # 흰색

    def _get_skeleton_color(self, action_label, is_violated=False):
        """
        행동 라벨과 위험 침범 여부에 따라 스켈레톤 색상 결정
        
        Args:
            action_label (str): 행동 라벨
            is_violated (bool): 위험 구역 침범 여부
            
        Returns:
            tuple: (B, G, R) 색상 튜플
        """
        # 위험 구역 침범 시 빨간색
        if is_violated:
            return (0, 0, 255)  # 빨간색
        
        # 행동에 따른 색상
        if action_label == 'falling':
            return (0, 0, 255)  # 빨간색
        elif action_label == 'fighting':
            return (0, 0, 192)  # 다크 레드
        elif action_label == 'standing':
            return (0, 255, 0)  # 초록색
        elif action_label == 'sitting':
            return (255, 165, 0)  # 주황색
        elif action_label == 'walking':
            return (255, 0, 0)  # 파란색
        else:
            return (255, 255, 255)  # 흰색 

    def _draw_keypoints(self, frame, keypoints, connections, color=(0, 255, 0)):
        """
        키포인트와 연결선 그리기
        
        Args:
            frame (numpy.ndarray): 이미지 프레임
            keypoints (list): 키포인트 리스트 [(x, y, confidence), ...]
            connections (list): 연결선 리스트 [(idx1, idx2), ...]
            color (tuple): (B, G, R) 색상
            
        Returns:
            numpy.ndarray: 키포인트가 그려진 이미지
        """
        # 키포인트가 없으면 원본 프레임 반환
        if not keypoints:
            return frame
            
        # 각 키포인트 그리기
        for i, keypoint in enumerate(keypoints):
            if len(keypoint) >= 3 and keypoint[2] > 0.3:  # 신뢰도가 0.3 이상인 경우만 그리기
                x, y = int(keypoint[0]), int(keypoint[1])
                cv2.circle(frame, (x, y), 4, color, -1)
                
        # 연결선 그리기
        for connection in connections:
            if len(connection) == 2:
                idx1, idx2 = connection
                if (idx1 < len(keypoints) and idx2 < len(keypoints) and
                    len(keypoints[idx1]) >= 3 and len(keypoints[idx2]) >= 3 and
                    keypoints[idx1][2] > 0.3 and keypoints[idx2][2] > 0.3):
                    x1, y1 = int(keypoints[idx1][0]), int(keypoints[idx1][1])
                    x2, y2 = int(keypoints[idx2][0]), int(keypoints[idx2][1])
                    cv2.line(frame, (x1, y1), (x2, y2), color, 2)
                    
        return frame
        
    def _draw_danger_zones(self, frame):
        """
        위험 구역 그리기
        
        Args:
            frame (numpy.ndarray): 이미지 프레임
            
        Returns:
            numpy.ndarray: 위험 구역이 그려진 이미지
        """
        if not DANGER_ZONE_CONFIG.get('enabled', False) or not self.show_danger_zones:
            return frame
        
        # 현재 프레임 크기
        frame_height, frame_width = frame.shape[:2]
        
        # 위험 구역 좌표 스케일링
        original_zone_width = DANGER_ZONE_CONFIG.get('original_frame_width', frame_width)
        original_zone_height = DANGER_ZONE_CONFIG.get('original_frame_height', frame_height)

        scale_x = frame_width / original_zone_width if original_zone_width > 0 else 1
        scale_y = frame_height / original_zone_height if original_zone_height > 0 else 1

        zone_color = DANGER_ZONE_CONFIG.get('zone_color', (0, 0, 255)) # BGR
        zone_opacity = DANGER_ZONE_CONFIG.get('zone_opacity', 0.3)
        line_thickness = self.line_thickness

        for i, zone_data in enumerate(DANGER_ZONE_CONFIG.get('zones', [])):
            zone_name = zone_data.get('name', f"Zone {i+1}") # 이름 가져오기, 없으면 기본값
            zone_coords_orig = zone_data.get('coordinates', [])

            if not zone_coords_orig or len(zone_coords_orig) < 3:
                logger.warning(f"잘못된 위험 구역 좌표: {zone_coords_orig} for {zone_name}. 최소 3개의 점 필요.")
                continue

            # 좌표 스케일링
            zone_points_scaled = np.array([(int(x * scale_x), int(y * scale_y)) for x, y in zone_coords_orig], dtype=np.int32)
            
            overlay = frame.copy()
            cv2.fillPoly(overlay, [zone_points_scaled], zone_color)
            frame = cv2.addWeighted(overlay, zone_opacity, frame, 1 - zone_opacity, 0)
            cv2.polylines(frame, [zone_points_scaled], isClosed=True, color=zone_color, thickness=line_thickness)

            # Zone 이름 표시
            label_pos_x = zone_points_scaled[0][0]
            label_pos_y = zone_points_scaled[0][1] - 10 # 첫번째 점 위에 표시
            # Pillow 사용
            frame = self._draw_text_pil(frame, zone_name, (label_pos_x, label_pos_y), (255,255,255))
            
            # 프레임 업데이트 (다음 구역 그리기를 위해)
            overlay = frame.copy() 

        return frame

    def _draw_ui_elements(self, frame):
        """
        UI 요소 그리기 (FPS, 설정 상태 등)
        
        Args:
            frame (numpy.ndarray): 이미지 프레임
            
        Returns:
            numpy.ndarray: UI 요소가 그려진 이미지
        """
        # FPS 표시
        if self.show_fps:
            # FPS 값을 소수점 한 자리까지 표시
            fps_value = self.current_fps if hasattr(self, 'current_fps') and self.current_fps is not None else 0
            fps_text = f"FPS: {fps_value:.1f}"
            # FPS는 영어/숫자이므로 cv2.putText 유지
            cv2.putText(frame, fps_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7 * self.font_size, (0, 255, 0), 
                        self.line_thickness, cv2.LINE_AA)
            
        # 날짜/시간 표시
        if self.show_timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # 영어/숫자이므로 cv2.putText 유지
            cv2.putText(frame, timestamp, (10, frame.shape[0] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7 * self.font_size, (255, 255, 255),
                        self.line_thickness, cv2.LINE_AA)
            
        # 컨트롤 도움말 표시 (영어로 변경)
        if self.show_controls_help:
            help_texts = [
                "Controls:",
                "+/- : Zoom in/out",
                "Mouse wheel: Zoom in/out", 
                "Mouse drag: Pan view",
                "R: Reset view",
                "H: Toggle help",
                "Z: Toggle danger zones"
            ]
            
            y_pos = 60
            for text in help_texts:
                # 영어이므로 cv2.putText 유지
                cv2.putText(frame, text, (10, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.font_size, (200, 200, 200), \
                            1, cv2.LINE_AA)
                y_pos += 20
        
        return frame 