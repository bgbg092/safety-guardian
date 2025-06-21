"""
행동 패턴 분석 시스템 알림 관리 모듈

다양한 알림 방법(이메일, 소리, API, 카카오 등)을 관리합니다.
"""

import os
import time
import logging
import cv2
import threading
from config import ALERT_CONFIG, DANGER_ZONE_CONFIG

# 로깅 설정
logger = logging.getLogger(__name__)

class AlertManager:
    """알림 관리 클래스"""
    
    def __init__(self):
        """알림 관리자 초기화"""
        # 알림 설정
        self.alert_actions = ALERT_CONFIG['alert_actions']
        self.min_consecutive_frames = ALERT_CONFIG['min_consecutive_frames']
        self.alert_enabled = ALERT_CONFIG['enabled']
        
        # 알림 추적용 변수
        self.alert_counts = {}              # 각 행동 연속 발생 프레임 수
        self.last_alert_time = {}           # 행동별/구역별 마지막 알림 시간
        self.method_last_alert_time = {     # 알림 방법별 마지막 알림 시간
            'console': {},                  # {행동/구역키: 마지막 시간}
            'file': {},
            'email': {},
            'api': {},
            'image': {},
            'sound': {},
            'kakao': {}
        }
        
        # 알림 이미지 저장 경로
        self.alert_image_dir = os.path.join('results', 'alert_images')
    
    def initialize(self):
        """
        알림 관련 폴더 및 파일 초기화
        """
        # 알림 관련 폴더 생성
        os.makedirs('results/alert', exist_ok=True)
        logger.info("알림 폴더 확인 및 생성: results/alert")
        
        # 알림 로그 파일 디렉토리 생성
        if 'file' in ALERT_CONFIG['alert_methods']:
            alert_file_path = ALERT_CONFIG['alert_file']
            alert_dir = os.path.dirname(alert_file_path)
            if alert_dir:
                os.makedirs(alert_dir, exist_ok=True)
                logger.info(f"알림 로그 디렉토리 확인 및 생성: {alert_dir}")
                
                # 알림 파일이 없으면 빈 파일 생성
                if not os.path.exists(alert_file_path):
                    try:
                        # 빈 파일 생성
                        with open(alert_file_path, 'w') as f:
                            pass
                        logger.info(f"알림 로그 파일 생성: {alert_file_path}")
                    except Exception as e:
                        logger.warning(f"알림 로그 파일 생성 실패: {e}, 첫 알림 발생 시 자동 생성됩니다.")
        
        # 알림 이미지 저장 디렉토리 생성
        os.makedirs(self.alert_image_dir, exist_ok=True)
        logger.info(f"알림 이미지 디렉토리 확인 및 생성: {self.alert_image_dir}")
        
        # 소리 알림용 디렉토리 생성
        if 'sound' in ALERT_CONFIG['alert_methods']:
            sound_file = ALERT_CONFIG['sound_config'].get('sound_file')
            if sound_file:
                sound_dir = os.path.dirname(sound_file)
                if sound_dir:
                    os.makedirs(sound_dir, exist_ok=True)
                    logger.info(f"소리 알림 디렉토리 확인 및 생성: {sound_dir}")
        
        return True
        
    def handle_alert(self, action, danger_violation=None, current_frame=None):
        """
        알림 처리
        
        Args:
            action (dict): 행동 분류 결과
            danger_violation (dict, optional): 위험 구역 침범 정보
            current_frame (numpy.ndarray, optional): 현재 비디오 프레임
        """
        # --- 1. 행동 기반 알림 처리 ---
        if action:
            action_label = action.get('action')
            if action_label and action_label in self.alert_actions:
                # 행동 카운트 증가
                self.alert_counts[action_label] = self.alert_counts.get(action_label, 0) + 1
                
                # 알림 발생 조건 체크
                if self.alert_counts[action_label] >= self.min_consecutive_frames:
                    now = time.time()
                    last_time = self.last_alert_time.get(action_label, 0)
                    
                    if now - last_time >= ALERT_CONFIG['cooldown_seconds']:
                        self.send_custom_alert(
                            f"[경고] {action_label} 행동 감지됨 (신뢰도: {action.get('confidence', 0):.2f})",
                            alert_type='action',
                            action_type=action_label,
                            frame=current_frame
                        )
                        self.last_alert_time[action_label] = now
                        self.alert_counts[action_label] = 0 # 해당 행동 알림 후 카운트 초기화
            elif action_label:
                # 관심 행동이 아니면 카운트만 초기화
                self.alert_counts[action_label] = 0
        
        # --- 2. 위험 구역 침범 알림 처리 (위의 로직과 완전히 독립적으로 실행) ---
        if danger_violation and danger_violation.get('violated'):
            zone_idx = danger_violation.get('zone_index')
            # zone_idx가 유효한지 확인
            if zone_idx is not None:
                alert_key = f"danger_zone_{zone_idx}"
                
                # 마지막 알림 시간 확인 (쿨다운)
                now = time.time()
                last_time = self.last_alert_time.get(alert_key, 0)
                
                if now - last_time >= ALERT_CONFIG['cooldown_seconds']:
                    alert_msg = f"[경고] {DANGER_ZONE_CONFIG.get('alert_message', '위험 구역 침범')} (구역: {zone_idx+1})"
                    self.send_custom_alert(
                        alert_msg,
                        alert_type='danger_zone',
                        zone_idx=zone_idx,
                        frame=current_frame
                    )
                    self.last_alert_time[alert_key] = now
    
    def send_custom_alert(self, alert_msg, alert_type=None, action_type=None, zone_idx=None, frame=None):
        """
        사용자 정의 알림 전송
        
        Args:
            alert_msg (str): 알림 메시지
            alert_type (str, optional): 알림 유형 ('action', 'danger_zone')
            action_type (str, optional): 행동 유형
            zone_idx (int, optional): 위험 구역 인덱스
            frame (numpy.ndarray, optional): 현재 비디오 프레임
        """
        # 현재 시간과 알림 키 생성
        now = time.time()
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        
        # 알림 키 생성 (행동 유형 또는 위험 구역 인덱스)
        if alert_type == 'action':
            alert_key = action_type
        elif alert_type == 'danger_zone':
            alert_key = f"danger_zone_{zone_idx}"
        else:
            alert_key = 'unknown'
            
        # 글로벌 로깅 (항상 기록)
        logger.warning(alert_msg)
        
        # 방법별 쿨다운 설정과 적용
        method_cooldowns = ALERT_CONFIG.get('method_cooldowns', {})
        
        # 이미지 캡처 및 저장
        image_path = None
        if ('image' in ALERT_CONFIG['alert_methods'] or 'file' in ALERT_CONFIG['alert_methods']) and frame is not None:
            method_key = 'image'
            cooldown = method_cooldowns.get(method_key, ALERT_CONFIG.get('image_save_cooldown', 60))
            last_time = self.method_last_alert_time[method_key].get(alert_key, 0)
            
            if now - last_time >= cooldown:
                try:
                    # 파일명 생성
                    if alert_type == 'action':
                        filename = f"{timestamp}_{alert_type}_{action_type}.jpg"
                    elif alert_type == 'danger_zone':
                        filename = f"{timestamp}_{alert_type}_zone{zone_idx+1}.jpg"
                    else:
                        filename = f"{timestamp}_alert.jpg"
                    
                    # 이미지 저장
                    image_path = os.path.join(self.alert_image_dir, filename)
                    cv2.imwrite(image_path, frame)
                    logger.info(f"알림 이미지 저장: {image_path}")
                    self.method_last_alert_time[method_key][alert_key] = now
                    
                    # 이미지가 저장되었을 때 알림도 함께 발생시킴
                    # 이 부분은 다른 알림 방법에 영향을 줄 수 있으므로, 필요시 조정
                    # alert_msg_with_image = f"{alert_msg} (이미지: {filename})" 
                except Exception as e:
                    logger.error(f"알림 이미지 저장 실패: {e}")
        
        # 콘솔 알림
        if 'console' in ALERT_CONFIG['alert_methods']:
            method_key = 'console'
            cooldown = method_cooldowns.get(method_key, 30)  # 기본값 30초
            last_time = self.method_last_alert_time[method_key].get(alert_key, 0)
            
            if now - last_time >= cooldown:
                print(f"\033[91m{alert_msg}\033[0m")  # 빨간색으로 출력
                self.method_last_alert_time[method_key][alert_key] = now
            
        # 파일 알림
        if 'file' in ALERT_CONFIG['alert_methods']:
            method_key = 'file'
            cooldown = method_cooldowns.get(method_key, 60)  # 기본값 60초
            last_time = self.method_last_alert_time[method_key].get(alert_key, 0)
            
            if now - last_time >= cooldown:
                try:
                    alert_message = alert_msg
                    if image_path:
                        alert_message = f"{alert_msg} (이미지: {os.path.basename(image_path)})"
                        
                    with open(ALERT_CONFIG['alert_file'], 'a') as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {alert_message}\n")
                    self.method_last_alert_time[method_key][alert_key] = now
                except Exception as e:
                    logger.error(f"알림 로그 파일 기록 실패: {e}")
                
        # 이메일 알림
        if 'email' in ALERT_CONFIG['alert_methods']:
            method_key = 'email'
            cooldown = method_cooldowns.get(method_key, 300)  # 기본값 300초
            last_time = self.method_last_alert_time[method_key].get(alert_key, 0)
            
            if now - last_time >= cooldown:
                self._send_email_alert(alert_msg, image_path, alert_type, action_type, zone_idx)
                self.method_last_alert_time[method_key][alert_key] = now
            
        # API 알림
        if 'api' in ALERT_CONFIG['alert_methods']:
            method_key = 'api'
            cooldown = method_cooldowns.get(method_key, 120)  # 기본값 120초
            last_time = self.method_last_alert_time[method_key].get(alert_key, 0)
            
            if now - last_time >= cooldown:
                self._send_api_alert(alert_key, 1.0, alert_msg)
                self.method_last_alert_time[method_key][alert_key] = now
                
        # 소리 알림
        if 'sound' in ALERT_CONFIG['alert_methods']:
            method_key = 'sound'
            cooldown = method_cooldowns.get(method_key, 30)  # 기본값 30초
            last_time = self.method_last_alert_time[method_key].get(alert_key, 0)
            
            if now - last_time >= cooldown:
                self._send_sound_alert()
                self.method_last_alert_time[method_key][alert_key] = now
                
        # 카카오 메시지 알림
        if 'kakao' in ALERT_CONFIG['alert_methods']:
            method_key = 'kakao'
            cooldown = method_cooldowns.get(method_key, 300)  # 기본값 300초
            last_time = self.method_last_alert_time[method_key].get(alert_key, 0)
            
            if now - last_time >= cooldown:
                self._send_kakao_alert(alert_msg, alert_type, action_type, zone_idx)
                self.method_last_alert_time[method_key][alert_key] = now

    def _send_email_alert(self, message, image_path=None, alert_type=None, action_type=None, zone_idx=None):
        """
        이메일 알림 전송
        
        Args:
            message (str): 알림 메시지
            image_path (str, optional): 첨부할 이미지 경로
            alert_type (str, optional): 알림 유형 ('action', 'danger_zone')
            action_type (str, optional): 행동 유형
            zone_idx (int, optional): 위험 구역 인덱스
        """
        
        def send_email():
            try:
                import smtplib
                from email.mime.text import MIMEText
                from email.mime.multipart import MIMEMultipart
                from email.mime.image import MIMEImage
                
                email_config = ALERT_CONFIG['email_config']
                msg_template = email_config.get('message_template', {})
                
                msg = MIMEMultipart()
                msg['From'] = email_config['sender_email']
                msg['To'] = email_config['receiver_email']
                msg['Subject'] = msg_template.get('subject', '행동 패턴 분석 시스템 경고')
                
                # 메시지 내용 포맷
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                location = f"영상 내 위치"
                confidence = "높음"
                
                if alert_type == 'action':
                    location = f"사람 위치"
                    confidence = "행동 감지"
                elif alert_type == 'danger_zone':
                    location = f"위험 구역 {zone_idx+1}"
                    confidence = "구역 침범"
                
                body_template = msg_template.get('body', '{message}')
                body = body_template.format(
                    message=message,
                    timestamp=timestamp,
                    location=location,
                    confidence=confidence
                )
                
                msg.attach(MIMEText(body, 'plain'))
                
                # 이미지 첨부
                if image_path and os.path.exists(image_path):
                    try:
                        with open(image_path, 'rb') as img_file:
                            img = MIMEImage(img_file.read())
                            img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
                            msg.attach(img)
                    except Exception as e:
                        logger.error(f"이메일 이미지 첨부 실패: {e}")
                
                server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
                server.starttls()
                server.login(email_config['sender_email'], email_config['password'])
                server.send_message(msg)
                server.quit()
                
                logger.info("이메일 알림 전송 성공")
                
            except Exception as e:
                logger.error(f"이메일 알림 전송 실패: {e}")

        # 별도 스레드에서 이메일 전송 (UI 블로킹 방지)
        email_thread = threading.Thread(target=send_email)
        email_thread.daemon = True
        email_thread.start()
        logger.info("이메일 전송 스레드 시작")
            
    def _send_sound_alert(self):
        """
        소리 알림 재생
        """
        try:
            sound_config = ALERT_CONFIG.get('sound_config', {})
            volume = sound_config.get('volume', 80)
            sound_file = sound_config.get('sound_file')
            default_sound = sound_config.get('default_sound', True)
            duration = sound_config.get('duration', 3) * 1000  # 밀리초 단위로 변환
            repeat = sound_config.get('repeat', 1)
            
            # Windows 환경에서는 winsound 사용
            if os.name == 'nt':
                import winsound
                
                def play_sound():
                    for _ in range(repeat):
                        if sound_file and os.path.exists(sound_file):
                            # 지정된 사운드 파일 재생
                            try:
                                winsound.PlaySound(sound_file, winsound.SND_FILENAME)
                            except Exception as e:
                                logger.error(f"사운드 파일 재생 실패: {e}")
                                if default_sound:
                                    winsound.Beep(1000, duration)  # 1000Hz 소리를 duration 밀리초 동안 재생
                        else:
                            # 기본 알림음 재생
                            if default_sound:
                                winsound.Beep(1000, duration)  # 1000Hz 소리를 duration 밀리초 동안 재생
                
                # 별도 스레드에서 소리 재생 (UI 블로킹 방지)
                sound_thread = threading.Thread(target=play_sound)
                sound_thread.daemon = True
                sound_thread.start()
                
            # Unix/Linux/Mac 환경에서는 print로 알림
            else:
                # 비Windows 환경에서는 콘솔에 알림만 출력
                print("\a")  # 터미널 벨 소리
                print("\033[91m소리 알림 발생! (Windows 환경이 아닌 경우 시스템 벨 소리만 재생됩니다)\033[0m")
                logger.info("소리 알림 재생 시도 (비Windows 환경)")
            
        except ImportError as e:
            logger.error(f"소리 알림 모듈 가져오기 실패: {e}")
            print("\033[91m소리 알림 기능을 사용할 수 없습니다: 모듈 가져오기 실패\033[0m")
        except Exception as e:
            logger.error(f"소리 알림 재생 실패: {e}")
            print("\033[91m소리 알림 재생 중 오류 발생: {e}\033[0m")
            
    def _send_kakao_alert(self, message, alert_type=None, action_type=None, zone_idx=None):
        """
        카카오 메시지 알림 전송
        
        Args:
            message (str): 알림 메시지
            alert_type (str, optional): 알림 유형 ('action', 'danger_zone')
            action_type (str, optional): 행동 유형
            zone_idx (int, optional): 위험 구역 인덱스
        """
        try:
            import requests
            import json
            
            kakao_config = ALERT_CONFIG.get('kakao_config', {})
            api_key = kakao_config.get('api_key')
            template_id = kakao_config.get('template_id')
            receiver_ids = kakao_config.get('receiver_ids', [])
            
            if not api_key or not template_id or not receiver_ids:
                logger.error("카카오 메시지 설정이 완료되지 않았습니다.")
                return
            
            # 메시지 내용 포맷
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            location = f"영상 내 위치"
            confidence = "높음"
            
            if alert_type == 'action':
                location = f"사람 위치"
                confidence = "행동 감지"
            elif alert_type == 'danger_zone':
                location = f"위험 구역 {zone_idx+1}"
                confidence = "구역 침범"
            
            msg_template = kakao_config.get('message_template', {})
            text_template = msg_template.get('text', '{message}')
            text = text_template.format(
                message=message,
                timestamp=timestamp,
                location=location,
                confidence=confidence
            )
            
            button_text = msg_template.get('button_text', '시스템 확인하기')
            button_url = msg_template.get('button_url', 'https://your-monitoring-system.com')
            
            # 카카오 API 호출 (메시지 전송)
            logger.info("카카오 메시지 알림 기능은 구현되었지만 현재 사용하지 않습니다.")
            logger.info(f"카카오 메시지 내용: {text}")
            
            # 실제 API 호출은 나중에 활성화할 수 있음
            # headers = {
            #     'Authorization': f'Bearer {api_key}',
            #     'Content-Type': 'application/json'
            # }
            # 
            # for receiver_id in receiver_ids:
            #     data = {
            #         'template_id': template_id,
            #         'receiver_id': receiver_id,
            #         'args': {
            #             'message': text,
            #             'button_text': button_text,
            #             'button_url': button_url
            #         }
            #     }
            #     
            #     response = requests.post(
            #         'https://kapi.kakao.com/v2/api/talk/memo/send',
            #         headers=headers,
            #         data=json.dumps(data)
            #     )
            #     
            #     if response.status_code == 200:
            #         logger.info(f"카카오 메시지 알림 전송 성공: {receiver_id}")
            #     else:
            #         logger.error(f"카카오 메시지 알림 전송 실패: {response.status_code}, {response.text}")
            
        except Exception as e:
            logger.error(f"카카오 메시지 알림 전송 실패: {e}")

    def _send_api_alert(self, action_label, confidence, message=None):
        """
        API 알림 전송 (메시지 인자 추가)
        
        Args:
            action_label (str): 행동 라벨
            confidence (float): 신뢰도
            message (str, optional): 사용자 정의 메시지
        """
        try:
            import requests
            import json
            
            api_config = ALERT_CONFIG['api_config']
            
            payload = {
                'timestamp': time.time(),
                'action': action_label,
                'confidence': confidence,
                'message': message or f"{action_label} 행동 감지됨"
            }
            
            headers = api_config['headers']
            headers['Authorization'] = f"Bearer {api_config['auth_token']}"
            
            response = requests.post(
                api_config['url'],
                data=json.dumps(payload),
                headers=headers
            )
            
            if response.status_code == 200:
                logger.info("API 알림 전송 성공")
            else:
                logger.error(f"API 알림 전송 실패: {response.status_code}, {response.text}")
                
        except Exception as e:
            logger.error(f"API 알림 전송 실패: {e}")

    def send_alert(self, action_label, confidence, frame=None):
        """
        행동 기반 알림 전송
        
        Args:
            action_label (str): 행동 라벨
            confidence (float): 신뢰도
            frame (numpy.ndarray, optional): 현재 비디오 프레임
        """
        alert_msg = f"[경고] {action_label} 행동 감지됨 (신뢰도: {confidence:.2f})"
        self.send_custom_alert(
            alert_msg,
            alert_type='action',
            action_type=action_label,
            frame=frame
        )