# 데이터 디렉토리

이 디렉토리는 행동 패턴 분석 시스템에서 사용되는 데이터를 저장하는 공간입니다.

## 포함 항목

- 샘플 비디오
- 데이터셋
- 테스트 결과

## 추천 데이터셋

### 1. 자세 추정 데이터셋
- **COCO Keypoints**: 인간 자세 추정을 위한 표준 데이터셋
  - 다운로드: [http://cocodataset.org/#keypoints-2017](http://cocodataset.org/#keypoints-2017)
  
- **MPII Human Pose**: 인간 자세 추정을 위한 고품질 데이터셋
  - 다운로드: [http://human-pose.mpi-inf.mpg.de](http://human-pose.mpi-inf.mpg.de)

### 2. 행동 인식 데이터셋
- **UCF101**: 101개 행동 범주의 비디오 클립
  - 다운로드: [https://www.crcv.ucf.edu/data/UCF101.php](https://www.crcv.ucf.edu/data/UCF101.php)
  
- **HMDB51**: 51개 행동 범주의 비디오 클립
  - 다운로드: [https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
  
- **Kinetics**: 대규모 인간 행동 비디오 데이터셋
  - 다운로드: [https://deepmind.com/research/open-source/kinetics](https://deepmind.com/research/open-source/kinetics)

### 3. 특수 행동 데이터셋
- **Fall Detection Dataset**: 넘어짐 감지 데이터셋
  - 다운로드: [http://falldataset.com](http://falldataset.com)
  
- **NTU RGB+D**: 3D 관절 위치 기반 행동 인식 데이터셋
  - 다운로드: [https://rose1.ntu.edu.sg/dataset/actionRecognition/](https://rose1.ntu.edu.sg/dataset/actionRecognition/)

## 샘플 데이터

다음 샘플 데이터를 사용하여 시스템을 테스트할 수 있습니다:

1. [Pexels](https://www.pexels.com/videos/) 또는 [Pixabay](https://pixabay.com/videos/)에서 무료 스톡 비디오
2. 테스트용 CCTV 영상 샘플: `sample_video.mp4`

## 데이터 사용 방법

1. 원하는 데이터셋을 다운로드
2. 이 디렉토리(`data/`)에 저장
3. 또는 S3 설정을 통해 원격 데이터 접근 구성 (config.py 참조) 