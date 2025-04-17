[![python](https://img.shields.io/badge/Python-3.10.16-0000FF.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![streamlit](https://img.shields.io/badge/Streamlit-v1.44.1-FF0040.svg?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)

# 모델 성능 비교 대시보드
- 다양한 Model, Optimizer, learning rate 조합에 따른 성능 비교 대시보드
- 모델별 성능 지표마다 가중치를 부여하여 종합 성능 점수 계산
- 종합 성능 점수를 기반으로 다양한 분석 및 비교 진행
  
(dashboard 초기 화면 캡처)

## ⚙️ Prepare
### 1. Anaconda 가상 환경 설정
```bash
# 가상 환경 생성
conda create -n dashboard python=3.10
# 가상 환경 활성화
conda activate dashboard
# 필요한 패키지 설치
pip install -r requirements.txt
```
### 2. 실행
```bash
# Streamlit 앱 실행
streamlit run main.py
```
- 웹 브라우저에서 대시보드 형태로 열림
- 기본 포트: 8501
- 다른 포트를 지정하고 싶다면 `--server.port={port_num}`를 함께 실행
#### `main.py`
- 전체 application 조정 및 통합
- 각 컴포넌트 초기화 및 연결
- 메인 UI 흐름 및 탭 구성 관리

## ✅ 사용 방법
(화면 캡처 첨부)

## 프로젝트 구조
     model_performance_dashboard/
      ├── main.py                  # 메인 실행 파일
      │
      ├── utils/                   # 유틸리티
      │   ├── __init__.py
      │   └── styles.py            # UI 스타일 정의
      │
      ├── data_handling/           # 데이터 처리
      │   ├── __init__.py
      │   ├── data_loader.py       # 데이터 로드 및 처리
      │   └── data_processor.py    # 데이터 전처리 및 검증
      │
      ├── analysis/                # 데이터 분석
      │   ├── __init__.py
      │   ├── metrics.py           # 성능 지표 정규화 및 분석
      │   └── performance.py       # 종합 성능 점수 계산
      │
      └── visualization/           # 데이터 시각화
          ├── __init__.py
          ├── ranking_viz.py       # 모델 성능 랭킹
          ├── parameter_viz.py     # 파라미터별 성능 분석
          ├── comparison_viz.py    # 모델별 지표 성능 비교
          ├── metrics_viz.py       # 성능 지표 상세분석
          └── data_viz.py          # 원본 데이터



