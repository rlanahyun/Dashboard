import streamlit as st


class UIStyles:
    """
    대시보드 UI 스타일 관리 클래스.
    """
    
    @staticmethod
    def set_custom_theme():
        """
        대시보드 색상 및 테마 설정.
        """
        # 커스텀 CSS 추가
        st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            border-radius: 4px 4px 0px 0px;
            padding: 10px 10px;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .stDataFrame {
            margin: 10px 0px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def set_page_config():
        """
        페이지 기본 설정 적용.
        """
        st.set_page_config(
            page_title="모델 성능 비교 대시보드",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    @staticmethod
    def set_page_header():
        """
        페이지 헤더 설정.
        """
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("📊 모델 성능 비교 대시보드")
            st.markdown("다양한 모델, 옵티마이저, 학습률 조합에 따른 성능을 비교하고 최적의 모델을 선택합니다.")
    
    @staticmethod
    def show_empty_state():
        """
        데이터가 로드되지 않았을 때 표시할 안내 메시지.
        """
        st.info("데이터를 로드하려면 사이드바에서 CSV/Excel 파일을 업로드하거나 경로를 입력하세요.")
        
        # 샘플 설명 추가
        st.markdown("""
        ### 대시보드 기능:
        
        - **📊 모델 성능 랭킹**: 종합 성능 점수 기반으로 모델 랭킹 및 최고 모델 분석
        - **🔍 파라미터별 성능 분석**: 모델, 옵티마이저, 학습률별 성능 비교 및 최적 조합 찾기
        - **📈 모델별 지표 성능 비교**: 그룹화 막대 그래프로 모델별 다중 지표 성능 비교
        - **⭐ 성능 지표 상세 분석**: 지표 간 상관관계 및 분포 분석 
        - **📋 원본 데이터**: 성능 점수가 계산된 전체 데이터 확인 및 다운로드
        """)
        
        st.markdown("""
        ### 예상 데이터 형식:
        
        다음과 같은 형식의 데이터를 지원합니다 (각 행에 모델 정보와 성능 지표):
        
        | Model | optimizer | lr | PCK1 | PCK2 | ICC | Pearson | ... |
        |-------|-----------|-----|------|------|-----|---------|-----|
        | Model1 | Adam | 0.001 | 0.80 | 0.95 | 0.90 | 0.90 | ... |
        | Model2 | SGD | 0.01 | 0.75 | 0.90 | 0.85 | 0.85 | ... |
        
        엑셀(.xls, .xlsx) 또는 CSV 파일을 업로드하여 분석을 시작하세요.
        """)