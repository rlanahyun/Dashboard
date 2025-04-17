import streamlit as st
import pandas as pd
import numpy as np
import warnings

from utils.styles import UIStyles
from data_handling.data_loader import DataLoader
from data_handling.data_processor import DataProcessor
from analysis.metrics import MetricsAnalyzer
from analysis.performance import PerformanceCalculator
from visualization.ranking_viz import RankingVisualizer
from visualization.parameter_viz import ParameterVisualizer
from visualization.comparison_viz import ComparisonVisualizer
from visualization.metrics_viz import MetricsVisualizer
from visualization.data_viz import DataVisualizer

# 경고 메시지 무시 설정
warnings.filterwarnings('ignore')


class DashboardApp:
    """
    모델 성능 비교 대시보드 애플리케이션 클래스.
    """
    
    def __init__(self):
        """
        DashboardApp 클래스 초기화
        """
        # 유틸리티 클래스 초기화
        self.ui_styles = UIStyles()
        self.data_loader = DataLoader()
        self.data_processor = DataProcessor()
        self.metrics_analyzer = MetricsAnalyzer()
        self.performance_calculator = PerformanceCalculator()
        
        # 시각화 클래스 초기화
        self.ranking_viz = RankingVisualizer()
        self.parameter_viz = ParameterVisualizer()
        self.comparison_viz = ComparisonVisualizer()
        self.metrics_viz = MetricsVisualizer()
        self.data_viz = DataVisualizer()
    
    def setup_page(self):
        """
        페이지 기본 설정 적용.
        """
        self.ui_styles.set_page_config()
        self.ui_styles.set_page_header()
        self.ui_styles.set_custom_theme()
    
    def load_data(self):
        """
        데이터 로드.
        
        Returns:
            로드된 데이터프레임 또는 None
        """
        # 사이드바: 데이터 로드 섹션
        st.sidebar.header("데이터 로드")
        load_option = st.sidebar.radio("데이터 로드 방법:", ["파일 업로드", "경로 입력"])
        
        df = None
        
        if load_option == "파일 업로드":
            uploaded_file = st.sidebar.file_uploader("CSV 또는 Excel 파일 업로드", type=["csv", "xlsx", "xls"])
            if uploaded_file is not None:
                with st.spinner("데이터 로드 중..."):
                    df = self.data_loader.process_uploaded_file(uploaded_file)
                    if not df.empty:
                        st.sidebar.success(f"파일 로드 완료. ({len(df)} 레코드)")
        else:
            file_path = st.sidebar.text_input("CSV 또는 Excel 파일/디렉토리 경로:", "")
            if file_path and st.sidebar.button("데이터 로드"):
                with st.spinner("데이터 로드 중..."):
                    df = self.data_loader.load_data(file_path)
                    if not df.empty:
                        st.sidebar.success(f"데이터 로드 완료. ({len(df)} 레코드)")
        
        return df
    
    def setup_weights(self, metric_columns: list):
        """
        가중치 설정 구성.
        
        Args:
            metric_columns: 지표 열 목록
            
        Returns:
            가중치 딕셔너리, 세부 정보 표시 여부
        """
        # 가중치 설정 섹션
        st.sidebar.header("성능 점수 설정")
        weight_option = st.sidebar.radio(
            "가중치 설정 방식:",
            ["차등 기본 가중치", "사용자 정의 가중치", "균등 가중치"],
            index=0
        )
        show_details = st.sidebar.checkbox("성능 점수 계산 과정 표시", value=False)

        # 가중치 옵션에 따라 설정
        if weight_option == "사용자 정의 가중치":
            st.sidebar.markdown("각 지표의 중요도 가중치 설정 (0-1 사이 값):")
            
            # 지표 그룹별로 가중치 설정
            total_weight = 0
            weights = {}

            # 지표 그룹화
            hva_metrics = [col for col in metric_columns if 'HVA' in col]
            ima_metrics = [col for col in metric_columns if 'IMA' in col]
            point_metrics = [col for col in metric_columns if not 'HVA' in col and not 'IMA' in col]
            
            # 지표 그룹별로 UI 구성
            if point_metrics:
                st.sidebar.markdown("#### Point Metrics:")
                for metric in point_metrics:
                    default_weight = 1.0 / len(metric_columns)
                    weight = st.sidebar.slider(
                        f"{metric}", 
                        min_value=0.0, 
                        max_value=1.0, 
                        value=default_weight,
                        step=0.01,
                        format="%.2f"
                    )
                    weights[metric] = weight
                    total_weight += weight
            
            if hva_metrics:
                st.sidebar.markdown("#### HVA Metrics:")
                for metric in hva_metrics:
                    default_weight = 1.0 / len(metric_columns)
                    weight = st.sidebar.slider(
                        f"{metric}", 
                        min_value=0.0, 
                        max_value=1.0, 
                        value=default_weight,
                        step=0.01,
                        format="%.2f"
                    )
                    weights[metric] = weight
                    total_weight += weight

            if ima_metrics:
                st.sidebar.markdown("#### IMA Metrics:")
                for metric in ima_metrics:
                    default_weight = 1.0 / len(metric_columns)
                    weight = st.sidebar.slider(
                        f"{metric}", 
                        min_value=0.0, 
                        max_value=1.0, 
                        value=default_weight,
                        step=0.01,
                        format="%.2f"
                    )
                    weights[metric] = weight
                    total_weight += weight
            
            # 가중치 합계가 1이 되도록 정규화
            if total_weight > 0:
                for metric in weights:
                    weights[metric] /= total_weight
            
            st.sidebar.markdown(f"**가중치 총합: {total_weight:.2f}**")
            return weights, show_details
            
        elif weight_option == "균등 가중치":
            # 모든 지표에 동일한 가중치 부여
            weights = self.performance_calculator.calculate_equal_weights(metric_columns)
            return weights, show_details
        else:
            # 차등 기본 가중치 (기본값)
            return None, show_details
    
    def run(self):
        """
        대시보드 애플리케이션 실행.
        """
        # 페이지 기본 설정
        self.setup_page()
        
        # 데이터 로드
        df = self.load_data()
        
        # 데이터가 로드되었으면 분석 및 시각화 진행
        if df is not None and not df.empty:
            # 데이터 유효성 검사
            is_valid, df, metric_columns = self.data_processor.validate_data(df)
            
            if not is_valid:
                st.stop()
            
            # 가중치 설정
            weights, show_details = self.setup_weights(metric_columns)

            # 성능 점수 계산
            try:
                scored_df, normalized_columns = self.performance_calculator.calculate_performance_score(
                    df, weights, show_details)
                scored_df = scored_df.sort_values('performance_score', ascending=False)
            except Exception as e:
                st.error(f"성능 점수 계산 중 오류 발생: {e}")
                st.stop()

            # 사이드바에 필터링 옵션 추가
            filters = self.data_processor.setup_sidebar_filters(scored_df)
            
            # 필터링 적용
            filtered_df = self.data_processor.filter_data(scored_df, filters)
            
            # 필터링된 데이터가 있는지 확인
            if filtered_df.empty:
                st.warning("선택한 필터에 맞는 데이터가 없습니다. 필터링 옵션을 변경해 주세요.")
                st.stop()
            
            # 중요 지표 선택 (표시용)
            important_metrics = self.metrics_analyzer.get_priority_metrics(
                metric_columns, ['PCK 1', 'PCK 2', 'ICC', 'Pearson'], 6)
            
            # 메인 컨텐츠 영역 구성 (탭)
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 모델 성능 랭킹", 
                "🔍 파라미터별 성능 분석", 
                "📈 모델별 지표 성능 비교",
                "⭐ 성능 지표 상세 분석",
                "📋 원본 데이터"
            ])
            
            # 탭 1: 모델 성능 랭킹
            with tab1:
                self.ranking_viz.display_top_models_ranking(
                    filtered_df, metric_columns, 10, ['PCK 1', 'PCK 2', 'ICC', 'Pearson'])
            
            # 탭 2: 파라미터별 성능 분석
            with tab2:
                st.subheader("파라미터별 성능 분석")
                
                # 옵티마이저별 성능 비교
                self.parameter_viz.display_optimizer_analysis(filtered_df, important_metrics)
                
                # 학습률별 성능 비교
                self.parameter_viz.display_learning_rate_analysis(filtered_df, important_metrics)
                
                # 모델별 성능 비교
                self.parameter_viz.display_model_analysis(filtered_df, important_metrics)
                
                # 모델-옵티마이저 히트맵
                self.parameter_viz.display_model_optimizer_heatmap(filtered_df)
                
                # 모델-학습률 히트맵
                self.parameter_viz.display_model_lr_heatmap(filtered_df)
                
                # 파라미터 조합 분석
                self.parameter_viz.display_optimal_param_combination(filtered_df)
            
            # 탭 3: 모델별 지표 성능 비교
            with tab3:
                self.comparison_viz.display_model_metric_comparison(filtered_df, metric_columns)
            
            # 탭 4: 성능 지표 상세 분석
            with tab4:
                self.metrics_viz.display_metrics_detail_analysis(filtered_df, metric_columns)
            
            # 탭 5: 원본 데이터
            with tab5:
                self.data_viz.display_original_data(filtered_df, metric_columns)
                self.data_viz.display_summary_statistics(filtered_df, metric_columns)
        else:
            # 데이터가 로드되지 않았을 때 안내 메시지 표시
            self.ui_styles.show_empty_state()


if __name__ == "__main__":
    app = DashboardApp()
    app.run()