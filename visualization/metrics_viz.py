import streamlit as st
import pandas as pd
import plotly.express as px
from typing import List, Dict, Tuple, Optional, Any
from analysis.metrics import MetricsAnalyzer


class MetricsVisualizer:
    """
    성능 지표 상세 분석 시각화를 담당하는 클래스.
    """
    
    def __init__(self):
        """
        MetricsVisualizer 클래스 초기화.
        """
        self.metrics_analyzer = MetricsAnalyzer()
    
    def display_correlation_analysis(self, df: pd.DataFrame, metric_columns: List[str]):
        """
        성능 지표 간 상관관계 분석 시각화.
        
        Args:
            df: 대상 데이터프레임
            metric_columns: 지표 열 목록
        """
        st.markdown("### 성능 지표 간 상관관계")
        
        # 상관관계 계산
        try:
            corr_metrics = df[metric_columns].corr()
            
            # 히트맵 생성
            fig = px.imshow(
                corr_metrics,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                title="성능 지표 간 상관관계 히트맵"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 지표별 상관관계 테이블
            with st.expander("상관관계 상세 테이블"):
                st.dataframe(
                    corr_metrics.style.format("{:.4f}").background_gradient(
                        cmap='RdBu_r', vmin=-1, vmax=1
                    ),
                    use_container_width=True
                )
        except Exception as e:
            st.warning(f"상관관계 분석 중 오류 발생: {e}")
    
    def find_default_metric(self, metric_columns: List[str], 
                           priority_metrics: Optional[List[str]] = None) -> Optional[str]:
        """
        우선순위에 따른 기본 지표 찾기.
        
        Args:
            metric_columns: 지표 열 목록
            priority_metrics: 우선순위 지표 목록
            
        Returns:
            기본 지표 이름 또는 None
        """
        if priority_metrics is None:
            priority_metrics = ['PCK 1', 'PCK 2', 'ICC', 'Pearson']
        
        default_metric = None
        
        # 우선순위 지표 중 첫 번째로 매칭되는 것 찾기
        for pattern in priority_metrics:
            matching = [col for col in metric_columns if pattern in col]
            if matching:
                default_metric = matching[0]
                break
        
        # 매칭되는 것이 없으면 첫 번째 지표 사용
        if default_metric is None and metric_columns:
            default_metric = metric_columns[0]
        
        return default_metric
    
    def display_metric_distribution(self, df: pd.DataFrame, metric_columns: List[str]):
        """
        지표별 분포 분석 시각화.
        
        Args:
            df: 대상 데이터프레임
            metric_columns: 지표 열 목록
        """
        st.markdown("### 성능 지표 분포 분석")
        
        # 지표 선택
        priority_metrics = ['PCK 1', 'PCK 2', 'ICC', 'Pearson']
        default_metric = self.find_default_metric(metric_columns, priority_metrics)
        
        selected_metric = st.selectbox(
            "분석할 지표 선택:", 
            metric_columns, 
            index=metric_columns.index(default_metric) if default_metric else 0
        )
        
        if not selected_metric:
            return
            
        # 히스토그램 및 통계 정보
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # 히스토그램 (모델별로 색상 구분)
            fig = px.histogram(
                df,
                x=selected_metric,
                color='Model',
                title=f"{selected_metric} 분포",
                marginal="box",
                opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 통계 정보
            st.markdown(f"#### {selected_metric} 통계 정보")
            
            stats = df[selected_metric].describe()
            stats_df = pd.DataFrame({
                '통계량': stats.index,
                '값': stats.values
            })
            
            st.dataframe(
                stats_df.style.format({'값': '{:.4f}'}),
                use_container_width=True
            )
            
            # 모델별 평균값
            st.markdown(f"#### 모델별 {selected_metric} 평균")
            model_metric = df.groupby('Model')[selected_metric].mean().sort_values(ascending=False).reset_index()
            
            st.dataframe(
                model_metric.style.format({selected_metric: '{:.4f}'}).background_gradient(
                    subset=[selected_metric], cmap='viridis'
                ),
                use_container_width=True
            )
    
    def display_metric_relationships(self, df: pd.DataFrame, metric_columns: List[str]):
        """
        성능 지표 간 관계 분석 시각화.
        
        Args:
            df: 대상 데이터프레임
            metric_columns: 지표 열 목록
        """
        st.markdown("### 성능 지표 간 관계 분석")
        
        # 지표 선택
        priority_metrics = ['PCK 1', 'PCK 2', 'ICC', 'Pearson']
        default_metric = self.find_default_metric(metric_columns, priority_metrics)
        
        if not default_metric:
            st.warning("분석할 지표가 없습니다.")
            return
            
        selected_metric = st.selectbox(
            "기준 지표 선택:", 
            metric_columns, 
            index=metric_columns.index(default_metric) if default_metric else 0,
            key="relationship_metric"
        )
        
        # 비교할 다른 지표 선택
        other_metrics = [m for m in metric_columns if m != selected_metric]
        
        # 우선순위 지표 찾기
        compare_default = []
        for pattern in priority_metrics:
            if len(compare_default) < 3:  # 최대 3개까지
                matching = [col for col in other_metrics if pattern in col and col not in compare_default]
                compare_default.extend(matching[:3-len(compare_default)])
        
        # 부족하면 다른 지표로 채우기
        if len(compare_default) < 3 and other_metrics:
            remaining = [m for m in other_metrics if m not in compare_default]
            compare_default.extend(remaining[:3-len(compare_default)])
        
        compare_metrics = st.multiselect(
            f"{selected_metric}와 비교할 다른 지표 선택:",
            options=other_metrics,
            default=compare_default
        )
        
        if not compare_metrics:
            st.warning("비교할 지표를 하나 이상 선택해주세요.")
            return
            
        # 산점도 매트릭스
        scatter_metrics = [selected_metric] + compare_metrics
        
        fig = px.scatter_matrix(
            df,
            dimensions=scatter_metrics,
            color='Model',
            title="성능 지표 간 산점도 매트릭스",
            opacity=0.7
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    def display_metrics_detail_analysis(self, df: pd.DataFrame, metric_columns: List[str]):
        """
        성능 지표 상세 분석 시각화.
        
        Args:
            df: 대상 데이터프레임
            metric_columns: 지표 열 목록
        """
        st.subheader("성능 지표 상세 분석")
        
        # 상관관계 분석
        self.display_correlation_analysis(df, metric_columns)
        
        # 지표별 분포 분석
        self.display_metric_distribution(df, metric_columns)
        
        # 지표 간 관계 분석
        self.display_metric_relationships(df, metric_columns)