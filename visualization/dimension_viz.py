import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Tuple, Optional, Any
from analysis.metrics import MetricsAnalyzer


class DimensionVisualizer:
    """
    다차원 성능 비교 시각화를 담당하는 클래스.
    """
    
    def __init__(self):
        """
        DimensionVisualizer 클래스 초기화.
        """
        self.metrics_analyzer = MetricsAnalyzer()
    
    def select_metrics_for_comparison(self, metric_columns: List[str], 
                                     default_count: int = 5) -> List[str]:
        """
        비교를 위한 지표 선택.
        
        Args:
            metric_columns: 모든 지표 열 목록
            default_count: 기본으로 선택할 지표 수
            
        Returns:
            선택된 지표 목록
        """
        # 표시할 지표 수 선택
        num_metrics = st.slider("비교할 지표 수:", 
                              min_value=3, 
                              max_value=min(8, len(metric_columns)), 
                              value=min(default_count, len(metric_columns)))
        
        # 기본 지표로 PCK1, PCK2, ICC, Pearson 위주로 선택
        priority_keywords = ['PCK 1', 'PCK 2', 'ICC', 'Pearson', 'RMSE', 'MD']
        default_metrics = self.metrics_analyzer.get_priority_metrics(
            metric_columns, priority_keywords, num_metrics)
        
        # 사용자 지표 선택
        selected_metrics = st.multiselect(
            "비교할 지표 선택:",
            options=metric_columns,
            default=default_metrics[:num_metrics]
        )
        
        return selected_metrics
    
    def create_radar_chart(self, radar_plot_data: pd.DataFrame, 
                          all_metrics: List[str], 
                          group_by: List[str]) -> go.Figure:
        """
        레이더 차트 생성.
        
        Args:
            radar_plot_data: 레이더 차트용 데이터
            all_metrics: 표시할 모든 지표 목록
            group_by: 그룹화 기준 컬럼 목록
            
        Returns:
            레이더 차트 객체
        """
        # 레이더 차트 생성
        fig = go.Figure()
        
        # 색상 팔레트 설정
        colors = px.colors.qualitative.Plotly
        
        for i, row in radar_plot_data.iterrows():
            # 모델 레이블 생성
            if len(group_by) > 1:
                label_parts = [f"{row[col]}" for col in group_by]
                name = " / ".join(label_parts)
            else:
                name = row['Model']
            
            fig.add_trace(go.Scatterpolar(
                r=row[all_metrics].values,
                theta=all_metrics,
                fill='toself',
                name=name,
                line_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="모델별 정규화된 성능 지표 비교 (레이더 차트)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        
        return fig
    
    def create_parallel_coordinates(self, radar_plot_data: pd.DataFrame,
                                   all_metrics: List[str]) -> px.parallel_coordinates:
        """
        병렬 좌표 차트 생성.
        
        Args:
            radar_plot_data: 병렬 좌표 차트용 데이터
            all_metrics: 표시할 모든 지표 목록
            
        Returns:
            병렬 좌표 차트 객체
        """
        # 색상 인덱스 추가
        radar_plot_data['color_idx'] = pd.Categorical(radar_plot_data['Model']).codes
        
        # 병렬 좌표 차트 생성
        fig = px.parallel_coordinates(
            radar_plot_data, 
            color='color_idx',
            dimensions=all_metrics,
            labels={'color_idx': 'Model'},
            color_continuous_scale=px.colors.qualitative.Plotly
        )
        
        # 차트 레이아웃 조정
        fig.update_layout(
            coloraxis_colorbar=dict(
                title='Model',
                tickvals=radar_plot_data['color_idx'].unique(),
                ticktext=radar_plot_data['Model'].unique()
            ),
            margin=dict(b=100)
        )
        
        return fig
    
    def display_multidimensional_comparison(self, df: pd.DataFrame, metric_columns: List[str]):
        """
        다차원 성능 비교 시각화 표시.
        
        Args:
            df: 대상 데이터프레임
            metric_columns: 모든 지표 열 목록
        """
        st.subheader("다차원 성능 비교")
        
        # 주요 지표 선택
        st.markdown("### 주요 성능 지표 선택")
        selected_metrics = self.select_metrics_for_comparison(metric_columns)
        
        if len(selected_metrics) < 3:
            st.warning("레이더 차트를 위해 최소 3개 이상의 지표를 선택해주세요.")
            return
            
        # 모델 선택 (상위 5개 기본 선택)
        top_model_names = df['Model'].unique()[:5]
        
        selected_models_radar = st.multiselect(
            "레이더 차트에 표시할 모델 선택:",
            options=df['Model'].unique(),
            default=top_model_names[:min(3, len(top_model_names))]
        )
        
        if not selected_models_radar:
            st.warning("최소 1개 이상의 모델을 선택해주세요.")
            return
        
        # 선택된 모델에 대한 데이터 준비
        radar_df = df[df['Model'].isin(selected_models_radar)].copy()
        
        # 그룹화 기준 선택
        col1, col2 = st.columns(2)
        
        group_by = ['Model']
        with col1:
            if 'optimizer' in radar_df.columns and len(radar_df['optimizer'].unique()) > 1:
                use_optimizer = st.checkbox("옵티마이저별로 구분하기", value=True)
                if use_optimizer:
                    group_by.append('optimizer')
        
        with col2:
            if 'lr' in radar_df.columns and len(radar_df['lr'].unique()) > 1:
                use_lr = st.checkbox("학습률별로 구분하기", value=False)
                if use_lr:
                    group_by.append('lr')
        
        # 레이더 차트 데이터 준비
        radar_data = radar_df.groupby(group_by)[selected_metrics + ['performance_score']].mean().reset_index()
        
        # 레이더 차트 생성
        st.markdown("### 레이더 차트: 다차원 성능 비교")
        
        # 데이터 정규화 (레이더 차트용)
        radar_plot_data = radar_data.copy()
        
        # 각 지표를 정규화
        for metric in selected_metrics:
            higher_is_better = self.metrics_analyzer.is_higher_better(metric)
            radar_plot_data[metric] = self.metrics_analyzer.normalize_metric(radar_plot_data, metric, higher_is_better)
        
        # 성능 점수도 포함
        all_metrics = selected_metrics + ['performance_score']
        
        # 레이더 차트 생성 및 표시
        fig = self.create_radar_chart(radar_plot_data, all_metrics, group_by)
        st.plotly_chart(fig, use_container_width=True)
        
        # 병렬 좌표 차트
        st.markdown("### 병렬 좌표 차트: 다차원 성능 비교")
        
        try:
            # 병렬 좌표 차트 생성 및 표시
            fig2 = self.create_parallel_coordinates(radar_plot_data, all_metrics)
            st.plotly_chart(fig2, use_container_width=True)
            
            # 상세 데이터 테이블
            with st.expander("선택된 모델 상세 데이터"):
                # 원본 값과 정규화된 값 모두 표시
                display_data = radar_data[group_by + selected_metrics + ['performance_score']].copy()
                st.dataframe(
                    display_data.style.format({
                        col: '{:.4f}' for col in selected_metrics + ['performance_score']
                    }),
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"병렬 좌표 차트 생성 중 오류 발생: {e}")