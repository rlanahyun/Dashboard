import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from typing import List, Dict, Tuple, Optional, Any


class ParameterVisualizer:
    """
    파라미터별 성능 분석 시각화를 담당하는 클래스.
    """
    
    def __init__(self):
        """
        ParameterVisualizer 클래스 초기화.
        """
        pass
    
    def display_optimizer_analysis(self, df: pd.DataFrame, important_metrics: List[str]):
        """
        옵티마이저별 성능 분석 시각화.
        
        Args:
            df: 대상 데이터프레임
            important_metrics: 표시할 중요 지표 목록
        """
        if 'optimizer' not in df.columns:
            return
        
        st.markdown("### 옵티마이저별 평균 성능")
        
        # 2개의 컬럼으로 분할
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # 옵티마이저별 평균 성능 점수 계산
            optimizer_perf = df.groupby('optimizer')['performance_score'].mean().reset_index()
            optimizer_perf = optimizer_perf.sort_values('performance_score', ascending=False)
            
            # 막대 그래프 생성
            fig = px.bar(
                optimizer_perf, 
                x='optimizer', 
                y='performance_score',
                title="옵티마이저별 평균 성능 점수",
                color='performance_score',
                text_auto='.3f'
            )
            fig.update_layout(xaxis_title="옵티마이저", yaxis_title="평균 성능 점수")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 옵티마이저별 주요 지표 평균값 테이블
            optimizer_metrics = df.groupby('optimizer')[important_metrics + ['performance_score']].mean().reset_index()
            st.markdown("#### 옵티마이저별 주요 지표 평균")
            st.dataframe(
                optimizer_metrics.style.format({
                    'performance_score': '{:.4f}',
                    **{col: '{:.4f}' for col in important_metrics if col in optimizer_metrics.columns}
                }),
                use_container_width=True
            )
    
    def display_learning_rate_analysis(self, df: pd.DataFrame, important_metrics: List[str]):
        """
        학습률별 성능 분석 시각화.
        
        Args:
            df: 대상 데이터프레임
            important_metrics: 표시할 중요 지표 목록
        """
        if 'lr' not in df.columns:
            return
        
        st.markdown("### 학습률별 평균 성능")
        
        # 숫자 형식으로 변환 시도
        try:
            df_copy = df.copy()
            df_copy['lr_numeric'] = pd.to_numeric(df_copy['lr'], errors='coerce')
            
            # 학습률별 성능 평균
            lr_perf = df_copy.groupby('lr_numeric')['performance_score'].mean().reset_index()
            lr_perf = lr_perf.sort_values('lr_numeric')
            
            # 2개의 컬럼으로 분할
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # 선 그래프 생성
                fig = px.line(
                    lr_perf, 
                    x='lr_numeric', 
                    y='performance_score',
                    title="학습률별 평균 성능 점수",
                    markers=True
                )
                fig.update_layout(xaxis_title="학습률 (lr)", yaxis_title="평균 성능 점수")
                fig.update_traces(line=dict(width=3), marker=dict(size=10))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 학습률별 주요 지표 평균값 테이블
                lr_metrics = df.groupby('lr')[important_metrics + ['performance_score']].mean().reset_index()
                st.markdown("#### 학습률별 주요 지표 평균")
                st.dataframe(
                    lr_metrics.style.format({
                        'performance_score': '{:.4f}',
                        **{col: '{:.4f}' for col in important_metrics if col in lr_metrics.columns}
                    }),
                    use_container_width=True
                )
                
        except Exception as e:
            st.warning(f"학습률을 숫자로 변환할 수 없어 상세 차트를 생성할 수 없습니다: {e}")
            
            # 문자열 형태로 처리
            lr_perf = df.groupby('lr')['performance_score'].mean().reset_index()
            fig = px.bar(
                lr_perf, 
                x='lr', 
                y='performance_score',
                title="학습률별 평균 성능 점수",
                color='performance_score',
                text_auto='.3f'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def display_model_analysis(self, df: pd.DataFrame, important_metrics: List[str]):
        """
        모델별 성능 분석 시각화.
        
        Args:
            df: 대상 데이터프레임
            important_metrics: 표시할 중요 지표 목록
        """
        st.markdown("### 모델별 평균 성능")
        
        # 2개의 컬럼으로 분할
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # 모델별 평균 성능 점수 계산
            model_perf = df.groupby('Model')['performance_score'].mean().reset_index()
            model_perf = model_perf.sort_values('performance_score', ascending=False)
            
            # 막대 그래프 생성
            fig = px.bar(
                model_perf, 
                x='Model', 
                y='performance_score',
                title="모델별 평균 성능 점수",
                color='performance_score',
                text_auto='.3f'
            )
            fig.update_layout(xaxis_title="모델", yaxis_title="평균 성능 점수")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 모델별 주요 지표 평균값 테이블
            model_metrics = df.groupby('Model')[important_metrics + ['performance_score']].mean().reset_index()
            st.markdown("#### 모델별 주요 지표 평균")
            st.dataframe(
                model_metrics.style.format({
                    'performance_score': '{:.4f}',
                    **{col: '{:.4f}' for col in important_metrics if col in model_metrics.columns}
                }),
                use_container_width=True
            )
    
    def display_model_optimizer_heatmap(self, df: pd.DataFrame):
        """
        모델-옵티마이저 조합 성능 히트맵 시각화.
        
        Args:
            df: 대상 데이터프레임
        """
        if 'optimizer' not in df.columns:
            return
            
        st.markdown("### 모델-옵티마이저 조합 성능 히트맵")
        try:
            # 모델과 옵티마이저 조합별 성능 점수 계산
            model_opt_perf = df.pivot_table(
                index='Model', 
                columns='optimizer', 
                values='performance_score',
                aggfunc='mean'
            )
            
            # 히트맵
            fig = px.imshow(
                model_opt_perf,
                text_auto='.3f',
                color_continuous_scale='viridis',
                title="모델-옵티마이저 조합별 성능 점수"
            )
            fig.update_layout(
                xaxis_title="옵티마이저",
                yaxis_title="모델"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 표 형태로도 제공
            with st.expander("표 형태로 보기"):
                st.dataframe(
                    model_opt_perf.style.format("{:.4f}").background_gradient(
                        cmap='viridis', axis=None
                    ),
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"히트맵 생성 중 오류 발생: {e}")
    
    def display_model_lr_heatmap(self, df: pd.DataFrame):
        """
        모델-학습률 조합 성능 히트맵 시각화.
        
        Args:
            df: 대상 데이터프레임
        """
        if 'lr' not in df.columns:
            return
            
        st.markdown("### 모델-학습률 조합 성능 히트맵")
        try:
            # 모델과 학습률 조합별 성능 점수 계산
            model_lr_perf = df.pivot_table(
                index='Model', 
                columns='lr', 
                values='performance_score',
                aggfunc='mean'
            )
            
            # 히트맵
            fig = px.imshow(
                model_lr_perf,
                text_auto='.3f',
                color_continuous_scale='viridis',
                title="모델-학습률 조합별 성능 점수"
            )
            fig.update_layout(
                xaxis_title="학습률",
                yaxis_title="모델"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 표 형태로도 제공
            with st.expander("표 형태로 보기"):
                st.dataframe(
                    model_lr_perf.style.format("{:.4f}").background_gradient(
                        cmap='viridis', axis=None
                    ),
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"히트맵 생성 중 오류 발생: {e}")
    
    def display_optimal_param_combination(self, df: pd.DataFrame):
        """
        모델 x 옵티마이저 x 학습률 최적 조합을 시각화.
        
        Args:
            df: 대상 데이터프레임
        """
        if 'optimizer' in df.columns and 'lr' in df.columns:
            st.markdown("### 최적 파라미터 조합 분석")
            
            # 파라미터 조합별 성능 테이블
            param_perf = df.pivot_table(
                index=['Model', 'optimizer'],
                columns='lr',
                values='performance_score',
                aggfunc='mean'
            ).reset_index()
            
            # 테이블 표시 (조건부 서식 적용)
            st.dataframe(
                param_perf.style.background_gradient(
                    cmap='viridis', subset=param_perf.columns[2:], axis=None
                ).format({col: '{:.4f}' for col in param_perf.columns[2:]}),
                use_container_width=True
            )