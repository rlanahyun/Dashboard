import streamlit as st
import pandas as pd
import plotly.express as px
from typing import List, Dict, Tuple, Optional, Any


class RankingVisualizer:
    """
    모델 성능 랭킹 시각화를 담당하는 클래스.
    """
    
    def __init__(self):
        """
        RankingVisualizer 클래스 초기화.
        """
        pass
    
    def create_model_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        모델 레이블을 생성.
        
        Args:
            df: 대상 데이터프레임
            
        Returns:
            모델 레이블이 추가된 데이터프레임
        """
        result_df = df.copy()
        
        if 'optimizer' in result_df.columns and 'lr' in result_df.columns:
            result_df['model_label'] = result_df.apply(
                lambda row: f"{row['Model']} ({row['optimizer']}, lr={row['lr']})", axis=1
            )
        elif 'optimizer' in result_df.columns:
            result_df['model_label'] = result_df.apply(
                lambda row: f"{row['Model']} ({row['optimizer']})", axis=1
            )
        elif 'lr' in result_df.columns:
            result_df['model_label'] = result_df.apply(
                lambda row: f"{row['Model']} (lr={row['lr']})", axis=1
            )
        else:
            result_df['model_label'] = result_df['Model']
        
        return result_df
    
    def plot_top_models(self, df: pd.DataFrame, top_n: int = 10) -> px.bar:
        """
        상위 N개 모델의 성능 점수를 막대 그래프로 시각화.
        
        Args:
            df: 대상 데이터프레임
            top_n: 표시할 상위 모델 수
            
        Returns:
            막대 그래프 시각화 객체
        """
        # 모델 레이블 생성 및 상위 N개 필터링
        df = self.create_model_labels(df)
        top_models_df = df.head(top_n)
        
        # 막대 그래프 생성
        fig = px.bar(
            top_models_df, 
            x='performance_score', 
            y='model_label',
            orientation='h',
            title=f"상위 {top_n}개 모델 종합 성능 점수",
            color='performance_score',
            color_continuous_scale='viridis',
            text_auto='.3f'
        )
        
        # 그래프 레이아웃 조정
        fig.update_layout(
            yaxis={'categoryorder':'total ascending'},
            height=max(400, 300 + 20 * top_n),  # 동적으로 높이 조정
            margin=dict(t=50, b=50, l=100, r=20)
        )
        
        return fig
    
    def display_top_models_ranking(self, df: pd.DataFrame, 
                                 metric_columns: List[str], 
                                 top_n: int = 10, 
                                 priority_metrics: Optional[List[str]] = None):
        """
        상위 모델 성능 랭킹 시각화.
        
        Args:
            df: 대상 데이터프레임
            metric_columns: 지표 열 목록
            top_n: 표시할 상위 모델 수
            priority_metrics: 우선순위 지표 목록
        """
        if priority_metrics is None:
            priority_metrics = ['PCK 1', 'PCK 2', 'ICC', 'Pearson']
        
        # 상위 N개 모델 표시 설정
        selected_top_n = st.slider("상위 모델 수:", min_value=1, max_value=min(20, len(df)), value=min(10, len(df)))
        
        # 데이터 준비
        df = self.create_model_labels(df)
        top_models_df = df.head(selected_top_n).copy()
        
        # 2개의 컬럼으로 분할
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 랭킹 막대 그래프
            fig = self.plot_top_models(df, selected_top_n)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 최고 성능 모델 하이라이트
            st.subheader("최고 성능 모델 정보")
            best_model = df.iloc[0]
            
            # 최고 모델 정보 표시
            st.markdown(f"**모델**: {best_model['Model']}")
            if 'optimizer' in best_model:
                st.markdown(f"**옵티마이저**: {best_model['optimizer']}")
            if 'lr' in best_model:
                st.markdown(f"**학습률**: {best_model['lr']}")
            
            st.markdown(f"**성능 점수**: {best_model['performance_score']:.4f}")
            
            # 주요 지표 표시
            st.markdown("#### 주요 성능 지표")
            
            # 모든 지표 중에서 필터링
            all_metrics = [col for col in best_model.index 
                         if col not in ['Model', 'optimizer', 'lr', 'source_file', 
                                      'performance_score', 'model_label', 'color_idx', 'lr_numeric'] 
                         and not col.endswith('_normalized')]
            
            # 특정 패턴과 일치하는 지표 찾기
            exact_metrics = self._find_exact_metrics(all_metrics)
            
            # 모든 지표 표시 (최대 6개)
            for metric in exact_metrics[:6]:
                if metric in best_model:
                    st.markdown(f"**{metric}**: {best_model[metric]:.4f}")
        
        # 랭킹 테이블 표시
        self._display_ranking_table(top_models_df, df, metric_columns, priority_metrics)
    
    def _find_exact_metrics(self, all_metrics: List[str]) -> List[str]:
        """
        특정 패턴과 일치하는 지표 찾기.
        
        Args:
            all_metrics: 모든 지표 목록
            
        Returns:
            선택된 지표 목록
        """
        # 이미지에 보이는 지표 패턴과 정확히 일치하는 열 찾기
        exact_metrics = []
        for pattern in ['HVA ICC', 'IMA ICC', 'HVA Pearson', 'IMA Pearson', 'PCK 1', 'PCK 2']:
            matching = [m for m in all_metrics if pattern in m]
            exact_metrics.extend(matching)
        
        # 원하는 형태의 지표가 없으면 "ICC"와 "Pearson"이 포함된 열 찾기
        if not exact_metrics:
            for pattern in ['ICC', 'Pearson', 'PCK']:
                matching = [m for m in all_metrics if pattern in m and m not in exact_metrics]
                exact_metrics.extend(matching)
        
        # 여전히 부족하면 다른 지표로 채우기
        if len(exact_metrics) < 6:
            remaining = [m for m in all_metrics if m not in exact_metrics]
            exact_metrics.extend(remaining[:6-len(exact_metrics)])
        
        return exact_metrics
    
    def _display_ranking_table(self, top_models_df: pd.DataFrame, filtered_df: pd.DataFrame, 
                             metric_columns: List[str], priority_metrics: List[str]):
        """
        랭킹 테이블 표시.
        
        Args:
            top_models_df: 상위 모델 데이터프레임
            filtered_df: 전체 필터링된 데이터프레임
            metric_columns: 지표 열 목록
            priority_metrics: 우선순위 지표 목록
        """
        st.subheader("상세 성능 랭킹 테이블")
        
        # 표시할 컬럼 선택 (PCK1, PCK2, ICC, Pearson 우선)
        display_columns = ['Model', 'performance_score']
        if 'optimizer' in top_models_df.columns:
            display_columns.append('optimizer')
        if 'lr' in top_models_df.columns:
            display_columns.append('lr')
        
        # 주요 성능 지표 추가
        important_metrics = []
        
        # 각 패턴에 대해 매칭되는 컬럼 찾기
        for pattern in priority_metrics:
            matching_columns = [col for col in metric_columns if pattern in col]
            important_metrics.extend(matching_columns)
        
        # 우선순위 지표가 충분하지 않으면 다른 지표 추가
        if len(important_metrics) < 6:
            remaining_metrics = [col for col in metric_columns if col not in important_metrics]
            important_metrics.extend(remaining_metrics[:6-len(important_metrics)])
        
        # 최대 표시 개수 설정: 6개
        display_columns.extend(important_metrics[:6])
        
        # 테이블 표시
        st.dataframe(
            top_models_df[display_columns].style.format({
                'performance_score': '{:.4f}',
                **{col: '{:.4f}' for col in important_metrics if col in top_models_df.columns}
            }),
            use_container_width=True
        )
        
        # 전체 테이블 확인 옵션
        with st.expander("전체 모델 성능 확인"):
            st.dataframe(
                filtered_df[display_columns].style.format({
                    'performance_score': '{:.4f}',
                    **{col: '{:.4f}' for col in important_metrics if col in filtered_df.columns}
                }),
                use_container_width=True
            )