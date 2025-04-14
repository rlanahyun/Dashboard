import streamlit as st
import pandas as pd
import plotly.express as px
from typing import List, Dict, Tuple, Optional, Any
from analysis.metrics import MetricsAnalyzer


class PerformanceCalculator:
    """
    모델 성능 점수를 계산하는 클래스.
    """
    
    def __init__(self):
        """
        PerformanceCalculator 클래스 초기화.
        """
        self.metrics_analyzer = MetricsAnalyzer()
        
        # 결과에서 제외할 열 목록
        self.exclude_columns = ['Model', 'optimizer', 'lr', 'source_file', 'performance_score', 
                               'model_label', 'color_idx', 'lr_numeric']
    
    def visualize_weights(self, weights: Dict[str, float], metric_columns: List[str]) -> Tuple[px.bar, pd.DataFrame]:
        """
        지표별 가중치 시각화.
        
        Args:
            weights: 지표별 가중치 딕셔너리
            metric_columns: 지표 열 목록
            
        Returns:
            가중치 시각화 그래프, 가중치 데이터프레임
        """
        # 가중치 데이터프레임 생성
        weight_df = pd.DataFrame({
            '지표': list(weights.keys()),
            '가중치': list(weights.values())
        })
        weight_df = weight_df.sort_values('가중치', ascending=False)
        
        # 차트 생성
        fig = px.bar(
            weight_df, 
            x='지표', 
            y='가중치',
            title="지표별 가중치 분포",
            color='가중치',
            color_continuous_scale='viridis',
            text_auto='.3f'
        )
        fig.update_layout(xaxis_title="지표", yaxis_title="가중치")
        
        return fig, weight_df
        
    def calculate_default_weights(self, metric_columns: List[str]) -> Dict[str, float]:
        """
        차등 기본 가중치 계산.
        
        Args:
            metric_columns: 지표 열 목록
            
        Returns:
            지표별 가중치 딕셔너리
        """
        # 차등 기본 가중치 설정
        weights = {}
        
        # 각 지표별 우선순위에 따른 가중치 부여
        for metric in metric_columns:
            # 지표의 우선순위에 따른 가중치 적용
            weights[metric] = self.metrics_analyzer.get_priority_weight(metric)
        
        # 가중치 합이 1이 되도록 정규화
        total_weight = sum(weights.values())
        for metric in weights:
            weights[metric] /= total_weight
        
        return weights
    
    def calculate_equal_weights(self, metric_columns: List[str]) -> Dict[str, float]:
        """
        균등 가중치 계산.
        
        Args:
            metric_columns: 지표 열 목록
            
        Returns:
            지표별 가중치 딕셔너리
        """
        # 모든 지표에 동일한 가중치 부여
        weight = 1.0 / len(metric_columns)
        return {metric: weight for metric in metric_columns}
    
    def calculate_performance_score(self, df: pd.DataFrame, weight_config: Optional[Dict[str, float]] = None, 
                                  log_details: bool = False) -> Tuple[pd.DataFrame, List[str]]:
        """
        각 지표의 정규화된 값에 가중치를 적용하여 종합 성능 점수 계산.
        
        Args:
            df: 대상 데이터프레임
            weight_config: 지표별 가중치 설정 (None인 경우 차등 기본 가중치 사용)
            log_details: 세부 계산 과정을 로그로 출력할지 여부
            
        Returns:
            성능 점수가 계산된 데이터프레임, 정규화된 열 목록
        """
        # 성능 지표 열 필터링
        metric_columns = [col for col in df.columns if col not in self.exclude_columns and 
                         not col.endswith('_normalized')]
        
        # 가중치 설정
        if weight_config is None:
            weights = self.calculate_default_weights(metric_columns)
        else:
            weights = weight_config
        
        # 계산 과정 로그 (옵션)
        if log_details:
            st.write("### 성능 점수 계산 세부 정보")
            
            # 가중치 시각화
            fig, weight_df = self.visualize_weights(weights, metric_columns)
            st.plotly_chart(fig, use_container_width=True)
            
            # 가중치 테이블 표시
            st.write("적용된 가중치:")
            st.dataframe(weight_df.style.format({'가중치': '{:.4f}'}).background_gradient(
                subset=['가중치'], cmap='viridis'
            ), use_container_width=True)
            
            # 지표 타입 정보 표시
            st.write("### 지표 특성 정보")
            metric_info = []
            for metric in metric_columns:
                higher_is_better = self.metrics_analyzer.is_higher_better(metric)
                metric_info.append({
                    '지표': metric,
                    '특성': '높을수록 좋음' if higher_is_better else '낮을수록 좋음',
                    '가중치': weights.get(metric, 0)
                })
            
            metric_info_df = pd.DataFrame(metric_info)
            st.dataframe(metric_info_df.style.format({'가중치': '{:.4f}'}).background_gradient(
                subset=['가중치'], cmap='viridis'
            ), use_container_width=True)
        
        # 최종 점수를 저장할 데이터프레임 생성
        result_df = df.copy()
        result_df['performance_score'] = 0
        
        # 각 지표별 정규화 및 가중치 적용
        normalized_columns = []
        for metric in metric_columns:
            # 숫자 형식으로 변환
            result_df[metric] = pd.to_numeric(result_df[metric], errors='coerce')
            
            # 낮은 값이 좋은지 여부 결정
            higher_is_better = self.metrics_analyzer.is_higher_better(metric)
            
            # 정규화된 지표 계산
            normalized_column = f'{metric}_normalized'
            result_df[normalized_column] = self.metrics_analyzer.normalize_metric(
                result_df, metric, higher_is_better)
            normalized_columns.append(normalized_column)
            
            # 가중치 적용하여 점수에 추가
            if metric in weights:
                result_df['performance_score'] += result_df[normalized_column] * weights[metric]
        
        return result_df, normalized_columns