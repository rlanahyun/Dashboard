import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Optional


class MetricsAnalyzer:
    """
    성능 지표 분석 및 정규화를 수행하는 클래스.
    """
    
    def __init__(self):
        """
        MetricsAnalyzer 클래스 초기화.
        """
        # 낮은 값이 좋은 지표 목록
        self.lower_is_better_metrics = ['RMSE', 'MD', '>3', 'Bland-Altman', 'Error', 'Loss', 'MAE', 'MSE']
        
        # 높은 값이 좋은 지표 목록
        self.higher_is_better_metrics = ['PCK', 'Pearson', 'ICC', 'Accuracy', 'Precision', 'Recall', 'F1']
        
        # 우선순위 지표 설정
        self.high_priority_metrics = ['PCK', 'Pearson', 'ICC', 'Accuracy', 'Precision', 'Recall', 'F1']
        self.medium_priority_metrics = ['RMSE', 'MD', 'MAE']
        self.low_priority_metrics = ['Loss', 'Error', '>3', 'Bland-Altman', 'MSE']
    
    def normalize_metric(self, df: pd.DataFrame, metric: str, higher_is_better: bool = True) -> pd.Series:
        """
        성능 지표를 0-1 사이로 정규화.
        
        Args:
            df: 대상 데이터프레임
            metric: 정규화할 지표 열 이름
            higher_is_better: 값이 높을수록 좋은 지표인지 여부 (True: PCK, Pearson, ICC 등, False: RMSE, MD 등)
            
        Returns:
            정규화된 지표 값 시리즈
        """
        # NaN 값 확인 및 처리
        if df[metric].isna().any():
            st.warning(f"'{metric}' 지표에 결측값이 있습니다. 정규화 시 이 값들은 제외됩니다.")
        
        min_val = df[metric].min()
        max_val = df[metric].max()
        
        if np.isnan(min_val) or np.isnan(max_val):
            return df[metric].copy()
        
        if max_val > min_val:
            # 높은 값이 좋은 지표 (PCK, Pearson, ICC 등)
            if higher_is_better:
                return (df[metric] - min_val) / (max_val - min_val)
            # 낮은 값이 좋은 지표 (RMSE, MD 등)
            else:
                return 1 - ((df[metric] - min_val) / (max_val - min_val))
        else:
            return df[metric] * 0  # 모든 값이 같으면 0으로 반환
    
    def is_higher_better(self, metric: str) -> bool:
        """
        주어진 지표의 값이 높을수록 좋은지 여부 판단.
        
        Args:
            metric: 지표 이름
            
        Returns:
            값이 높을수록 좋은 지표인지 여부
        """
        return not any(lower_better in metric for lower_better in self.lower_is_better_metrics)
    
    def get_metric_priority(self, metric: str) -> str:
        """
        지표의 우선순위 결정.
        
        Args:
            metric: 지표 이름
            
        Returns:
            우선순위 ('high', 'medium', 'low', 'normal')
        """
        if any(high_metric in metric for high_metric in self.high_priority_metrics):
            return 'high'
        elif any(medium_metric in metric for medium_metric in self.medium_priority_metrics):
            return 'medium'
        elif any(low_metric in metric for low_metric in self.low_priority_metrics):
            return 'low'
        else:
            return 'normal'
    
    def get_priority_weight(self, metric: str) -> float:
        """
        지표의 우선순위에 따른 가중치를 반환.
        
        Args:
            metric: 지표 이름
            
        Returns:
            가중치 값
        """
        priority = self.get_metric_priority(metric)
        
        # 우선순위별 가중치 매핑
        priority_weights = {
            'high': 1.5,
            'medium': 1.0,
            'low': 0.5,
            'normal': 0.8
        }
        
        return priority_weights.get(priority, 0.8)
    
    def get_priority_metrics(self, metric_columns: List[str], 
                             priority_keywords: List[str] = None, 
                             count: int = 5) -> List[str]:
        """
        우선순위에 따라 지표 열 선택.
        
        Args:
            metric_columns: 모든 지표 열 목록
            priority_keywords: 우선순위 키워드 목록
            count: 선택할 지표 수
            
        Returns:
            선택된 지표 열 목록
        """
        if priority_keywords is None:
            priority_keywords = ['PCK 1', 'PCK 2', 'ICC', 'Pearson', 'RMSE', 'MD']
        
        selected_metrics = []
        
        # 우선순위 키워드에 따라 지표 선택
        for keyword in priority_keywords:
            if len(selected_metrics) < count:
                matched_metrics = [m for m in metric_columns if keyword in m and m not in selected_metrics]
                selected_metrics.extend(matched_metrics[:count - len(selected_metrics)])
        
        # 부족한 경우 나머지 지표로 채우기
        if len(selected_metrics) < count:
            remaining = [m for m in metric_columns if m not in selected_metrics]
            selected_metrics.extend(remaining[:count - len(selected_metrics)])
        
        return selected_metrics[:count]