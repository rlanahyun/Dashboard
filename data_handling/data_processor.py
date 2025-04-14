import streamlit as st
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional


class DataProcessor:
    """
    데이터 전처리 및 검증을 수행하는 클래스.
    """
    
    def __init__(self):
        """
        DataProcessor 클래스 초기화.
        """
        # 데이터 필터링 시 제외할 기본 열
        self.exclude_columns = ['Model', 'optimizer', 'lr', 'source_file', 'performance_score', 
                              'model_label', 'color_idx', 'lr_numeric']
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, pd.DataFrame, List[str]]:
        """
        로드된 데이터의 유효성 검사 및 필요한 변환을 수행.
        
        Args:
            df: 검증할 데이터프레임
            
        Returns:
            검증 결과(성공/실패), 처리된 데이터프레임, 지표 열 목록
        """
        if df.empty:
            return False, df, []
        
        # 필수 열 확인
        if 'Model' not in df.columns:
            st.error("데이터에 'Model' 열이 없습니다. 데이터 형식을 확인해주세요.")
            return False, df, []
        
        # 성능 지표 열 추출
        exclude_columns = ['Model', 'optimizer', 'lr', 'source_file']
        metric_columns = [col for col in df.columns if col not in exclude_columns]
        
        if len(metric_columns) == 0:
            st.error("성능 지표 열을 찾을 수 없습니다. 데이터 형식을 확인하세요.")
            return False, df, []
        
        # 데이터 타입 변환
        for col in metric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 결측치 확인
        missing_values = df[metric_columns].isna().sum()
        if missing_values.sum() > 0:
            st.warning("일부 성능 지표에 결측값이 있습니다:")
            for col, count in missing_values[missing_values > 0].items():
                st.warning(f"- {col}: {count}개 결측값")
        
        return True, df, metric_columns
    
    def setup_sidebar_filters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        사이드바에 데이터 필터링 옵션 설정.
        
        Args:
            df: 필터링할 데이터프레임
            
        Returns:
            사용자가 선택한 필터 옵션
        """
        st.sidebar.header("필터링 옵션")
        
        filters = {}
        
        # 모델 필터링
        if 'Model' in df.columns:
            models = df['Model'].unique()
            filters['selected_models'] = st.sidebar.multiselect("모델 선택:", models, default=models)
        
        # 옵티마이저 필터링 (있는 경우)
        if 'optimizer' in df.columns:
            optimizers = df['optimizer'].unique()
            filters['selected_optimizers'] = st.sidebar.multiselect("옵티마이저 선택:", optimizers, default=optimizers)
        
        # 학습률 필터링 (있는 경우)
        if 'lr' in df.columns:
            lr_values = sorted(df['lr'].astype(str).unique())
            filters['selected_lr'] = st.sidebar.multiselect("학습률 선택:", lr_values, default=lr_values)
        
        return filters
    
    def filter_data(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        필터 설정에 따라 데이터 필터링.
        
        Args:
            df: 필터링할 데이터프레임
            filters: 필터 설정
            
        Returns:
            필터링된 데이터프레임
        """
        filtered_df = df.copy()
        
        # 모델 필터 적용
        if 'selected_models' in filters and len(filters['selected_models']) > 0:
            filtered_df = filtered_df[filtered_df['Model'].isin(filters['selected_models'])]
        
        # 옵티마이저 필터 적용
        if 'selected_optimizers' in filters and len(filters['selected_optimizers']) > 0:
            filtered_df = filtered_df[filtered_df['optimizer'].isin(filters['selected_optimizers'])]
        
        # 학습률 필터 적용
        if 'selected_lr' in filters and len(filters['selected_lr']) > 0:
            filtered_df = filtered_df[filtered_df['lr'].astype(str).isin(filters['selected_lr'])]
        
        return filtered_df
    
    def get_metric_columns(self, df: pd.DataFrame) -> List[str]:
        """
        데이터프레임에서 지표 열 추출.
        
        Args:
            df: 대상 데이터프레임
            
        Returns:
            지표 열 목록
        """
        return [col for col in df.columns if col not in self.exclude_columns and 
                not col.endswith('_normalized')]
    
    def prepare_display_columns(self, df: pd.DataFrame, metric_columns: List[str], 
                                 priority_metrics: List[str] = None) -> List[str]:
        """
        우선순위에 따라 표시할 열 목록 준비.
        
        Args:
            df: 대상 데이터프레임
            metric_columns: 모든 지표 열 목록
            priority_metrics: 우선순위가 높은 지표 목록 (기본값: PCK1, PCK2, ICC, Pearson)
            
        Returns:
            표시할 열 목록
        """
        if priority_metrics is None:
            priority_metrics = ['PCK 1', 'PCK 2', 'ICC', 'Pearson']
        
        # 기본 표시 열
        display_columns = ['Model', 'performance_score']
        
        # 옵티마이저, 학습률 추가 (있는 경우)
        if 'optimizer' in df.columns:
            display_columns.append('optimizer')
        if 'lr' in df.columns:
            display_columns.append('lr')
        
        # 우선순위 지표 찾기
        important_metrics = []
        for pattern in priority_metrics:
            matching_columns = [col for col in metric_columns if pattern in col]
            important_metrics.extend(matching_columns)
        
        # 부족하면 다른 지표 추가
        if len(important_metrics) < 6:
            remaining_metrics = [col for col in metric_columns if col not in important_metrics]
            important_metrics.extend(remaining_metrics[:6-len(important_metrics)])
        
        # 최대 6개 지표 추가
        display_columns.extend(important_metrics[:6])
        
        return display_columns