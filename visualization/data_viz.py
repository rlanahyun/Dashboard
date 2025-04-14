import streamlit as st
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any


class DataVisualizer:
    """
    원본 데이터 시각화를 담당하는 클래스.
    """
    
    def __init__(self):
        """
        DataVisualizer 클래스 초기화.
        """
        pass
    
    def display_original_data(self, df: pd.DataFrame, metric_columns: List[str]):
        """
        원본 데이터 시각화.
        
        Args:
            df: 대상 데이터프레임
            metric_columns: 지표 열 목록
        """
        st.subheader("원본 데이터")
        
        # 종합 성능 점수 포함 데이터
        st.markdown("### 성능 점수가 계산된 전체 데이터")
        
        # 표시할 열 선택
        show_normalized = st.checkbox("정규화된 지표 값 표시", value=False)
        
        if show_normalized:
            display_cols = df.columns
        else:
            display_cols = [col for col in df.columns if not col.endswith('_normalized')]
        
        # 데이터프레임 표시
        st.dataframe(df[display_cols], use_container_width=True)
        
        # 데이터 다운로드 버튼
        csv = df.to_csv(index=False)
        st.download_button(
            label="CSV로 다운로드",
            data=csv,
            file_name="model_performance_scores.csv",
            mime="text/csv",
        )
    
    def display_summary_statistics(self, df: pd.DataFrame, metric_columns: List[str]):
        """
        데이터 요약 통계 시각화.
        
        Args:
            df: 대상 데이터프레임
            metric_columns: 지표 열 목록
        """
        st.markdown("### 데이터 요약 통계")
        
        # 요약 통계 계산
        summary_stats = df[metric_columns].describe().T
        
        # 열 이름 변경
        summary_stats.columns = ['개수', '평균', '표준편차', '최소값', '25%', '50%', '75%', '최대값']
        
        # 인덱스 재설정
        summary_stats.reset_index(inplace=True)
        summary_stats.rename(columns={'index': '지표'}, inplace=True)
        
        # 테이블 표시
        st.dataframe(
            summary_stats.style.format({
                col: '{:.4f}' for col in summary_stats.columns if col not in ['개수', '지표']
            }),
            use_container_width=True
        )