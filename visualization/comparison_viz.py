import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Tuple, Optional, Any
from analysis.metrics import MetricsAnalyzer


class ComparisonVisualizer:
    """
    모델별 지표 성능 비교 시각화를 담당하는는 클래스.
    """
    
    # 색상 팔레트 상수 정의
    COLORS = [
        '#1f77b4',  # 파란색
        '#ff7f0e',  # 주황색
        '#2ca02c',  # 녹색
        '#d62728',  # 빨간색
        '#9467bd',  # 보라색
        '#8c564b',  # 갈색
        '#e377c2',  # 분홍색
        '#7f7f7f',  # 회색
        '#bcbd22',  # 황록색
        '#17becf'   # 청록색
    ]
    
    # 방향 심볼 상수
    UP_SYMBOL = "↑"
    DOWN_SYMBOL = "↓"
    
    def __init__(self):
        """
        ComparisonVisualizer 클래스 초기화.
        """
        self.metrics_analyzer = MetricsAnalyzer()   

    def display_model_metric_comparison(self, df: pd.DataFrame, metric_columns: List[str]) -> None:
        """
        모델별 다중 지표 성능 비교 시각화 표시.
        그룹화 막대 그래프.

        Args:
            df: 대상 데이터프레임
            metric_columns: 모든 지표 열 목록
        """
        st.subheader("모델별 지표 성능 비교")
        
        # 데이터 전처리
        numbered_df = self._preprocess_dataframe(df)
        
        # 지표 선택 UI
        selected_metrics = self._create_metrics_selection_ui(metric_columns)
        if not selected_metrics:
            st.warning("최소 1개 이상의 지표를 선택해주세요.")
            return
        
        # 표시할 컬럼 지정
        display_columns = [col for col in df.columns if not col.endswith('_normalized')]
        
        # 모델 선택 UI
        selected_model_numbers = self._create_model_selection_ui(numbered_df, display_columns)
        if not selected_model_numbers:
            st.warning("최소 1개 이상의 모델을 선택해주세요.")
            return
        
        # 선택된 모델 번호에 대한 데이터 필터링
        selected_df = numbered_df[numbered_df['No.'].isin(selected_model_numbers)].copy()
        
        try:
            # 지표 방향 정보 수집 (높을수록 좋은지, 낮을수록 좋은지)
            metric_directions = self._get_metric_directions(selected_metrics)
            
            # 그래프 생성 및 표시
            st.markdown(f"### 선택된 지표 모델별 그룹화")
            self._create_and_display_chart(selected_df, selected_metrics, metric_directions)
            
            # 지표 정보 테이블 추가
            self._display_metric_info_table(selected_df, selected_metrics, metric_directions)
                
        except Exception as e:
            st.error(f"차트 생성 중 오류 발생: {e}")
    
    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        데이터프레임 전처리: 번호(No.)와 모델 ID 추가
        
        Args:
            df: 원본 데이터프레임
            
        Returns:
            전처리된 데이터프레임
        """
        # 데이터프레임에 번호 컬럼 추가
        numbered_df = df.copy().reset_index(drop=True)
        numbered_df['No.'] = numbered_df.index + 1
        
        # 모델 식별을 위한 고유 식별자 생성
        has_optimizer = 'optimizer' in numbered_df.columns
        has_lr = 'lr' in numbered_df.columns
        
        if has_optimizer or has_lr:
            numbered_df['model_id'] = numbered_df.apply(
                lambda row: f"{row['Model']}" + 
                        (f" ({row['optimizer']})" if has_optimizer else "") +
                        (f" (lr={row['lr']})" if has_lr else ""),
                axis=1
            )
        else:
            numbered_df['model_id'] = numbered_df['Model']
        
        return numbered_df
    
    def _create_metrics_selection_ui(self, metric_columns: List[str]) -> List[str]:
        """
        지표 선택 UI 생성
        
        Args:
            metric_columns: 사용 가능한 지표 열 목록
            
        Returns:
            선택된 지표 목록
        """
        st.markdown("### 성능 지표 선택")
        default_metrics = [metric_columns[0]] if metric_columns else []
        
        return st.multiselect(
            "비교할 성능 지표를 선택하세요 (여러 개 선택 가능):",
            options=metric_columns,
            default=default_metrics
        )
    
    def _create_model_selection_ui(self, df: pd.DataFrame, display_columns: List[str]) -> List[int]:
        """
        모델 선택 UI 생성
        
        Args:
            df: 전처리된 데이터프레임
            display_columns: 표시할 컬럼 목록
            
        Returns:
            선택된 모델 번호 목록
        """
        st.markdown("### 비교할 모델 선택")

        # 전체 데이터 테이블을 확장/축소 가능한 섹션으로 표시
        with st.expander("전체 모델 데이터 표 보기", expanded=False):
            st.markdown("아래 표에서 모델 번호와 데이터를 확인하세요:")
            
            # 데이터프레임 포맷 설정
            format_dict = self._create_format_dict(df)
            
            # 데이터프레임 표시
            st.dataframe(
                df[display_columns].style.format(format_dict),
                use_container_width=True,
                hide_index=True
            )
            
        # 모델 선택 UI 생성
        model_labels = self._create_model_labels(df)
        all_model_numbers = df['No.'].tolist()
        default_models = all_model_numbers[:min(5, len(all_model_numbers))]
        
        return st.multiselect(
            "비교할 모델 번호를 선택하세요:",
            options=all_model_numbers,
            format_func=lambda x: model_labels[x],  # 표시 형식 설정
            default=default_models
        )
    
    def _create_format_dict(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        데이터프레임 포맷 딕셔너리 생성
        
        Args:
            df: 데이터프레임
            
        Returns:
            포맷 딕셔너리
        """
        format_dict = {}
        
        # 모든 열에 대해 확인
        for col in df.columns:
            # 숫자형 열이고 'No.'가 아닌 경우
            if pd.api.types.is_numeric_dtype(df[col]) and col != 'No.':
                if col == 'lr':
                    format_dict[col] = '{:.5f}'
                else:
                    format_dict[col] = '{:.4f}'
        
        return format_dict
    
    def _create_model_labels(self, df: pd.DataFrame) -> Dict[int, str]:
        """
        모델 라벨 딕셔너리 생성
        
        Args:
            df: 데이터프레임
            
        Returns:
            모델 번호를 키로 하는 라벨 딕셔너리
        """
        model_labels = {}
        has_optimizer = 'optimizer' in df.columns
        has_lr = 'lr' in df.columns
        
        for idx, row in df.iterrows():
            model_no = row['No.']
            label = f"No.{model_no}. {row['Model']}"
            
            if has_optimizer:
                label += f", {row['optimizer']}"
            
            if has_lr:
                label += f", {row['lr']:.5f}"
                
            model_labels[model_no] = label
            
        return model_labels
    
    def _get_metric_directions(self, metrics: List[str]) -> Dict[str, bool]:
        """
        각 지표의 방향(높을수록 좋은지, 낮을수록 좋은지) 정보 반환
        
        Args:
            metrics: 지표 목록
            
        Returns:
            지표를 키로 하고 방향을 값으로 하는 딕셔너리
        """
        return {metric: self.metrics_analyzer.is_higher_better(metric) for metric in metrics}
    
    def _create_and_display_chart(
        self, 
        df: pd.DataFrame, 
        metrics: List[str], 
        metric_directions: Dict[str, bool]
    ) -> None:
        """
        차트 생성 및 표시
        
        Args:
            df: 데이터프레임
            metrics: 지표 목록
            metric_directions: 지표 방향 정보
        """
        fig = go.Figure()
        
        # x축 값 설정
        x_values = df['No.'].astype(str).tolist()
        
        # 범례용 더미 트레이스 추가
        self._add_dummy_traces(fig, metrics, metric_directions)
        
        # 데이터 트레이스 추가
        self._add_data_traces(fig, df, metrics, metric_directions, x_values)
        
        # 그래프 레이아웃 설정
        self._setup_chart_layout(fig)
        
        # 차트 표시
        st.plotly_chart(fig, use_container_width=True)
        
        # 주의 사항 추가
        self._display_chart_info()
    
    def _add_dummy_traces(
        self, 
        fig: go.Figure, 
        metrics: List[str], 
        metric_directions: Dict[str, bool]
    ) -> None:
        """
        범례용 더미 트레이스 추가
        
        Args:
            fig: Plotly 그래프 객체
            metrics: 지표 목록
            metric_directions: 지표 방향 정보
        """
        for i, metric in enumerate(metrics):
            # 지표의 방향 (높을수록 좋은지, 낮을수록 좋은지)
            is_higher_better = metric_directions[metric]
            direction_symbol = self.UP_SYMBOL if is_higher_better else self.DOWN_SYMBOL
            
            # 색상 선택 (색상 순환)
            color_idx = i % len(self.COLORS)
            bar_color = self.COLORS[color_idx]
            
            # 범례용 더미 트레이스 추가 (데이터 없음, 범례에만 표시)
            fig.add_trace(
                go.Bar(
                    x=[None],
                    y=[None],
                    name=f"{metric} {direction_symbol}",
                    marker_color=bar_color,
                    marker_opacity=0.7
                )
            )
    
    def _add_data_traces(
        self, 
        fig: go.Figure, 
        df: pd.DataFrame, 
        metrics: List[str], 
        metric_directions: Dict[str, bool], 
        x_values: List[str]
    ) -> None:
        """
        데이터 트레이스 추가
        
        Args:
            fig: Plotly 그래프 객체
            df: 데이터프레임
            metrics: 지표 목록
            metric_directions: 지표 방향 정보
            x_values: x축 값
        """
        for i, metric in enumerate(metrics):
            # 지표의 방향 확인
            is_higher_better = metric_directions[metric]
            direction_symbol = self.UP_SYMBOL if is_higher_better else self.DOWN_SYMBOL
            
            # 색상 선택
            color_idx = i % len(self.COLORS)
            bar_color = self.COLORS[color_idx]
            
            # 최적 모델 인덱스 계산
            if is_higher_better:
                optimal_idx = df[metric].values.argmax()
            else:
                optimal_idx = df[metric].values.argmin()
            
            # 최적 모델을 위한 패턴 및 불투명도 설정
            marker_pattern = [''] * len(df)
            marker_pattern[optimal_idx] = '/'
            
            opacity = [0.7] * len(df)
            opacity[optimal_idx] = 1.0
            
            # 막대 추가
            fig.add_trace(
                go.Bar(
                    x=x_values,
                    y=df[metric],
                    name=f"{metric} {direction_symbol}",
                    marker_color=bar_color,
                    marker_opacity=opacity,
                    marker_pattern_shape=marker_pattern,
                    marker_line=dict(
                        width=[2 if j == optimal_idx else 0 for j in range(len(df))],
                        color='black'
                    ),
                    text=df[metric].apply(lambda x: f'{x:.4f}'),
                    textposition='outside',
                    textfont=dict(size=10),
                    hovertemplate='<b>모델 번호: %{x}</b><br>' +
                                '모델: %{customdata}<br>' +
                                f'{metric}: %{{y:.4f}} {direction_symbol}<br>' +
                                '<b>%{marker.pattern.shape}</b><extra></extra>',
                    customdata=df['model_id'],
                    showlegend=False  # 범례에 표시하지 않음
                )
            )
    
    def _setup_chart_layout(self, fig: go.Figure) -> None:
        """
        차트 레이아웃 설정
        
        Args:
            fig: Plotly 그래프 객체
        """
        fig.update_layout(
            title='모델별 다중 지표 성능 비교',
            xaxis_title='모델 번호',
            yaxis_title='지표 값',
            barmode='group',  # 그룹화된 막대 차트
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=20, r=20, t=70, b=20),
        )
    
    def _display_chart_info(self) -> None:
        """
        차트 해석 정보 표시
        """
        st.info("""
        **그래프 해석 방법:**
        - 지표명 옆의 화살표는 해당 지표가 높을수록 좋은지(↑), 낮을수록 좋은지(↓)를 나타냅니다.
        - 각 지표는 서로 다른 색상으로 표시됩니다.
        - 패턴(/)이 있거나 진한 색상의 막대는 해당 지표에서 최적 성능을 보이는 모델입니다.
        - 지표 단위가 서로 다를 수 있으므로, Y축 값의 직접적인 비교보다는 각 지표별 상대적 비교에 집중하세요.
        """)
    
    def _display_metric_info_table(
        self, 
        df: pd.DataFrame, 
        metrics: List[str], 
        metric_directions: Dict[str, bool]
    ) -> None:
        """
        지표 정보 테이블 표시
        
        Args:
            df: 데이터프레임
            metrics: 지표 목록
            metric_directions: 지표 방향 정보
        """
        st.markdown("### 지표 정보")
        metric_info_data = self._prepare_metric_info_data(df, metrics, metric_directions)
        
        metric_info_df = pd.DataFrame(metric_info_data)
        
        # HTML로 색상 표시를 포함한 테이블 생성
        st.write(metric_info_df.to_html(escape=False, index=False), unsafe_allow_html=True)
    
    def _prepare_metric_info_data(
        self, 
        df: pd.DataFrame, 
        metrics: List[str], 
        metric_directions: Dict[str, bool]
    ) -> List[Dict[str, str]]:
        """
        지표 정보 테이블 데이터 준비
        
        Args:
            df: 데이터프레임
            metrics: 지표 목록
            metric_directions: 지표 방향 정보
            
        Returns:
            테이블 데이터
        """
        metric_info_data = []
        
        for i, metric in enumerate(metrics):
            is_higher_better = metric_directions[metric]
            direction = f"높을수록 좋음 {self.UP_SYMBOL}" if is_higher_better else f"낮을수록 좋음 {self.DOWN_SYMBOL}"
            
            # 최적 모델 찾기
            if is_higher_better:
                best_idx = df[metric].idxmax()
            else:
                best_idx = df[metric].idxmin()
            
            best_model = df.loc[best_idx]
            
            # 색상 선택
            color_idx = i % len(self.COLORS)
            color_hex = self.COLORS[color_idx]
            
            metric_info_data.append({
                "지표명": metric,
                "방향": direction,
                "최적 모델": f"모델 {best_model['No.']} ({best_model['model_id']})",
                "최적값": f"{best_model[metric]:.4f}",
                "색상": f'<span style="color:{color_hex}">■</span>'
            })
            
        return metric_info_data