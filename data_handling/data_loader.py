import os
import tempfile
import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple


class DataLoader:
    """
    파일 또는 디렉토리에서 데이터 로드하는 클래스.
    """
    
    SUPPORTED_EXTENSIONS = ['.csv', '.xlsx', '.xls']
    
    def __init__(self):
        """
        DataLoader 클래스 초기화.
        """
        pass
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def load_data(file_path: str) -> pd.DataFrame:
        """
        파일 또는 디렉토리에서 데이터 로드.
        
        Args:
            file_path: 로드할 파일 또는 디렉토리 경로
            
        Returns:
            로드된 데이터프레임
        """
        path = Path(file_path)
        
        if path.is_file():
            return DataLoader._load_file(path)
        elif path.is_dir():
            return DataLoader._load_directory(path)
        else:
            st.error(f"'{file_path}'가 유효한 파일 또는 디렉토리가 아닙니다.")
            return pd.DataFrame()
    
    @staticmethod
    def _load_file(path: Path) -> pd.DataFrame:
        """
        단일 파일 로드.
        
        Args:
            path: 로드할 파일 경로
            
        Returns:
            로드된 데이터프레임
        """
        if path.suffix.lower() in DataLoader.SUPPORTED_EXTENSIONS:
            try:
                return DataLoader._read_file(path)
            except Exception as e:
                st.error(f"파일 로드 중 오류 발생: {e}")
                return pd.DataFrame()
        else:
            st.error(f"지원되지 않는 파일 형식입니다. {', '.join(DataLoader.SUPPORTED_EXTENSIONS)} 파일만 업로드해 주세요.")
            return pd.DataFrame()
    
    @staticmethod
    def _load_directory(path: Path) -> pd.DataFrame:
        """
        디렉토리 내 지원되는 모든 파일을 로드하고 병합.
        
        Args:
            path: 로드할 디렉토리 경로
            
        Returns:
            병합된 데이터프레임
        """
        # 디렉토리 내 모든 지원 파일 찾기
        all_files = []
        for ext in DataLoader.SUPPORTED_EXTENSIONS:
            all_files.extend(list(path.glob(f'*{ext}')))
        
        if not all_files:
            st.error(f"'{path}' 디렉토리에 지원되는 파일이 없습니다.")
            return pd.DataFrame()
        
        # 모든 파일 로드 및 병합
        dfs = []
        for file in all_files:
            try:
                df = DataLoader._read_file(file)
                # 파일 이름을 데이터에 추가
                df['source_file'] = file.name
                dfs.append(df)
            except Exception as e:
                st.warning(f"파일 '{file.name}' 로드 중 오류 발생: {e}")
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()
    
    @staticmethod
    def _read_file(file_path: Path) -> pd.DataFrame:
        """
        파일 형식에 따라 파일 읽기.
        
        Args:
            file_path: 읽을 파일 경로
            
        Returns:
            로드된 데이터프레임
        """
        suffix = file_path.suffix.lower()
        if suffix == '.csv':
            return pd.read_csv(file_path)
        elif suffix in ['.xlsx', '.xls']:
            return pd.read_excel(file_path, engine='openpyxl')
        else:
            raise ValueError(f"지원되지 않는 파일 형식: {suffix}")
    
    @staticmethod
    def process_uploaded_file(uploaded_file) -> pd.DataFrame:
        """
        Streamlit에서 업로드된 파일 처리.
        
        Args:
            uploaded_file: Streamlit의 업로드된 파일 객체
            
        Returns:
            로드된 데이터프레임
        """
        try:
            # 임시 파일로 저장하여 처리
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # 파일 확장자에 따라 적절한 방법으로 로드
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(tmp_path)
            else:  # Excel 파일
                df = pd.read_excel(tmp_path, engine='openpyxl')
            
            # 임시 파일 삭제
            os.unlink(tmp_path)
            
            return df
        except Exception as e:
            st.sidebar.error(f"파일 로드 중 오류 발생: {e}")
            return pd.DataFrame()