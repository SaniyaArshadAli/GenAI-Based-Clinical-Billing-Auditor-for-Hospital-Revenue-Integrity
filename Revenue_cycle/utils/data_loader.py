"""
Data loading and preprocessing module for Medicare Provider Data and IPPS Final Rule.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import streamlit as st
import os

class MedicareDataPipeline:
    """Pipeline for loading and processing Medicare and IPPS datasets."""
    
    def __init__(self):
        self.df_provider = None
        self.df_ipps = None
        self.df_merged = None
        self.provider_file_path = None
        self.ipps_file_path = None
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
    
    def find_files(self) -> bool:
        """Auto-detect required files."""
        search_dirs = [
            self.base_dir,
            os.path.dirname(self.base_dir),
            os.path.join(os.path.expanduser("~"), "Desktop", "revenue_billing"),
            os.path.join(os.path.expanduser("~"), "Desktop", "revenue_billing", "utils"),
        ]
        
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
            try:
                files = os.listdir(search_dir)
            except:
                continue
            
            for file in files:
                file_path = os.path.join(search_dir, file)
                
                if 'MUP' in file.upper() and not self.provider_file_path:
                    self.provider_file_path = file_path
                    st.success(f"📁 Found: {file}")
                
                if 'IPPS' in file.upper() and not self.ipps_file_path:
                    self.ipps_file_path = file_path
                    st.success(f"📁 Found: {file}")
        
        if not self.provider_file_path or not self.ipps_file_path:
            st.error("❌ Could not find required files")
            return False
        return True
    
    def load_provider_data(self) -> Optional[pd.DataFrame]:
        """Load Medicare Provider data."""
        try:
            cols = ['Rndrng_Prvdr_Org_Name', 'Rndrng_Prvdr_City', 'Rndrng_Prvdr_State_Abrvtn',
                    'DRG_Cd', 'DRG_Desc', 'Tot_Dschrgs', 'Avg_Submtd_Cvrd_Chrg']
            
            for enc in ['latin1', 'iso-8859-1', 'cp1252', 'utf-8']:
                try:
                    df = pd.read_csv(self.provider_file_path, usecols=cols, 
                                   encoding=enc, low_memory=False, nrows=50000)
                    break
                except:
                    continue
            
            df = df.dropna(subset=['DRG_Cd', 'Rndrng_Prvdr_Org_Name'])
            df['Avg_Submtd_Cvrd_Chrg'] = pd.to_numeric(df['Avg_Submtd_Cvrd_Chrg'], errors='coerce').fillna(0)
            df['Tot_Dschrgs'] = pd.to_numeric(df['Tot_Dschrgs'], errors='coerce').fillna(0)
            
            # Clean DRG codes
            df['DRG_Cd'] = df['DRG_Cd'].apply(lambda x: str(x).strip().zfill(3) if pd.notna(x) else '000')
            
            self.df_provider = df
            st.success(f"✅ Loaded {len(df):,} provider records")
            return df
        except Exception as e:
            st.error(f"Provider error: {e}")
            return None
    
    def load_ipps_data(self) -> Optional[pd.DataFrame]:
        """Load IPPS Table 5 data."""
        try:
            # Read the file
            if self.ipps_file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(self.ipps_file_path, skiprows=1)
            else:
                df = pd.read_csv(self.ipps_file_path, skiprows=1, encoding='latin1')
            
            # Remove duplicate columns
            df = df.loc[:, ~df.columns.duplicated()]
            
            # Clean column names
            df.columns = [str(c).strip() for c in df.columns]
            
            st.write("📋 IPPS columns found:", df.columns.tolist()[:15])
            
            # Find DRG and Weight columns
            drg_col = None
            weight_col = None
            title_col = None
            los_col = None
            
            for col in df.columns:
                cu = col.upper()
                if 'DRG' in cu and drg_col is None and 'TITLE' not in cu and 'DESC' not in cu:
                    drg_col = col
                if 'WEIGHT' in cu and weight_col is None:
                    weight_col = col
                if ('TITLE' in cu or 'DESC' in cu) and title_col is None:
                    title_col = col
                if 'LOS' in cu or 'LENGTH' in cu or 'GEOMETRIC' in cu:
                    los_col = col
            
            if drg_col is None or weight_col is None:
                st.error(f"Cannot find columns. DRG={drg_col}, Weight={weight_col}")
                st.write("All columns:", df.columns.tolist())
                return None
            
            st.success(f"Using columns - DRG: '{drg_col}', Weight: '{weight_col}'")
            
            # Create clean dataframe with EXPECTED column names (matching app.py)
            clean_df = pd.DataFrame()
            
            # Clean DRG codes
            clean_df['MS-DRG'] = df[drg_col].apply(
                lambda x: str(int(float(str(x).strip()))) if pd.notna(x) and str(x).strip().replace('.','').isdigit() else '000'
            )
            clean_df['MS-DRG'] = clean_df['MS-DRG'].apply(lambda x: x.zfill(3))
            
            # Keep original column name for weights (app.py expects 'Weights - Before Cap')
            clean_df['Weights - Before Cap'] = pd.to_numeric(df[weight_col], errors='coerce').fillna(0)
            
            # Add title if available (app.py expects 'MS-DRG Title')
            if title_col:
                clean_df['MS-DRG Title'] = df[title_col].astype(str)
            else:
                clean_df['MS-DRG Title'] = 'Unknown DRG'
            
            # Add LOS if available
            if los_col:
                clean_df['Geometric mean LOS'] = pd.to_numeric(df[los_col], errors='coerce').fillna(0)
            
            # Filter valid rows
            clean_df = clean_df[clean_df['Weights - Before Cap'] > 0]
            clean_df = clean_df[clean_df['MS-DRG'].str.match(r'^\d{3}$')]
            
            self.df_ipps = clean_df
            st.success(f"✅ Loaded {len(clean_df):,} IPPS records")
            st.write(f"   Sample DRGs: {clean_df['MS-DRG'].head(3).tolist()}")
            st.write(f"   Sample Weights: {clean_df['Weights - Before Cap'].head(3).tolist()}")
            return clean_df
            
        except Exception as e:
            st.error(f"IPPS error: {e}")
            import traceback
            st.code(traceback.format_exc())
            return None
    
    def merge_datasets(self) -> Optional[pd.DataFrame]:
        """Merge datasets."""
        try:
            merged = pd.merge(
                self.df_provider,
                self.df_ipps[['MS-DRG', 'Weights - Before Cap', 'MS-DRG Title']],
                left_on='DRG_Cd',
                right_on='MS-DRG',
                how='inner'
            )
            
            # Calculate estimated revenue
            merged['Estimated_Revenue'] = (
                merged['Weights - Before Cap'].astype(float) * 
                merged['Tot_Dschrgs'].astype(float) * 6500
            )
            
            st.success(f"✅ Merged: {len(merged):,} records across {merged['Rndrng_Prvdr_Org_Name'].nunique():,} providers")
            return merged
        except Exception as e:
            st.error(f"Merge error: {e}")
            import traceback
            st.code(traceback.format_exc())
            return None
    
    def run_pipeline(self):
        """Run the full pipeline."""
        if not self.find_files():
            return None, None
        
        st.write("---")
        st.write("### 📊 Data Loading Progress")
        
        provider = self.load_provider_data()
        if provider is None:
            return None, None
        
        ipps = self.load_ipps_data()
        if ipps is None:
            return None, None
        
        merged = self.merge_datasets()
        if merged is None:
            return None, None
        
        st.balloons()
        return merged, ipps


@st.cache_data(ttl=3600)
def load_and_merge_data():
    """Cached wrapper."""
    pipeline = MedicareDataPipeline()
    return pipeline.run_pipeline()