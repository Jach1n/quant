# quant_app.py (优化版)
import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import os
import warnings
import sqlite3
import hashlib
from typing import List, Dict, Tuple
import io
import re
import json

warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="幻方量化增强分析系统", 
    layout="wide",
    page_icon="🎯"
)

# ==================== 数据库缓存模块 ====================
class DataCache:
    def __init__(self, db_path="stock_data_cache.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建股票数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_data (
                stock_code TEXT,
                date TEXT,
                open REAL,
                close REAL,
                high REAL,
                low REAL,
                volume REAL,
                turnover REAL,
                amplitude REAL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (stock_code, date)
            )
        ''')
        
        # 创建技术指标缓存表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS technical_indicators (
                stock_code TEXT,
                calculation_date TEXT,
                indicators_blob BLOB,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (stock_code, calculation_date)
            )
        ''')
        
        # 创建股票代码-名称映射表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_name_mapping (
                stock_code TEXT PRIMARY KEY,
                stock_name TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建股票基本信息表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_basic_info (
                stock_code TEXT PRIMARY KEY,
                stock_name TEXT NOT NULL,
                market_type TEXT,
                listing_date TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建自选股表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS watchlist (
                stock_code TEXT PRIMARY KEY,
                stock_name TEXT,
                added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建模型验证记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_validation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_code TEXT,
                stock_name TEXT,
                model_type TEXT,
                prediction_date TEXT,
                signal TEXT,
                confidence REAL,
                actual_return REAL,
                actual_direction TEXT,
                is_correct INTEGER,
                base_score REAL,
                quality_bonus REAL,
                confidence_weighted REAL,
                total_score REAL,
                cumulative_score REAL,
                validation_status TEXT,
                created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建用户验证数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_validation_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_code TEXT,
                stock_name TEXT,
                validation_date TEXT,
                actual_return REAL,
                actual_direction TEXT,
                user_notes TEXT,
                created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(stock_code, validation_date)
            )
        ''')
        
        # 创建自动验证结果表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS auto_validation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_code TEXT,
                stock_name TEXT,
                validation_date TEXT,
                prediction_signal TEXT,
                actual_return REAL,
                is_correct INTEGER,
                validation_type TEXT,
                created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(stock_code, validation_date, validation_type)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_cached_stock_data(self, stock_code: str, days: int) -> pd.DataFrame:
        """从缓存获取股票数据"""
        conn = sqlite3.connect(self.db_path)
        
        # 获取最近days天的数据
        query = '''
            SELECT date, open, close, high, low, volume, turnover, amplitude
            FROM stock_data 
            WHERE stock_code = ? 
            ORDER BY date DESC 
            LIMIT ?
        '''
        df = pd.read_sql_query(query, conn, params=[stock_code, days])
        conn.close()
        
        if not df.empty:
            df['日期'] = pd.to_datetime(df['date'])
            df = df.drop('date', axis=1)
            # 重新排列列顺序以匹配原始格式
            df = df[['日期', 'open', 'close', 'high', 'low', 'volume', 'turnover', 'amplitude']]
            df.columns = ['日期', '开盘', '收盘', '最高', '最低', '成交量(万手)', '成交额(万元)', '振幅(%)']
            return df.sort_values('日期').reset_index(drop=True)
        return pd.DataFrame()
    
    def cache_stock_data(self, stock_code: str, df: pd.DataFrame):
        """缓存股票数据到数据库"""
        conn = sqlite3.connect(self.db_path)
        
        # 准备数据
        cache_data = []
        for _, row in df.iterrows():
            cache_data.append((
                stock_code,
                row['日期'].strftime('%Y-%m-%d'),
                row['开盘'],
                row['收盘'],
                row['最高'],
                row['最低'],
                row['成交量(万手)'],
                row['成交额(万元)'],
                row['振幅(%)']
            ))
        
        # 使用INSERT OR REPLACE来更新已存在的数据
        cursor = conn.cursor()
        cursor.executemany('''
            INSERT OR REPLACE INTO stock_data 
            (stock_code, date, open, close, high, low, volume, turnover, amplitude)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', cache_data)
        
        conn.commit()
        conn.close()
    
    def get_cached_indicators(self, stock_code: str, calculation_date: str) -> pd.DataFrame:
        """获取缓存的技术指标"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT indicators_blob 
            FROM technical_indicators 
            WHERE stock_code = ? AND calculation_date = ?
        '''
        cursor = conn.cursor()
        cursor.execute(query, (stock_code, calculation_date))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            # 从blob数据重建DataFrame
            return pd.read_pickle(io.BytesIO(result[0]))
        return pd.DataFrame()
    
    def cache_indicators(self, stock_code: str, calculation_date: str, df: pd.DataFrame):
        """缓存技术指标数据"""
        conn = sqlite3.connect(self.db_path)
        
        # 将DataFrame转换为blob
        blob = io.BytesIO()
        pd.to_pickle(df, blob)
        blob_value = blob.getvalue()
        
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO technical_indicators 
            (stock_code, calculation_date, indicators_blob)
            VALUES (?, ?, ?)
        ''', (stock_code, calculation_date, blob_value))
        
        conn.commit()
        conn.close()
    
    def cache_stock_name(self, stock_code: str, stock_name: str):
        """缓存股票名称"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO stock_name_mapping 
            (stock_code, stock_name)
            VALUES (?, ?)
        ''', (stock_code, stock_name))
        conn.commit()
        conn.close()
    
    def get_cached_stock_name(self, stock_code: str) -> str:
        """获取缓存的股票名称"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT stock_name FROM stock_name_mapping WHERE stock_code = ?', (stock_code,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None

    def get_stock_name_from_basic_info(self, stock_code: str) -> str:
        """从股票基本信息表获取股票名称"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT stock_name FROM stock_basic_info WHERE stock_code = ?', (stock_code,))
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else None
        except:
            return None

    def get_all_stocks_from_basic_info(self) -> List[Dict]:
        """从股票基本信息表获取所有股票"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT stock_code, stock_name, market_type FROM stock_basic_info ORDER BY stock_code')
            results = cursor.fetchall()
            conn.close()
            
            stocks = []
            for code, name, market in results:
                stocks.append({
                    'code': code,
                    'name': name,
                    'market': market
                })
            return stocks
        except:
            return []

    def search_stocks_from_basic_info(self, keyword: str) -> List[Dict]:
        """从股票基本信息表搜索股票"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            query = '''
                SELECT stock_code, stock_name, market_type 
                FROM stock_basic_info 
                WHERE stock_code LIKE ? OR stock_name LIKE ?
                ORDER BY stock_code
                LIMIT 20
            '''
            search_term = f"%{keyword}%"
            cursor.execute(query, (search_term, search_term))
            results = cursor.fetchall()
            conn.close()
            
            stocks = []
            for code, name, market in results:
                stocks.append({
                    'code': code,
                    'name': name,
                    'market': market
                })
            return stocks
        except:
            return []
    
    def get_stock_count(self) -> int:
        """获取股票基本信息表中的股票数量"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM stock_basic_info")
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else 0
        except Exception as e:
            print(f"获取股票数量失败: {e}")
            return 0

    # ==================== 自选股管理功能 ====================
    def add_to_watchlist(self, stock_code: str, stock_name: str):
        """添加股票到自选股"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO watchlist 
            (stock_code, stock_name)
            VALUES (?, ?)
        ''', (stock_code, stock_name))
        conn.commit()
        conn.close()
    
    def remove_from_watchlist(self, stock_code: str):
        """从自选股移除股票"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM watchlist WHERE stock_code = ?', (stock_code,))
        conn.commit()
        conn.close()
    
    def get_watchlist(self) -> List[Dict]:
        """获取自选股列表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT stock_code, stock_name FROM watchlist ORDER BY added_date DESC')
        results = cursor.fetchall()
        conn.close()
        
        watchlist = []
        for stock_code, stock_name in results:
            watchlist.append({
                'code': stock_code,
                'name': stock_name
            })
        return watchlist
    
    def clear_old_cache(self, days_to_keep: int = 30):
        """清理过期缓存数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 删除过期的股票数据（保留最近days_to_keep天的数据）
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).strftime('%Y-%m-%d')
        cursor.execute('DELETE FROM stock_data WHERE date < ?', (cutoff_date,))
        
        # 删除过期的技术指标缓存（保留最近7天的计算）
        indicator_cutoff = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        cursor.execute('DELETE FROM technical_indicators WHERE calculation_date < ?', (indicator_cutoff,))
        
        conn.commit()
        conn.close()

    # ==================== 模型验证功能 ====================
    def save_validation_record(self, record: Dict):
        """保存模型验证记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_validation 
            (stock_code, stock_name, model_type, prediction_date, signal, confidence,
             actual_return, actual_direction, is_correct, base_score, quality_bonus,
             confidence_weighted, total_score, cumulative_score, validation_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            record['stock_code'], record['stock_name'], record['model_type'],
            record['prediction_date'], record['signal'], record['confidence'],
            record.get('actual_return', 0), record.get('actual_direction', ''),
            record.get('is_correct', 0), record.get('base_score', 0),
            record.get('quality_bonus', 0), record.get('confidence_weighted', 0),
            record.get('total_score', 0), record.get('cumulative_score', 0),
            record.get('validation_status', '待验证')
        ))
        
        conn.commit()
        conn.close()
    
    def get_validation_records(self, limit: int = 100) -> pd.DataFrame:
        """获取模型验证记录（去重版本）"""
        conn = sqlite3.connect(self.db_path)
        
        # 使用DISTINCT去重，基于股票代码、预测日期和模型类型
        query = f'''
            SELECT DISTINCT 
                stock_code, stock_name, model_type, prediction_date, 
                signal, confidence, actual_return, actual_direction, 
                is_correct, validation_status, created_time
            FROM model_validation 
            ORDER BY created_time DESC 
            LIMIT {limit}
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def get_unique_validation_records(self) -> pd.DataFrame:
        """获取去重后的验证记录，每个股票每个预测日期只保留最新的一条"""
        conn = sqlite3.connect(self.db_path)
        
        # 使用子查询获取每个股票在每个预测日期的最新记录
        query = '''
            SELECT mv.* 
            FROM model_validation mv
            INNER JOIN (
                SELECT stock_code, prediction_date, MAX(created_time) as max_time
                FROM model_validation 
                GROUP BY stock_code, prediction_date
            ) latest 
            ON mv.stock_code = latest.stock_code 
            AND mv.prediction_date = latest.prediction_date 
            AND mv.created_time = latest.max_time
            ORDER BY mv.created_time DESC
            LIMIT 50
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def update_validation_result(self, record_id: int, actual_return: float, is_correct: bool):
        """更新验证结果"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        actual_direction = "上涨" if actual_return > 0 else "下跌" if actual_return < 0 else "平盘"
        
        cursor.execute('''
            UPDATE model_validation 
            SET actual_return = ?, actual_direction = ?, is_correct = ?, validation_status = '已验证'
            WHERE id = ?
        ''', (actual_return, actual_direction, 1 if is_correct else 0, record_id))
        
        conn.commit()
        conn.close()
    
    def save_user_validation_data(self, stock_code: str, stock_name: str, validation_date: str, 
                                actual_return: float, user_notes: str = ""):
        """保存用户验证数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        actual_direction = "上涨" if actual_return > 0 else "下跌" if actual_return < 0 else "平盘"
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_validation_data 
            (stock_code, stock_name, validation_date, actual_return, actual_direction, user_notes)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (stock_code, stock_name, validation_date, actual_return, actual_direction, user_notes))
        
        conn.commit()
        conn.close()
    
    def get_user_validation_data(self, validation_date: str = None) -> pd.DataFrame:
        """获取用户验证数据"""
        conn = sqlite3.connect(self.db_path)
        
        if validation_date:
            query = '''
                SELECT * FROM user_validation_data 
                WHERE validation_date = ?
                ORDER BY created_time DESC
            '''
            df = pd.read_sql_query(query, conn, params=[validation_date])
        else:
            query = '''
                SELECT * FROM user_validation_data 
                ORDER BY validation_date DESC, created_time DESC
            '''
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        return df
    
    def save_auto_validation_result(self, stock_code: str, stock_name: str, validation_date: str,
                                  prediction_signal: str, actual_return: float, validation_type: str = "auto"):
        """保存自动验证结果"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 判断预测是否正确
        if "买入" in prediction_signal:
            is_correct = 1 if actual_return > 0 else 0
        elif "卖出" in prediction_signal:
            is_correct = 1 if actual_return < 0 else 0
        else:  # 观望信号
            is_correct = 1 if abs(actual_return) < 0.01 else 0
        
        cursor.execute('''
            INSERT OR REPLACE INTO auto_validation_results 
            (stock_code, stock_name, validation_date, prediction_signal, actual_return, is_correct, validation_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (stock_code, stock_name, validation_date, prediction_signal, actual_return, is_correct, validation_type))
        
        conn.commit()
        conn.close()
    
    def get_auto_validation_results(self, days: int = 30) -> pd.DataFrame:
        """获取自动验证结果"""
        conn = sqlite3.connect(self.db_path)
        
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        query = '''
            SELECT * FROM auto_validation_results 
            WHERE validation_date >= ?
            ORDER BY validation_date DESC, created_time DESC
        '''
        df = pd.read_sql_query(query, conn, params=[cutoff_date])
        conn.close()
        return df

# ==================== 配置模块 ====================
class QuantConfig:
    REQUEST_TIMEOUT = 10
    REQUEST_DELAY = 0.5
    DEFAULT_DAYS = 250
    
    # 删除硬编码的股票映射，改为从数据库读取
    TECH_FACTORS = ['momentum', 'volatility', 'volume', 'reversal', 'liquidity', 'trend']
    WEIGHTS = {
        'momentum': 0.22,
        'volatility': 0.18,
        'volume': 0.16,
        'reversal': 0.14,
        'liquidity': 0.12,
        'trend': 0.18
    }
    MAX_POSITION_SIZE = 0.1
    STOP_LOSS = -0.05
    TAKE_PROFIT = 0.10

# 初始化数据缓存
data_cache = DataCache()

# ==================== 股票搜索和名称获取 ====================
def search_stock_by_name(stock_name: str) -> List[Dict]:
    """根据股票名称搜索股票 - 使用本地数据库"""
    try:
        # 使用本地数据库搜索
        results = data_cache.search_stocks_from_basic_info(stock_name)
        return results
    except Exception as e:
        st.error(f"搜索股票失败: {e}")
        return []

def extract_stock_code(user_input: str) -> str:
    """从用户输入中提取股票代码"""
    # 移除空格和特殊字符
    cleaned = re.sub(r'[^\w]', '', user_input)
    
    # 尝试匹配股票代码模式 (6位数字)
    match = re.search(r'(\d{6})', cleaned)
    if match:
        return match.group(1)
    
    return None

def get_stock_name_from_api(stock_code: str) -> str:
    """从API获取股票名称"""
    try:
        # 使用东方财富API获取股票名称
        if stock_code.startswith('6'):
            secid = f"1.{stock_code}"
        elif stock_code.startswith('0') or stock_code.startswith('3'):
            secid = f"0.{stock_code}"
        else:
            return f"股票{stock_code}"
            
        url = f"https://push2.eastmoney.com/api/qt/stock/get"
        params = {
            "secid": secid,
            "fields": "f58,f14",  # f14:股票名称, f58:行业
            "invt": "2"
        }
        headers = {"User-Agent": "Mozilla/5.0"}
        
        resp = requests.get(url, params=params, headers=headers, timeout=QuantConfig.REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        
        stock_name = data.get("f14", "")
        if stock_name:
            # 缓存股票名称
            data_cache.cache_stock_name(stock_code, stock_name)
            return stock_name
    except Exception as e:
        st.sidebar.warning(f"无法获取 {stock_code} 的股票名称: {e}")
    
    return f"股票{stock_code}"

def get_stock_name(code: str) -> str:
    """获取股票名称（优先使用本地数据库）"""
    if not code:
        return "未知股票"
    
    # 确保代码是6位
    code = str(code).zfill(6)
    
    # 1. 首先从股票基本信息表查询
    basic_name = data_cache.get_stock_name_from_basic_info(code)
    if basic_name:
        return basic_name
    
    # 2. 然后检查缓存
    cached_name = data_cache.get_cached_stock_name(code)
    if cached_name:
        return cached_name
    
    # 3. 最后从API获取
    api_name = get_stock_name_from_api(code)
    if api_name and api_name != f"股票{code}":
        # 获取到后，缓存股票名称
        data_cache.cache_stock_name(code, api_name)
        return api_name
    
    # 4. 所有方法都失败，返回带代码的未知名称
    return f"股票{code}"

def validate_stock_code(stock_code: str) -> bool:
    """验证股票代码格式"""
    if not stock_code or len(stock_code) != 6:
        return False
    
    # 检查是否为数字
    if not stock_code.isdigit():
        return False
    
    # 检查市场代码 (6-沪市, 0/2/3-深市，其中2是中小板，3是创业板)
    first_char = stock_code[0]
    if first_char not in ['0', '2', '3', '6']:
        return False
    
    return True

def add_stock_to_watchlist(stock_input: str) -> bool:
    """添加股票到自选股的统一函数"""
    if not stock_input:
        return False
    
    # 清理输入
    cleaned_input = stock_input.strip()
    
    # 检查是否是6位数字股票代码
    if cleaned_input.isdigit() and len(cleaned_input) == 6:
        stock_code = cleaned_input
        
        # 直接尝试添加，不进行额外的验证
        # 因为有些新股票可能无法通过API获取到名称
        stock_name = get_stock_name(stock_code)
        
        # 即使获取不到名称，也允许添加，使用代码作为名称
        if not stock_name or stock_name == f"股票{stock_code}":
            stock_name = f"股票{stock_code}"
        
        data_cache.add_to_watchlist(stock_code, stock_name)
        return True
    
    # 如果不是纯数字，尝试搜索
    else:
        with st.spinner("搜索中..."):
            search_results = search_stock_by_name(cleaned_input)
        
        if search_results:
            # 如果有多个结果，让用户选择
            if len(search_results) == 1:
                # 只有一个结果，直接添加
                stock = search_results[0]
                data_cache.add_to_watchlist(stock['code'], stock['name'])
                return True
            else:
                # 多个结果，保存到session state供选择
                st.session_state.search_results = search_results
                return False
        else:
            st.error(f"未找到相关股票: {cleaned_input}")
            return False

# ==================== 数据获取（带缓存） ====================
def fetch_kline_data(stock_code: str, days: int = QuantConfig.DEFAULT_DAYS, use_cache: bool = True) -> pd.DataFrame:
    """从东方财富获取股票K线数据（带缓存功能）"""
    
    # 首先尝试从缓存获取数据
    if use_cache:
        cached_data = data_cache.get_cached_stock_data(stock_code, days)
        if not cached_data.empty:
            # 使用占位符显示缓存状态，不占用主界面空间
            if 'cache_status' not in st.session_state:
                st.session_state.cache_status = {}
            st.session_state.cache_status[stock_code] = "📦 使用缓存数据"
            return cached_data
    
    # 缓存中没有或强制刷新，从API获取
    if 'cache_status' not in st.session_state:
        st.session_state.cache_status = {}
    st.session_state.cache_status[stock_code] = "🌐 从API获取数据"
    
    # 修正：002开头的股票属于深市中小板，secid应该是0.002714
    if stock_code.startswith('6'):
        secid = f"1.{stock_code}"
    elif stock_code.startswith('0') or stock_code.startswith('3'):
        secid = f"0.{stock_code}"
    else:
        raise ValueError(f"不支持的股票代码格式: {stock_code}")
        
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
        "klt": "101",
        "fqt": "1",
        "end": "20500101",
        "lmt": days
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=QuantConfig.REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        klines = data.get("klines", [])
        
        if not klines:
            raise ValueError("未获取到K线数据")
        
        df = pd.DataFrame([x.split(",") for x in klines],
                          columns=["日期", "开盘", "收盘", "最高", "最低", "成交量(万手)", "成交额(万元)", "振幅(%)"])
        numeric_columns = ["开盘", "收盘", "最高", "最低", "成交量(万手)", "成交额(万元)", "振幅(%)"]
        df[numeric_columns] = df[numeric_columns].astype(float)
        df["日期"] = pd.to_datetime(df["日期"])
        
        # 缓存获取到的数据
        data_cache.cache_stock_data(stock_code, df)
        
        return df
        
    except Exception as e:
        st.error(f"获取 {stock_code} 数据失败: {e}")
        # 如果API失败，尝试返回缓存数据（即使不完整）
        cached_data = data_cache.get_cached_stock_data(stock_code, days)
        if not cached_data.empty:
            st.session_state.cache_status[stock_code] = "⚠️ 使用缓存的旧数据"
            return cached_data
        raise

# ==================== 指标计算（带缓存） ====================
def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['最高'] - df['最低']
    high_close = np.abs(df['最高'] - df['收盘'].shift())
    low_close = np.abs(df['最低'] - df['收盘'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    return true_range.rolling(period).mean()

def calculate_obv(df: pd.DataFrame) -> pd.Series:
    price_change = df['收盘'].diff()
    obv_dir = np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0))
    obv = (df['成交量(万手)'] * obv_dir).cumsum()
    return obv

def calculate_enhanced_adaptive_ma(df: pd.DataFrame, price_col: str = '收盘', period: int = 20) -> pd.Series:
    """改进的自适应移动平均线"""
    returns = df[price_col].pct_change().abs()
    volatility = returns.rolling(period, min_periods=10).std()
    
    # 使用更复杂的自适应参数
    base_alpha = 2 / (period + 1)
    volatility_factor = volatility * np.sqrt(252)
    adaptive_alpha = base_alpha * (1 + np.tanh(volatility_factor * 5 - 1))
    adaptive_alpha = np.clip(adaptive_alpha, 0.01, 0.3)  # 更严格的限制
    
    ema = df[price_col].copy()
    for i in range(1, len(ema)):
        if pd.notna(adaptive_alpha.iloc[i]):
            ema.iloc[i] = (adaptive_alpha.iloc[i] * df[price_col].iloc[i] + 
                          (1 - adaptive_alpha.iloc[i]) * ema.iloc[i-1])
    
    return ema

def calculate_enhanced_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """改进的ATR计算"""
    high_low = df['最高'] - df['最低']
    high_close = np.abs(df['最高'] - df['收盘'].shift())
    low_close = np.abs(df['最低'] - df['收盘'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    
    # 使用Wilder的平滑方法
    atr = true_range.ewm(alpha=1/period, adjust=False).mean()
    return atr

def calculate_enhanced_volume_profile(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """改进的成交量分析"""
    # 计算VWAP
    df['VWAP'] = (df['成交额(万元)'] * 10000) / (df['成交量(万手)'] * 10000 + 1e-8)
    
    # 成交量动量
    df['Volume_Momentum'] = df['成交量(万手)'] / (df['成交量(万手)'].rolling(period, min_periods=10).mean() + 1e-8)
    
    # 价量背离检测
    price_trend = df['收盘'].rolling(5).mean() - df['收盘'].rolling(20).mean()
    volume_trend = df['Volume_Momentum'].rolling(5).mean() - df['Volume_Momentum'].rolling(20).mean()
    df['Volume_Price_Divergence'] = price_trend * volume_trend
    
    # 成交量集中度
    df['Volume_Concentration'] = df['成交量(万手)'] / df['成交量(万手)'].rolling(period).sum()
    
    return df

def compute_enhanced_technical_indicators(df: pd.DataFrame, stock_code: str = "", use_cache: bool = True) -> pd.DataFrame:
    """计算技术指标（带缓存功能）- 优化版"""
    
    # 生成缓存键
    calculation_date = datetime.now().strftime('%Y-%m-%d')
    
    # 尝试从缓存获取技术指标
    if use_cache and stock_code:
        cached_indicators = data_cache.get_cached_indicators(stock_code, calculation_date)
        if not cached_indicators.empty:
            return cached_indicators
    
    # 优化1: 更准确的移动平均线计算
    for period in [5, 10, 20, 30, 60]:
        df[f'MA{period}'] = df['收盘'].rolling(period, min_periods=1).mean()
        df[f'EMA{period}'] = df['收盘'].ewm(span=period, adjust=False).mean()
    
    # 优化2: 改进的自适应移动平均线
    df['AMA_20'] = calculate_enhanced_adaptive_ma(df)
    
    # 优化3: 更准确的MACD计算
    ema_12 = df['收盘'].ewm(span=12, adjust=False).mean()
    ema_26 = df['收盘'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    df['MACD_Histogram_Change'] = df['MACD_Histogram'].diff()
    
    # 优化4: 动量指标增强
    for period in [5, 10, 20, 60]:
        df[f'Momentum_{period}D'] = (df['收盘'] / df['收盘'].shift(period) - 1) * 100
        df[f'ROC_{period}'] = df['收盘'].pct_change(periods=period) * 100
    
    # 动量加速度
    df['Momentum_Accel_5D'] = df['Momentum_5D'] - df['Momentum_10D']
    df['Momentum_Accel_10D'] = df['Momentum_10D'] - df['Momentum_20D']
    
    # 优化5: 改进的RSI计算，防止除零错误
    for period in [6, 14, 24]:
        delta = df['收盘'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        
        # 避免除零
        rs = gain / (loss + 1e-10)
        df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    
    # 优化6: 更稳健的KDJ计算
    low_9 = df["最低"].rolling(9, min_periods=1).min()
    high_9 = df["最高"].rolling(9, min_periods=1).max()
    rsv = ((df["收盘"] - low_9) / (high_9 - low_9 + 1e-10)) * 100
    
    # 使用更稳定的KDJ计算方法
    df["K"] = 50.0
    df["D"] = 50.0
    
    for i in range(1, len(df)):
        if pd.notna(rsv.iloc[i]):
            df.loc[df.index[i], "K"] = (2/3) * df.loc[df.index[i-1], "K"] + (1/3) * rsv.iloc[i]
            df.loc[df.index[i], "D"] = (2/3) * df.loc[df.index[i-1], "D"] + (1/3) * df.loc[df.index[i], "K"]
    
    df["J"] = 3 * df["K"] - 2 * df["D"]
    
    # KDJ信号
    df['KDJ_Golden_Cross'] = (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1))
    df['KDJ_Death_Cross'] = (df['K'] < df['D']) & (df['K'].shift(1) >= df['D'].shift(1))
    df['KDJ_Overbought'] = (df['K'] > 80) & (df['D'] > 80)
    df['KDJ_Oversold'] = (df['K'] < 20) & (df['D'] < 20)
    
    # 优化7: 改进的波动率计算
    returns = df['收盘'].pct_change()
    df['VOLATILITY_20D'] = returns.rolling(20, min_periods=10).std() * np.sqrt(252)
    df['VOLATILITY_60D'] = returns.rolling(60, min_periods=30).std() * np.sqrt(252)
    df['REALIZED_VOL'] = np.sqrt((returns**2).rolling(20, min_periods=10).sum() * 252)
    
    # 优化8: 增强的布林带
    df['BB_Middle'] = df['收盘'].rolling(20, min_periods=10).mean()
    df['BB_Std'] = df['收盘'].rolling(20, min_periods=10).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    df['BB_WIDTH_NORM'] = (df['BB_Upper'] - df['BB_Lower']) / (df['BB_Middle'] + 1e-10)
    df['BB_Position'] = (df['收盘'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-10)
    
    # 布林带突破信号
    df['BB_Breakout_Up'] = df['收盘'] > df['BB_Upper']
    df['BB_Breakout_Down'] = df['收盘'] < df['BB_Lower']
    
    # 优化9: ATR计算
    df['ATR'] = calculate_enhanced_atr(df)
    df['ATR_Ratio'] = df['ATR'] / (df['收盘'] + 1e-10)
    
    # 优化10: 成交量分析增强
    df = calculate_enhanced_volume_profile(df)
    df['Volume_MA20'] = df['成交量(万手)'].rolling(20, min_periods=10).mean()
    df['Volume_Ratio'] = df['成交量(万手)'] / (df['Volume_MA20'] + 1e-10)
    df['Volume_Spike'] = df['Volume_Ratio'] > 2.0  # 成交量突增信号
    
    df['OBV'] = calculate_obv(df)
    df['OBV_Momentum'] = df['OBV'].pct_change(5)
    df['OBV_Trend'] = df['OBV'].rolling(20).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1 if x.iloc[-1] < x.iloc[0] else 0)
    
    # 优化11: 趋势强度指标
    df['MA20_Slope'] = df['MA20'].diff(5) / (df['MA20'].shift(5) + 1e-10) * 100
    df['Trend_Strength'] = ((df['MA5'] > df['MA10']).astype(int) + 
                           (df['MA10'] > df['MA20']).astype(int) + 
                           (df['MA20'] > df['MA60']).astype(int)) / 3.0
    
    # 优化12: 价格位置指标
    rolling_min = df['最低'].rolling(20, min_periods=10).min()
    rolling_max = df['最高'].rolling(20, min_periods=10).max()
    df['Price_Position_20D'] = (df['收盘'] - rolling_min) / (rolling_max - rolling_min + 1e-10)
    
    # 相对强弱指标
    df['Relative_Strength'] = df['收盘'] / (df['收盘'].rolling(60, min_periods=30).mean() + 1e-10) - 1
    
    # 新增: 支撑阻力指标
    df['Support_Level'] = df['最低'].rolling(20, min_periods=10).min()
    df['Resistance_Level'] = df['最高'].rolling(20, min_periods=10).max()
    df['Support_Distance'] = (df['收盘'] - df['Support_Level']) / df['收盘']
    df['Resistance_Distance'] = (df['Resistance_Level'] - df['收盘']) / df['收盘']
    
    # 缓存技术指标
    if stock_code:
        data_cache.cache_indicators(stock_code, calculation_date, df)
    
    return df

# ==================== 信号生成（完整实现） ====================
class MarketRegimeDetector:
    @staticmethod
    def detect_market_regime(df: pd.DataFrame, lookback: int = 60):
        if len(df) < lookback:
            return "unknown"
        recent = df.tail(lookback)
        returns = recent['收盘'].pct_change().dropna()
        if len(returns) < 20:
            return "unknown"
        volatility = returns.std() * np.sqrt(252)
        autocorr = returns.autocorr()
        if 'TREND_CONSISTENCY' in recent.columns:
            trend_strength = abs(recent['TREND_CONSISTENCY'].iloc[-1])
        else:
            ma_trend = (recent['收盘'] > recent['MA20']).tail(10).mean()
            trend_strength = ma_trend
        if volatility > 0.30:
            return "high_volatility"
        elif volatility < 0.15:
            if trend_strength > 0.7:
                return "strong_trend"
            else:
                return "low_volatility"
        elif abs(autocorr) < 0.05:
            return "trending"
        elif autocorr > 0.1:
            return "momentum"
        else:
            return "mean_reverting"

class DynamicWeightAllocator:
    REGIME_WEIGHTS = {
        'strong_trend': {'momentum':0.28,'volatility':0.12,'volume':0.14,'reversal':0.08,'liquidity':0.10,'trend':0.28},
        'trending': {'momentum':0.24,'volatility':0.16,'volume':0.16,'reversal':0.10,'liquidity':0.12,'trend':0.22},
        'momentum': {'momentum':0.30,'volatility':0.14,'volume':0.18,'reversal':0.06,'liquidity':0.10,'trend':0.22},
        'mean_reverting': {'momentum':0.12,'volatility':0.16,'volume':0.14,'reversal':0.26,'liquidity':0.12,'trend':0.20},
        'high_volatility': {'momentum':0.18,'volatility':0.26,'volume':0.16,'reversal':0.12,'liquidity':0.08,'trend':0.20},
        'low_volatility': {'momentum':0.20,'volatility':0.14,'volume':0.18,'reversal':0.16,'liquidity':0.12,'trend':0.20},
        'unknown': QuantConfig.WEIGHTS
    }
    
    @staticmethod
    def get_regime_weights(regime: str):
        return DynamicWeightAllocator.REGIME_WEIGHTS.get(regime, QuantConfig.WEIGHTS)

class QuantSignalGenerator:
    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        self.weight_allocator = DynamicWeightAllocator()
        # 市场状态中英文映射
        self.regime_translation = {
            'strong_trend': '强势趋势',
            'trending': '趋势行情',
            'momentum': '动量行情',
            'mean_reverting': '均值回归',
            'high_volatility': '高波动',
            'low_volatility': '低波动',
            'unknown': '未知状态'
        }
        # 因子名称中英文映射
        self.factor_translation = {
            'momentum': '动量',
            'volatility': '波动率', 
            'volume': '成交量',
            'reversal': '反转',
            'liquidity': '流动性',
            'trend': '趋势'
        }
    
    def generate_technical_signal_with_score(self, df: pd.DataFrame) -> Tuple[str, str, float]:
        """使用传统技术指标生成买卖信号和评分 - 优化版"""
        if len(df) < 50:
            return "数据不足", "需要至少50个交易日数据", 0.0
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        signals = []
        reasons = []
        score_components = []
        
        # 优化1: 多时间框架均线分析 (权重: 20%)
        ma_score = 0.0
        ma_signals = []
        
        if all(col in df.columns for col in ['MA5', 'MA10', 'MA20', 'MA60']):
            # 多头排列检查
            ma_bullish = (latest['MA5'] > latest['MA10'] > latest['MA20'] > latest['MA60'])
            ma_bearish = (latest['MA5'] < latest['MA10'] < latest['MA20'] < latest['MA60'])
            
            if ma_bullish:
                ma_signals.append("均线呈完美多头排列")
                ma_score += 0.5
            elif ma_bearish:
                ma_signals.append("均线呈空头排列")
                ma_score -= 0.4
            
            # 短期均线关系
            ma5_above_ma20 = latest['MA5'] > latest['MA20']
            ma5_above_ma20_prev = prev['MA5'] > prev['MA20']
            
            if ma5_above_ma20 and not ma5_above_ma20_prev:
                ma_signals.append("MA5上穿MA20形成金叉")
                ma_score += 0.3
            elif not ma5_above_ma20 and ma5_above_ma20_prev:
                ma_signals.append("MA5下穿MA20形成死叉")
                ma_score -= 0.3
            
            # 价格相对于均线的位置
            price_vs_ma20 = (latest['收盘'] - latest['MA20']) / latest['MA20']
            if price_vs_ma20 > 0.05:  # 价格在MA20上方5%
                ma_signals.append("价格显著高于20日均线")
                ma_score += 0.2
            elif price_vs_ma20 < -0.05:  # 价格在MA20下方5%
                ma_signals.append("价格显著低于20日均线")
                ma_score -= 0.2
        
        score_components.append(ma_score * 0.20)
        
        # 优化2: 多维度MACD分析 (权重: 20%)
        macd_score = 0.0
        macd_signals = []
        
        if all(col in df.columns for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
            macd_above_signal = latest['MACD'] > latest['MACD_Signal']
            macd_above_signal_prev = prev['MACD'] > prev['MACD_Signal']
            
            # 金叉死叉
            if macd_above_signal and not macd_above_signal_prev:
                macd_signals.append("MACD金叉，买入信号增强")
                macd_score += 0.4
            elif not macd_above_signal and macd_above_signal_prev:
                macd_signals.append("MACD死叉，卖出信号增强")
                macd_score -= 0.4
            
            # MACD柱状图分析
            histogram = latest['MACD_Histogram']
            histogram_prev = prev['MACD_Histogram']
            
            if histogram > 0 and histogram > histogram_prev:
                macd_signals.append("MACD柱状图放大，动能强劲")
                macd_score += 0.3
            elif histogram > 0:
                macd_signals.append("MACD柱状图为正，动能向上")
                macd_score += 0.2
            elif histogram < 0 and histogram < histogram_prev:
                macd_signals.append("MACD柱状图收缩，动能减弱")
                macd_score -= 0.2
            
            # MACD零轴位置
            if latest['MACD'] > 0:
                macd_signals.append("MACD位于零轴上方，趋势偏多")
                macd_score += 0.3
            else:
                macd_signals.append("MACD位于零轴下方，趋势偏空")
                macd_score -= 0.2
        
        score_components.append(macd_score * 0.20)
        
        # 优化3: 多周期RSI分析 (权重: 15%)
        rsi_score = 0.0
        rsi_signals = []
        
        rsi_periods = ['RSI_6', 'RSI_14', 'RSI_24']
        rsi_values = []
        
        for rsi_col in rsi_periods:
            if rsi_col in df.columns and pd.notna(latest[rsi_col]):
                rsi_values.append(latest[rsi_col])
        
        if rsi_values:
            avg_rsi = np.mean(rsi_values)
            rsi_signals.append(f"平均RSI: {avg_rsi:.1f}")
            
            if avg_rsi < 30:
                rsi_signals.append("RSI进入超卖区域，反弹概率大")
                rsi_score += 0.6
            elif avg_rsi > 70:
                rsi_signals.append("RSI进入超买区域，回调风险高")
                rsi_score -= 0.6
            elif 45 <= avg_rsi <= 55:
                rsi_signals.append("RSI处于中性平衡区域")
                rsi_score += 0.1
            elif avg_rsi > 60:
                rsi_signals.append("RSI偏强但未超买")
                rsi_score += 0.2
            else:
                rsi_signals.append("RSI偏弱但未超卖")
                rsi_score += 0.1
        
        score_components.append(rsi_score * 0.15)
        
        # 优化4: 增强KDJ分析 (权重: 20%)
        kdj_score = 0.0
        kdj_signals = []
        
        if all(col in df.columns for col in ['K', 'D', 'J']):
            k, d, j = latest['K'], latest['D'], latest['J']
            
            # KDJ金叉死叉
            k_above_d = k > d
            k_above_d_prev = prev['K'] > prev['D']
            
            if k_above_d and not k_above_d_prev:
                kdj_signals.append("KDJ金叉，短期动能转强")
                kdj_score += 0.4
            elif not k_above_d and k_above_d_prev:
                kdj_signals.append("KDJ死叉，短期动能转弱")
                kdj_score -= 0.4
            
            # KDJ超买超卖区域分析
            if k < 20 and d < 20:
                kdj_signals.append(f"KDJ双线超卖(K:{k:.1f},D:{d:.1f})，反弹在即")
                kdj_score += 0.5
            elif k > 80 and d > 80:
                kdj_signals.append(f"KDJ双线超买(K:{k:.1f},D:{d:.1f})，回调风险")
                kdj_score -= 0.5
            elif 30 <= k <= 70 and 30 <= d <= 70:
                kdj_signals.append("KDJ处于健康波动区间")
                kdj_score += 0.2
            
            # J值极端情况
            if j > 100:
                kdj_signals.append(f"J值({j:.1f})极度超买，警惕反转")
                kdj_score -= 0.3
            elif j < 0:
                kdj_signals.append(f"J值({j:.1f})极度超卖，反弹可期")
                kdj_score += 0.3
            
            # KDJ位置关系强度
            if j > k > d:
                kdj_signals.append("J>K>D，强势多头排列")
                kdj_score += 0.3
            elif j < k < d:
                kdj_signals.append("J<K<D，弱势空头排列")
                kdj_score -= 0.3
        
        score_components.append(kdj_score * 0.20)
        
        # 优化5: 布林带多信号分析 (权重: 15%)
        bb_score = 0.0
        bb_signals = []
        
        if all(col in df.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
            price = latest['收盘']
            bb_upper = latest['BB_Upper']
            bb_lower = latest['BB_Lower']
            bb_middle = latest['BB_Middle']
            
            # 布林带位置分析
            bb_position = (price - bb_lower) / (bb_upper - bb_lower + 1e-10)
            
            if price <= bb_lower:
                bb_signals.append("价格触及布林带下轨，强烈反弹信号")
                bb_score += 0.6
            elif price >= bb_upper:
                bb_signals.append("价格触及布林带上轨，强烈回调信号")
                bb_score -= 0.6
            elif bb_position < 0.2:
                bb_signals.append("价格位于布林带低位，反弹概率较高")
                bb_score += 0.4
            elif bb_position > 0.8:
                bb_signals.append("价格位于布林带高位，回调压力较大")
                bb_score -= 0.4
            elif 0.4 <= bb_position <= 0.6:
                bb_signals.append("价格位于布林带中轨附近，方向待定")
                bb_score += 0.1
            
            # 布林带宽度分析
            bb_width = (bb_upper - bb_lower) / bb_middle
            if bb_width > 0.15:  # 布林带很宽，波动大
                bb_signals.append("布林带大幅扩张，波动率极高")
                bb_score += 0.2
            elif bb_width < 0.05:  # 布林带很窄，波动小
                bb_signals.append("布林带极度收缩，突破在即")
                bb_score += 0.1
        
        score_components.append(bb_score * 0.15)
        
        # 优化6: 成交量多维度分析 (权重: 10%)
        volume_score = 0.0
        volume_signals = []
        
        if 'Volume_Ratio' in df.columns and pd.notna(latest['Volume_Ratio']):
            volume_ratio = latest['Volume_Ratio']
            price_change = (latest['收盘'] / prev['收盘'] - 1) if len(df) > 1 else 0
            
            if volume_ratio > 2.0:
                volume_signals.append(f"成交量放大{volume_ratio:.1f}倍，异常活跃")
                if price_change > 0.02:
                    volume_signals.append("价涨量增，强势特征明显")
                    volume_score += 0.8
                elif price_change < -0.02:
                    volume_signals.append("放量下跌，风险较大")
                    volume_score -= 0.3
                else:
                    volume_signals.append("巨量震荡，方向待确认")
                    volume_score += 0.2
            elif volume_ratio > 1.5:
                volume_signals.append(f"成交量放大{volume_ratio:.1f}倍，活跃度提升")
                if price_change > 0:
                    volume_score += 0.4
            elif volume_ratio < 0.5:
                volume_signals.append(f"成交量萎缩至{volume_ratio:.1f}倍，交投清淡")
                volume_score -= 0.2
            else:
                volume_signals.append("成交量处于正常水平")
                volume_score += 0.1
            
            # OBV趋势确认
            if 'OBV_Trend' in df.columns:
                if latest['OBV_Trend'] > 0:
                    volume_signals.append("OBV趋势向上，资金流入")
                    volume_score += 0.2
                elif latest['OBV_Trend'] < 0:
                    volume_signals.append("OBV趋势向下，资金流出")
                    volume_score -= 0.2
        
        score_components.append(volume_score * 0.10)
        
        # 计算技术指标总评分
        technical_score = sum(score_components)
        technical_score = max(-1.0, min(1.0, technical_score))  # 限制在-1到1之间
        
        # 将技术评分转换为0-1范围
        technical_score_normalized = (technical_score + 1) / 2
        
        # 生成技术信号描述
        all_signals = ma_signals + macd_signals + rsi_signals + kdj_signals + bb_signals + volume_signals
        technical_description = f"技术评分{technical_score_normalized:.3f}。"
        if all_signals:
            # 只显示最重要的信号
            important_signals = [sig for sig in all_signals if any(keyword in sig for keyword in ['强烈', '极度', '完美', '异常', '巨量'])]
            if important_signals:
                technical_description += "关键信号：" + "；".join(important_signals[:3])
            else:
                technical_description += "主要信号：" + "；".join(all_signals[:4])
        
        # 生成技术信号评级
        if technical_score_normalized >= 0.8:
            technical_signal = "🔥 技术强烈买入"
        elif technical_score_normalized >= 0.7:
            technical_signal = "📈 技术买入"
        elif technical_score_normalized >= 0.6:
            technical_signal = "⚠️ 技术偏多"
        elif technical_score_normalized <= 0.3:
            technical_signal = "💀 技术强烈卖出"
        elif technical_score_normalized <= 0.4:
            technical_signal = "📉 技术卖出"
        elif technical_score_normalized <= 0.5:
            technical_signal = "⚠️ 技术偏空"
        else:
            technical_signal = "⏸️ 技术中性"
        
        return technical_signal, technical_description, technical_score_normalized

    def generate_quant_signal_with_score(self, df: pd.DataFrame) -> Tuple[str, str, float, Dict]:
        """生成量化信号和评分"""
        if len(df) < 100:
            return "数据不足", "需要至少100个交易日数据", 0.0, {}
        
        # 确保基本技术指标存在
        self._ensure_basic_indicators(df)
        
        market_regime = self.regime_detector.detect_market_regime(df)
        market_regime_cn = self.regime_translation.get(market_regime, market_regime)
        regime_weights = self.weight_allocator.get_regime_weights(market_regime)
        factor_scores = self.calculate_factor_scores(df, regime_weights)
        total_score = self.calculate_total_score(factor_scores, regime_weights)
        risk_adjusted_score = self.apply_risk_adjustment(total_score, df)
        signal, reasons = self.generate_signal_from_score(risk_adjusted_score, factor_scores, market_regime_cn)
        
        # 生成量化信号描述
        quant_description = "量化评分{:.3f}。".format(risk_adjusted_score)
        
        # 添加因子分析
        strong_factors = []
        weak_factors = []
        for factor, score in factor_scores.items():
            factor_cn = self.factor_translation.get(factor, factor)
            if score > 0.7:
                strong_factors.append(f"{factor_cn}因子强势({score:.2f})")
            elif score < 0.3:
                weak_factors.append(f"{factor_cn}因子弱势({score:.2f})")
        
        if strong_factors:
            quant_description += "强势因子：" + "、".join(strong_factors)
        if weak_factors:
            if strong_factors:
                quant_description += "；"
            quant_description += "弱势因子：" + "、".join(weak_factors)
        
        details = {
            'market_regime': market_regime_cn,
            'factor_scores': factor_scores,
            'total_score': total_score,
            'risk_adjusted_score': risk_adjusted_score,
            'regime_weights': regime_weights
        }
        return signal, quant_description, risk_adjusted_score, details

    def calculate_comprehensive_signal(self, quant_signal: str, technical_signal: str, 
                                    quant_score: float, technical_score: float,
                                    quant_description: str, tech_description: str) -> Tuple[str, float, str]:
        """计算综合信号和评分，考虑信号一致性"""
        
        # 信号一致性判断
        quant_direction = 0
        tech_direction = 0
        
        if "买入" in quant_signal or "强烈买入" in quant_signal:
            quant_direction = 1
        elif "卖出" in quant_signal or "强烈卖出" in quant_signal:
            quant_direction = -1
        
        if "买入" in technical_signal or "偏多" in technical_signal:
            tech_direction = 1
        elif "卖出" in technical_signal or "偏空" in technical_signal:
            tech_direction = -1
        
        # 基础综合评分（加权平均）
        base_comprehensive_score = (quant_score * 0.6 + technical_score * 0.4)
        
        # 一致性奖励
        consistency_bonus = 0.0
        if quant_direction == tech_direction and quant_direction != 0:
            if quant_direction == 1:  # 一致看多
                consistency_bonus = 0.15
                consistency_reason = "量化模型与技术分析共振看多，信号可靠性高"
            else:  # 一致看空
                consistency_bonus = 0.10
                consistency_reason = "量化模型与技术分析共振看空，风险信号明确"
        elif quant_direction != 0 and tech_direction != 0 and quant_direction != tech_direction:
            # 信号冲突惩罚
            consistency_bonus = -0.10
            consistency_reason = "量化模型与技术分析信号分歧，建议谨慎操作"
        else:
            consistency_reason = "多空信号平衡，市场处于震荡状态"
        
        # 计算最终综合评分
        comprehensive_score = base_comprehensive_score + consistency_bonus
        comprehensive_score = max(0.0, min(1.0, comprehensive_score))  # 限制在0-1范围内
        
        # 生成综合评级
        if comprehensive_score >= 0.75:
            comprehensive_signal = "🔥 强烈买入"
            comprehensive_reason = f"综合评分{comprehensive_score:.3f}，{consistency_reason}"
        elif comprehensive_score >= 0.65:
            comprehensive_signal = "📈 买入"
            comprehensive_reason = f"综合评分{comprehensive_score:.3f}，{consistency_reason}"
        elif comprehensive_score >= 0.55:
            comprehensive_signal = "⚠️ 谨慎买入"
            comprehensive_reason = f"综合评分{comprehensive_score:.3f}，{consistency_reason}"
        elif comprehensive_score >= 0.45:
            comprehensive_signal = "⏸️ 观望"
            comprehensive_reason = f"综合评分{comprehensive_score:.3f}，{consistency_reason}"
        elif comprehensive_score >= 0.35:
            comprehensive_signal = "⚠️ 谨慎卖出"
            comprehensive_reason = f"综合评分{comprehensive_score:.3f}，{consistency_reason}"
        else:
            comprehensive_signal = "📉 卖出"
            comprehensive_reason = f"综合评分{comprehensive_score:.3f}，{consistency_reason}"
        
        return comprehensive_signal, comprehensive_score, comprehensive_reason

    def _ensure_basic_indicators(self, df: pd.DataFrame):
        """确保基本技术指标存在"""
        # 计算移动平均线
        for period in [5, 10, 20, 60]:
            if f'MA{period}' not in df.columns:
                df[f'MA{period}'] = df['收盘'].rolling(period).mean()
        
        # 计算RSI
        if 'RSI_14' not in df.columns:
            delta = df['收盘'].diff()
            gain = delta.where(delta>0,0).rolling(14).mean()
            loss = (-delta.where(delta<0,0)).rolling(14).mean()
            rs = gain / (loss + 1e-8)
            df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # 计算MACD
        if 'MACD' not in df.columns:
            ema_12 = df['收盘'].ewm(span=12).mean()
            ema_26 = df['收盘'].ewm(span=26).mean()
            df['MACD'] = ema_12 - ema_26
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # 计算KDJ
        if 'K' not in df.columns:
            low_9 = df["最低"].rolling(9).min()
            high_9 = df["最高"].rolling(9).max()
            rsv = (df["收盘"] - low_9) / (high_9 - low_9 + 1e-8) * 100
            
            # 初始化K、D、J值
            df["K"] = 50.0  # 默认值
            df["D"] = 50.0  # 默认值
            df["J"] = 50.0  # 默认值
            
            # 计算K、D、J值（使用指数移动平均）
            for i in range(len(df)):
                if i == 0:
                    df.loc[df.index[i], "K"] = 50  # 初始值
                    df.loc[df.index[i], "D"] = 50  # 初始值
                else:
                    # K值 = 2/3 * 前一日K值 + 1/3 * 当日RSV
                    df.loc[df.index[i], "K"] = (2/3) * df.loc[df.index[i-1], "K"] + (1/3) * rsv.iloc[i]
                    # D值 = 2/3 * 前一日D值 + 1/3 * 当日K值
                    df.loc[df.index[i], "D"] = (2/3) * df.loc[df.index[i-1], "D"] + (1/3) * df.loc[df.index[i], "K"]
            
            # J值 = 3*K - 2*D
            df["J"] = 3 * df["K"] - 2 * df["D"]
        
        # 计算布林带
        if 'BB_Upper' not in df.columns:
            df['BB_Middle'] = df['收盘'].rolling(20).mean()
            df['BB_Std'] = df['收盘'].rolling(20).std()
            df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
            df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
        
        # 计算成交量比率
        if 'Volume_Ratio' not in df.columns:
            df['Volume_MA20'] = df['成交量(万手)'].rolling(20).mean()
            df['Volume_Ratio'] = df['成交量(万手)'] / (df['Volume_MA20'] + 1e-8)

    def calculate_factor_scores(self, df, weights):
        scores = {}
        scores['momentum'] = self.calculate_momentum_score(df)
        scores['volatility'] = self.calculate_volatility_score(df)
        scores['volume'] = self.calculate_volume_score(df)
        scores['reversal'] = self.calculate_reversal_score(df)
        scores['liquidity'] = self.calculate_liquidity_score(df)
        scores['trend'] = self.calculate_trend_score(df)
        return scores
    
    def calculate_momentum_score(self, df):
        latest = df.iloc[-1]
        score = 0.0
        
        try:
            # 使用实际存在的指标
            if 'Momentum_10D' in df.columns and latest.get('Momentum_10D', 0) > 1:
                score += 0.2
            
            # RSI动量
            if 'RSI_14' in df.columns:
                rsi = latest.get('RSI_14', 50)
                if 40 < rsi < 70:
                    score += 0.15
                elif rsi > 50:
                    score += 0.1
            
            # MACD动量
            if 'MACD' in df.columns and 'MACD_Histogram' in df.columns:
                if latest.get('MACD', 0) > 0:
                    score += 0.15
                if latest.get('MACD_Histogram', 0) > 0:
                    score += 0.1
            
            # 价格位置动量
            if 'Price_Position_20D' in df.columns:
                price_pos = latest.get('Price_Position_20D', 0.5)
                if price_pos > 0.5:
                    score += 0.1
            
            # 简单的价格动量
            returns_5d = (latest['收盘'] / df['收盘'].iloc[-6] - 1) if len(df) >= 6 else 0
            if returns_5d > 0.02:
                score += 0.15
                
        except Exception as e:
            # 如果计算失败，使用基础动量
            try:
                if len(df) >= 20:
                    returns_20d = (latest['收盘'] / df['收盘'].iloc[-21] - 1)
                    if returns_20d > 0:
                        score += 0.3
            except:
                pass
        
        return min(score, 1.0)
    
    def calculate_volatility_score(self, df):
        latest = df.iloc[-1]; score=0.0
        
        try:
            # 使用实际波动率计算
            if len(df) >= 20:
                returns = df['收盘'].pct_change().tail(20)
                volatility = returns.std() * np.sqrt(252)
                
                if 0.15 <= volatility <= 0.35:
                    score += 0.4
                elif 0.1 <= volatility <= 0.4:
                    score += 0.2
            
            # 布林带宽度
            if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
                bb_width = (latest['BB_Upper'] - latest['BB_Lower']) / latest['收盘']
                if 0.04 <= bb_width <= 0.12:
                    score += 0.3
                elif 0.02 <= bb_width <= 0.15:
                    score += 0.15
            
            # ATR比率
            if 'ATR_Ratio' in df.columns:
                atr_ratio = latest.get('ATR_Ratio', 0.02)
                if atr_ratio < 0.03:
                    score += 0.2
                elif atr_ratio < 0.05:
                    score += 0.1
                    
        except:
            pass
        
        return min(score, 1.0)
    
    def calculate_volume_score(self, df):
        latest = df.iloc[-1]; score=0.0
        
        try:
            # 成交量比率
            if 'Volume_Ratio' in df.columns:
                vol_ratio = latest.get('Volume_Ratio', 1)
                if 1.0 <= vol_ratio <= 2.5:
                    score += 0.4
                elif vol_ratio > 0.8:
                    score += 0.2
            
            # OBV动量
            if 'OBV' in df.columns and len(df) >= 6:
                obv_current = latest['OBV']
                obv_prev = df['OBV'].iloc[-6]
                if obv_current > obv_prev:
                    score += 0.3
            
            # 简单的价量关系
            price_change = (latest['收盘'] / df['收盘'].iloc[-2] - 1) if len(df) >= 2 else 0
            volume_change = (latest['成交量(万手)'] / df['成交量(万手)'].iloc[-2] - 1) if len(df) >= 2 else 0
            
            if price_change > 0 and volume_change > 0:
                score += 0.3
                
        except:
            pass
        
        return min(score, 1.0)
    
    def calculate_reversal_score(self, df):
        latest = df.iloc[-1]; score=0.0
        
        try:
            # RSI反转
            if 'RSI_14' in df.columns:
                rsi = latest.get('RSI_14', 50)
                if rsi < 30:
                    score += 0.4
                elif rsi > 70:
                    score -= 0.2  # 超买时降低反转评分
            
            # 价格位置反转
            if 'Price_Position_20D' in df.columns:
                price_pos = latest.get('Price_Position_20D', 0.5)
                if price_pos < 0.2:
                    score += 0.3
                elif price_pos > 0.8:
                    score -= 0.1
            
            # 布林带位置反转
            if 'BB_Position' in df.columns:
                bb_pos = latest.get('BB_Position', 0.5)
                if bb_pos < 0.1:
                    score += 0.3
                elif bb_pos > 0.9:
                    score += 0.2
            
            # 简单的价格反转
            if len(df) >= 10:
                returns_10d = (latest['收盘'] / df['收盘'].iloc[-11] - 1)
                if returns_10d < -0.05:  # 近期下跌
                    score += 0.3
                    
        except:
            pass
        
        return max(0, min(score, 1.0))  # 确保分数在0-1之间
    
    def calculate_liquidity_score(self, df):
        latest = df.iloc[-1]; score=0.0
        
        try:
            # 使用成交额作为流动性代理
            turnover = latest.get('成交额(万元)', 0)
            if turnover > 100000:  # 10亿以上
                score += 0.6
            elif turnover > 50000:  # 5-10亿
                score += 0.4
            elif turnover > 10000:  # 1-5亿
                score += 0.2
            
            # 成交量
            volume = latest.get('成交量(万手)', 0)
            if volume > 50:  # 50万手以上
                score += 0.4
            elif volume > 20:  # 20-50万手
                score += 0.2
                
        except:
            pass
        
        return min(score, 1.0)
    
    def calculate_trend_score(self, df):
        latest = df.iloc[-1]; score=0.0
        
        try:
            # 移动平均线趋势
            if 'MA5' in df.columns and 'MA20' in df.columns and 'MA60' in df.columns:
                if latest['MA5'] > latest['MA20'] > latest['MA60']:
                    score += 0.4
                elif latest['MA5'] > latest['MA20']:
                    score += 0.2
            
            # MACD趋势
            if 'MACD' in df.columns:
                if latest['MACD'] > 0:
                    score += 0.3
                if latest.get('MACD_Histogram', 0) > 0:
                    score += 0.2
            
            # 简单的价格趋势
            if len(df) >= 20:
                price_20d_ago = df['收盘'].iloc[-21]
                if latest['收盘'] > price_20d_ago:
                    score += 0.3
                    
        except:
            pass
        
        return min(score, 1.0)
    
    def calculate_total_score(self, factor_scores, weights):
        total_score = 0.0
        for factor, score in factor_scores.items():
            total_score += score * weights.get(factor,0)
        return total_score
    
    def apply_risk_adjustment(self, score, df):
        latest = df.iloc[-1]
        try:
            volatility = latest.get('VOLATILITY_20D',0.25)
            if volatility > 0.4: score *= 0.6
            elif volatility > 0.3: score *= 0.8
            if latest.get('AMIHUD_ILLIQUIDITY',1e-8) > 1e-7: score *= 0.7
            rsi = latest.get('RSI_14',50)
            if rsi > 85 or rsi < 15: score *= 0.8
            elif rsi > 75 or rsi < 25: score *= 0.9
            price_position = latest.get('Price_Position_20D',0.5)
            if price_position > 0.9 or price_position < 0.1: score *= 0.85
        except: pass
        return max(0, min(score, 1.0))
    
    def generate_signal_from_score(self, score, factor_scores, market_regime):
        reasons = []
        threshold_config = {
            '强势趋势': (0.65,0.80),'趋势行情':(0.60,0.75),'动量行情':(0.62,0.78),
            '均值回归':(0.55,0.70),'高波动':(0.58,0.72),'低波动':(0.57,0.73),'未知状态':(0.60,0.75)
        }
        buy_threshold, strong_buy_threshold = threshold_config.get(market_regime,(0.60,0.75))
        if score >= strong_buy_threshold:
            signal = "📈 量化买入"
            reasons.append(f"综合评分{score:.2f}超过强烈买入阈值")
        elif score >= buy_threshold:
            signal = "📈 量化买入"
            reasons.append(f"综合评分{score:.2f}超过买入阈值")
        elif score >= 0.4:
            signal = "⚠️ 量化观望"
            reasons.append("评分适中，建议谨慎操作")
        elif score <= 0.2:
            signal = "📉 量化卖出"
            reasons.append("综合评分显示卖出信号")
        elif score <= 0.35:
            signal = "⚠️ 量化减仓"
            reasons.append("评分偏低，建议减仓")
        else:
            signal = "⏸️ 量化中性"
            reasons.append("评分中性，建议观望")
        
        # 修改：将因子名称转换为中文
        top_factors = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        for factor, fs in top_factors:
            factor_cn = self.factor_translation.get(factor, factor)
            if fs > 0.7:
                reasons.append(f"{factor_cn}因子表现优秀({fs:.2f})")
            elif fs > 0.5:
                reasons.append(f"{factor_cn}因子表现良好({fs:.2f})")
        reasons.append(f"市场状态: {market_regime}")
        return signal, reasons

# ==================== 风控 ====================
class RiskManager:
    @staticmethod
    def calculate_position_size(score: float, volatility: float, portfolio_value: float = 100000) -> str:
        if score >= 0.7:
            return "重仓"
        elif score >= 0.5:
            return "中等仓位"
        elif score >= 0.3:
            return "轻仓"
        else:
            return "不建议持仓"
    
    @staticmethod
    def generate_stop_loss_take_profit(current_price: float, signal: str, volatility: float) -> Tuple[float, float]:
        vol_multiplier = max(0.8, min(1.5, volatility * 10))
        if '买入' in signal:
            stop_loss_pct = QuantConfig.STOP_LOSS * vol_multiplier
            take_profit_pct = QuantConfig.TAKE_PROFIT * (1 + (1 - vol_multiplier) * 0.5)
            stop_loss = current_price * (1 + stop_loss_pct)
            take_profit = current_price * (1 + take_profit_pct)
        elif '卖出' in signal:
            stop_loss_pct = QuantConfig.STOP_LOSS * vol_multiplier
            take_profit_pct = QuantConfig.TAKE_PROFIT * (1 + (1 - vol_multiplier) * 0.5)
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = current_price * (1 - take_profit_pct)
        else:
            stop_loss = take_profit = current_price
        return stop_loss, take_profit

# ==================== 主分析逻辑（带缓存和导出功能） ====================
def quant_enhanced_analysis(stock_codes: List[str], days: int = QuantConfig.DEFAULT_DAYS, use_cache: bool = True):
    signal_gen = QuantSignalGenerator()
    risk_mgr = RiskManager()
    
    # 进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []
    
    # 初始化缓存状态
    if 'cache_status' not in st.session_state:
        st.session_state.cache_status = {}
    
    for i, code in enumerate(stock_codes):
        try:
            stock_name = get_stock_name(code)
            status_text.text(f"分析中: {stock_name} ({code})")
            progress_bar.progress((i) / len(stock_codes))
            
            df = fetch_kline_data(code, days, use_cache=use_cache)
            df = compute_enhanced_technical_indicators(df, stock_code=code, use_cache=use_cache)
            
            # 生成量化信号和评分
            quant_signal, quant_description, quant_score, quant_details = signal_gen.generate_quant_signal_with_score(df)
            
            # 生成技术信号和评分
            technical_signal, tech_description, technical_score = signal_gen.generate_technical_signal_with_score(df)
            
            # 计算综合信号和评分
            comprehensive_signal, comprehensive_score, comprehensive_reason = signal_gen.calculate_comprehensive_signal(
                quant_signal, technical_signal, quant_score, technical_score, quant_description, tech_description
            )
            
            latest = df.iloc[-1]
            position_recommendation = risk_mgr.calculate_position_size(comprehensive_score, latest.get('VOLATILITY_20D',0.25))
            sl, tp = risk_mgr.generate_stop_loss_take_profit(latest['收盘'], comprehensive_signal, latest.get('VOLATILITY_20D',0.25))
            
            # 记录预测结果
            prediction_date = datetime.now().strftime('%Y-%m-%d')
            validation_record = {
                'stock_code': code,
                'stock_name': stock_name,
                'model_type': "量化模型",
                'prediction_date': prediction_date,
                'signal': comprehensive_signal,
                'confidence': comprehensive_score,
                'validation_status': '待验证'
            }
            data_cache.save_validation_record(validation_record)
            
            result = {
                '股票代码': code,
                '股票名称': stock_name,
                '当前价格': round(latest['收盘'],2),
                '量化信号': f"{quant_signal}\n\n{quant_description}",
                '技术信号': f"{technical_signal}\n\n{tech_description}",
                '综合评级': comprehensive_signal,
                '市场状态': quant_details.get('market_regime','未知状态'),
                '波动率': f"{latest.get('VOLATILITY_20D',0):.1%}",
                '操作建议': position_recommendation,
                '止损价位': round(sl,2),
                '止盈价位': round(tp,2),
                '分析日期': prediction_date,
                '信号一致性': comprehensive_reason
            }
            results.append(result)
            
        except Exception as e:
            stock_name = get_stock_name(code)
            st.error(f"分析{code}时出错: {str(e)}")
            results.append({
                '股票代码': code,
                '股票名称': stock_name,
                '当前价格': 'N/A',
                '量化信号': '分析失败',
                '技术信号': '分析失败',
                '综合评级': '分析失败',
                '市场状态': '未知',
                '波动率': 'N/A',
                '操作建议': 'N/A',
                '止损价位': 'N/A',
                '止盈价位': 'N/A',
                '分析日期': datetime.now().strftime('%Y-%m-%d'),
                '信号一致性': f"错误: {str(e)}"
            })
    
    progress_bar.progress(1.0)
    status_text.text("分析完成!")
    
    df_res = pd.DataFrame(results)
    
    return df_res

# ==================== 数据导出功能 ====================
def export_to_csv(df: pd.DataFrame, filename: str = None) -> str:
    """将DataFrame导出为CSV字符串"""
    if filename is None:
        filename = f"quant_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    csv_string = df.to_csv(index=False, encoding='utf-8-sig')
    return csv_string, filename

def export_detailed_report(results_df: pd.DataFrame) -> str:
    """生成详细的文本报告"""
    report = "幻方量化分析报告\n"
    report += "=" * 50 + "\n\n"
    report += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"分析股票数量: {len(results_df)}\n\n"
    
    # 信号统计
    if '综合评级' in results_df.columns:
        signal_counts = results_df['综合评级'].value_counts()
        report += "综合评级分布:\n"
        report += "-" * 30 + "\n"
        for signal, count in signal_counts.items():
            report += f"{signal}: {count} 只\n"
        report += "\n"
    
    # 详细股票分析
    report += "个股分析详情:\n"
    report += "=" * 50 + "\n"
    
    for _, row in results_df.iterrows():
        report += f"\n股票: {row.get('股票代码', '')} - {row.get('股票名称', '')}\n"
        report += f"当前价格: {row.get('当前价格', '')}\n"
        report += f"量化信号: {row.get('量化信号', '')}\n"
        report += f"技术信号: {row.get('技术信号', '')}\n"
        report += f"综合评级: {row.get('综合评级', '')}\n"
        report += f"操作建议: {row.get('操作建议', '')}\n"
        report += f"信号一致性: {row.get('信号一致性', '')}\n"
        report += "-" * 30 + "\n"
    
    return report

# ==================== 显示股票卡片的通用函数 ====================
def display_stock_cards(recommendations_df, category):
    """显示股票卡片的通用函数"""
    # 根据类别设置颜色
    color_map = {
        "strong_buy": "#e74c3c",  # 红色，强烈买入
        "buy": "#f39c12",         # 橙色，建议买入
        "watch": "#3498db",       # 蓝色，观望
        "technical": "#2ecc71"    # 绿色，技术亮点
    }
    
    border_color = color_map.get(category, "#e0e0e0")
    
    # 创建紧凑布局
    num_stocks = len(recommendations_df)
    cols_per_row = min(3, num_stocks)
    
    # 按行显示股票卡片
    for i in range(0, num_stocks, cols_per_row):
        cols = st.columns(cols_per_row)
        row_stocks = recommendations_df.iloc[i:i+cols_per_row]
        
        for idx, (_, stock) in enumerate(row_stocks.iterrows()):
            with cols[idx]:
                # 根据信号类型设置不同的背景色
                bg_color = "#fef5f5" if category == "strong_buy" else "#fffaf2" if category == "buy" else "#f5f9ff" if category == "watch" else "#f2f9f5"
                
                # 股票卡片
                st.markdown(f"""
                <div style="border: 2px solid {border_color}; border-radius: 8px; padding: 12px; margin: 8px 0; background-color: {bg_color};">
                    <div style="font-weight: bold; font-size: 14px;">{stock['股票名称']}</div>
                    <div style="font-size: 12px; color: #666;">{stock['股票代码']}</div>
                    <div style="font-size: 13px; color: {border_color}; font-weight: bold; margin: 4px 0;">{stock['综合评级']}</div>
                    <div style="font-size: 12px;">价格: {stock['当前价格']}</div>
                    <div style="font-size: 12px;">操作: {stock['操作建议']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # 展开按钮 - 修复嵌套列问题
                with st.expander("详细分析", expanded=False):
                    # 使用单列布局避免嵌套
                    st.write(f"**当前价格**: {stock['当前价格']}")
                    st.write(f"**市场状态**: {stock['市场状态']}")
                    st.write(f"**波动率**: {stock['波动率']}")
                    st.write(f"**操作建议**: {stock['操作建议']}")
                    st.write(f"**止损价位**: {stock['止损价位']}")
                    st.write(f"**止盈价位**: {stock['止盈价位']}")
                    
                    # 显示详细的信号分析
                    st.write("**量化信号分析:**")
                    st.info(stock['量化信号'])
                    st.write("**技术信号分析:**")
                    st.info(stock['技术信号'])
                    st.write("**信号一致性分析:**")
                    st.success(stock['信号一致性'])
                    
                    # 添加关注理由
                    st.write("**关注理由:**")
                    if category == "strong_buy":
                        st.success("🔴 强烈买入信号：量化模型和技术分析均给出强烈买入信号，信号一致性高")
                    elif category == "buy":
                        st.warning("🟠 建议买入：存在较好的买入机会，建议关注")
                    elif category == "watch":
                        st.info("🔵 建议观望：存在潜在机会，需要进一步观察确认")
                    elif category == "technical":
                        st.success("🟢 技术面亮点：技术指标显示有积极信号，值得关注")

# ==================== 自动验证功能 ====================
def perform_auto_validation():
    """执行自动验证"""
    try:
        st.info("正在执行自动验证...")
        
        # 获取今天的日期
        today = datetime.now().strftime('%Y-%m-%d')
        
        # 检查是否是交易日（周一到周五）
        weekday = datetime.now().weekday()
        if weekday >= 5:  # 5=周六, 6=周日
            st.warning("今天是周末，无法获取最新数据")
            return
        
        # 获取当前时间
        current_time = datetime.now().time()
        market_close_time = datetime.strptime("15:00", "%H:%M").time()
        
        # 如果还没到收盘时间，提示用户
        if current_time < market_close_time:
            st.warning(f"当前时间 {current_time.strftime('%H:%M')} 还未到收盘时间(15:00)，建议收盘后再执行自动验证")
            return
        
        # 获取自选股列表
        watchlist = data_cache.get_watchlist()
        if not watchlist:
            st.warning("自选股为空，无法执行自动验证")
            return
        
        # 获取今天的预测记录
        prediction_records = data_cache.get_unique_validation_records()
        today_predictions = prediction_records[prediction_records['prediction_date'] == today]
        
        if today_predictions.empty:
            st.warning(f"今天({today})没有预测记录，请先进行量化分析")
            return
        
        # 进度条
        progress_bar = st.progress(0)
        validated_count = 0
        
        for idx, record in today_predictions.iterrows():
            stock_code = record['stock_code']
            stock_name = record['stock_name']
            prediction_signal = record['signal']
            
            try:
                # 获取最新数据（包含今天的收盘价）
                df = fetch_kline_data(stock_code, 2, use_cache=False)  # 获取2天数据
                if len(df) < 2:
                    continue
                
                # 计算今日涨跌幅
                today_close = df.iloc[-1]['收盘']
                yesterday_close = df.iloc[-2]['收盘']
                actual_return = (today_close - yesterday_close) / yesterday_close
                
                # 保存自动验证结果
                data_cache.save_auto_validation_result(
                    stock_code, stock_name, today, 
                    prediction_signal, actual_return, "auto"
                )
                
                # 更新验证记录
                if '买入' in prediction_signal:
                    is_correct = 1 if actual_return > 0 else 0
                elif '卖出' in prediction_signal:
                    is_correct = 1 if actual_return < 0 else 0
                else:  # 观望信号
                    is_correct = 1 if abs(actual_return) < 0.01 else 0
                
                data_cache.update_validation_result(record['id'], actual_return, is_correct)
                validated_count += 1
                
            except Exception as e:
                st.error(f"验证 {stock_name}({stock_code}) 失败: {e}")
            
            progress_bar.progress((idx + 1) / len(today_predictions))
        
        st.success(f"自动验证完成！共验证 {validated_count} 只股票")
        
        # 显示验证结果统计
        auto_results = data_cache.get_auto_validation_results(1)  # 获取今天的结果
        if not auto_results.empty:
            correct_predictions = len(auto_results[auto_results['is_correct'] == 1])
            total_predictions = len(auto_results)
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            st.metric("今日验证准确率", f"{accuracy:.2%}", 
                     f"{correct_predictions}/{total_predictions}")
    
    except Exception as e:
        st.error(f"自动验证过程出错: {e}")

def model_validation_page():
    st.header("🔍 模型验证与性能分析")
    
    # 1. 自动验证功能
    st.subheader("🤖 自动验证")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("一键执行自动验证，系统将获取自选股最新数据并验证模型预测准确性")
    with col2:
        if st.button("🔄 执行自动验证", use_container_width=True, type="primary"):
            perform_auto_validation()
            st.rerun()
    
    st.markdown("---")
    
    # 获取自动验证记录
    auto_records = data_cache.get_auto_validation_results(days=90)  # 获取最近90天的记录
    
    if auto_records.empty:
        st.info("暂无自动验证记录，请先执行自动验证")
        # 将说明移到页面底部
        st.markdown("---")
        st.subheader("📝 功能说明")
        st.info("""
        **自动验证功能说明:**
        - 系统会自动获取自选股今日收盘价
        - 计算实际涨跌幅并与预测信号对比
        - 只有在交易日15:00后才会执行完整验证
        - 验证结果将永久保存到数据库
        - 系统会自动判断预测是否正确并记录结果
        
        **验证说明**：
        - 系统自动记录预测日期当天的实际涨跌幅
        - 涨跌幅 = (当日收盘价 - 前日收盘价) / 前日收盘价 × 100%
        - 验证数据将永久保存到数据库
        """)
        return
    
    # 2. 详细验证记录（按日期分组）
    st.subheader("📋 模型数据统计")
    
    # 日期筛选
    col1, col2 = st.columns(2)
    with col1:
        # 获取所有有记录的日期
        all_dates = sorted(auto_records['validation_date'].unique(), reverse=True)
        selected_date = st.selectbox(
            "选择查看日期",
            options=all_dates,
            index=0 if all_dates else None
        )
    
    with col2:
        # 信号类型筛选
        signal_filter = st.selectbox(
            "筛选信号类型",
            options=["全部", "买入信号", "卖出信号", "观望信号"],
            index=0
        )
    
    # 筛选记录
    filtered_records = auto_records[auto_records['validation_date'] == selected_date] if selected_date else auto_records
    
    if signal_filter != "全部":
        if signal_filter == "买入信号":
            filtered_records = filtered_records[filtered_records['prediction_signal'].str.contains('买入', na=False)]
        elif signal_filter == "卖出信号":
            filtered_records = filtered_records[filtered_records['prediction_signal'].str.contains('卖出', na=False)]
        else:  # 观望信号
            filtered_records = filtered_records[~filtered_records['prediction_signal'].str.contains('买入|卖出', na=False)]
    
    if not filtered_records.empty:
        # 计算当日统计
        date_correct = len(filtered_records[filtered_records['is_correct'] == 1])
        date_accuracy = date_correct / len(filtered_records) if len(filtered_records) > 0 else 0
        
        # 显示当日统计
        st.markdown("#### 📊 每日统计")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总记录数", len(filtered_records))
        with col2:
            st.metric("正确预测", date_correct)
        with col3:
            st.metric("当日准确率", f"{date_accuracy:.2%}")
        with col4:
            # 计算相对于整体准确率的表现
            overall_accuracy = len(auto_records[auto_records['is_correct'] == 1]) / len(auto_records) if len(auto_records) > 0 else 0
            if overall_accuracy > 0:
                performance_diff = (date_accuracy - overall_accuracy) * 100
                st.metric(
                    "相对表现", 
                    f"{performance_diff:+.1f}%",
                    delta=f"{performance_diff:+.1f}%"
                )
            else:
                st.metric("相对表现", "N/A")
        
        # 计算月度统计
        if selected_date:
            # 获取选中日期所在的月份
            selected_month = selected_date[:7]  # 格式: YYYY-MM
            
            # 筛选该月份的所有记录
            month_records = auto_records[auto_records['validation_date'].str.startswith(selected_month)]
            
            # 应用相同的信号类型筛选
            if signal_filter != "全部":
                if signal_filter == "买入信号":
                    month_records = month_records[month_records['prediction_signal'].str.contains('买入', na=False)]
                elif signal_filter == "卖出信号":
                    month_records = month_records[month_records['prediction_signal'].str.contains('卖出', na=False)]
                else:  # 观望信号
                    month_records = month_records[~month_records['prediction_signal'].str.contains('买入|卖出', na=False)]
            
            if not month_records.empty:
                # 计算月度统计
                month_total = len(month_records)
                month_correct = len(month_records[month_records['is_correct'] == 1])
                month_accuracy = month_correct / month_total if month_total > 0 else 0
                
                # 显示月度统计
                st.markdown("#### 📈 月度统计")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("总记录数", month_total)
                with col2:
                    st.metric("正确预测", month_correct)
                with col3:
                    st.metric("月度准确率", f"{month_accuracy:.2%}")
                with col4:
                    # 计算相对于整体准确率的表现
                    if overall_accuracy > 0:
                        month_performance_diff = (month_accuracy - overall_accuracy) * 100
                        st.metric(
                            "相对表现", 
                            f"{month_performance_diff:+.1f}%",
                            delta=f"{month_performance_diff:+.1f}%"
                        )
                    else:
                        st.metric("相对表现", "N/A")
        
        # 使用expander实现折叠功能
        with st.expander("📋 详细记录", expanded=True):
            for i, record in filtered_records.iterrows():
                # 根据正确性设置颜色
                if record['is_correct'] == 1:
                    border_color = "#2ecc71"  # 绿色 - 正确
                    bg_color = "#f2f9f5"
                else:
                    border_color = "#e74c3c"  # 红色 - 错误
                    bg_color = "#fef5f5"
                
                # 使用卡片形式显示每条记录 - 移除红黄绿圆圈
                st.markdown(f"""
                <div style="border: 2px solid {border_color}; border-radius: 8px; padding: 12px; margin: 8px 0; background-color: {bg_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="font-weight: bold; font-size: 14px;">{record['stock_name']} ({record['stock_code']})</div>
                        <div style="font-size: 16px;">{record['prediction_signal']}</div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 13px;">
                        <div>实际涨跌幅: <strong>{record.get('actual_return', 0)*100:.2f}%</strong></div>
                        <div>验证结果: <strong>{'✅ 正确' if record['is_correct'] == 1 else '❌ 错误'}</strong></div>
                        <div>验证时间: {record['created_time'][:19] if record.get('created_time') else 'N/A'}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("选定条件下暂无验证记录")
    
    # 3. 功能说明 - 移到底部
    st.markdown("---")
    st.subheader("📝 功能说明")
    st.info("""
    **自动验证功能说明:**
    - 系统会自动获取自选股今日收盘价
    - 计算实际涨跌幅并与预测信号对比
    - 只有在交易日15:00后才会执行完整验证
    - 验证结果将永久保存到数据库
    - 系统会自动判断预测是否正确并记录结果
    
    **统计说明**：
    - 当日统计：显示选定日期的验证结果统计
    - 月度统计：显示选定日期所在月份的验证结果统计
    - 详细记录：可按日期和信号类型筛选查看具体验证结果
    - 相对表现：当日/月度准确率相对于整体准确率的变化
    """)

# ==================== 数据管理功能 ====================
def data_management_page():
    st.header("💾 数据缓存管理")
    
    # 显示数据库状态
    stock_count = data_cache.get_stock_count()
    if stock_count == 0:
        st.error("❌ 股票数据库未初始化！")
        st.info("""
        请运行以下命令初始化股票数据库：
        ```bash
        python update_stock_basic_info.py
        ```
        或者使用下面的按钮自动更新。
        """)
    else:
        st.success(f"✅ 股票数据库已初始化，包含 {stock_count} 只股票")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("缓存统计")
        try:
            conn = sqlite3.connect("stock_data_cache.db")
            
            # 添加股票基本信息统计
            basic_count = data_cache.get_stock_count()
            stock_count_cache = pd.read_sql_query("SELECT COUNT(DISTINCT stock_code) as count FROM stock_data", conn)['count'].iloc[0]
            data_count = pd.read_sql_query("SELECT COUNT(*) as count FROM stock_data", conn)['count'].iloc[0]
            indicator_count = pd.read_sql_query("SELECT COUNT(*) as count FROM technical_indicators", conn)['count'].iloc[0]
            watchlist_count = pd.read_sql_query("SELECT COUNT(*) as count FROM watchlist", conn)['count'].iloc[0]
            validation_count = pd.read_sql_query("SELECT COUNT(*) as count FROM model_validation", conn)['count'].iloc[0]
            auto_validation_count = pd.read_sql_query("SELECT COUNT(*) as count FROM auto_validation_results", conn)['count'].iloc[0]
            conn.close()
            
            st.metric("股票基本信息", f"{basic_count:,} 只")
            st.metric("缓存股票数量", stock_count_cache)
            st.metric("K线数据条数", f"{data_count:,}")
            st.metric("技术指标缓存", indicator_count)
            st.metric("自选股数量", watchlist_count)
            st.metric("验证记录数", validation_count)
            st.metric("自动验证结果", auto_validation_count)
        except Exception as e:
            st.info("暂无缓存数据或数据库未初始化")
    
    with col2:
        st.subheader("数据更新")
        
        use_cache = st.checkbox("启用数据缓存", value=True, 
                               help="启用后可以避免重复下载数据，提高分析速度")
        
        st.subheader("缓存操作")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 清理过期缓存", use_container_width=True):
                data_cache.clear_old_cache()
                st.success("过期缓存已清理!")
                st.rerun()
            
            if st.button("🔥 清空所有缓存", use_container_width=True):
                if os.path.exists("stock_data_cache.db"):
                    os.remove("stock_data_cache.db")
                    data_cache.init_database()
                    st.success("所有缓存已清空!")
                    st.rerun()
        
        with col2:
            if st.button("🔄 重置验证记录", use_container_width=True):
                conn = sqlite3.connect("stock_data_cache.db")
                cursor = conn.cursor()
                cursor.execute("DELETE FROM model_validation")
                cursor.execute("DELETE FROM auto_validation_results")
                conn.commit()
                conn.close()
                st.success("验证记录已重置!")
        
        st.subheader("股票基本信息")
        if st.button("📈 更新股票基本信息", use_container_width=True, type="primary"):
            with st.spinner("正在更新股票基本信息，这可能需要几分钟..."):
                try:
                    # 导入更新脚本中的函数
                    import subprocess
                    import sys
                    
                    result = subprocess.run([sys.executable, "update_stock_basic_info.py"], 
                                          capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0:
                        st.success("股票基本信息更新成功!")
                        st.code(result.stdout)
                    else:
                        st.error("股票基本信息更新失败!")
                        st.code(f"错误信息:\n{result.stderr}")
                        
                except Exception as e:
                    st.error(f"更新过程中发生错误: {e}")
                    st.info("""
                    您也可以手动运行更新脚本：
                    ```bash
                    python update_stock_basic_info.py
                    ```
                    """)
    
    st.markdown("---")
    st.subheader("缓存说明")
    st.info("""
    - **股票基本信息**: 存储完整的股票代码-名称映射数据
    - **数据缓存**: 存储从API获取的原始K线数据，自动清理30天前数据
    - **指标缓存**: 存储计算完成的技术指标数据，按日期缓存
    - **自选股**: 存储用户关注的股票列表
    - **验证记录**: 存储模型预测和验证结果
    - **自动验证结果**: 存储系统自动验证的结果
    
    💡 **建议**: 在网络环境良好时定期清理缓存，确保数据及时更新
    """)

# ==================== 自选股管理功能 ====================
def watchlist_management_page():
    st.header("⭐ 自选股批量分析与管理")
    
    watchlist = data_cache.get_watchlist()
    
    # 修改：只保留一列布局，移除右侧设置
    col_main = st.columns([1])[0]
    
    with col_main:
        # 自选股管理功能
        st.subheader("📋 自选股管理")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 添加股票到自选股
            add_stock_input = st.text_input(
                "添加股票到自选股",
                placeholder="输入股票代码或名称...",
                key="add_stock_input_widget"
            )
        
        with col2:
            st.write("")  # 空行用于对齐
            if st.button("➕ 添加股票", use_container_width=True, key="add_stock_btn"):
                if add_stock_input:
                    success = add_stock_to_watchlist(add_stock_input)
                    if success:
                        st.success("股票添加成功!")
                        st.rerun()
        
        # 显示搜索结果（如果有）
        if 'search_results' in st.session_state and st.session_state.search_results:
            st.markdown("**搜索结果:**")
            for stock in st.session_state.search_results:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"{stock['name']} ({stock['code']}) - {stock.get('market', '')}")
                with col2:
                    if st.button("➕", key=f"add_{stock['code']}"):
                        data_cache.add_to_watchlist(stock['code'], stock['name'])
                        st.success(f"已添加 {stock['name']}!")
                        # 清除搜索结果
                        if 'search_results' in st.session_state:
                            del st.session_state.search_results
                        st.rerun()
        
        # 修改1：固定今日关注股票窗口界面
        st.subheader("🎯 今日关注股票")
        
        # 检查是否有分析结果
        if 'watchlist_analysis_results' in st.session_state and st.session_state.watchlist_analysis_results is not None:
            results_df = st.session_state.watchlist_analysis_results
            
            # 扩展筛选逻辑：包含多个关注级别
            # 强烈买入信号
            strong_buy_recommendations = results_df[
                results_df['综合评级'].str.contains('强烈买入|🔥', na=False)
            ]
            
            # 买入信号（包括谨慎买入）
            buy_recommendations = results_df[
                results_df['综合评级'].str.contains('买入|📈', na=False) & 
                ~results_df['综合评级'].str.contains('强烈买入|🔥', na=False)
            ]
            
            # 观望信号（潜在机会）
            watch_recommendations = results_df[
                results_df['综合评级'].str.contains('观望|⏸️', na=False)
            ]
            
            # 技术面有亮点的股票（即使综合评级不是买入，但技术信号有亮点）
            technical_highlights = results_df[
                results_df['技术信号'].str.contains('金叉|上穿|超卖|反弹|强势', na=False) &
                ~results_df['综合评级'].str.contains('卖出|📉', na=False)
            ]
            
            # 合并所有推荐股票，去重
            all_recommendations = pd.concat([
                strong_buy_recommendations,
                buy_recommendations,
                watch_recommendations,
                technical_highlights
            ]).drop_duplicates(subset=['股票代码'])
            
            if not all_recommendations.empty:
                # 按信号强度排序：强烈买入 > 买入 > 观望 > 技术亮点
                def get_signal_priority(row):
                    if '强烈买入' in row['综合评级'] or '🔥' in row['综合评级']:
                        return 1
                    elif '买入' in row['综合评级'] or '📈' in row['综合评级']:
                        return 2
                    elif '观望' in row['综合评级'] or '⏸️' in row['综合评级']:
                        return 3
                    else:
                        return 4
                
                all_recommendations['priority'] = all_recommendations.apply(get_signal_priority, axis=1)
                all_recommendations = all_recommendations.sort_values(['priority'])
                
                # 按关注级别分组显示
                if not strong_buy_recommendations.empty:
                    st.markdown("##### 🔥 强烈买入")
                    display_stock_cards(strong_buy_recommendations, "strong_buy")
                
                if not buy_recommendations.empty:
                    st.markdown("##### 📈 建议买入")
                    display_stock_cards(buy_recommendations, "buy")
                
                if not watch_recommendations.empty:
                    st.markdown("##### ⏸️ 建议观望（潜在机会）")
                    display_stock_cards(watch_recommendations, "watch")
                
                if not technical_highlights.empty and len(technical_highlights) > len(all_recommendations) - len(strong_buy_recommendations) - len(buy_recommendations) - len(watch_recommendations):
                    st.markdown("##### 💡 技术面亮点")
                    display_stock_cards(technical_highlights, "technical")
                    
            else:
                st.info("今日暂无推荐关注的股票")
        else:
            st.info("请先点击下方的'分析自选股'按钮获取今日关注股票")
        
        # 显示当前自选股
        if watchlist:
            st.subheader("📊 当前自选股")
            
            # 批量操作按钮
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 分析自选股", use_container_width=True, type="primary"):
                    # 设置自动分析标志和股票代码
                    st.session_state.auto_analyze = True
                    st.session_state.auto_stock_codes = ",".join([stock['code'] for stock in watchlist])
                    st.session_state.navigation = "量化分析"  # 设置导航到量化分析页面
                    st.session_state.is_watchlist_analysis = True  # 标记为自选股分析
                    st.success("正在跳转到量化分析页面...")
                    st.rerun()
            with col2:
                if st.button("🗑️ 清空自选股", use_container_width=True):
                    for stock in watchlist:
                        data_cache.remove_from_watchlist(stock['code'])
                    st.success("自选股已清空!")
                    # 清空分析结果
                    if 'watchlist_analysis_results' in st.session_state:
                        st.session_state.watchlist_analysis_results = None
                    st.rerun()
            
            # 修改2：优化自选股列表排版，使用表格形式
            st.write(f"**当前自选股 ({len(watchlist)} 只):**")
            
            # 创建紧凑的表格布局
            cols_per_row = 4  # 每行显示4个股票
            rows = (len(watchlist) + cols_per_row - 1) // cols_per_row
            
            for row in range(rows):
                cols = st.columns(cols_per_row)
                for col_idx in range(cols_per_row):
                    stock_idx = row * cols_per_row + col_idx
                    if stock_idx < len(watchlist):
                        stock = watchlist[stock_idx]
                        with cols[col_idx]:
                            with st.container():
                                # 股票信息卡片
                                st.markdown(f"""
                                <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; margin: 4px 0; background-color: white;">
                                    <div style="font-weight: bold; font-size: 14px;">{stock['name']}</div>
                                    <div style="font-size: 12px; color: #666;">{stock['code']}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # 移除按钮
                                if st.button("移除", key=f"del_{stock['code']}", use_container_width=True):
                                    data_cache.remove_from_watchlist(stock['code'])
                                    # 如果移除了股票，清空分析结果
                                    if 'watchlist_analysis_results' in st.session_state:
                                        st.session_state.watchlist_analysis_results = None
                                    st.rerun()
        
        else:
            st.info("您的自选股为空，请先添加股票到自选股")
            return

# ==================== 量化分析功能 ====================
def quantitative_analysis_page():
    st.header("📈 量化增强分析")
    
    # 参数设置区域 - 优化布局
    st.subheader("⚙️ 分析参数设置")
    
    # 使用三列布局，让界面更紧凑
    col1, col2, col3 = st.columns([3, 2, 1])
    
    with col1:
        # 股票代码输入 - 支持代码和名称
        if 'stock_input_value' not in st.session_state:
            st.session_state.stock_input_value = ""
        
        # 检查是否有自动分析的股票代码
        if st.session_state.get('auto_analyze', False):
            auto_stock_codes = st.session_state.get('auto_stock_codes', '')
            # 清除自动分析标志
            st.session_state.auto_analyze = False
            auto_analyze = True
        else:
            auto_stock_codes = ""
            auto_analyze = False
        
        stock_input = st.text_input(
            "输入股票代码或名称",
            value=auto_stock_codes if auto_analyze else st.session_state.stock_input_value,
            placeholder="例如：600519 或 贵州茅台 或 600519,000858,贵州茅台...",
            help="支持输入股票代码(6位数字)或股票名称，多个用逗号分隔",
            key="main_stock_input"
        )
    
    with col2:
        days = st.slider("分析天数", min_value=60, max_value=800, 
                        value=QuantConfig.DEFAULT_DAYS, key="analysis_days")
    
    with col3:
        use_cache = st.checkbox("使用数据缓存", value=True, 
                              help="启用后避免重复下载数据", key="analysis_cache")
        # 将分析按钮放在第三列
        run_btn = st.button("🚀 开始分析", type="primary", use_container_width=True)
    
    # 分析按钮逻辑
    if run_btn or auto_analyze:
        # 使用手动输入的代码或自动分析的代码
        if auto_analyze:
            stock_codes_input = auto_stock_codes
            # 如果是自选股分析，保存分析结果到session_state
            st.session_state.is_watchlist_analysis = True
        else:
            stock_codes_input = stock_input
            st.session_state.is_watchlist_analysis = False
        
        if not stock_codes_input:
            st.error("请输入股票代码或名称")
        else:
            # 解析输入的股票代码或名称
            stock_codes = []
            input_items = re.split(r'[,\s]+', stock_codes_input.strip())
            
            for item in input_items:
                if not item.strip():
                    continue
                    
                cleaned_item = item.strip()
                
                # 1. 首先尝试作为股票代码处理
                code = extract_stock_code(cleaned_item)
                if code and validate_stock_code(code):
                    stock_codes.append(code)
                    continue
                
                # 2. 如果不是纯数字代码，尝试作为股票名称搜索
                if not cleaned_item.isdigit():
                    with st.spinner(f"搜索股票: {cleaned_item}"):
                        search_results = search_stock_by_name(cleaned_item)
                    
                    if search_results:
                        if len(search_results) == 1:
                            # 只有一个匹配结果，直接使用
                            stock_codes.append(search_results[0]['code'])
                            st.success(f"找到股票: {search_results[0]['name']} ({search_results[0]['code']})")
                        else:
                            # 多个匹配结果，使用第一个
                            stock_codes.append(search_results[0]['code'])
                            st.warning(f"找到多个匹配，使用: {search_results[0]['name']} ({search_results[0]['code']})")
                            # 显示其他匹配选项
                            other_options = [f"{s['name']}({s['code']})" for s in search_results[1:3]]
                            if other_options:
                                st.info(f"其他匹配: {', '.join(other_options)}")
                    else:
                        st.warning(f"未找到匹配的股票: {cleaned_item}")
                else:
                    st.warning(f"'{cleaned_item}' 不是有效的股票代码")
            
            if stock_codes:
                # 去重
                stock_codes = list(dict.fromkeys(stock_codes))
                
                # 显示简洁的缓存状态
                if st.session_state.get('cache_status'):
                    cache_info = " | ".join([f"{code}: {status}" for code, status in st.session_state.cache_status.items()])
                    st.caption(f"数据状态: {cache_info}")
                
                # 显示将要分析的股票
                stock_names = [get_stock_name(code) for code in stock_codes]
                analysis_list = [f"{name}({code})" for name, code in zip(stock_names, stock_codes)]
                st.info(f"即将分析: {', '.join(analysis_list)}")
                
                with st.spinner("正在执行量化分析，请稍候..."):
                    results_df = quant_enhanced_analysis(stock_codes, days, use_cache=use_cache)
                
                st.success("✅ 分析完成!")
                
                # 如果是自选股分析，保存结果到session_state
                if st.session_state.get('is_watchlist_analysis', False):
                    st.session_state.watchlist_analysis_results = results_df
                
                # 显示结果表格
                st.dataframe(results_df, use_container_width=True)
                
                # 导出功能
                st.subheader("📥 导出结果")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # CSV导出
                    csv_data, csv_filename = export_to_csv(results_df)
                    st.download_button(
                        label="📊 下载CSV报告",
                        data=csv_data,
                        file_name=csv_filename,
                        mime="text/csv",
                        help="下载完整的分析结果CSV文件",
                        use_container_width=True
                    )
                
                with col2:
                    # 文本报告导出
                    report_text = export_detailed_report(results_df)
                    st.download_button(
                        label="📄 下载文本报告",
                        data=report_text,
                        file_name=f"quant_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        help="下载详细的分析报告文本文件",
                        use_container_width=True
                    )
            else:
                st.error("没有找到有效的股票代码")
    else:
        st.info("💡 在上方输入股票代码或名称，点击 **开始分析** 按钮运行分析。")
        st.markdown("""
        **使用说明:**
        - 支持输入股票代码 (6位数字，如: 600519)
        - 支持输入股票名称 (如: 贵州茅台)
        - 支持混合输入 (如: 600519,贵州茅台,000858)
        - 多个股票用逗号或空格分隔
        """)

# ==================== 帮助页面 ====================
def help_page():
    st.header("❓ 帮助与使用说明")
    
    st.markdown("""
    ### 🚀 快速开始指南
    
    1. **选择分析功能**: 在左侧导航栏选择需要的功能
    2. **输入股票代码**: 在量化分析页面输入6位股票代码
    3. **设置分析参数**: 调整分析天数和缓存设置
    4. **运行分析**: 点击开始量化分析按钮
    5. **查看结果**: 查看分析结果、图表和导出报告
    
    ### 💡 核心功能说明
    
    #### 📈 量化分析
    - 基于多因子模型的智能买卖点分析
    - 动态市场状态识别和权重调整
    - 完整的风险管理建议
    
    #### ⭐ 自选股管理  
    - 添加和管理关注的股票
    - 批量分析自选股组合
    - 实时监控信号变化
    
    #### 🔍 模型验证
    - 记录和验证模型预测准确性
    - 分析不同信号类型的表现
    - 置信度分布统计
    - 自动验证功能（收盘后自动获取数据）
    
    #### 💾 数据管理
    - 智能缓存系统管理
    - 数据统计和清理
    - 性能优化设置
    
    ### 📊 支持的股票市场
    
    - **沪市主板**: 6开头（如600519贵州茅台）
    - **深市主板**: 0开头（如000858五粮液）  
    - **中小板**: 002开头（如002714牧原股份）
    - **创业板**: 3开头（如300750宁德时代）
    - **科创板**: 688开头（如688981中芯国际）
    
    ### ⚠️ 重要提示
    
    - 系统使用东方财富接口，请确保网络连接正常
    - 建议合理使用缓存功能避免频繁请求
    - 模型验证需要手动输入实际涨跌幅进行验证
    - 自动验证功能在交易日15:00后执行
    - 本系统提供策略参考，不构成投资建议
    
    ### 🔧 技术特性
    
    - 多层级缓存系统，提升分析速度
    - 模块化设计，易于维护和扩展
    - 完整的错误处理和用户提示
    - 响应式界面设计，支持多种导出格式
    - 数据持久化存储，验证结果永久保存
    """)

# ==================== 主界面 ====================
def main():
    # 初始化session state
    if 'navigation' not in st.session_state:
        st.session_state.navigation = "量化分析"
    if 'auto_analyze' not in st.session_state:
        st.session_state.auto_analyze = False
    if 'auto_stock_codes' not in st.session_state:
        st.session_state.auto_stock_codes = ""
    if 'watchlist_analysis_results' not in st.session_state:
        st.session_state.watchlist_analysis_results = None
    if 'is_watchlist_analysis' not in st.session_state:
        st.session_state.is_watchlist_analysis = False
    
    # 检查股票基本信息表是否已初始化
    if 'db_initialized' not in st.session_state:
        stock_count = data_cache.get_stock_count()
        if stock_count == 0:
            # 如果表为空，显示初始化提示
            st.session_state.show_db_init_warning = True
        else:
            st.session_state.show_db_init_warning = False
        st.session_state.db_initialized = True
    
    # 显示数据库初始化警告（如果需要）
    if st.session_state.get('show_db_init_warning', False):
        st.warning("""
        📊 **股票数据库未初始化**
        
        系统检测到股票基本信息数据库为空，这可能会影响股票搜索和名称显示功能。
        
        请执行以下步骤初始化数据库：
        
        1. 运行股票数据更新脚本：
        ```bash
        python update_stock_basic_info.py
        ```
        
        2. 重启本应用
        
        或者，您可以在"数据管理"页面手动更新股票基本信息。
        """)
    
    # 重新设计的侧边栏CSS样式 - TradingAgents-CN风格
    st.markdown("""
        <style>
        /* 主内容区域样式 */
        .main {background-color: #ffffff; color: #1e1e1e;}
        
        /* 侧边栏样式 - 白色背景黑色字体 */
        .sidebar .sidebar-content {
            background-color: #ffffff !important;
            color: #1e1e1e !important;
            border-right: 1px solid #e0e0e0;
        }
        
        /* 确保侧边栏所有文字可见 */
        .sidebar .sidebar-content * {
            color: #1e1e1e !important;
        }
        
        /* 侧边栏标题样式 - 专业风格 */
        .sidebar-title {
            color: #1e1e1e !important;
            font-size: 1.6rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            text-align: center;
            padding: 0.8rem;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 8px;
            border: none; /* 移除黑色框线 */
        }
        
        /* 导航按钮样式 - 专业简约 */
        .nav-btn {
            width: 100%;
            padding: 0.75rem 1rem;
            margin: 0.25rem 0;
            text-align: left;
            border: none;
            border-radius: 6px;
            background-color: #ffffff;
            color: #1e1e1e !important;
            font-size: 0.95rem;
            font-weight: 500;
            transition: all 0.2s ease;
            cursor: pointer;
            border: 1px solid #e9ecef;
        }
        
        .nav-btn:hover {
            background-color: #f8f9fa;
            border-color: #007bff;
            transform: translateX(4px);
        }
        
        .nav-btn.active {
            background-color: #007bff;
            color: white !important;
            font-weight: 600;
            border-color: #007bff;
            box-shadow: 0 2px 4px rgba(0,123,255,0.2);
        }
        
        /* 主内容区域按钮样式 */
        .stButton>button {
            background-color: #007bff; 
            color: white; 
            border-radius: 6px;
            border: none;
            padding: 0.6rem 1.2rem;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            background-color: #0056b3;
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .stDownloadButton>button {
            background-color: #28a745; 
            color: white; 
            border-radius: 6px;
            border: none;
            transition: all 0.2s ease;
        }
        .stDownloadButton>button:hover {
            background-color: #1e7e34;
            transform: translateY(-1px);
        }
        
        .stTextInput>div>div>input {
            color: #1e1e1e !important;
            background-color: #ffffff;
            border: 1px solid #ced4da;
            border-radius: 6px;
            transition: all 0.2s ease;
        }
        .stTextInput>div>div>input:focus {
            border-color: #007bff;
            box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25);
        }
        
        h1, h2, h3 {
            color: #1e1e1e;
            font-weight: 600;
        }
        
        .stAlert {
            border-radius: 6px;
            border: 1px solid;
        }
        
        .stProgress > div > div > div {
            background-color: #007bff;
        }
        
        .stDataFrame {
            border-radius: 6px;
            border: 1px solid #e0e0e0;
        }
        
        /* 系统信息样式 - 专业卡片 */
        .system-info {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 6px;
            margin: 1rem 0;
            font-size: 0.85rem;
            border: 1px solid #e9ecef;
        }
        
        .system-info h3 {
            color: #1e1e1e !important;
            margin-bottom: 0.8rem;
            font-weight: 600;
            text-align: center;
        }
        
        /* 卡片样式 */
        .card {
            background-color: white;
            border-radius: 6px;
            padding: 1rem;
            margin: 0.5rem 0;
            border: 1px solid #e0e0e0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        /* 响应式调整 */
        @media (max-width: 768px) {
            .sidebar .sidebar-content {
                width: 220px;
            }
            .sidebar-title {
                font-size: 1.4rem;
            }
        }
        
        /* 分隔线样式 */
        .sidebar-divider {
            height: 1px;
            background-color: #e0e0e0;
            margin: 1.5rem 0;
        }
        
        /* 版权信息样式 */
        .copyright {
            text-align: center;
            font-size: 0.75rem;
            color: #6c757d;
            margin-top: 1rem;
        }
        
        /* 指标卡片样式 */
        .metric-card {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid #dee2e6;
        }
        
        /* 标签页样式 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f8f9fa;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #007bff;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        # 标题 - 专业风格
        st.markdown('<div class="sidebar-title">🎯 幻方量化分析系统</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        
        # 导航菜单
        nav_options = [
            ("📈", "量化分析"),
            ("⭐", "自选股管理"), 
            ("🔍", "模型验证"),
            ("💾", "数据管理"),
            ("❓", "帮助")
        ]
        
        for icon, page_name in nav_options:
            is_active = st.session_state.navigation == page_name
            btn_class = "nav-btn active" if is_active else "nav-btn"
            
            if st.button(f"{icon} {page_name}", key=f"nav_{page_name}", use_container_width=True):
                st.session_state.navigation = page_name
                st.rerun()
        
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        
        # 系统信息
        st.markdown('<div class="system-info">', unsafe_allow_html=True)
        st.markdown("### 📊 系统信息")
        
        # 当前时间信息
        current_time = datetime.now()
        st.markdown(f"**📅 日期**: {current_time.strftime('%Y-%m-%d')}")
        st.markdown(f"**⏰ 时间**: {current_time.strftime('%H:%M:%S')}")

        # 显示股票数据库信息
        stock_count = data_cache.get_stock_count()
        if stock_count > 0:
            st.markdown(f"**📈 股票数据库**: {stock_count:,} 只")
        else:
            st.markdown("**📈 股票数据库**: 未初始化")

        st.markdown("**🟢 状态**: 运行中")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="copyright">© 2025 幻方量化分析系统 v2.9.0</div>', unsafe_allow_html=True)
    
    # 根据选择的页面显示内容
    if st.session_state.navigation == "量化分析":
        quantitative_analysis_page()
    elif st.session_state.navigation == "自选股管理":
        watchlist_management_page()
    elif st.session_state.navigation == "模型验证":
        model_validation_page()
    elif st.session_state.navigation == "数据管理":
        data_management_page()
    elif st.session_state.navigation == "帮助":
        help_page()

if __name__ == "__main__":
    main()
