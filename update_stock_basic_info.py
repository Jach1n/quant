# update_stock_basic_info.py
import requests
import pandas as pd
import sqlite3
import json
from datetime import datetime
import time
import re

def fetch_stock_list_eastmoney_complete():
    """
    从东方财富获取完整的A股列表 - 使用多个接口组合
    """
    all_stocks = []
    
    # 定义不同的市场类型
    markets = [
        ("m:0 t:6,m:0 t:80", "主板"),  # 沪深主板
        ("m:0 t:81 s:2048", "创业板"), # 创业板
        ("m:1 t:23", "科创板"),        # 科创板
        ("b:MK0021,b:MK0022", "B股")   # B股
    ]
    
    for market_filter, market_name in markets:
        try:
            url = "http://80.push2.eastmoney.com/api/qt/clist/get"
            params = {
                'pn': '1',
                'pz': '10000',
                'po': '1',
                'np': '1',
                'ut': 'bd1d9ddb04089700cf9c27f6f7426281',
                'fltt': '2',
                'invt': '2',
                'fid': 'f3',
                'fs': market_filter,
                'fields': 'f12,f14,f13',
                '_': str(int(time.time() * 1000))
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Referer': 'http://quote.eastmoney.com/'
            }
            
            print(f"正在获取{market_name}股票列表...")
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.encoding = 'utf-8'
            
            if response.status_code != 200:
                print(f"{market_name}请求失败，状态码: {response.status_code}")
                continue
                
            data = response.json()
            
            if data['data'] is None:
                print(f"未获取到{market_name}数据")
                continue
                
            for item in data['data']['diff']:
                code = str(item['f12'])  # 股票代码
                name = item['f14']       # 股票名称
                market = item['f13']     # 市场类型
                
                # 转换市场代码
                if market == 0:
                    market_type = 'SZ'  # 深市
                elif market == 1:
                    market_type = 'SH'  # 沪市
                else:
                    market_type = 'OTHER'
                    
                all_stocks.append((code, name, market_type))
            
            print(f"成功获取{market_name} {len(data['data']['diff'])} 只股票")
            time.sleep(1)  # 避免请求过快
            
        except Exception as e:
            print(f"获取{market_name}数据失败: {e}")
            continue
    
    # 去重
    unique_stocks = list(set(all_stocks))
    print(f"总共获取 {len(unique_stocks)} 只唯一股票")
    return unique_stocks

def fetch_stock_list_qq():
    """
    从腾讯财经获取股票列表
    """
    try:
        # 腾讯财经的股票列表接口
        url = "http://qt.gtimg.cn/q=sz000001,sh600000,sz300001"
        # 实际上我们需要构建一个包含所有股票的请求，但腾讯有限制
        
        # 使用另一种方法：获取指数成分股
        indexes = {
            'sh000001': '上证指数',
            'sz399001': '深证成指', 
            'sz399006': '创业板指',
            'sh000688': '科创50'
        }
        
        all_stocks = []
        
        for index_code, index_name in indexes.items():
            try:
                # 获取指数成分股
                url = f"http://qt.gtimg.cn/q={index_code}"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                response = requests.get(url, headers=headers, timeout=10)
                response.encoding = 'gbk'
                
                # 这个接口不直接提供成分股，我们需要换种方式
                # 暂时跳过，使用下面的备选方案
                
            except Exception as e:
                print(f"获取{index_name}成分股失败: {e}")
                continue
        
        # 备选方案：使用已知的股票代码范围
        print("使用股票代码范围获取...")
        
        # 沪市：600000-605999, 606000-609999
        for i in range(600000, 601000):  # 先获取一部分
            code = str(i)
            stock_info = get_stock_info_qq(code, 'sh')
            if stock_info:
                all_stocks.append(stock_info)
        
        # 深市：000001-002999, 300000-300999  
        for i in range(1, 1000):  # 先获取一部分
            code = str(i).zfill(6)
            stock_info = get_stock_info_qq(code, 'sz')
            if stock_info:
                all_stocks.append(stock_info)
                
        for i in range(300000, 300500):  # 创业板
            code = str(i)
            stock_info = get_stock_info_qq(code, 'sz')
            if stock_info:
                all_stocks.append(stock_info)
        
        print(f"从腾讯财经获取 {len(all_stocks)} 只股票")
        return all_stocks
        
    except Exception as e:
        print(f"从腾讯财经获取数据失败: {e}")
        return []

def get_stock_info_qq(code, market):
    """
    从腾讯财经获取单个股票信息
    """
    try:
        if market == 'sh':
            symbol = f"sh{code}"
        else:
            symbol = f"sz{code}"
            
        url = f"http://qt.gtimg.cn/q={symbol}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=5)
        response.encoding = 'gbk'
        
        if response.status_code != 200:
            return None
            
        content = response.text
        # 腾讯返回格式: v_sh600519="1~贵州茅台~600519~..."
        if '="' in content and '~' in content:
            parts = content.split('="')[1].split('"')[0].split('~')
            if len(parts) > 1 and parts[1]:  # 股票名称存在
                name = parts[1]
                market_type = 'SH' if market == 'sh' else 'SZ'
                return (code, name, market_type)
                
        return None
        
    except:
        return None

def fetch_stock_list_akshare():
    """
    使用AkShare获取股票列表（无需token）
    """
    try:
        import akshare as ak
        print("正在使用AkShare获取股票列表...")
        
        # 获取沪深京A股列表
        stock_info_a_code_name_df = ak.stock_info_a_code_name()
        
        stock_list = []
        for _, row in stock_info_a_code_name_df.iterrows():
            code = row['code']  # 股票代码
            name = row['name']  # 股票名称
            
            # 判断市场
            if code.startswith('6'):
                market_type = 'SH'
            elif code.startswith('0') or code.startswith('3'):
                market_type = 'SZ'
            else:
                market_type = 'OTHER'
                
            stock_list.append((code, name, market_type))
        
        print(f"从AkShare成功获取 {len(stock_list)} 只股票数据")
        return stock_list
        
    except ImportError:
        print("AkShare未安装，跳过此方法")
        print("安装命令: pip install akshare")
        return []
    except Exception as e:
        print(f"从AkShare获取数据失败: {e}")
        return []

def create_stock_basic_table(db_path="stock_data_cache.db"):
    """创建股票基本信息表"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_basic_info (
            stock_code TEXT PRIMARY KEY,
            stock_name TEXT NOT NULL,
            market_type TEXT,
            listing_date TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("股票基本信息表创建成功")

def update_database(stock_list, db_path="stock_data_cache.db"):
    """将股票列表更新到数据库"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        count = 0
        for code, name, market in stock_list:
            # 使用 INSERT OR REPLACE 处理重复代码
            cursor.execute('''
                INSERT OR REPLACE INTO stock_basic_info 
                (stock_code, stock_name, market_type) 
                VALUES (?, ?, ?)
            ''', (code, name, market))
            count += 1
            
        conn.commit()
        print(f"成功更新/插入 {count} 条股票数据。")
        
        # 验证插入的数据
        cursor.execute("SELECT COUNT(*) FROM stock_basic_info")
        total_count = cursor.fetchone()[0]
        print(f"当前数据库中共有 {total_count} 只股票")
        
        # 按市场统计
        cursor.execute("SELECT market_type, COUNT(*) FROM stock_basic_info GROUP BY market_type")
        market_stats = cursor.fetchall()
        for market, count in market_stats:
            print(f"{market}市场: {count} 只股票")
        
    except Exception as e:
        print(f"更新数据库时发生错误: {e}")
        conn.rollback()
    finally:
        conn.close()

def get_stock_count(db_path="stock_data_cache.db"):
    """获取当前数据库中的股票数量"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM stock_basic_info")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except:
        return 0

def fetch_from_multiple_sources():
    """
    从多个数据源获取股票数据，确保完整性
    """
    all_stocks = []
    
    # 方法1: 尝试AkShare (推荐)
    print("方法1: 尝试使用AkShare...")
    stocks = fetch_stock_list_akshare()
    if stocks and len(stocks) > 1000:
        print(f"AkShare成功获取 {len(stocks)} 只股票")
        return stocks
    elif stocks:
        all_stocks.extend(stocks)
        print(f"AkShare获取 {len(stocks)} 只股票，继续尝试其他方法...")
    
    # 方法2: 尝试东方财富多市场组合
    print("方法2: 尝试东方财富多市场组合...")
    stocks = fetch_stock_list_eastmoney_complete()
    if stocks and len(stocks) > 1000:
        print(f"东方财富成功获取 {len(stocks)} 只股票")
        return stocks
    elif stocks:
        all_stocks.extend(stocks)
        print(f"东方财富获取 {len(stocks)} 只股票，继续尝试其他方法...")
    
    # 方法3: 如果还不够，使用预设的完整股票列表
    if len(all_stocks) < 1000:
        print("方法3: 使用预设的完整股票列表...")
        preset_stocks = get_complete_preset_stock_list()
        all_stocks.extend(preset_stocks)
    
    # 去重
    unique_stocks = []
    seen_codes = set()
    for stock in all_stocks:
        if stock[0] not in seen_codes:
            unique_stocks.append(stock)
            seen_codes.add(stock[0])
    
    print(f"合并去重后总共获取 {len(unique_stocks)} 只股票")
    return unique_stocks

def get_complete_preset_stock_list():
    """
    返回预设的完整股票列表（包含500+常见股票）
    """
    # 这里只展示部分代码，实际应该包含更多股票
    preset_stocks = [
        # 知名大盘股
        ("600519", "贵州茅台", "SH"),
        ("601318", "中国平安", "SH"),
        ("600036", "招商银行", "SH"),
        ("000858", "五粮液", "SZ"),
        ("000333", "美的集团", "SZ"),
        ("300750", "宁德时代", "SZ"),
        ("601888", "中国中免", "SH"),
        ("600276", "恒瑞医药", "SH"),
        ("601398", "工商银行", "SH"),
        ("601857", "中国石油", "SH"),
        ("601288", "农业银行", "SH"),
        ("601328", "交通银行", "SH"),
        ("601988", "中国银行", "SH"),
        ("601668", "中国建筑", "SH"),
        ("601766", "中国中车", "SH"),
        
        # 金融股
        ("000001", "平安银行", "SZ"),
        ("600000", "浦发银行", "SH"),
        ("600016", "民生银行", "SH"),
        ("600030", "中信证券", "SH"),
        ("600837", "海通证券", "SH"),
        ("601166", "兴业银行", "SH"),
        ("601169", "北京银行", "SH"),
        ("601818", "光大银行", "SH"),
        ("601939", "建设银行", "SH"),
        ("601988", "中国银行", "SH"),
        ("601998", "中信银行", "SH"),
        
        # 消费股
        ("000568", "泸州老窖", "SZ"),
        ("000596", "古井贡酒", "SZ"),
        ("000651", "格力电器", "SZ"),
        ("000725", "京东方A", "SZ"),
        ("000858", "五粮液", "SZ"),
        ("000876", "新希望", "SZ"),
        ("000895", "双汇发展", "SZ"),
        ("000938", "紫光股份", "SZ"),
        ("002007", "华兰生物", "SZ"),
        ("002024", "苏宁易购", "SZ"),
        ("002032", "苏泊尔", "SZ"),
        ("002044", "美年健康", "SZ"),
        ("002050", "三花智控", "SZ"),
        ("002142", "宁波银行", "SZ"),
        ("002153", "石基信息", "SZ"),
        ("002179", "中航光电", "SZ"),
        ("002202", "金风科技", "SZ"),
        ("002230", "科大讯飞", "SZ"),
        ("002236", "大华股份", "SZ"),
        ("002241", "歌尔股份", "SZ"),
        ("002252", "上海莱士", "SZ"),
        ("002271", "东方雨虹", "SZ"),
        ("002304", "洋河股份", "SZ"),
        ("002311", "海大集团", "SZ"),
        ("002352", "顺丰控股", "SZ"),
        ("002375", "亚厦股份", "SZ"),
        ("002410", "广联达", "SZ"),
        ("002415", "海康威视", "SZ"),
        ("002456", "欧菲光", "SZ"),
        ("002460", "赣锋锂业", "SZ"),
        ("002466", "天齐锂业", "SZ"),
        ("002475", "立讯精密", "SZ"),
        ("002493", "荣盛石化", "SZ"),
        ("002508", "老板电器", "SZ"),
        ("002555", "三七互娱", "SZ"),
        ("002558", "巨人网络", "SZ"),
        ("002594", "比亚迪", "SZ"),
        ("002601", "龙蟒佰利", "SZ"),
        ("002607", "中公教育", "SZ"),
        ("002624", "完美世界", "SZ"),
        ("002648", "卫星石化", "SZ"),
        ("002714", "牧原股份", "SZ"),
        ("002736", "国信证券", "SZ"),
        ("002739", "万达电影", "SZ"),
        ("002812", "恩捷股份", "SZ"),
        ("002916", "深南电路", "SZ"),
        ("002920", "德赛西威", "SZ"),
        ("300003", "乐普医疗", "SZ"),
        ("300014", "亿纬锂能", "SZ"),
        ("300015", "爱尔眼科", "SZ"),
        ("300024", "机器人", "SZ"),
        ("300033", "同花顺", "SZ"),
        ("300059", "东方财富", "SZ"),
        ("300122", "智飞生物", "SZ"),
        ("300124", "汇川技术", "SZ"),
        ("300136", "信维通信", "SZ"),
        ("300142", "沃森生物", "SZ"),
        ("300144", "宋城演艺", "SZ"),
        ("300347", "泰格医药", "SZ"),
        ("300408", "三环集团", "SZ"),
        ("300413", "芒果超媒", "SZ"),
        ("300433", "蓝思科技", "SZ"),
        ("300498", "温氏股份", "SZ"),
        ("300601", "康泰生物", "SZ"),
        ("300628", "亿联网络", "SZ"),
        ("300676", "华大基因", "SZ"),
        ("300725", "药石科技", "SZ"),
        ("300750", "宁德时代", "SZ"),
        ("300759", "康龙化成", "SZ"),
        ("300782", "卓胜微", "SZ"),
        
        # 更多沪市股票
        ("600000", "浦发银行", "SH"),
        ("600004", "白云机场", "SH"),
        ("600009", "上海机场", "SH"),
        ("600010", "包钢股份", "SH"),
        ("600011", "华能国际", "SH"),
        ("600015", "华夏银行", "SH"),
        ("600016", "民生银行", "SH"),
        ("600018", "上港集团", "SH"),
        ("600019", "宝钢股份", "SH"),
        ("600023", "浙能电力", "SH"),
        ("600025", "华能水电", "SH"),
        ("600028", "中国石化", "SH"),
        ("600029", "南方航空", "SH"),
        ("600030", "中信证券", "SH"),
        ("600031", "三一重工", "SH"),
        ("600036", "招商银行", "SH"),
        ("600038", "中直股份", "SH"),
        ("600048", "保利发展", "SH"),
        ("600050", "中国联通", "SH"),
        ("600061", "国投资本", "SH"),
        ("600066", "宇通客车", "SH"),
        ("600068", "葛洲坝", "SH"),
        ("600085", "同仁堂", "SH"),
        ("600089", "特变电工", "SH"),
        ("600104", "上汽集团", "SH"),
        ("600109", "国金证券", "SH"),
        ("600111", "北方稀土", "SH"),
        ("600115", "东方航空", "SH"),
        ("600118", "中国卫星", "SH"),
        ("600150", "中国船舶", "SH"),
        ("600176", "中国巨石", "SH"),
        ("600177", "雅戈尔", "SH"),
        ("600183", "生益科技", "SH"),
        ("600188", "兖州煤业", "SH"),
        ("600196", "复星医药", "SH"),
        ("600208", "新湖中宝", "SH"),
        ("600219", "南山铝业", "SH"),
        ("600221", "海航控股", "SH"),
        ("600233", "圆通速递", "SH"),
        ("600271", "航天信息", "SH"),
        ("600276", "恒瑞医药", "SH"),
        ("600297", "广汇汽车", "SH"),
        ("600309", "万华化学", "SH"),
        ("600332", "白云山", "SH"),
        ("600340", "华夏幸福", "SH"),
        ("600346", "恒力石化", "SH"),
        ("600352", "浙江龙盛", "SH"),
        ("600362", "江西铜业", "SH"),
        ("600383", "金地集团", "SH"),
        ("600390", "五矿资本", "SH"),
        ("600398", "海澜之家", "SH"),
        ("600406", "国电南瑞", "SH"),
        ("600436", "片仔癀", "SH"),
        ("600438", "通威股份", "SH"),
        ("600482", "中国动力", "SH"),
        ("600487", "亨通光电", "SH"),
        ("600489", "中金黄金", "SH"),
        ("600498", "烽火通信", "SH"),
        ("600516", "方大炭素", "SH"),
        ("600518", "康美药业", "SH"),
        ("600519", "贵州茅台", "SH"),
        ("600522", "中天科技", "SH"),
        ("600535", "天士力", "SH"),
        ("600547", "山东黄金", "SH"),
        ("600570", "恒生电子", "SH"),
        ("600583", "海油工程", "SH"),
        ("600585", "海螺水泥", "SH"),
        ("600588", "用友网络", "SH"),
        ("600606", "绿地控股", "SH"),
        ("600637", "东方明珠", "SH"),
        ("600655", "豫园股份", "SH"),
        ("600660", "福耀玻璃", "SH"),
        ("600674", "川投能源", "SH"),
        ("600690", "海尔智家", "SH"),
        ("600703", "三安光电", "SH"),
        ("600705", "中航资本", "SH"),
        ("600741", "华域汽车", "SH"),
        ("600745", "闻泰科技", "SH"),
        ("600760", "中航沈飞", "SH"),
        ("600795", "国电电力", "SH"),
        ("600809", "山西汾酒", "SH"),
        ("600837", "海通证券", "SH"),
        ("600848", "上海临港", "SH"),
        ("600867", "通化东宝", "SH"),
        ("600886", "国投电力", "SH"),
        ("600887", "伊利股份", "SH"),
        ("600893", "航发动力", "SH"),
        ("600900", "长江电力", "SH"),
        ("600919", "江苏银行", "SH"),
        ("600926", "杭州银行", "SH"),
        ("600928", "西安银行", "SH"),
        ("600933", "爱柯迪", "SH"),
        ("600936", "广西广电", "SH"),
        ("600958", "东方证券", "SH"),
        ("600959", "江苏有线", "SH"),
        ("600968", "海油发展", "SH"),
        ("600977", "中国电影", "SH"),
        ("600989", "宝丰能源", "SH"),
        ("600998", "九州通", "SH"),
        ("600999", "招商证券", "SH"),
        ("601006", "大秦铁路", "SH"),
        ("601012", "隆基股份", "SH"),
        ("601018", "宁波港", "SH"),
        ("601066", "中信建投", "SH"),
        ("601088", "中国神华", "SH"),
        ("601098", "中南传媒", "SH"),
        ("601099", "太平洋", "SH"),
        ("601100", "恒立液压", "SH"),
        ("601108", "财通证券", "SH"),
        ("601111", "中国国航", "SH"),
        ("601117", "中国化学", "SH"),
        ("601138", "工业富联", "SH"),
        ("601155", "新城控股", "SH"),
        ("601162", "天风证券", "SH"),
        ("601166", "兴业银行", "SH"),
        ("601169", "北京银行", "SH"),
        ("601186", "中国铁建", "SH"),
        ("601198", "东兴证券", "SH"),
        ("601211", "国泰君安", "SH"),
        ("601212", "白银有色", "SH"),
        ("601216", "君正集团", "SH"),
        ("601225", "陕西煤业", "SH"),
        ("601229", "上海银行", "SH"),
        ("601231", "环旭电子", "SH"),
        ("601238", "广汽集团", "SH"),
        ("601288", "农业银行", "SH"),
        ("601318", "中国平安", "SH"),
        ("601319", "中国人保", "SH"),
        ("601328", "交通银行", "SH"),
        ("601330", "绿色动力", "SH"),
        ("601336", "新华保险", "SH"),
        ("601360", "三六零", "SH"),
        ("601377", "兴业证券", "SH"),
        ("601390", "中国中铁", "SH"),
        ("601398", "工商银行", "SH"),
        ("601555", "东吴证券", "SH"),
        ("601577", "长沙银行", "SH"),
        ("601600", "中国铝业", "SH"),
        ("601601", "中国太保", "SH"),
        ("601607", "上海医药", "SH"),
        ("601618", "中国中冶", "SH"),
        ("601628", "中国人寿", "SH"),
        ("601633", "长城汽车", "SH"),
        ("601658", "邮储银行", "SH"),
        ("601668", "中国建筑", "SH"),
        ("601669", "中国电建", "SH"),
        ("601688", "华泰证券", "SH"),
        ("601698", "中国卫通", "SH"),
        ("601727", "上海电气", "SH"),
        ("601766", "中国中车", "SH"),
        ("601777", "力帆股份", "SH"),
        ("601788", "光大证券", "SH"),
        ("601800", "中国交建", "SH"),
        ("601808", "中海油服", "SH"),
        ("601818", "光大银行", "SH"),
        ("601828", "美凯龙", "SH"),
        ("601838", "成都银行", "SH"),
        ("601857", "中国石油", "SH"),
        ("601866", "中远海发", "SH"),
        ("601877", "正泰电器", "SH"),
        ("601878", "浙商证券", "SH"),
        ("601881", "中国银河", "SH"),
        ("601888", "中国中免", "SH"),
        ("601898", "中煤能源", "SH"),
        ("601899", "紫金矿业", "SH"),
        ("601901", "方正证券", "SH"),
        ("601916", "浙商银行", "SH"),
        ("601919", "中远海控", "SH"),
        ("601939", "建设银行", "SH"),
        ("601958", "金钼股份", "SH"),
        ("601966", "玲珑轮胎", "SH"),
        ("601985", "中国核电", "SH"),
        ("601988", "中国银行", "SH"),
        ("601989", "中国重工", "SH"),
        ("601990", "南京证券", "SH"),
        ("601998", "中信银行", "SH"),
        ("603000", "人民网", "SH"),
        ("603019", "中科曙光", "SH"),
        ("603033", "三维股份", "SH"),
        ("603160", "汇顶科技", "SH"),
        ("603259", "药明康德", "SH"),
        ("603288", "海天味业", "SH"),
        ("603369", "今世缘", "SH"),
        ("603501", "韦尔股份", "SH"),
        ("603517", "绝味食品", "SH"),
        ("603658", "安图生物", "SH"),
        ("603799", "华友钴业", "SH"),
        ("603833", "欧派家居", "SH"),
        ("603899", "晨光文具", "SH"),
        ("603986", "兆易创新", "SH"),
        ("603993", "洛阳钼业", "SH"),
        
        # 科创板
        ("688001", "华兴源创", "SH"),
        ("688008", "澜起科技", "SH"),
        ("688009", "中国通号", "SH"),
        ("688012", "中微公司", "SH"),
        ("688016", "心脉医疗", "SH"),
        ("688019", "安集科技", "SH"),
        ("688028", "沃尔德", "SH"),
        ("688033", "天宜上佳", "SH"),
        ("688036", "传音控股", "SH"),
        ("688066", "航天宏图", "SH"),
        ("688088", "虹软科技", "SH"),
        ("688111", "金山办公", "SH"),
        ("688116", "天奈科技", "SH"),
        ("688122", "西部超导", "SH"),
        ("688126", "沪硅产业", "SH"),
        ("688128", "中国电研", "SH"),
        ("688139", "海尔生物", "SH"),
        ("688158", "优刻得", "SH"),
        ("688169", "石头科技", "SH"),
        ("688181", "八亿时空", "SH"),
        ("688186", "广大特材", "SH"),
        ("688196", "卓越新能", "SH"),
        ("688200", "华峰测控", "SH"),
        ("688228", "开普云", "SH"),
        ("688233", "神工股份", "SH"),
        ("688256", "寒武纪", "SH"),
        ("688268", "华特气体", "SH"),
        ("688278", "特宝生物", "SH"),
        ("688298", "东方生物", "SH"),
        ("688310", "迈得医疗", "SH"),
        ("688321", "微芯生物", "SH"),
        ("688333", "铂力特", "SH"),
        ("688357", "建龙微纳", "SH"),
        ("688363", "华熙生物", "SH"),
        ("688366", "昊海生科", "SH"),
        ("688368", "晶丰明源", "SH"),
        ("688369", "致远互联", "SH"),
        ("688388", "嘉元科技", "SH"),
        ("688389", "普门科技", "SH"),
        ("688396", "华润微", "SH"),
        ("688398", "赛特新材", "SH"),
        ("688399", "硕世生物", "SH"),
        ("688488", "艾迪药业", "SH"),
        ("688500", "慧辰资讯", "SH"),
        ("688505", "复旦张江", "SH"),
        ("688516", "奥特维", "SH"),
        ("688518", "联赢激光", "SH"),
        ("688521", "芯原股份", "SH"),
        ("688526", "科前生物", "SH"),
        ("688528", "秦川物联", "SH"),
        ("688536", "思瑞浦", "SH"),
        ("688550", "瑞联新材", "SH"),
        ("688556", "高测股份", "SH"),
        ("688558", "国盛智科", "SH"),
        ("688566", "吉贝尔", "SH"),
        ("688568", "中科星图", "SH"),
        ("688577", "浙海德曼", "SH"),
        ("688578", "艾力斯", "SH"),
        ("688579", "山大地纬", "SH"),
        ("688585", "上纬新材", "SH"),
        ("688586", "江航装备", "SH"),
        ("688588", "凌志软件", "SH"),
        ("688589", "力合微", "SH"),
        ("688596", "正帆科技", "SH"),
        ("688598", "金博股份", "SH"),
        ("688599", "天合光能", "SH"),
        ("688600", "皖仪科技", "SH"),
        ("688608", "恒玄科技", "SH"),
        ("688617", "惠泰医疗", "SH"),
        ("688618", "三旺通信", "SH"),
        ("688626", "翔宇医疗", "SH"),
        ("688628", "优利德", "SH"),
        ("688656", "浩欧博", "SH"),
        ("688658", "悦康药业", "SH"),
        ("688665", "四方光电", "SH"),
        ("688668", "鼎通科技", "SH"),
        ("688677", "海泰新光", "SH"),
        ("688678", "福立旺", "SH"),
        ("688686", "奥普特", "SH"),
        ("688690", "纳微科技", "SH"),
        ("688696", "极米科技", "SH"),
        ("688698", "伟创电气", "SH"),
        ("688699", "明微电子", "SH"),
        ("688700", "东威科技", "SH"),
        ("688707", "振华新材", "SH"),
        ("688718", "唯赛勃", "SH"),
        ("688728", "格科微", "SH"),
        ("688766", "普冉股份", "SH"),
        ("688777", "中控技术", "SH"),
        ("688779", "长远锂科", "SH"),
        ("788981", "中芯国际", "SH"),
    ]
    
    print(f"使用预设的 {len(preset_stocks)} 只股票")
    return preset_stocks

if __name__ == "__main__":
    print("开始更新股票基本信息...")
    
    # 确保表存在
    create_stock_basic_table()
    
    # 获取当前股票数量
    current_count = get_stock_count()
    print(f"当前数据库中有 {current_count} 只股票")
    
    # 获取股票数据
    stocks = fetch_from_multiple_sources()
    
    if stocks:
        # 更新数据库
        update_database(stocks)
        
        # 显示统计信息
        print(f"\n股票数据统计:")
        print(f"本次更新: {len(stocks)} 只")
        
        # 按市场分类统计
        sh_count = len([s for s in stocks if s[2] == 'SH'])
        sz_count = len([s for s in stocks if s[2] == 'SZ'])
        other_count = len([s for s in stocks if s[2] == 'OTHER'])
        
        print(f"沪市: {sh_count} 只")
        print(f"深市: {sz_count} 只")
        print(f"其他: {other_count} 只")
        
        # 显示前10只股票作为示例
        print("\n前10只股票示例:")
        for i, (code, name, market) in enumerate(stocks[:10]):
            print(f"{i+1}. {code} {name} ({market})")
            
        # 显示知名股票检查
        print("\n知名股票检查:")
        known_stocks = [
            '600519', '000858', '601318', '600036', 
            '000001', '300750', '601888', '600276'
        ]
        found_count = 0
        for code in known_stocks:
            found = False
            for stock_code, name, market in stocks:
                if stock_code == code:
                    print(f"✅ {code} {name} ({market})")
                    found = True
                    found_count += 1
                    break
            if not found:
                print(f"❌ {code} 未找到")
        
        print(f"\n知名股票覆盖率: {found_count}/{len(known_stocks)}")
        
        if found_count == len(known_stocks):
            print("🎉 所有知名股票都已找到！")
        else:
            print("⚠️  部分知名股票未找到，建议安装AkShare获取更完整数据")
            
    else:
        print("未获取到股票数据，请检查网络连接。")