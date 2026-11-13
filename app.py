import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from catboost import CatBoostClassifier
from models import db, User, PredictionHistory, City, CityPolicy
from datetime import datetime
import pymysql
import random
import traceback

pymysql.install_as_MySQLdb()

app = Flask(__name__)

# 配置
app.config['SECRET_KEY'] = 'dev-secret-key-change-in-production'
# MySQL连接: user:password@host:port/dbname
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:1234%40Aliyun@114.132.83.76:3306/happiness_system'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 初始化扩展
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# 全局变量
MODEL = None
DEFAULTS = {}
FEATURE_COLUMNS = []


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def load_resources():
    """加载模型和资源"""
    global MODEL, DEFAULTS, FEATURE_COLUMNS

    script_dir = os.path.dirname(os.path.abspath(__file__))
    mode_dir = os.path.join(script_dir, 'mode')

    # 1. 加载模型
    try:
        model_path = os.path.join(mode_dir, 'catboost_model.cbm')
        MODEL = CatBoostClassifier()
        MODEL.load_model(model_path)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")

    # 2. 加载默认值
    try:
        with open(os.path.join(mode_dir, 'model_defaults.json'), 'r', encoding='utf-8') as f:
            DEFAULTS = json.load(f)
        print("Defaults loaded")
    except Exception as e:
        print(f"Error loading defaults: {e}")

    # 3. 加载特征列表
    try:
        with open(os.path.join(mode_dir, 'feature_columns.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
            FEATURE_COLUMNS = data['feature_columns']
        print(f"Feature columns loaded: {len(FEATURE_COLUMNS)} features")
    except Exception as e:
        print(f"Error loading feature columns: {e}")


# --- 辅助函数 ---

def preprocess_csv_row(row):
    """
    预处理CSV行数据，确保与模型兼容
    """
    processed = {}

    # 映射CSV字段到模型字段
    field_mapping = {
        'gender': ('gender', lambda x: 1 if str(x) == '1' else 2),
        'birth': ('birth_year', lambda x: int(float(x))),
        'edu': ('edu', lambda x: int(float(x))),
        'income': ('income', lambda x: float(x)),
        'family_income': ('family_income', lambda x: float(x)),
        'family_m': ('family_m', lambda x: int(float(x))),
        'house': ('house', lambda x: int(float(x))),
        'car': ('car', lambda x: int(float(x))),
        'equity': ('equity', lambda x: int(float(x))),
        'depression': ('depression', lambda x: int(float(x))),
        'class': ('class', lambda x: int(float(x))),
        'status_peer': ('status_peer', lambda x: int(float(x))),
        'province': ('province', lambda x: int(float(x))),
        'political': ('political', lambda x: int(float(x))),
        'marital': ('marital_status', lambda x: int(float(x))),
        'hukou': ('hukou', lambda x: int(float(x))),
        'work_status': ('work_status', lambda x: int(float(x))),
        'work_type': ('work_type', lambda x: int(float(x))),
        'work_manage': ('work_manage', lambda x: int(float(x))),
        'work_yr': ('work_yr', lambda x: int(float(x))),
        'religion': ('religion', lambda x: int(float(x))),
        'nationality': ('nationality', lambda x: int(float(x))),
        'health': ('health', lambda x: int(float(x))),
        'health_problem': ('health_problem', lambda x: int(float(x))),
        'socialize': ('socialize', lambda x: int(float(x))),
        'relax': ('relax', lambda x: int(float(x))),
        'learn': ('learn', lambda x: int(float(x))),
        'floor_area': ('floor_area', lambda x: float(x)),
        'height_cm': ('height_cm', lambda x: float(x)),
        'weight_jin': ('weight_jin', lambda x: float(x)),
        'view': ('view', lambda x: int(float(x))),
        'inc_ability': ('inc_ability', lambda x: int(float(x))),
        'religion_freq': ('religion_freq', lambda x: int(float(x)))
    }

    # 首先处理原始数据
    for csv_field in row.index:
        if pd.notna(row[csv_field]):
            value = row[csv_field]
            # 检查是否有映射关系
            if csv_field in field_mapping:
                model_field, converter = field_mapping[csv_field]
                try:
                    processed[model_field] = converter(value)
                except (ValueError, TypeError):
                    # 如果转换失败，使用默认值
                    if model_field in ['gender', 'edu', 'political', 'marital_status',
                                       'province', 'hukou', 'religion']:
                        processed[model_field] = str(value)
                    else:
                        processed[model_field] = 0
            else:
                # 如果没有映射，直接添加
                try:
                    processed[csv_field] = float(value)
                except (ValueError, TypeError):
                    processed[csv_field] = str(value)

    # 计算年龄（如果只有出生年份）
    if 'birth_year' in processed and 'age' not in processed:
        try:
            birth_year = int(processed['birth_year'])
            current_year = datetime.now().year
            age = max(18, min(80, current_year - birth_year))
            processed['age'] = age
        except:
            processed['age'] = 35

    # 确保必需字段有值
    defaults = {
        'age': 35,
        'gender': 1,
        'edu': 12,
        'income': 50000,
        'family_income': 120000,
        'family_m': 3,
        'house': 1,
        'car': 0,
        'equity': 3,
        'depression': 2,
        'class': 5,
        'status_peer': 2,
        'political': 4,  # 群众
        'province': 11,  # 北京
        'hukou': 2,  # 城镇户口
        'marital_status': 3,  # 已婚
        'work_status': 5,  # 在职
        'health': 4,  # 健康
        'nationality': 1,  # 汉族
        'religion': 0  # 无宗教信仰
    }

    for field, default_value in defaults.items():
        if field not in processed:
            processed[field] = default_value

    return processed


def predict_single_resident(data):
    """
    预测单个居民的幸福感
    """
    try:
        # 使用与主预测函数相同的逻辑
        features = DEFAULTS.copy()

        # 更新特征，确保类型正确
        for key, value in data.items():
            if value is not None and value != '':
                # 处理特定字段的转换
                if key in ['gender', 'edu', 'political', 'marital_status', 'work_status',
                           'province', 'city', 'county', 'nationality', 'religion',
                           'survey_type', 'hukou', 'work_type']:
                    # 分类特征转换为字符串
                    try:
                        features[key] = str(int(float(value)))
                    except:
                        features[key] = str(value)
                else:
                    # 数值特征
                    try:
                        features[key] = float(value)
                    except (ValueError, TypeError):
                        features[key] = 0.0

        # 计算衍生特征
        family_income = float(features.get('family_income', 0))
        family_m = float(features.get('family_m', 3))
        house = float(features.get('house', 0))
        car = float(features.get('car', 0))
        depression = float(features.get('depression', 0))
        equity = float(features.get('equity', 0))

        # 人均收入
        if family_m > 0:
            features['income_per_member'] = family_income / family_m
        else:
            features['income_per_member'] = family_income

        # 贫困指数
        poverty_index = 0
        if family_income < 30000:
            poverty_index += 1
        if house == 0:
            poverty_index += 1
        if car == 0:
            poverty_index += 1
        features['poverty_index'] = float(poverty_index)

        # 高压力指数
        if depression >= 4 and equity <= 2:
            features['high_stress'] = 1.0
        else:
            features['high_stress'] = 0.0

        # 房车总数
        features['house_car_count'] = house + car

        # Log变换
        if 'income' in features:
            features['income_log'] = np.log1p(float(features['income']))
        if 'family_income' in features:
            features['family_income_log'] = np.log1p(family_income)

        # 构建模型输入
        cat_indices = set(MODEL.get_cat_feature_indices())
        input_row = []
        for i, col in enumerate(FEATURE_COLUMNS):
            val = features.get(col, DEFAULTS.get(col))

            if i in cat_indices:
                val = str(val) if val is not None else "Unknown"
            else:
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    val = 0.0

            input_row.append(val)

        input_df = pd.DataFrame([input_row], columns=FEATURE_COLUMNS)

        # 预测
        pred_label = MODEL.predict(input_df)[0][0]
        pred_proba = MODEL.predict_proba(input_df)[0]

        # 规则干预
        if poverty_index >= 2 and features.get('high_stress') == 1.0:
            original_label = int(pred_label)
            if original_label > 2:
                pred_label = 2
                max_prob_idx = np.argmax(pred_proba)
                if max_prob_idx != 1:  # Level 2 is index 1
                    pred_proba[1] += pred_proba[max_prob_idx]
                    pred_proba[max_prob_idx] = 0.0

        # 计算幸福指数
        weighted_sum = (
                pred_proba[0] * 1 + pred_proba[1] * 2 + pred_proba[2] * 3 +
                pred_proba[3] * 4 + pred_proba[4] * 5
        )
        happiness_score = (weighted_sum / 5) * 100

        return {
            'happiness_level': int(pred_label),
            'happiness_score': round(happiness_score, 1),
            'probability': {
                '1': f"{pred_proba[0] * 100:.1f}%",
                '2': f"{pred_proba[1] * 100:.1f}%",
                '3': f"{pred_proba[2] * 100:.1f}%",
                '4': f"{pred_proba[3] * 100:.1f}%",
                '5': f"{pred_proba[4] * 100:.1f}%",
            }
        }

    except Exception as e:
        print(f"Single resident prediction error: {e}")
        return {'error': str(e)}


def get_happiness_message(level):
    messages = {
        1: "非常不幸福 - 需要更多关注和改善",
        2: "不幸福 - 存在较大的生活压力",
        3: "一般 - 生活平淡，有待提升",
        4: "幸福 - 生活状态良好",
        5: "非常幸福 - 生活质量极高"
    }
    return messages.get(level, "未知结果")


# --- 路由 ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('用户名或密码错误')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            flash('两次密码不一致')
            return render_template('register.html')

        if User.query.filter_by(username=username).first():
            flash('用户名已存在')
            return render_template('register.html')

        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        login_user(new_user)
        return redirect(url_for('index'))

    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/history')
@login_required
def history():
    records = PredictionHistory.query.filter_by(user_id=current_user.id).order_by(
        PredictionHistory.created_at.desc()).all()
    # 转换JSON以便前端使用
    history_data = []
    for r in records:
        history_data.append({
            'created_at': r.created_at.strftime('%Y-%m-%d %H:%M'),
            'level': r.prediction_level,
            'input': r.get_input_dict(),
            'prob': r.get_prob_dict()
        })
    return render_template('history.html', history=history_data)


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        data = request.json
        print("Received data:", data)

        # 1. 准备特征字典 (使用默认值初始化)
        features = DEFAULTS.copy()

        # 2. 更新用户输入的特征
        for key, value in data.items():
            if value is not None and value != '':
                try:
                    features[key] = float(value)
                except ValueError:
                    features[key] = str(value)

        # 3. 计算衍生特征
        # --- 基本转化 ---
        family_income = float(features.get('family_income', 0))
        family_m = float(features.get('family_m', 3))  # 默认3口之家
        house = float(features.get('house', 0))
        car = float(features.get('car', 0))
        depression = float(features.get('depression', 0))
        equity = float(features.get('equity', 0))

        # 3.1 人均收入
        features['income_per_member'] = family_income / (family_m + 1)

        # 3.2 贫困指数 (0-3)
        # family_income < 30000 -> +1
        # house == 0 -> +1
        # car == 0 -> +1
        poverty_index = 0
        if family_income < 30000:
            poverty_index += 1
        if house == 0:
            poverty_index += 1
        if car == 0:
            poverty_index += 1
        features['poverty_index'] = float(poverty_index)

        # 3.3 高压力指数 (1 or 0)
        # depression >= 4 (经常/总是) AND equity <= 2 (不公平/完全不公平)
        if depression >= 4 and equity <= 2:
            features['high_stress'] = 1.0
        else:
            features['high_stress'] = 0.0

        # 3.4 房车总数
        features['house_car_count'] = house + car

        # 3.5 Log变换 (必须在用到原始值的逻辑之后)
        if 'income' in features:
            features['income_log'] = np.log1p(float(features['income']))
        if 'family_income' in features:
            features['family_income_log'] = np.log1p(family_income)

        # 4. 构建模型输入
        cat_indices = set(MODEL.get_cat_feature_indices())
        input_row = []
        for i, col in enumerate(FEATURE_COLUMNS):
            val = features.get(col, DEFAULTS.get(col))

            if i in cat_indices:
                val = str(val) if val is not None else "Unknown"
            else:
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    val = 0.0

            input_row.append(val)

        input_df = pd.DataFrame([input_row], columns=FEATURE_COLUMNS)

        # 5. 预测
        pred_label = MODEL.predict(input_df)[0][0]
        pred_proba = MODEL.predict_proba(input_df)[0]

        # --- 规则干预 (针对极端情况的硬性修正) ---
        # 如果 贫困指数高 (>=2) 且 高压力 (1.0), 强制限制幸福感不超过 2
        if poverty_index >= 2 and features.get('high_stress') == 1.0:
            original_label = int(pred_label)
            if original_label > 2:
                print(f"Triggering Rule Override: Level {original_label} -> 2")
                pred_label = 2

                # 调整概率分布 (将高概率转移给 Level 2)
                # 简单处理：将当前最大概率转移给 Level 2
                max_prob_idx = np.argmax(pred_proba)
                if max_prob_idx != 1:  # Index 1 is Level 2
                    pred_proba[1] += pred_proba[max_prob_idx]
                    pred_proba[max_prob_idx] = 0.0  # 简单的转移逻辑

        # 6. 保存历史记录
        prob_dict = {
            '1': f"{pred_proba[0] * 100:.1f}%",
            '2': f"{pred_proba[1] * 100:.1f}%",
            '3': f"{pred_proba[2] * 100:.1f}%",
            '4': f"{pred_proba[3] * 100:.1f}%",
            '5': f"{pred_proba[4] * 100:.1f}%",
        }

        history_record = PredictionHistory(
            user_id=current_user.id,
            input_data=json.dumps(data, ensure_ascii=False),
            prediction_level=int(pred_label),
            probability=json.dumps(prob_dict, ensure_ascii=False)
        )
        db.session.add(history_record)
        db.session.commit()

        # 7. 返回结果
        # 计算幸福指数 (Weighted Score: 20-100)
        # Score = (P1*1 + P2*2 + P3*3 + P4*4 + P5*5) / 5 * 100
        weighted_sum = (
                pred_proba[0] * 1 + pred_proba[1] * 2 + pred_proba[2] * 3 + pred_proba[3] * 4 + pred_proba[4] * 5)
        happiness_score = (weighted_sum / 5) * 100

        result = {
            'happiness_level': int(pred_label),
            'happiness_score': round(happiness_score, 1),
            'probability': prob_dict,
            'message': get_happiness_message(int(pred_label))
        }

        return jsonify(result)

    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/batch_predict', methods=['GET', 'POST'])
@login_required
def batch_predict():
    if request.method == 'GET':
        return render_template('batch_predict.html')

    if request.method == 'POST':
        try:
            # 检查是否有文件上传
            if 'file' not in request.files:
                flash('请选择文件')
                return redirect(url_for('batch_predict'))

            file = request.files['file']
            if file.filename == '':
                flash('请选择文件')
                return redirect(url_for('batch_predict'))

            # 检查文件类型
            if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
                flash('仅支持 CSV 或 Excel 文件')
                return redirect(url_for('batch_predict'))

            # 读取文件
            if file.filename.endswith('.csv'):
                # 尝试不同的编码
                try:
                    df = pd.read_csv(file, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(file, encoding='gbk')
                    except UnicodeDecodeError:
                        df = pd.read_csv(file, encoding='latin1')
            else:  # .xlsx or .xls
                df = pd.read_excel(file)

            print(f"文件加载成功: {len(df)} 行, {len(df.columns)} 列")
            print(f"列名: {df.columns.tolist()}")
            print(f"前5行数据:")
            print(df.head().to_string())

            # 标准化列名（去除空格、转换为小写）
            df.columns = df.columns.str.strip().str.lower()
            print(f"标准化后的列名: {df.columns.tolist()}")

            # 批量预测
            results = []
            success_count = 0
            error_count = 0

            for idx, row in df.iterrows():
                try:
                    # 预处理CSV行数据
                    data = preprocess_csv_row(row)

                    # 打印第一条数据的样例
                    if idx == 0:
                        print(f"第一条数据预处理后: {data}")

                    # 调用预测函数
                    prediction_result = predict_single_resident(data)

                    if 'happiness_level' in prediction_result:
                        results.append({
                            'row_id': idx + 1,
                            'age': int(data.get('age', 0)),
                            'gender': '男' if str(data.get('gender', '1')) == '1' else '女',
                            'income': int(data.get('income', 0)),
                            'family_income': int(data.get('family_income', 0)),
                            'happiness_level': prediction_result['happiness_level'],
                            'happiness_score': prediction_result['happiness_score'],
                            'message': get_happiness_message(prediction_result['happiness_level']),
                            'probabilities': prediction_result['probability']
                        })
                        success_count += 1
                    else:
                        results.append({
                            'row_id': idx + 1,
                            'error': prediction_result.get('error', '预测失败'),
                            'age': int(data.get('age', 0)) if 'age' in data else 0,
                            'gender': '男' if str(data.get('gender', '1')) == '1' else '女',
                            'income': int(data.get('income', 0)) if 'income' in data else 0,
                            'family_income': int(data.get('family_income', 0)) if 'family_income' in data else 0,
                        })
                        error_count += 1

                except Exception as e:
                    print(f"处理第 {idx + 1} 行时出错: {e}")
                    results.append({
                        'row_id': idx + 1,
                        'error': f'数据处理错误: {str(e)[:50]}',
                        'age': 0,
                        'gender': '未知',
                        'income': 0,
                        'family_income': 0,
                    })
                    error_count += 1

            print(f"批量预测完成: 成功 {success_count} 条, 失败 {error_count} 条")

            # 保存到 session 以便下载
            session['batch_results'] = {
                'data': results,
                'filename': f'batch_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            }

            flash(f'批量预测完成: 成功 {success_count} 条, 失败 {error_count} 条')
            return render_template('batch_predict.html', results=results, total=len(results))

        except Exception as e:
            print(f"批量预测错误: {e}")
            traceback.print_exc()
            flash(f'批量预测失败: {str(e)}')
            return redirect(url_for('batch_predict'))


@app.route('/download_batch_results')
@login_required
def download_batch_results():
    try:
        results_data = session.get('batch_results')
        if not results_data:
            flash('没有可下载的结果')
            return redirect(url_for('batch_predict'))

        results = results_data['data']
        filename = results_data['filename']

        # 创建 CSV
        import csv
        from io import StringIO

        output = StringIO()
        writer = csv.writer(output)

        # 写入表头
        writer.writerow(['序号', '年龄', '性别', '个人收入', '家庭收入',
                         '幸福等级', '幸福指数', '评估结果',
                         '1级概率', '2级概率', '3级概率', '4级概率', '5级概率'])

        # 写入数据
        for result in results:
            if 'error' in result:
                writer.writerow([
                    result['row_id'],
                    result['age'],
                    result['gender'],
                    result['income'],
                    result['family_income'],
                    '错误',
                    '0.0',
                    result['error'],
                    '', '', '', '', ''
                ])
            else:
                writer.writerow([
                    result['row_id'],
                    result['age'],
                    result['gender'],
                    result['income'],
                    result['family_income'],
                    result['happiness_level'],
                    result['happiness_score'],
                    result['message'],
                    result['probabilities']['1'],
                    result['probabilities']['2'],
                    result['probabilities']['3'],
                    result['probabilities']['4'],
                    result['probabilities']['5']
                ])

        # 返回 CSV 文件
        from flask import make_response
        response = make_response(output.getvalue())
        response.headers["Content-Disposition"] = f"attachment; filename={filename}"
        response.headers["Content-type"] = "text/csv; charset=utf-8"

        return response

    except Exception as e:
        flash(f'下载失败: {str(e)}')
        return redirect(url_for('batch_predict'))


@app.route('/cities')
@login_required
def cities():
    """查看我的城市"""
    user_cities = City.query.filter_by(user_id=current_user.id).order_by(City.updated_at.desc()).all()
    return render_template('cities.html', cities=user_cities)


@app.route('/city/new', methods=['GET', 'POST'])
@login_required
def new_city():
    """创建新城市"""
    if request.method == 'POST':
        try:
            name = request.form.get('name')
            description = request.form.get('description', '')

            # 创建默认城市
            new_city = City(
                user_id=current_user.id,
                name=name,
                description=description,
                avg_age=float(request.form.get('avg_age', 35)),
                avg_income=float(request.form.get('avg_income', 50000)),
                avg_family_income=float(request.form.get('avg_family_income', 120000)),
                avg_house=float(request.form.get('avg_house', 1.0)),
                avg_car=float(request.form.get('avg_car', 0.5)),
                avg_edu=float(request.form.get('avg_edu', 12)),
                avg_equity=float(request.form.get('avg_equity', 3.0)),
                avg_depression=float(request.form.get('avg_depression', 2.5)),
                avg_class=float(request.form.get('avg_class', 5.0))
            )

            db.session.add(new_city)
            db.session.commit()

            # 立即进行预测
            simulate_city_population(new_city)

            return redirect(url_for('city_detail', city_id=new_city.id))

        except Exception as e:
            flash(f'创建城市失败: {str(e)}')

    return render_template('new_city.html')


@app.route('/city/<int:city_id>')
@login_required
def city_detail(city_id):
    """查看城市详情"""
    city = City.query.get_or_404(city_id)

    # 验证权限
    if city.user_id != current_user.id:
        flash('无权访问此城市')
        return redirect(url_for('cities'))

    # 获取政策历史
    policies = CityPolicy.query.filter_by(city_id=city_id).order_by(CityPolicy.applied_at.desc()).all()

    return render_template('city_detail.html', city=city, policies=policies)


@app.route('/city/<int:city_id>/simulate', methods=['POST'])
@login_required
def simulate_city(city_id):
    """模拟城市居民幸福感"""
    city = City.query.get_or_404(city_id)

    # 验证权限
    if city.user_id != current_user.id:
        return jsonify({'error': '无权操作'}), 403

    try:
        # 模拟城市人口
        result = simulate_city_population(city)

        return jsonify({
            'success': True,
            'avg_happiness': city.avg_happiness,
            'distribution': city.get_distribution()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/city/<int:city_id>/apply_policy', methods=['POST'])
@login_required
def apply_policy(city_id):
    """应用政策到城市"""
    city = City.query.get_or_404(city_id)

    # 验证权限
    if city.user_id != current_user.id:
        return jsonify({'error': '无权操作'}), 403

    try:
        data = request.json

        # 创建政策记录
        policy = CityPolicy(
            city_id=city_id,
            name=data.get('name', '未命名政策'),
            description=data.get('description', ''),
            income_multiplier=float(data.get('income_multiplier', 1.0)),
            housing_bonus=float(data.get('housing_bonus', 0.0)),
            equity_bonus=float(data.get('equity_bonus', 0.0)),
            depression_reduction=float(data.get('depression_reduction', 0.0))
        )

        db.session.add(policy)

        # 更新城市参数
        city.policy_income_effect += (float(data.get('income_multiplier', 1.0)) - 1.0) * 0.1
        city.policy_housing_effect += float(data.get('housing_bonus', 0.0)) * 0.1
        city.policy_equity_effect += float(data.get('equity_bonus', 0.0)) * 0.1
        city.policy_mental_effect -= float(data.get('depression_reduction', 0.0)) * 0.1

        # 重新模拟
        simulate_city_population(city)

        db.session.commit()

        return jsonify({
            'success': True,
            'avg_happiness': city.avg_happiness,
            'distribution': city.get_distribution()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/city/<int:city_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_city(city_id):
    """编辑城市参数"""
    city = City.query.get_or_404(city_id)

    # 验证权限
    if city.user_id != current_user.id:
        flash('无权编辑此城市')
        return redirect(url_for('cities'))

    if request.method == 'POST':
        try:
            city.name = request.form.get('name', city.name)
            city.description = request.form.get('description', city.description)

            # 更新参数
            city.avg_age = float(request.form.get('avg_age', city.avg_age))
            city.avg_income = float(request.form.get('avg_income', city.avg_income))
            city.avg_family_income = float(request.form.get('avg_family_income', city.avg_family_income))
            city.avg_house = float(request.form.get('avg_house', city.avg_house))
            city.avg_car = float(request.form.get('avg_car', city.avg_car))
            city.avg_edu = float(request.form.get('avg_edu', city.avg_edu))
            city.avg_equity = float(request.form.get('avg_equity', city.avg_equity))
            city.avg_depression = float(request.form.get('avg_depression', city.avg_depression))
            city.avg_class = float(request.form.get('avg_class', city.avg_class))

            # 重新模拟
            simulate_city_population(city)

            db.session.commit()

            flash('城市参数已更新')
            return redirect(url_for('city_detail', city_id=city.id))

        except Exception as e:
            flash(f'更新失败: {str(e)}')

    return render_template('edit_city.html', city=city)


@app.route('/city/<int:city_id>/delete', methods=['POST'])
@login_required
def delete_city(city_id):
    """删除城市"""
    city = City.query.get_or_404(city_id)

    # 验证权限
    if city.user_id != current_user.id:
        return jsonify({'error': '无权删除'}), 403

    try:
        # 删除关联的政策
        CityPolicy.query.filter_by(city_id=city_id).delete()
        db.session.delete(city)
        db.session.commit()

        return jsonify({'success': True})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def simulate_city_population(city, sample_size=100):
    """
    模拟城市人口的幸福感分布
    """
    try:
        # 收集所有预测结果
        happiness_levels = []
        happiness_scores = []
        distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

        for i in range(sample_size):
            # 生成随机居民数据（基于城市平均值）
            resident_data = generate_resident_data(city)

            # 预测单个居民
            result = predict_single_resident(resident_data)

            if 'happiness_level' in result:
                level = result['happiness_level']
                score = result.get('happiness_score', 0)

                happiness_levels.append(level)
                happiness_scores.append(score)
                distribution[level] += 1

        # 计算统计数据
        if happiness_levels:
            city.avg_happiness = sum(happiness_scores) / len(happiness_scores)

            # 转换为百分比
            total = sum(distribution.values())
            for level in distribution:
                distribution[level] = round(distribution[level] / total * 100, 1)

            city.set_distribution(distribution)

        db.session.commit()

        return {
            'avg_happiness': city.avg_happiness,
            'distribution': distribution
        }

    except Exception as e:
        print(f"City simulation error: {e}")
        raise


def generate_resident_data(city):
    """
    基于城市参数生成随机居民数据
    """
    # 基础参数
    data = {
        'gender': random.choice([1, 2]),  # 1=男, 2=女
        'age': max(18, min(80, random.gauss(city.avg_age, 10))),
        'edu': max(6, min(14, random.gauss(city.avg_edu, 2))),
        'income': max(0, random.gauss(city.avg_income, city.avg_income * 0.5)),
        'family_income': max(0, random.gauss(city.avg_family_income, city.avg_family_income * 0.5)),
        'family_m': random.choice([1, 2, 3, 4, 5]),
        'house': max(0, int(random.gauss(city.avg_house, 1))),
        'car': max(0, int(random.gauss(city.avg_car, 0.5))),
        'equity': max(1, min(5, random.gauss(city.avg_equity + city.policy_equity_effect, 1))),
        'depression': max(1, min(5, random.gauss(city.avg_depression + city.policy_mental_effect, 1))),
        'class': max(1, min(10, random.gauss(city.avg_class, 2))),
        'status_peer': random.choice([1, 2, 3]),
        'province': random.randint(1, 31)
    }

    # 应用政策影响
    if city.policy_income_effect != 0:
        data['income'] *= (1 + city.policy_income_effect)
        data['family_income'] *= (1 + city.policy_income_effect)

    if city.policy_housing_effect != 0:
        data['house'] = max(0, data['house'] + city.policy_housing_effect)

    return data


# 初始化数据库 (仅用于开发环境，生产环境应手动迁移)
with app.app_context():
    # 尝试创建表
    try:
        db.create_all()
        print("Database tables initialized.")
    except Exception as e:
        print(f"Database init error (might be okay if tables exist): {e}")

# 启动时加载资源
load_resources()

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')