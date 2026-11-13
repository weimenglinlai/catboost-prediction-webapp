from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
import json
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """用户模型"""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # 关系：一个用户有多个历史记录和城市
    predictions = db.relationship('PredictionHistory', backref='user', lazy=True)
    cities = db.relationship('City', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class PredictionHistory(db.Model):
    """预测历史记录模型"""
    __tablename__ = 'prediction_history'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    # 输入数据 (JSON String)
    input_data = db.Column(db.Text, nullable=False)

    # 预测结果
    prediction_level = db.Column(db.Integer, nullable=False)

    # 概率分布 (JSON String)
    probability = db.Column(db.Text, nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def get_input_dict(self):
        try:
            return json.loads(self.input_data)
        except:
            return {}

    def get_prob_dict(self):
        try:
            return json.loads(self.probability)
        except:
            return {}


class City(db.Model):
    """城市模型"""
    __tablename__ = 'cities'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)

    # 城市参数
    avg_age = db.Column(db.Float, default=35.0)
    avg_income = db.Column(db.Float, default=50000.0)
    avg_family_income = db.Column(db.Float, default=120000.0)
    avg_house = db.Column(db.Float, default=1.0)
    avg_car = db.Column(db.Float, default=0.5)
    avg_edu = db.Column(db.Float, default=12.0)

    # 社会心理参数
    avg_equity = db.Column(db.Float, default=3.0)
    avg_depression = db.Column(db.Float, default=2.5)
    avg_class = db.Column(db.Float, default=5.0)

    # 政策影响
    policy_income_effect = db.Column(db.Float, default=0.0)
    policy_housing_effect = db.Column(db.Float, default=0.0)
    policy_equity_effect = db.Column(db.Float, default=0.0)
    policy_mental_effect = db.Column(db.Float, default=0.0)

    # 预测结果
    avg_happiness = db.Column(db.Float, default=0.0)
    happiness_distribution = db.Column(db.Text)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    policies = db.relationship('CityPolicy', backref='city', lazy=True)

    def get_distribution(self):
        try:
            return json.loads(self.happiness_distribution) if self.happiness_distribution else {}
        except:
            return {}

    def set_distribution(self, distribution):
        self.happiness_distribution = json.dumps(distribution)


class CityPolicy(db.Model):
    """城市政策模型"""
    __tablename__ = 'city_policies'

    id = db.Column(db.Integer, primary_key=True)
    city_id = db.Column(db.Integer, db.ForeignKey('cities.id'), nullable=False)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)

    # 政策效果参数
    income_multiplier = db.Column(db.Float, default=1.0)
    housing_bonus = db.Column(db.Float, default=0.0)
    equity_bonus = db.Column(db.Float, default=0.0)
    depression_reduction = db.Column(db.Float, default=0.0)

    applied_at = db.Column(db.DateTime, default=datetime.utcnow)