"""
CatBoost 幸福感预测模型训练脚本
使用清洗后的数据训练多分类模型
"""

import pandas as pd
import numpy as np
import json
import os
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')


class HappinessModelTrainer:
    """幸福感预测模型训练器"""
    
    def __init__(self, data_path, model_dir):
        """
        初始化训练器
        
        Args:
            data_path: 清洗后数据文件路径
            model_dir: 模型保存目录
        """
        self.data_path = data_path
        self.model_dir = model_dir
        self.df = None
        self.model = None
        self.feature_columns = []
        self.cat_features = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # 确保输出目录存在
        os.makedirs(model_dir, exist_ok=True)
        
    def load_data(self):
        """加载数据"""
        print("=" * 60)
        print("步骤1: 加载数据")
        print("=" * 60)
        
        self.df = pd.read_csv(self.data_path)
        print(f"原始数据: {self.df.shape[0]} 行 x {self.df.shape[1]} 列")
        
        # 过滤无效样本 (happiness = -8)
        valid_mask = self.df['happiness'] > 0
        invalid_count = (~valid_mask).sum()
        self.df = self.df[valid_mask].copy()
        print(f"过滤无效样本: 删除 {invalid_count} 条 (happiness=-8)")
        print(f"有效数据: {self.df.shape[0]} 行")
        
        return self
    
    def prepare_features(self):
        """准备特征"""
        print("\n" + "=" * 60)
        print("步骤2: 准备特征")
        print("=" * 60)
        
        # 排除不需要的列
        exclude_cols = ['id', 'happiness', 'survey_time', 'birth_year', 'survey_year']
        
        # 获取特征列
        self.feature_columns = [col for col in self.df.columns if col not in exclude_cols]
        
        # 识别类别特征
        # CatBoost可以直接处理类别特征,无需编码
        self.cat_features = []
        cat_feature_names = []
        
        for i, col in enumerate(self.feature_columns):
            # 检查是否为类别特征
            if self.df[col].dtype == 'object':
                self.cat_features.append(i)
                cat_feature_names.append(col)
            elif col in ['province', 'city', 'county', 'gender', 'nationality', 
                        'religion', 'edu', 'political', 'marital_status', 
                        'work_status', 'survey_type']:
                self.cat_features.append(i)
                cat_feature_names.append(col)
        
        print(f"特征数量: {len(self.feature_columns)}")
        print(f"类别特征: {len(self.cat_features)} 个")
        if cat_feature_names:
            print(f"  包括: {cat_feature_names[:10]}...")
        
        return self
    
    def split_data(self, test_size=0.2, random_state=42):
        """划分训练集和测试集"""
        print("\n" + "=" * 60)
        print("步骤3: 划分数据集")
        print("=" * 60)
        
        X = self.df[self.feature_columns].copy()
        y = self.df['happiness']
        
        # 获取类别特征列名
        cat_feature_names = [self.feature_columns[i] for i in self.cat_features]
        
        # 填充缺失值并转换类型
        for col in X.columns:
            if col in cat_feature_names or X[col].dtype == 'object':
                # 类别特征：转换为字符串并填充
                X[col] = X[col].fillna('Unknown').astype(str)
            else:
                # 数值特征：用中位数填充
                X[col] = X[col].fillna(X[col].median())
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"训练集: {len(self.X_train)} 样本")
        print(f"测试集: {len(self.X_test)} 样本")
        print(f"\n目标变量分布 (训练集):")
        train_dist = self.y_train.value_counts().sort_index()
        for val, count in train_dist.items():
            print(f"  happiness={val}: {count} ({count/len(self.y_train)*100:.1f}%)")
        
        return self
    
    def train_model(self):
        """训练CatBoost模型"""
        print("\n" + "=" * 60)
        print("步骤4: 训练模型")
        print("=" * 60)
        
        # 配置CatBoost模型
        self.model = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.03,
            depth=8,  # 加深树深度
            l2_leaf_reg=3, # 正则化
            loss_function='MultiClass',
            auto_class_weights='Balanced',  # 启用平衡权重，提升少数类识别率
            cat_features=self.cat_features,
            random_seed=42,
            verbose=100,
            early_stopping_rounds=100,
            use_best_model=True
        )
        
        print("模型配置 (优化版+平衡权重):")
        print("  - iterations: 2000")
        print("  - learning_rate: 0.03")
        print("  - depth: 8")
        print("  - l2_leaf_reg: 3")
        print("  - loss_function: MultiClass")
        print("  - auto_class_weights: Balanced (已启用)")
        print("  - early_stopping_rounds: 100")
        
        print("\n开始训练...")
        
        # 创建训练和验证数据池
        train_pool = Pool(
            self.X_train, 
            self.y_train, 
            cat_features=self.cat_features
        )
        eval_pool = Pool(
            self.X_test, 
            self.y_test, 
            cat_features=self.cat_features
        )
        
        # 训练模型
        self.model.fit(
            train_pool,
            eval_set=eval_pool,
            plot=False
        )
        
        print("\n模型训练完成!")
        print(f"最佳迭代次数: {self.model.get_best_iteration()}")
        
        return self
    
    def evaluate_model(self):
        """评估模型"""
        print("\n" + "=" * 60)
        print("步骤5: 评估模型")
        print("=" * 60)
        
        # 预测
        y_pred = self.model.predict(self.X_test)
        
        # 计算指标
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        
        print(f"\n准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\n分类报告:\n{report}")
        print(f"\n混淆矩阵:\n{conf_matrix}")
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.get_feature_importance()
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 重要特征:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.2f}")
        
        # 保存评估报告
        report_path = os.path.join(self.model_dir, 'evaluation_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("CatBoost 幸福感预测模型评估报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
            f.write("分类报告:\n")
            f.write(report + "\n")
            f.write("混淆矩阵:\n")
            f.write(str(conf_matrix) + "\n\n")
            f.write("Top 20 重要特征:\n")
            for i, row in feature_importance.head(20).iterrows():
                f.write(f"  {row['feature']}: {row['importance']:.2f}\n")
        
        print(f"\n评估报告已保存: {report_path}")
        
        return self, accuracy, report
    
    def save_model(self):
        """保存模型"""
        print("\n" + "=" * 60)
        print("步骤6: 保存模型")
        print("=" * 60)
        
        # 保存CatBoost模型
        model_path = os.path.join(self.model_dir, 'catboost_model.cbm')
        self.model.save_model(model_path)
        print(f"模型已保存: {model_path}")
        
        # 保存特征列表
        features_path = os.path.join(self.model_dir, 'feature_columns.json')
        with open(features_path, 'w', encoding='utf-8') as f:
            json.dump({
                'feature_columns': self.feature_columns,
                'cat_features_indices': self.cat_features,
                'cat_features_names': [self.feature_columns[i] for i in self.cat_features]
            }, f, ensure_ascii=False, indent=2)
        print(f"特征列表已保存: {features_path}")
        
        return self
    
    def run(self):
        """执行完整训练流程"""
        self.load_data() \
            .prepare_features() \
            .split_data() \
            .train_model()
        
        _, accuracy, _ = self.evaluate_model()
        self.save_model()
        
        print("\n" + "=" * 60)
        print("训练完成!")
        print("=" * 60)
        print(f"模型准确率: {accuracy*100:.2f}%")
        print(f"模型文件: {os.path.join(self.model_dir, 'catboost_model.cbm')}")
        
        return self


def main():
    """主函数"""
    # 配置路径
    DATA_PATH = '../happiness_cleaned.csv'
    MODEL_DIR = 'mode'
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, DATA_PATH)
    model_dir = os.path.join(script_dir, MODEL_DIR)
    
    # 训练模型
    trainer = HappinessModelTrainer(data_path, model_dir)
    trainer.run()


if __name__ == '__main__':
    main()
