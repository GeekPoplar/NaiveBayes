#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：NaiveBayes
@File ：naive_bayes.py
@Author ：poplar
@Date ：2021/4/1 15:40
"""
import pandas as pd
from pandas.core.frame import DataFrame
import json


class NaiveBayes:
    """
        朴素贝叶斯，可能是最原始的实现，对于类别较少，且每个属性的可选值较少的数据集（如uci的mushroom数据集，即agaricus-lepiota.csv）效果还行
    """

    def fit(self, x_train, y_train):
        """
            输入训练集，得到预测模型
        """
        print("-------------开始训练模型-------------")
        self.classes = self._get_classes(y_train)
        self.features = self._get_features(x_train)
        self.class_probability = self._get_class_probability(y_train=y_train)
        self.feature_values_probability_for_every_class = self._get_feature_values_probability_for_every_class(x_train,
                                                                                                               y_train)
        print("-------------完成模型训练-------------")

    def evaluation(self, x_test: DataFrame, y_test: DataFrame):
        """
            输入测试集，返回评估结果（分类准确率）
        """
        print("-------------开始评价模型-------------")
        total = len(y_test)
        right_total = 0
        x_test_array = x_test.to_numpy()
        y_test_array = y_test.to_numpy()
        for i, row in enumerate(x_test_array):
            x = row
            y = y_test_array[i][0]
            if self._check(x, y):
                print("正确")
                right_total += 1
            else:
                print("错误")
        print("-------------完成评价模型-------------")
        return right_total / total

    def forecast(self, x_validation, y):
        """
            输入数据，进行验证
        """
        result = self._check(x_validation, y)
        print(result)
        return result

    def save(self, path):
        """
            将模型持久化到文件
        """
        print("-------------开始保存模型-------------")
        model = dict()
        model["classes"] = self.classes
        model["features"] = self.features
        model["class_probability"] = self.class_probability
        model["feature_values_probability_for_every_class"] = self.feature_values_probability_for_every_class
        with open(path, "w") as f:
            f.write(json.dumps(model, ensure_ascii=False, indent=4, separators=(',', ':')))
        print("-------------完成模型保存-------------")

    def load_model(self, path):
        """
            从文件中加载模型
        """
        print("-------------开始加载模型-------------")
        with open(path, "r", encoding="utf-8") as f:
            model = json.load(f)
        self.classes = model["classes"]
        self.features = model["features"]
        self.class_probability = model["class_probability"]
        self.feature_values_probability_for_every_class = model["feature_values_probability_for_every_class"]
        print("-------------完成模型加载-------------")

    def _check(self, x, y):
        # x = [str(i) for i in x]
        probability_list = list()
        for c in self.classes:
            probability = 1
            for i, feature in enumerate(self.features):
                try:
                    feature_values_probability = self.feature_values_probability_for_every_class[c][feature][x[i]]
                    probability *= feature_values_probability * self.class_probability[c]
                except KeyError:
                    print("key_error")
                    probability = 0
            probability_list.append(probability)
        print("预测结果：", self.classes[probability_list.index(max(probability_list))])
        return y == self.classes[probability_list.index(max(probability_list))]

    def _get_classes(self, y_train: DataFrame):
        """
            返回所有类别组成的列表
        """
        column = y_train.columns[0]
        return list(dict(y_train[column].value_counts()).keys())

    def _get_features(self, x_train: DataFrame):
        """
            返回所有属性组成的列表
        """
        return list(x_train.columns)

    def _get_class_probability(self, y_train):
        """
            计算各个类别的先验概率
        """
        count = len(y_train)
        class_probability_dict = dict(y_train[y_train.columns[0]].value_counts())
        for k in class_probability_dict.keys():
            class_probability_dict[k] = class_probability_dict[k] / count
        return class_probability_dict

    def _get_feature_values_probability_for_every_class(self, x_train, y_train):
        """
            计算各个特征的特征值在各个已知类别下的出现概率
        """
        # 将训练数据合并起来
        train = pd.concat([y_train, x_train], axis=1)
        classes = self.classes
        features = self.features
        d = dict()
        for c in classes:
            d[c] = dict()
            class_data = train[train[y_train.columns[0]] == c]
            class_data_count = len(class_data)
            for feature in features:
                tmp_d = dict(class_data[feature].value_counts())
                for k in tmp_d.keys():
                    tmp_d[k] = tmp_d[k] / class_data_count
                d[c][feature] = tmp_d
        return d
