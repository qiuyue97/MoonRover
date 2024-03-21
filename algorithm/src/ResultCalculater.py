import json
from collections import defaultdict
import pandas as pd


def load_results_from_json(file_path):
    # 从JSON文件中加载数据
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def calculate_performance_metrics(results):
    # 用于存储算法性能指标的字典
    metrics = defaultdict(lambda: defaultdict(list))

    # 遍历结果，按算法名称分组
    for result in results:
        alg_name = result['算法名称']
        metrics[alg_name]['路径规划长度'].append(result['路径规划长度'])
        metrics[alg_name]['计算时间'].append(result['计算时间'])
        metrics[alg_name]['扩展节点数'].append(result['扩展节点数'])
        metrics[alg_name]['路径最大高程差'].append(result['路径最大高程差'])
        metrics[alg_name]['路径累计高程差'].append(result['路径累计高程差'])

    # 计算平均性能指标和最大路径累计高程差
    df = pd.DataFrame()

    for alg, values in metrics.items():
        df.loc[alg, '平均路径规划长度'] = round(sum(values['路径规划长度']) / len(values['路径规划长度']), 2)
        df.loc[alg, '平均计算时间'] = round(sum(values['计算时间']) / len(values['计算时间']), 2)
        df.loc[alg, '平均扩展节点数'] = round(sum(values['扩展节点数']) / len(values['扩展节点数']), 2)
        df.loc[alg, '平均路径最大高程差'] = round(sum(values['路径最大高程差']) / len(values['路径最大高程差']), 2)
        df.loc[alg, '路径最大高程差'] = round(max(values['路径最大高程差']), 2)
        df.loc[alg, '平均路径累计高程差'] = round(sum(values['路径累计高程差']) / len(values['路径累计高程差']), 2)
        df.loc[alg, '最大路径累计高程差'] = round(max(values['路径累计高程差']), 2)
    return df


# 例如，假设JSON文件位于 'results.json'
json_path = 'results.json'
results = load_results_from_json(json_path)
performance_metrics_df = calculate_performance_metrics(results)
xlsx_path = 'performance_metrics.csv'

# 打印结果
performance_metrics_df.to_csv(xlsx_path, index=True, encoding='utf-8-sig')
