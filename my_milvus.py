from pymilvus import __version__,MilvusClient
import numpy as np
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import asyncio
# from langchain_community.embeddings import HuggingFaceEmbeddings
import sys
sys.path.append("/data/python_project")
from EfficientRetireve.llm_inference.insightful_embeddings.embeddings import EmbeddingModel
import os
import json
print(__version__)

client = MilvusClient(
    uri="http://192.168.96.55:19530",
    db_name="synapse"
)
print(client.list_collections())

# 嵌入模型配置
embedding_model_dict = {
    "Conan-embedding-v1": "/data/python_project/EfficientRetireve/Conan-embedding-v1"  # 修改为您的模型路径
}

# 全局模型实例
_embedding_model_instance = None

def load_embedding_model(model_name: str = "Conan-embedding-v1"):
    """使用insightful_embeddings框架加载嵌入模型"""
    model_path = embedding_model_dict[model_name]
    return EmbeddingModel(model_name, model_path)

def get_embedding_model(model_name: str = "Conan-embedding-v1"):
    """获取嵌入模型实例（单例模式）"""
    global _embedding_model_instance
    if _embedding_model_instance is None:
        print("加载 embedding_model 实例...")
        _embedding_model_instance = load_embedding_model(model_name)
    else:
        print("使用已存在的 embedding_model 实例...")
    return _embedding_model_instance

def generate_vector_from_text(text: str, model_name: str = "Conan-embedding-v1") -> List[float]:
    """使用嵌入模型将文本转换为向量"""
    embedding_model = get_embedding_model(model_name)
    # 使用insightful_embeddings的方法生成嵌入
    vectors = embedding_model.generate_embeddings([text])
    return vectors[0]  # 返回第一个结果

async def generate_vector_async(text: str, model_name: str = "Conan-embedding-v1") -> List[float]:
    """异步获取文本的嵌入向量"""
    # 对于异步版本，可以使用to_thread将同步操作转为异步
    embedding_model = get_embedding_model(model_name)
    vectors = await asyncio.to_thread(embedding_model.generate_embeddings, [text])
    return vectors[0]

def detect_cliff(
    similarity_scores: List[float],
    cliff_threshold: float = 0.05,
    min_results: int = 5
) -> int:
    """
    检测相似度分数中的断崖位置
    
    参数:
        similarity_scores: 排序后的相似度分数列表
        cliff_threshold: 断崖检测阈值
        min_results: 最少返回结果数
        
    返回:
        断崖位置的索引
    """
    if len(similarity_scores) <= min_results:
        return len(similarity_scores)
    
    # 计算相邻分数的差值
    score_diffs = [similarity_scores[i] - similarity_scores[i+1] 
                  for i in range(len(similarity_scores)-1)]
    
    # 找到差值超过阈值的位置
    for i in range(min_results-1, len(score_diffs)):
        if score_diffs[i] > cliff_threshold:
            return i + 1  # 返回断崖位置（包含断崖前的元素）
    
    # 如果没有找到明显的断崖，返回所有结果
    return len(similarity_scores)

def detect_by_curvature(
    similarity_scores: List[float], 
    window_size: int = 5, 
    min_results: int = 5, 
    percentile_threshold: float = 90
) -> int:
    """
    使用曲率分析找到相似度曲线的拐点
    
    参数:
        similarity_scores: 相似度分数列表
        window_size: 用于平滑曲率的窗口大小
        min_results: 最少返回结果数
        percentile_threshold: 曲率百分位阈值，超过此百分位的点被视为显著点
        
    返回:
        拐点位置的索引
    """
    if len(similarity_scores) <= min_results:
        return len(similarity_scores)
    
    # 将列表转换为numpy数组以便使用numpy函数
    scores = np.array(similarity_scores)
    
    # 计算差分（一阶导数的近似）
    first_derivative = np.gradient(scores)
    
    # 计算二阶导数的近似（曲率的一个指标）
    second_derivative = np.gradient(first_derivative)
    
    # 使用绝对值，因为我们关注变化率的大小而非方向
    curvature = np.abs(second_derivative)
    
    # 平滑曲率曲线
    if len(curvature) > window_size:
        smoothed_curvature = np.convolve(curvature, np.ones(window_size)/window_size, mode='valid')
        offset = window_size // 2
    else:
        smoothed_curvature = curvature
        offset = 0
    
    # 找到曲率值超过阈值的点
    threshold = np.percentile(smoothed_curvature, percentile_threshold)
    significant_points = np.where(smoothed_curvature > threshold)[0]
    
    # 排除开始的几个点，因为开始部分通常曲率较大
    start_idx = min(20, len(smoothed_curvature) // 10)
    filtered_points = [p for p in significant_points if p >= start_idx]
    
    if len(filtered_points) > 0:
        # 返回第一个显著点，加上偏移以补偿平滑窗口
        return filtered_points[0] + offset
    else:
        # 如果没有找到显著点，尝试返回曲率最大点
        max_curvature_idx = start_idx + np.argmax(smoothed_curvature[start_idx:])
        return min(len(similarity_scores), max_curvature_idx + offset)

def detect_by_slope_change(
    similarity_scores: List[float], 
    window_size: int = 10, 
    min_results: int = 5
) -> int:
    """
    检测斜率变化率最大的点
    
    参数:
        similarity_scores: 相似度分数列表
        window_size: 计算斜率的窗口大小
        min_results: 最少返回结果数
        
    返回:
        斜率变化最大点的索引
    """
    if len(similarity_scores) <= min_results:
        return len(similarity_scores)
    
    if len(similarity_scores) <= window_size:
        return len(similarity_scores)
    
    # 计算移动窗口内的斜率
    slopes = []
    for i in range(len(similarity_scores) - window_size):
        y1 = similarity_scores[i]
        y2 = similarity_scores[i + window_size]
        slope = (y2 - y1) / window_size
        slopes.append(slope)
    
    if not slopes:
        return min_results
    
    # 将列表转换为numpy数组
    slopes = np.array(slopes)
    
    # 计算斜率的变化率
    slope_changes = np.abs(np.gradient(slopes))
    
    # 排除开始的几个点，因为开始部分通常变化较大
    start_idx = min(20, len(slope_changes) // 10)
    if start_idx >= len(slope_changes):
        return min(len(similarity_scores), min_results + 10)
    
    # 找到斜率变化最大的点
    max_change_idx = start_idx + np.argmax(slope_changes[start_idx:])
    
    # 如果最大变化点太靠前，返回一个合理的默认值
    if max_change_idx < min_results:
        return min(len(similarity_scores), min_results + 10)
    
    return max_change_idx + window_size // 2  # 加上偏移以补偿计算窗口

def detect_by_percentage(
    similarity_scores: List[float], 
    percentage_drop: float = 5, 
    min_results: int = 5
) -> int:
    """
    基于与最高分数的百分比差值来截取结果
    
    参数:
        similarity_scores: 相似度分数列表
        percentage_drop: 与最高分数的百分比差值阈值
        min_results: 最少返回结果数
        
    返回:
        满足百分比差值条件的点的索引
    """
    if len(similarity_scores) <= min_results:
        return len(similarity_scores)
    
    max_score = similarity_scores[0]
    threshold = max_score * (1 - percentage_drop / 100)
    
    # 找到第一个低于阈值的点
    for i, score in enumerate(similarity_scores):
        if score < threshold and i >= min_results:
            return i
    
    return len(similarity_scores)

def plot_similarity_distribution(similarity_scores: List[float], cliff_position: int = None, detection_method: str = "断崖检测"):
    """绘制相似度分布图并标记断崖位置"""
    plt.figure(figsize=(10, 6))
    plt.plot(similarity_scores, marker='o')
    
    if cliff_position is not None:
        plt.axvline(x=cliff_position-0.5, color='r', linestyle='--', 
                    label=f'{detection_method}位置 (k={cliff_position})')
    
    plt.ylabel('相似度分数')
    plt.xlabel('结果排名')
    plt.title('相似度分数分布')
    plt.legend()
    plt.grid(True)
    plt.savefig('similarity_distribution.png')
    plt.close()

# 在cliff_detection_search中使用新算法
def cliff_detection_search(
    collection_name: str,
    query_vector: List[float],
    field_name: str = "vector",
    max_candidates: int = 10000,
    cliff_threshold: float = 0.05,
    min_results: int = 5,
    output_fields: List[str] = None,
    plot: bool = False,
    detection_method: str = "curvature"  # 新参数：检测方法
) -> Tuple[List[Dict[str, Any]], int]:
    """
    使用断崖检测方法进行相似度查询
    
    参数:
        collection_name: Milvus集合名称
        query_vector: 查询向量
        field_name: 向量字段名称
        max_candidates: 最大候选结果数
        cliff_threshold: 断崖检测阈值，相邻相似度差值超过此值视为断崖
        min_results: 最少返回结果数
        output_fields: 需要返回的字段列表
        plot: 是否绘制相似度分布图
        detection_method: 检测方法，可选值: "cliff"(原始断崖检测), "curvature"(曲率分析), 
                          "slope"(斜率变化), "percentage"(百分比阈值)
        
    返回:
        检测到断崖之前的结果列表，以及断崖位置k
    """
    # 执行相似度查询获取候选结果
    search_params = {
        "metric_type": "COSINE",  # 或 "L2", "IP" 等
        "params": {"nprobe": 10}
    }
    
    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=max_candidates,
        output_fields=output_fields,
        search_params=search_params,
        anns_field=field_name
    )
    # results = results.split(":")
    # print("===============results===============",type(results),results)
    
    # 打印结果的示例以了解其结构
    if len(results) > 0 and len(results[0]) > 0:
        print("搜索结果示例：")
        # 把所有的搜索结果都保存到文件中,保存的内容要解码成字符串
        with open(f"/data/python_project/EfficientRetireve/output/search_results_{detection_method}.json", "w") as f:
            # json.dump(results[0], f, ensure_ascii=False)
            json.dump([dict(hit) for hit in results[0]], f, ensure_ascii=False)
        print(results[0][0])  # 打印第一个结果的结构
    else:
        print("搜索结果为空")
        return [], 0
    
    # 获取相似度分数列表 - 根据Milvus返回的实际键名调整
    # 尝试不同可能的键名
    score_key = None
    possible_keys = ["distance"] # , "distance", "similarity", "metric"]
    
    if len(results[0]) > 0:
        sample_hit = results[0][0]
        for key in possible_keys:
            if key in sample_hit:
                score_key = key
                print(f"找到分数键名: {score_key}")
                break
    
    if score_key is None:
        print("警告: 无法在结果中找到相似度分数的键名")
        print("可用的键名:", list(sample_hit.keys()) if len(results[0]) > 0 else "无结果")
        return results[0][:min_results], min_results
    
    similarity_scores = [hit[score_key] for hit in results[0]]
    
    # 根据选择的方法执行断点检测
    method_display_name = ""
    if detection_method == "cliff":
        cliff_position = detect_cliff(
            similarity_scores, 
            cliff_threshold=cliff_threshold,
            min_results=min_results
        )
        method_display_name = "断崖检测"
    elif detection_method == "curvature":
        cliff_position = detect_by_curvature(
            similarity_scores,
            min_results=min_results
        )
        method_display_name = "曲率分析"
    elif detection_method == "slope":
        cliff_position = detect_by_slope_change(
            similarity_scores,
            min_results=min_results
        )
        method_display_name = "斜率变化"
    elif detection_method == "percentage":
        cliff_position = detect_by_percentage(
            similarity_scores,
            min_results=min_results
        )
        method_display_name = "百分比阈值"
    else:
        # 默认使用曲率分析
        cliff_position = detect_by_curvature(
            similarity_scores,
            min_results=min_results
        )
        method_display_name = "曲率分析"
    
    # 绘制相似度分布图并标记断崖位置
    if plot:
        plot_similarity_distribution(similarity_scores, cliff_position, method_display_name)
    
    # 返回断崖位置之前的结果
    return results[0][:cliff_position], cliff_position

# 更新demo函数使用新的检测方法
async def demo_async():
    # 您需要替换为实际的集合名称
    collection_name = "entities"
    
    # 示例查询文本 - 可替换为您的实际查询
    query_text = "甲硝锉有什么不良反应？"
    print(f"查询文本: {query_text}")
    
    # 使用嵌入模型将文本转换为向量
    query_vector = await generate_vector_async(query_text)
    print(f"生成向量维度: {len(query_vector)}")
    
    # 执行断崖检测查询
    try:
        # 尝试四种不同的断点检测方法
        detection_methods = ["cliff", "curvature", "slope", "percentage"]
        
        for method in detection_methods:
            print(f"\n===== 使用{method}方法 =====")
            results, k = cliff_detection_search(
                collection_name=collection_name,
                query_vector=query_vector,
                output_fields=["description"],  # 替换为您的字段
                plot=True,  # 绘制相似度分布图
                detection_method=method  # 指定检测方法
            )
            
            print(f"检测到断点位置在 k={k}")
            print(f"返回的结果数量: {len(results)}")
            
            # 仅打印前5个结果
            for i, hit in enumerate(results[:5]):
                print(f"{i+1}. ID: {hit.get('id')}")
                print(f"   分数: {hit.get('score', hit.get('distance', hit.get('similarity', '未知')))}")
                if "description" in hit:
                    print(f"   描述: {hit.get('description')[:100]}...")
            
            if len(results) > 5:
                print(f"... 还有 {len(results)-5} 个结果 ...")
                
            # 重命名图片，防止被覆盖
            if os.path.exists('similarity_distribution.png'):
                os.rename('similarity_distribution.png', f'similarity_distribution_{method}.png')
    
    except Exception as e:
        print(f"执行查询时发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 运行异步版本的示例
    asyncio.run(demo_async())