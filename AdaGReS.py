from my_milvus import generate_vector_from_text
from pymilvus import __version__,MilvusClient
import numpy as np
import random
from typing import List, Dict, Tuple, Any
import requests

print(__version__)

client = MilvusClient(
    uri="http://192.168.96.55:19530",
    db_name="synapse"
)
print(client.list_collections())


def get_top_n_candidates(
    query_vector: List[float],
    collection_name: str = "entities",
    top_n: int = 1000,
    field_name: str = "vector",
    output_fields: List[str] = None,
    include_vectors: bool = True
) -> List[Dict[str, Any]]:
    """
    通过query_vector从向量数据库中获取top n个最相关的候选chunks
    
    参数:
        query_vector: 查询向量
        collection_name: Milvus集合名称
        top_n: 返回的top候选数量
        field_name: 向量字段名称
        output_fields: 需要返回的额外字段列表
        include_vectors: 是否包含向量数据（用于ARCS算法计算相似度）
        
    返回:
        候选chunks列表，格式适合作为select_chunks_pipeline的candidate_chunks参数
    """
    # 设置搜索参数
    search_params = {
        "metric_type": "COSINE",  # 或 "L2", "IP" 等
        "params": {"nprobe": 10}
    }
    
    # 默认输出字段
    if output_fields is None:
        output_fields = ["description"]  # 根据实际数据字段调整
    
    # 如果需要包含向量，添加向量字段到输出字段中
    final_output_fields = output_fields.copy()
    if include_vectors and field_name not in final_output_fields:
        final_output_fields.append(field_name)
    
    try:
        # 执行向量搜索
        results = client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=top_n,
            output_fields=final_output_fields,
            search_params=search_params,
            anns_field=field_name,
            filter="knowledge_base_id in [37441783377985536]"#, 37441720429871104, 37441675718590464]"
        )
        # print("===============results===============",type(results),results)
        
        if not results or not results[0]:
            print("搜索结果为空")
            return []
        
        # 格式化结果为candidate_chunks格式
        candidate_chunks = []
        for hit in results[0]:
            chunk = {}
            
            # 从hit中获取基本信息（id, distance等）
            chunk['id'] = hit.get('id')
            chunk['distance'] = hit.get('distance')
            
            # 从entity中获取实际的字段数据
            if 'entity' in hit:
                entity_data = hit['entity']
                
                # 复制entity中的所有字段
                for key, value in entity_data.items():
                    chunk[key] = value
                
                # 确保向量字段存在且命名正确（ARCS算法期望的字段名）
                if include_vectors:
                    if field_name in entity_data:
                        # 如果向量字段名不是"vector"，也复制一份到"vector"字段
                        if field_name != "vector":
                            chunk["vector"] = entity_data[field_name]
                    else:
                        print(f"警告: entity中缺少向量字段 {field_name}")
                        print(f"entity中可用字段: {list(entity_data.keys())}")
                        continue
            else:
                print("警告: hit中缺少entity字段")
                print(f"hit中可用字段: {list(hit.keys())}")
                continue
            
            candidate_chunks.append(chunk)
        
        print(f"成功获取 {len(candidate_chunks)} 个候选chunks")
        
        # 避免IndexError，只在有数据时打印示例
        if candidate_chunks:
            print(f"格式示例：{candidate_chunks[0]}")
        else:
            print("警告: 没有成功获取到任何候选chunks")
            
        return candidate_chunks
        
    except Exception as e:
        print(f"获取候选chunks时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return []


# 1. 随机抽样n个chunks，计算平均长度，基于Tmax得到预算k
def estimate_k_by_tokens_api(Tmax: int = 1500,
                             sample_n: int = 1000,
                             candidate_chunks: List[Dict] = None, 
                             content_field: str = "description") -> int:
    """
    计算平均长度，得到预算k
    
    参数:
        Tmax: 最大token数量
        sample_n: 抽样数量
        candidate_chunks: 候选chunks列表，如果提供则直接从中抽样，否则通过API获取
        content_field: 用于计算长度的字段名（默认为"description"）
        
    返回:
        预算k值
    """
    chunks = []
    if candidate_chunks is not None:
        # 如果提供了candidate_chunks，直接从中抽样
        print(f"从提供的 {len(candidate_chunks)} 个候选chunks中抽样 {sample_n} 个")
        
        if len(candidate_chunks) <= sample_n:
            # 如果候选数量不足，使用全部
            chunks = candidate_chunks
            print(f"候选数量不足，使用全部 {len(chunks)} 个chunks")
        else:
            # 随机抽样
            chunks = random.sample(candidate_chunks, sample_n)
            print(f"随机抽样了 {len(chunks)} 个chunks")
    else:
        # 原有的API获取逻辑
        print(f"通过API获取 {sample_n} 个随机chunks")
        entity_ids = []  # 用于存储已获取的entity id
        while len(chunks) < sample_n:
            remaining = sample_n - len(chunks)
            base_url = f"https://knowledge.public.yzint.cn/knowledge/v1/graph/entity/random?count={remaining}"
            # 如果已有实体ID，则拼接到URL中
            if entity_ids:
                ids_param = ",".join(entity_ids)
                url = f"{base_url}&entityIds={ids_param}"
            else:
                url = base_url
            resp = requests.get(url)
            resp.raise_for_status()
            
            new_chunks = resp.json()["data"]
            if not new_chunks:  # 如果API没有返回任何数据，避免无限循环
                break
            # 添加新获取的chunks并记录它们的ID
            chunks.extend(new_chunks)
            for chunk in new_chunks:
                entity_id = str(chunk["id"])
                entity_ids.append(entity_id)
                # print(f"获取到新的实体{entity_id}")
        

    

    if not chunks:
        if candidate_chunks is not None:
            raise ValueError("提供的candidate_chunks为空")
        else:
            raise ValueError("API未返回任何chunks数据")
        
    
    # 获取的数据格式如下：
    #     {
    #   "code": 0,
    #   "success": true,
    #   "msg": "Operation successful",
    #   "timestamp": "1748008264272",
    #   "data": [
    #     {
    #       "name_vector": null,
    #       "vectors": null,
    #       "vector_id": "456318473179889681",
    #       "id": "60727173372084224",
    #       "knowledge_base_id": "201",
    #       "docId": "1892530413013061634",
    #       "name": "生育力对男性的影响",
    #       "type": "fertility",
    #       "description": "生育力对男性的影响：根据药物作用机制，可能会损害具有生殖潜力的男性的生育能力。",
    #       "community_id": "0"
    #     },
    #     {
    #       "name_vector": null,
    #       "vectors": null,
    #       "vector_id": "456318473179889658",
    #       "id": "60727173371166720",
    #       "knowledge_base_id": "201",
    #       "docId": "1892530413013061634",
    #       "name": "注射用醋酸曲普瑞林微球",
    #       "type": "drug",
    #       "description": "注射用醋酸曲普瑞林微球是一种生殖泌尿系统和性激素类药物，属于性激素和生殖系统调节药。该药物的活性成分是醋酸曲普瑞林。LZ-102804研究是一项在子宫内膜异位症患者中开展的多中心、随机、双盲、阳性药(达菲林®)对照的Ⅲ期临床研究，评估本品的有效性和安全性。研究中，392例患者入组，试验组和对照组各有196例患者分别接受本品3.75mg和达菲林®3.75mg，肌肉注射，每4周注射一次。试验组接受本品治疗时间超过1个月和超过3个月的患者比例分别为100%和76.5%。试验组和对照组所有级别的不良反应发生率分别为59.2%和57.7%。发生率≥2%的不良反应包括潮热、阴道出血、外阴阴道干燥、性交困难、失眠、焦虑、性欲降低、多汗、关节痛和头痛。上市后信息显示，同类GnRH激动剂使用期间发现了以下不良反应：过敏性疾病（类过敏或哮喘过程、皮疹、荨麻疹和光敏反应）、心血管系统（低血压、心肌梗塞、肺栓塞）、中枢/周围神经系统（惊厥、周围神经病变、脊柱骨折/瘫痪）、内分泌系统（垂体卒中、糖尿病）、肝胆疾病（药物性肝损伤）、血液学（白细胞）、精神科（情绪波动，包括抑郁、自杀意念和企图自杀）、呼吸、胸部和纵隔疾病（间质性肺疾病）、肌肉骨骼系统（骨密度降低、腱鞘炎样症状、纤维肌痛）、皮肤和皮下（注射部位反应）、泌尿生殖系统（前列腺疼痛）。禁忌症包括对促性腺激素释放激素、促性腺激素释放激素类似物或本品任何一种成分过敏者禁用。孕妇及哺乳期妇女禁用，因为理论上同时应用GnRH激动剂具有流产或胎儿致畸的风险。治疗期间，有生育能力的妇女应采取非激素方法避孕。哺乳期间不应使用曲普瑞林。生育力方面，根据药物作用机制，可能会损害具有生殖潜力的男性生育能力。儿童用药方面，暂无儿童使用本品的疗效和安全性数据。老年用药方面，可参见【注意事项】和【禁忌】。药物过量时，应给予对症治疗。",
    #       "community_id": "0"
    #     }
    #   ]
    # }
    # 把description字段的内容取出来计算出平均长度
    # 计算平均长度
    # 尝试不同的字段名来获取内容
    valid_chunks = []
    for chunk in chunks:
        content = None
        if content_field in chunk:
            content = chunk[content_field]
        elif "description" in chunk:
            content = chunk["description"]
        elif "text" in chunk:
            content = chunk["text"]
        elif "content" in chunk:
            content = chunk["content"]
        
        if content and isinstance(content, str):
            valid_chunks.append(content)
    
    if not valid_chunks:
        print(f"警告: 在chunks中未找到有效的内容字段")
        print(f"尝试的字段: {content_field}")
        if chunks:
            print(f"可用字段示例: {list(chunks[0].keys())}")
        return 1  # 返回默认值
    
    avg_len = np.mean([len(chunk["description"]) for chunk in chunks])
    print(f"从 {len(valid_chunks)} 个有效chunks计算出平均长度: {avg_len}")
    # 打印一个样例
    # print(chunks[0])
   
    k = int(Tmax // avg_len) if avg_len > 0 else 1
    return max(1, k)

# 2. 计算候选集合中元素对query的平均相似度
def average_query_chunk_similarity(query_vec: List[float], chunks: List[Dict], vec_field: str = "vector") -> float:
    """
    计算chunks中所有元素与query的平均相似度
    """
    sims = []
    q = np.array(query_vec)
    for chunk in chunks:
        c = np.array(chunk[vec_field])
        sim = np.dot(q, c) / (np.linalg.norm(q) * np.linalg.norm(c) + 1e-8)
        sims.append(sim)
    return float(np.mean(sims)) if sims else 0.0

# 3. 计算候选集合两两之间的平均相似度
def average_chunk_pair_similarity(chunks: List[Dict], vec_field: str = "vector") -> float:
    """
    计算chunks集合内两两之间的平均相似度
    """
    n = len(chunks)
    if n < 2:
        return 0.0
    sims = []
    vectors = [np.array(chunk[vec_field]) for chunk in chunks]
    for i in range(n):
        for j in range(i+1, n):
            sim = np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]) + 1e-8)
            sims.append(sim)
    return float(np.mean(sims)) if sims else 0.0

# 4. 用上面3个方法计算动态β（假设α=1）
def dynamic_beta(
    query_vec: List[float],
    candidate_chunks: List[Dict],
    Tmax: int = 1600,
    sample_n: int = 500,
    vec_field: str = "vector",
) -> float:
    """
    动态自适应beta的计算（α=1）
    """
    k = estimate_k_by_tokens_api(Tmax=Tmax, sample_n=sample_n,candidate_chunks=candidate_chunks)
    mean_q_sim = average_query_chunk_similarity(query_vec, candidate_chunks, vec_field)
    mean_pair_sim = average_chunk_pair_similarity(candidate_chunks, vec_field)
    if mean_pair_sim == 0 or k <= 1:
        return 0.01  # 避免除0，返回极小值
    beta = mean_q_sim / (((k-1)/2) * mean_pair_sim)
    return float(beta),int(k)

# 评估函数F(q, C)，用于排序
def eval_fn(query_vec: List[float], chunk_vecs: List[np.ndarray], alpha: float, beta: float) -> float:
    """
    F(q, C) = α * sum(query·c) - β * sum_{i<j}(c_i·c_j)
    """
    q = np.array(query_vec)
    n = len(chunk_vecs)
    # 第一项：相关性
    rel = sum([np.dot(q, c) / (np.linalg.norm(q) * np.linalg.norm(c) + 1e-8) for c in chunk_vecs])
    # 第二项：冗余惩罚
    redun = 0.0
    for i in range(n):
        for j in range(i+1, n):
            redun += np.dot(chunk_vecs[i], chunk_vecs[j]) / (np.linalg.norm(chunk_vecs[i]) * np.linalg.norm(chunk_vecs[j]) + 1e-8)
    return alpha * rel - beta * redun

# 给所有chunks打分并排序
def score_and_sort_chunks(
    query_vec: List[float],
    candidate_chunks: List[Dict],
    alpha: float,
    beta: float,
    vec_field: str = "vector",
    top_k: int = None
) -> List[Dict]:
    """
    对所有chunks按评估函数打分排序，返回排序后的chunks
    """
    # 单chunk评分（贪心/启发式先打分，后面可扩展为多步贪心选集）
    scores = []
    for chunk in candidate_chunks:
        chunk_vec = np.array(chunk[vec_field])
        rel = np.dot(query_vec, chunk_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec) + 1e-8)
        # 冗余项单独加权（这里只作为打分排序的第一步，后续可用贪心或beam search扩展）
        score = alpha * rel - beta * 0  # 暂不考虑冗余（单chunk评分），用于初步筛选
        scores.append((score, chunk))
    # 排序
    sorted_chunks = [x[1] for x in sorted(scores, key=lambda x: x[0], reverse=True)]
    # 截断到top_k
    if top_k is not None:
        return sorted_chunks[:top_k]
    return sorted_chunks

# 多步贪心集选实现
def greedy_select_chunks(
    query_vec: List[float],
    candidate_chunks: List[Dict],
    alpha: float,
    beta: float,
    vec_field: str = "vector",
    k: int = 8
) -> List[Dict]:
    """
    贪心选k个chunks，使得评估函数最大化
    """
    selected = []
    remain = candidate_chunks.copy()
    while len(selected) < k and remain:
        best_score, best_chunk, best_idx = None, None, None
        for idx, chunk in enumerate(remain):
            trial_chunks = selected + [chunk]
            chunk_vecs = [np.array(c[vec_field]) for c in trial_chunks]
            score = eval_fn(query_vec, chunk_vecs, alpha, beta)
            if (best_score is None) or (score > best_score):
                best_score = score
                best_chunk = chunk
                best_idx = idx
        selected.append(best_chunk)
        remain.pop(best_idx)
    return selected

# ===================== 用法演示 ======================
# 假设有如下chunks数据和query
# candidate_chunks = [{'text': '...', 'vector': [...], 'token_count': ...}, ...]
# query_text = "你的问题"
# query_vec = generate_vector_from_text(query_text)
# alpha = 1.0


def select_chunks_pipeline(
    query_text: str, 
    k: int|None = None,
    candidate_chunks: List[Dict] = None,
    collection_name: str = "entities",
    top_n_candidates: int = 2000,
    Tmax: int = 1500, 
    beta: float = None,  # 新增参数：如果提供则使用固定beta值
    sample_n: int = 500,
    selection_method: str = "greedy"  # "greedy" 或 "simple"
):
    """
    完整的chunks选择流水线
    
    参数:
        query_text: 查询文本
        k: 选择的chunks数量，如果为None则动态计算
        candidate_chunks: 候选chunks列表，如果为None则从向量数据库获取
        collection_name: Milvus集合名称（当candidate_chunks为None时使用）
        top_n_candidates: 从向量数据库获取的候选数量（当candidate_chunks为None时使用）
        Tmax: 最大token数量（当beta为None时用于动态计算beta）
        beta: 固定的beta值，如果提供则不使用动态计算
        sample_n: 用于估算的样本数量
        selection_method: 选择方法，"greedy" 或 "simple"
        
    返回:
        (selected_chunks, beta, k)
    """
    if selection_method == "simple":
        assert k is not None
    
    # 如果没有提供候选chunks，从向量数据库获取
    if candidate_chunks is None:
        print("候选集合没有，自行到数据库中抽取...")
        
        # 生成查询向量
        query_vec = generate_vector_from_text(query_text)
        
        # 从向量数据库获取候选chunks
        candidate_chunks = get_top_n_candidates(
            collection_name=collection_name,
            query_vector=query_vec,
            top_n=top_n_candidates,
            include_vectors=True
        )
        
        if not candidate_chunks:
            print("未能获取到候选chunks")
            return [], 0.0, 0
    
    # 1. 获取query向量（如果还没有的话）
    query_vec = generate_vector_from_text(query_text)
    
    # 2. beta值的处理：固定beta vs 动态beta
    if beta is not None:
        # 使用固定的beta值
        print(f"使用固定beta值: {beta}")
        used_beta = beta
        # 如果k为None，仍需要估算k值
        if k is None:
            k = estimate_k_by_tokens_api(Tmax=Tmax, sample_n=sample_n, candidate_chunks=candidate_chunks)
    else:
        # 动态自适应β和k
        print("使用动态计算beta值")
        used_beta, k_ = dynamic_beta(query_vec, candidate_chunks, Tmax=Tmax, sample_n=sample_n)
        if k is None:
            k = k_
    
    alpha = 1.0
    
    # 3. 选择chunks的方法
    if selection_method == "simple":
        # 使用简单的打分排序方法
        sorted_chunks = score_and_sort_chunks(query_vec, candidate_chunks, alpha, used_beta)
        selected_chunks = sorted_chunks[:k]
    else:
        # 使用贪心方法（默认）
        selected_chunks = greedy_select_chunks(query_vec, candidate_chunks, alpha, used_beta, k=k)
    
    return selected_chunks, used_beta, k

# =====================
# 如果时主入口
if __name__ == "__main__":
    # 用法示例：
    
    # 方式1: 自动从向量数据库获取候选chunks，使用动态beta（原有方式）
    print("=== 使用动态beta计算 ===")
    selected_chunks, beta, k = select_chunks_pipeline(
        query_text="甲硝唑的不良反应？",
        top_n_candidates = 1000,
        # collection_name="entities",
        #    top_n_candidates=1000,
        selection_method="greedy",
        # k= 18,  
        # selection_method="simple"
    )
    selected_chunks_description = [chunk["description"] for chunk in selected_chunks]
    print("selected_chunks_description:",selected_chunks_description)
    print("动态计算的beta:",beta)
    print("k:",k)
    
    print("\n" + "="*50 + "\n")
    
    # 方式2: 使用固定的beta值
    print("=== 使用固定beta值 ===")
    selected_chunks_fixed, beta_fixed, k_fixed = select_chunks_pipeline(
        query_text="甲硝唑的不良反应？",
        top_n_candidates = 1000,
        beta=0.5,  # 使用固定beta值
        selection_method="greedy",
    )
    selected_chunks_description_fixed = [chunk["description"] for chunk in selected_chunks_fixed]
    print("selected_chunks_description:",selected_chunks_description_fixed)
    print("固定beta:",beta_fixed)
    print("k:",k_fixed)
    
    # 方式3: 手动提供候选chunks（原有方式）
    # selected_chunks, beta, k = select_chunks_pipeline(
    #        query_text="甲硝唑的不良反应？",
    #        candidate_chunks=your_candidate_chunks,
    #        beta=0.3  # 可选：使用固定beta
    #    )

