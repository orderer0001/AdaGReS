from flask import Flask, request, jsonify
from typing import Optional
import traceback
from AdaGReS import select_chunks_pipeline

app = Flask(__name__)

@app.route('/api/select_chunks', methods=['POST'])
def select_chunks_api():
    """
    AdaGReS chunks选择API接口
    
    参数:
    - query_text (必填): 查询文本
    - top_n_candidates (可选): 候选chunks数量
    - k (可选): 选择的chunks数量，当selection_method为simple时必填
    - selection_method (可选): 选择方法，可选值为"greedy"或"simple"，默认"greedy"
    - beta (可选): 固定的beta值，如果提供则不使用动态计算
    - Tmax (可选): 最大token数量，当beta为空时用于动态计算beta，默认1500
    - sample_n (可选): 用于估算的样本数量，默认500
    
    返回:
    - selected_chunks: 选中的chunks列表
    - beta: 使用的beta值（固定值或动态计算值）
    - k: 实际使用的k值
    """
    try:
        # 获取请求数据
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "message": "请求体不能为空",
                "code": 400
            }), 400
        
        # 验证必填参数
        query_text = data.get('query_text')
        if not query_text:
            return jsonify({
                "success": False,
                "message": "query_text参数是必填的",
                "code": 400
            }), 400
        
        # 获取可选参数
        top_n_candidates = data.get('top_n_candidates', 3000)
        k = data.get('k')
        selection_method = data.get('selection_method', 'greedy')
        beta = data.get('beta')  # 新增：固定beta值
        Tmax = data.get('Tmax', 1500)  # 新增：最大token数量
        sample_n = data.get('sample_n', 500)  # 新增：样本数量
        
        # 验证参数类型
        if not isinstance(query_text, str):
            return jsonify({
                "success": False,
                "message": "query_text必须是字符串类型",
                "code": 400
            }), 400
        
        if not isinstance(top_n_candidates, int) or top_n_candidates <= 0:
            return jsonify({
                "success": False,
                "message": "top_n_candidates必须是正整数",
                "code": 400
            }), 400
        
        if not isinstance(Tmax, int) or Tmax <= 0:
            return jsonify({
                "success": False,
                "message": "Tmax必须是正整数",
                "code": 400
            }), 400
        
        if not isinstance(sample_n, int) or sample_n <= 0:
            return jsonify({
                "success": False,
                "message": "sample_n必须是正整数",
                "code": 400
            }), 400
        
        # 验证selection_method
        if selection_method not in ['greedy', 'simple']:
            return jsonify({
                "success": False,
                "message": "selection_method只能是'greedy'或'simple'",
                "code": 400
            }), 400
        
        # 验证k参数
        if k is not None:
            if not isinstance(k, int) or k <= 0:
                return jsonify({
                    "success": False,
                    "message": "k必须是正整数",
                    "code": 400
                }), 400
        
        # 验证beta参数
        if beta is not None:
            if not isinstance(beta, (int, float)) or beta <= 0:
                return jsonify({
                    "success": False,
                    "message": "beta必须是正数",
                    "code": 400
                }), 400
        
        # 当selection_method为simple时，k必须提供
        if selection_method == 'simple' and k is None:
            return jsonify({
                "success": False,
                "message": "当selection_method为'simple'时，k参数是必填的",
                "code": 400
            }), 400
        
        # 调用 AdaGReS pipeline
        selected_chunks, actual_beta, actual_k = select_chunks_pipeline(
            query_text=query_text,
            k=k,
            top_n_candidates=top_n_candidates,
            Tmax=Tmax,
            beta=beta,
            sample_n=sample_n,
            selection_method=selection_method
        )
        
        # 返回结果
        return jsonify({
            "success": True,
            "message": "chunks选择成功",
            "code": 200,
            "data": {
                "selected_chunks": selected_chunks,
                "beta": float(actual_beta),
                "k": int(actual_k),
                "total_selected": len(selected_chunks),
                "beta_source": "fixed" if beta is not None else "dynamic"  # 标识beta来源
            }
        }), 200
        
    except Exception as e:
        # 记录错误信息
        error_msg = str(e)
        print(f"API调用出错: {error_msg}")
        print(f"错误详情: {traceback.format_exc()}")
        
        return jsonify({
            "success": False,
            "message": f"服务器内部错误: {error_msg}",
            "code": 500
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "success": True,
        "message": "AdaGReS API服务运行正常",
        "code": 200
    }), 200

@app.route('/api/select_chunks/help', methods=['GET'])
def api_help():
    """API使用说明"""
    return jsonify({
        "success": True,
        "message": "AdaGReS chunks选择API使用说明",
        "code": 200,
        "data": {
            "endpoint": "/api/select_chunks",
            "method": "POST",
            "parameters": {
                "query_text": {
                    "type": "string",
                    "required": True,
                    "description": "查询文本"
                },
                "top_n_candidates": {
                    "type": "integer",
                    "required": False,
                    "default": 3000,
                    "description": "候选chunks数量"
                },
                "k": {
                    "type": "integer",
                    "required": False,
                    "description": "选择的chunks数量，当selection_method为simple时必填"
                },
                "selection_method": {
                    "type": "string",
                    "required": False,
                    "default": "greedy",
                    "options": ["greedy", "simple"],
                    "description": "选择方法"
                },
                "beta": {
                    "type": "number",
                    "required": False,
                    "description": "固定的beta值，如果提供则不使用动态计算"
                },
                "Tmax": {
                    "type": "integer",
                    "required": False,
                    "default": 1500,
                    "description": "最大token数量，当beta为空时用于动态计算beta"
                },
                "sample_n": {
                    "type": "integer",
                    "required": False,
                    "default": 500,
                    "description": "用于估算的样本数量"
                }
            },
            "usage_examples": {
                "dynamic_beta": {
                    "description": "使用动态beta计算",
                    "request": {
                        "query_text": "甲硝唑的不良反应？",
                        "top_n_candidates": 2000,
                        "k": 18,
                        "selection_method": "greedy",
                        "Tmax": 1500,
                        "sample_n": 500
                    }
                },
                "fixed_beta": {
                    "description": "使用固定beta值",
                    "request": {
                        "query_text": "甲硝唑的不良反应？",
                        "top_n_candidates": 2000,
                        "k": 18,
                        "selection_method": "greedy",
                        "beta": 0.5
                    }
                }
            },
            "notes": [
                "beta和Tmax参数二选一：提供beta时使用固定值，不提供beta时通过Tmax动态计算",
                "当selection_method为'simple'时，k参数必填",
                "返回结果中会包含beta_source字段，标识beta是'fixed'还是'dynamic'"
            ]
        }
    }), 200

if __name__ == '__main__':
    print("启动AdaGReS API服务...")
    app.run(host='0.0.0.0', port=5000, debug=True)
