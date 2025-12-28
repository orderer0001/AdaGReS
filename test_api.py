import requests
import json

# API服务地址
API_BASE_URL = "http://localhost:5000"

def test_api():
    """测试 AdaGReS API 接口"""
    
    print("=== AdaGReS API 测试 ===\n")
    
    # 1. 健康检查
    print("1. 健康检查测试:")
    try:
        response = requests.get(f"{API_BASE_URL}/api/health")
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.json()}\n")
    except Exception as e:
        print(f"健康检查失败: {e}\n")
    
    # 2. 获取API帮助信息
    print("2. API帮助信息:")
    try:
        response = requests.get(f"{API_BASE_URL}/api/select_chunks/help")
        print(f"状态码: {response.status_code}")
        help_data = response.json()
        print(f"API说明: {json.dumps(help_data, ensure_ascii=False, indent=2)}\n")
    except Exception as e:
        print(f"获取帮助信息失败: {e}\n")
    
    # 3. 测试greedy方法（不指定k）
    print("3. 测试greedy方法（不指定k）:")
    test_data_greedy = {
        "query_text": "甲硝唑的不良反应？",
        "top_n_candidates": 1000,
        "selection_method": "greedy"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/select_chunks",
            json=test_data_greedy,
            headers={"Content-Type": "application/json"}
        )
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"成功: {result.get('success')}")
        print(f"消息: {result.get('message')}")
        if result.get('success'):
            data = result.get('data', {})
            print(f"Beta值: {data.get('beta')}")
            print(f"实际K值: {data.get('k')}")
            print(f"选中chunks数量: {data.get('total_selected')}")
            print(f"第一个chunk示例: {data.get('selected_chunks', [{}])[0] if data.get('selected_chunks') else '无'}")
        print()
    except Exception as e:
        print(f"greedy方法测试失败: {e}\n")
    
    # 4. 测试simple方法（指定k）
    print("4. 测试simple方法（指定k）:")
    test_data_simple = {
        "query_text": "甲硝唑的不良反应？",
        "top_n_candidates": 1000,
        "k": 15,
        "selection_method": "simple"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/select_chunks",
            json=test_data_simple,
            headers={"Content-Type": "application/json"}
        )
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"成功: {result.get('success')}")
        print(f"消息: {result.get('message')}")
        if result.get('success'):
            data = result.get('data', {})
            print(f"Beta值: {data.get('beta')}")
            print(f"实际K值: {data.get('k')}")
            print(f"选中chunks数量: {data.get('total_selected')}")
        print()
    except Exception as e:
        print(f"simple方法测试失败: {e}\n")
    
    # 5. 测试参数验证（缺少query_text）
    print("5. 测试参数验证（缺少query_text）:")
    test_data_invalid = {
        "top_n_candidates": 1000,
        "selection_method": "greedy"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/select_chunks",
            json=test_data_invalid,
            headers={"Content-Type": "application/json"}
        )
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"成功: {result.get('success')}")
        print(f"错误消息: {result.get('message')}")
        print()
    except Exception as e:
        print(f"参数验证测试失败: {e}\n")
    
    # 6. 测试simple方法缺少k参数
    print("6. 测试simple方法缺少k参数:")
    test_data_simple_no_k = {
        "query_text": "甲硝唑的不良反应？",
        "selection_method": "simple"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/select_chunks",
            json=test_data_simple_no_k,
            headers={"Content-Type": "application/json"}
        )
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"成功: {result.get('success')}")
        print(f"错误消息: {result.get('message')}")
        print()
    except Exception as e:
        print(f"simple方法无k参数测试失败: {e}\n")

if __name__ == "__main__":
    print("请确保AdaGReS API服务已启动 (python AdaGReS_api.py)")
    print("如果服务未启动，请先运行: python AdaGReS_api.py")
    print()
    
    # 等待用户确认
    input("按Enter键开始测试...")
    test_api() 