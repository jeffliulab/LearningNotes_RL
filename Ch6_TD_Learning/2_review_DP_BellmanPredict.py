
import numpy as np

# 定义状态
states = ["town", "forest", "castle"]

# 折扣因子
gamma = 0.95

# 初始化状态价值函数
V = {"town": 0.0, "forest": 0.0, "castle": 0.0}

# 收敛阈值
theta = 1e-4

while True:
    delta = 0.0
    V_new = V.copy()
    
    # 更新 town 状态的价值
    # 策略：在 town 选择去 castle
    v_town = 0.1 * (8 + gamma * V["castle"]) + 0.9 * (-5 + gamma * V["town"])
    delta = max(delta, abs(v_town - V["town"]))
    V_new["town"] = v_town
    
    # 更新 forest 状态的价值
    # 策略：在 forest 选择去 castle
    # 在现在的策略下，实际上不会到forest
    v_forest = 0.1 * (8 + gamma * V["castle"]) + 0.9 * (-5 + gamma * V["forest"])
    delta = max(delta, abs(v_forest - V["forest"]))
    V_new["forest"] = v_forest
    
    # 更新 castle 状态的价值
    # 策略：在 castle 选择去 town
    v_castle = 0.3 * (5 + gamma * V["town"]) + 0.7 * (-1 + gamma * V["castle"])
    delta = max(delta, abs(v_castle - V["castle"]))
    V_new["castle"] = v_castle
    
    V = V_new
    
    if delta < theta:
        break

print("DP 策略评估得到的状态价值函数:")
print("V(town):", V["town"])
print("V(forest):", V["forest"])
print("V(castle):", V["castle"])
