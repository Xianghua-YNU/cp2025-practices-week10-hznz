import numpy as np
import matplotlib.pyplot as plt
import os

def q3a(T):
    """计算3-α反应速率中与温度相关的部分 q/(ρ²Y³)"""
    T8 = T / 1e8  # 转换为T8单位（以1e8 K为单位）
    return 5.09e11 * (T8)**(-3) * np.exp(-44.027 / T8)

def calculate_nu(T0, h=1e-8):
    """使用前向差分法计算温度敏感性指数ν"""
    q0 = q3a(T0)
    dT = h * T0  # 计算绝对温度步长
    q_perturbed = q3a(T0 + dT)
    dq_dT = (q_perturbed - q0) / dT  # 前向差分
    nu = (T0 / q0) * dq_dT
    return nu

def plot_rate(filename="results/rate_vs_temp.png"):
    """绘制速率因子随温度变化的log-log图"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    T_values = np.logspace(8, np.log10(5e9), 100)  # 1e8到5e9对数间隔
    q_values = q3a(T_values)
    
    plt.figure(figsize=(10, 6))
    plt.loglog(T_values, q_values, label='q/(ρ²Y³)')
    plt.xlabel('Temperature (K)')
    plt.ylabel('q/(ρ²Y³) [erg cm$^6$ g$^{-3}$ s$^{-1}$]')
    plt.title('3-α Reaction Rate vs Temperature (log-log scale)')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    temperatures_K = [1.0e8, 2.5e8, 5.0e8, 1.0e9, 2.5e9, 5.0e9]
    h = 1e-8
    
    print("温度 T (K)    :   ν 值")
    print("----------------------------")
    for T0 in temperatures_K:
        nu = calculate_nu(T0, h)
        print(f"{T0:12.1e} : {nu:.3f}")
    
    plot_rate()
