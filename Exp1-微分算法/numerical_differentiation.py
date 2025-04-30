# numerical_differentiation.py
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def f(x):
    """目标函数：f(x) = 1 + 0.5*tanh(2x)"""
    return 1 + 0.5 * np.tanh(2 * x)

def calculate_central_difference(f, x, h=1e-5):
    """
    中心差分法计算导数（函数名与测试用例一致）
    :param f: 目标函数
    :param x: 计算点的位置（支持向量化输入）
    :param h: 步长（默认1e-5）
    :return: 导数近似值
    """
    return (f(x + h) - f(x - h)) / (2 * h)

def get_analytical_derivative():
    """使用Sympy计算解析导数并转换为可调用函数"""
    x = sp.symbols('x')
    f_sym = 1 + 0.5 * sp.tanh(2 * x)
    df_sym = sp.diff(f_sym, x)
    df_func = sp.lambdify(x, df_sym, 'numpy')  # 转换为NumPy函数
    return df_func

def richardson_extrapolation(f, x, n=5, h0=0.1):
    """
    Richardson外推法计算导数
    :param f: 目标函数
    :param x: 计算点的位置
    :param n: 外推阶数（默认5）
    :param h0: 初始步长（默认0.1）
    :return: 最高阶的导数近似值
    """
    d = np.zeros((n+1, n+1))
    h = h0
    for i in range(n+1):
        # 计算D_{i,0}
        d[i, 0] = (f(x + h) - f(x - h)) / (2 * h)
        # 递推计算高阶外推
        for j in range(1, i+1):
            factor = 4**j
            d[i, j] = d[i, j-1] + (d[i, j-1] - d[i-1, j-1]) / (factor - 1)
        h /= 2  # 步长折半
    return d[n, n]

# 添加测试所需的别名（与测试用例名称匹配）
richardson_derivative_all_orders = richardson_extrapolation

def analyze_errors():
    """分析步长对误差的影响并绘图"""
    # 生成步长序列：0.1, 0.01, ..., 1e-6
    hs = np.logspace(-1, -6, num=6, base=10)
    x_test = 0.5  # 测试点
    
    df_analytical = get_analytical_derivative()
    exact = df_analytical(x_test)
    
    # 计算误差
    errors_central = []
    errors_richardson = []
    
    for h in hs:
        # 中心差分误差
        approx_central = calculate_central_difference(f, x_test, h)
        errors_central.append(np.abs(approx_central - exact))
        
        # Richardson外推（n=3, h0=h）
        approx_rich = richardson_extrapolation(f, x_test, n=3, h0=h)
        errors_richardson.append(np.abs(approx_rich - exact))
    
    # 绘制log-log图
    plt.figure(figsize=(10, 6))
    plt.loglog(hs, errors_central, 'o-', label='Central Difference')
    plt.loglog(hs, errors_richardson, 's-', label='Richardson (n=3)')
    plt.xlabel('Step size h (log scale)')
    plt.ylabel('Absolute error (log scale)')
    plt.title('Error vs. Step Size')
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.savefig('error_analysis.png')
    plt.show()

def plot_derivative_comparison():
    """绘制三种方法在区间[-2, 2]上的导数对比"""
    x_vals = np.linspace(-2, 2, 400)
    df_analytical = get_analytical_derivative()
    
    # 计算各方法导数
    exact_vals = df_analytical(x_vals)
    central_vals = calculate_central_difference(f, x_vals, h=1e-5)
    richardson_vals = np.array([richardson_extrapolation(f, x, n=3, h0=0.1) for x in x_vals])
    
    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(x_vals, exact_vals, 'k-', lw=2, label='Analytical')
    plt.plot(x_vals, central_vals, 'r--', label='Central Difference (h=1e-5)')
    plt.plot(x_vals, richardson_vals, 'b:', label='Richardson (n=3, h0=0.1)')
    plt.xlabel('x')
    plt.ylabel("f'(x)")
    plt.title('Derivative Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('derivative_comparison.png')
    plt.show()

if __name__ == "__main__":
    analyze_errors()
    plot_derivative_comparison()
