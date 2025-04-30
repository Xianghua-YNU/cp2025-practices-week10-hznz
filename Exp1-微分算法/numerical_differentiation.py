import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def f(x):
    return 1 + 0.5 * np.tanh(2 * x)

def get_analytical_derivative():
    """返回解析导数函数"""
    x_sym = sp.symbols('x')
    f_expr = 1 + 0.5 * sp.tanh(2 * x_sym)
    f_prime_expr = sp.diff(f_expr, x_sym)
    return sp.lambdify(x_sym, f_prime_expr, 'numpy')

df_analytical = get_analytical_derivative()

def calculate_central_difference(x_points, func, h=0.1):
    """计算中心差分导数（支持数组输入）"""
    derivatives = []
    for x in x_points:
        deriv = (func(x + h) - func(x - h)) / (2 * h)
        derivatives.append(deriv)
    return np.array(derivatives)

def richardson_derivative_all_orders(x, func, h, max_order=3):
    """Richardson外推法（返回各阶结果列表）"""
    D = np.zeros((max_order+1, max_order+1))
    results = []
    for i in range(max_order+1):
        current_h = h / (2 ** i)
        D[i, 0] = (func(x + current_h) - func(x - current_h)) / (2 * current_h)
        for j in range(1, i+1):
            factor = 4 ** j
            D[i, j] = D[i, j-1] + (D[i, j-1] - D[i-1, j-1]) / (factor - 1)
        results.append(D[i, i])
    return results

def analyze_errors():
    """分析步长对误差的影响"""
    hs = np.logspace(-1, -6, num=6, base=10)  # 生成 [0.1, 0.01, ..., 1e-6]
    x_test = 0.5  # 测试点
    
    df_analytical = get_analytical_derivative()
    exact = df_analytical(x_test)  # 解析解
    
    # 计算误差
    errors_central = []
    errors_richardson = []
    
    for h in hs:
        # 中心差分误差（明确传递函数对象f）
        approx_central = calculate_central_difference(f, x_test, h)
        errors_central.append(np.abs(approx_central - exact))
        
        # Richardson外推（使用max_order参数）
        approx_rich = richardson_extrapolation(f, x_test, max_order=3, h0=h)
        errors_richardson.append(np.abs(approx_rich - exact))
    
    # 绘制log-log图
    plt.figure(figsize=(10, 6))
    plt.loglog(hs, errors_central, 'o-', label='Central Difference')
    plt.loglog(hs, errors_richardson, 's-', label='Richardson (max_order=3)')
    plt.xlabel('Step size h (log scale)')
    plt.ylabel('Absolute error (log scale)')
    plt.title('Error vs. Step Size')
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.savefig('error_analysis.png')
    plt.show()

def plot_derivative_comparison():
    """绘制三种方法在区间[-2, 2]上的导数对比图"""
    x_vals = np.linspace(-2, 2, 400)
    df_analytical = get_analytical_derivative()
    
    # 计算各方法导数
    exact_vals = df_analytical(x_vals)
    central_vals = calculate_central_difference(f, x_vals, h=1e-5)
    richardson_vals = np.array([richardson_extrapolation(f, x, max_order=3, h0=0.1) for x in x_vals])
    
    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(x_vals, exact_vals, 'k-', lw=2, label='Analytical')
    plt.plot(x_vals, central_vals, 'r--', label='Central Difference (h=1e-5)')
    plt.plot(x_vals, richardson_vals, 'b:', label='Richardson (max_order=3, h0=0.1)')
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
