import numpy as np
import matplotlib.pyplot as plt
from sympy import tanh, symbols, diff, lambdify

def f(x):
    """原始函数 f(x) = 1 + 0.5*tanh(2x)"""
    return 1 + 0.5 * np.tanh(2 * x)

def get_analytical_derivative():
    """获取解析导数函数"""
    x = symbols('x')
    expr = diff(1 + 0.5 * tanh(2 * x), x)
    return lambdify(x, expr)

def calculate_central_difference(x, f_func):
    """使用中心差分法计算数值导数"""
    dy = []
    for i in range(1, len(x)-1):
        h = x[i+1] - x[i]  # 动态计算步长
        dy.append((f_func(x[i+1]) - f_func(x[i-1])) / (2 * h))
    return np.array(dy)

def richardson_derivative_all_orders(x, f_func, h, max_order=3):
    """Richardson外推法计算不同阶数的导数值"""
    R = np.zeros((max_order + 1, max_order + 1))
    
    # 计算第一列（不同步长的中心差分）
    for i in range(max_order + 1):
        hi = h / (2**i)
        R[i, 0] = (f_func(x + hi) - f_func(x - hi)) / (2 * hi)
    
    # Richardson外推填充表
    for j in range(1, max_order + 1):
        for i in range(max_order - j + 1):
            R[i, j] = (4**j * R[i+1, j-1] - R[i, j-1]) / (4**j - 1)
    
    return [R[0, j] for j in range(1, max_order + 1)]

def create_comparison_plot(x, x_central, dy_central, dy_richardson, df_analytical):
    """生成包含四幅子图的对比分析图"""
    plt.style.use('seaborn')  # 设置绘图风格
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 14))
    
    # 解析解计算
    analytical = df_analytical(x)
    analytical_central = df_analytical(x_central)

    ax1.plot(x, analytical, 'b-', lw=2, label='解析解')
    ax1.plot(x_central, dy_central, 'ro', markersize=5, label='中心差分法')
    ax1.plot(x, dy_richardson[:, 1], 'g^', markersize=5, label='Richardson外推（二阶）')
    ax1.set_title('导数对比（区间 [-2, 2]）', fontsize=12)
    ax1.set_xlabel('x', fontsize=10)
    ax1.set_ylabel("f'(x)", fontsize=10)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    error_central = np.abs(dy_central - analytical_central)
    error_richardson = np.abs(dy_richardson[:, 1] - analytical)
    
    ax2.semilogy(x_central, error_central, 'ro', markersize=5, label='中心差分法误差')
    ax2.semilogy(x, error_richardson, 'g^', markersize=5, label='Richardson外推误差')
    ax2.set_title('误差分析（对数坐标）', fontsize=12)
    ax2.set_xlabel('x', fontsize=10)
    ax2.set_ylabel('绝对误差', fontsize=10)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    for i, order in enumerate(['一阶', '二阶', '三阶'], 1):
        error = np.abs(dy_richardson[:, i-1] - analytical)
        ax3.semilogy(x, error, marker='^', markersize=5, label=f'Richardson外推（{order}）')
    ax3.set_title('不同阶数的Richardson外推误差', fontsize=12)
    ax3.set_xlabel('x', fontsize=10)
    ax3.set_ylabel('绝对误差', fontsize=10)
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # 子图4：步长敏感性分析（x=0）
    h_values = np.logspace(-6, -1, 20)
    x_test = 0.0
    central_errors = []
    richardson_errors = []
    expected = df_analytical(x_test)
    
    for h in h_values:
        # 中心差分误差
        central_result = (f(x_test + h) - f(x_test - h)) / (2 * h)
        central_errors.append(abs(central_result - expected))
        
        # Richardson外推误差（二阶）
        rich_result = richardson_derivative_all_orders(x_test, f, h, max_order=3)[1]
        richardson_errors.append(abs(rich_result - expected))
    
    ax4.loglog(h_values, central_errors, 'ro-', label='中心差分法')
    ax4.loglog(h_values, richardson_errors, 'g^-', label='Richardson外推（二阶）')
    ax4.set_title('步长敏感性分析（x=0）', fontsize=12)
    ax4.set_xlabel('步长 h', fontsize=10)
    ax4.set_ylabel('绝对误差', fontsize=10)
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('numerical_derivative_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数，运行数值微分实验"""
    # 参数设置
    h_initial = 0.1     # 初始步长
    max_order = 3       # Richardson最大外推阶数
    N_points = 200      # 采样点数
    
    # 生成均匀分布的x值（包含边界点）
    x = np.linspace(-2, 2, N_points)
    df_analytical = get_analytical_derivative()  # 解析导数函数
    
    # 计算中心差分导数（注意：中心差分结果比输入x少两个点）
    dy_central = calculate_central_difference(x, f)
    x_central = x[1:-1]  # 去除首尾无法计算的点
    
    # 计算Richardson外推导数（每个x点独立计算）
    dy_richardson = np.array([
        richardson_derivative_all_orders(xi, f, h_initial, max_order=max_order)
        for xi in x
    ])
    
    # 生成对比分析图
    create_comparison_plot(x, x_central, dy_central, dy_richardson, df_analytical)

if __name__ == '__main__':
    main()
