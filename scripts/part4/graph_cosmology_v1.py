╔═════════════════════════════════════════════════════════════════╗
║ ЭКСПЕРИМЕНТ 3.1: ТЁМНАЯ ЭНЕРГИЯ И ГОРИЗОНТ ХАББЛА ║
║ Моделирование ускоренного расширения Вселенной ║
╚═════════════════════════════════════════════════════════════════╝
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings("ignore")

=====================================================================
SBE DARK UNIVERSE LABORATORY (Dark Energy & Hubble Horizon)
=====================================================================
def generate_dynamic_phases(N_total):
"""Генерация 6D фаз для растущей Вселенной"""
D = 6
omega = 2 * np.sin(np.pi * np.array([1,4,5,7,8,11]) / 12)
phases = np.zeros((N_total, D))
phases[0] = np.random.uniform(0, 2*np.pi, D)
for n in range(N_total - 1):
phases[n+1] = (phases[n] + omega + 0.1 * np.sin(phases[n])) % (2 * np.pi)
return phases

def run_dark_energy_horizon():
print("╔" + "═"*65 + "╗")
print("║ ЭКСПЕРИМЕНТ 3.1: ТЁМНАЯ ЭНЕРГИЯ И ГОРИЗОНТ ХАББЛА ║")
print("║ Моделирование ускоренного расширения Вселенной ║")
print("╚" + "═"*65 + "╝")

text

epochs = 25
N_initial = 500
N_add_per_epoch = 150
N_total = N_initial + epochs * N_add_per_epoch

phases = generate_dynamic_phases(N_total)

G = nx.Graph()
G.add_nodes_from(range(N_initial))
for i in range(N_initial - 1):
    G.add_edge(i, i + 1)
    
epsilon_base = 1.5 

time_epochs = []
scale_factor = []       
dark_matter =[]        

print("\n[+] Запуск Космологического Времени с Горизонтом Событий...")
print(f"    {'Эпоха':<6s} | {'Узлов':<6s} | {'Тёмная Материя':<18s} | {'Размер a(t)':<15s} | {'Горизонт λ':<10s}")
print("    " + "-"*65)

start_time = time.time()

for epoch in range(epochs):
    current_N = N_initial + epoch * N_add_per_epoch
    new_nodes = range(current_N, current_N + N_add_per_epoch)
    
    G.add_nodes_from(new_nodes)
    
    # Хронологическая стрела времени
    for i in new_nodes:
        G.add_edge(i - 1, i)
        
    # =========================================================
    # ВВОДИМ ТЁМНУЮ ЭНЕРГИЮ: Динамический Горизонт Хаббла
    # Чем старше Вселенная, тем сильнее экспоненциальное затухание связей с прошлым
    # lambda_decay растет со временем, отрезая глубокое прошлое
    # =========================================================
    lambda_decay = 0.001 + 0.0005 * (epoch ** 1.5)
    
    degrees = np.array([G.degree(n) for n in range(current_N)])
    
    for i in new_nodes:
        # Расстояние во времени до всех узлов в прошлом
        delta_t = i - np.arange(current_N)
        
        # Вероятность связи: Гравитация (степень узла) минус Тёмная Энергия (затухание по времени)
        scores = (degrees + 1) * np.exp(-lambda_decay * delta_t)
        scores_sum = np.sum(scores)
        
        if scores_sum > 0:
            weights = scores / scores_sum
        else:
            weights = np.ones(current_N) / current_N
            
        candidates = np.random.choice(range(current_N), size=min(current_N, 80), p=weights, replace=False)
        
        connections_made = 0
        for j in candidates:
            if connections_made >= 5: # Жесткий бюджет связей (сохранение энергии)
                break
                
            diff = np.abs(phases[i] - phases[j])
            diff = np.minimum(diff, 2*np.pi - diff)
            dist = np.linalg.norm(diff)
            
            # Если фазы совпали - образуется кротовина
            if dist < epsilon_base:
                G.add_edge(i, j)
                connections_made += 1
                
    time_epochs.append(epoch)
    
    # Замеряем кластеризацию (Тёмная Материя)
    clustering = nx.average_clustering(G)
    dark_matter.append(clustering)
    
    # Замеряем размер Вселенной (топологическое расстояние)
    largest_cc = max(nx.connected_components(G), key=len)
    subgraph = G.subgraph(largest_cc)
    sample_nodes = np.random.choice(list(subgraph.nodes()), min(len(subgraph), 50), replace=False)
    
    path_lengths =[]
    for n in sample_nodes:
        lengths = nx.single_source_shortest_path_length(subgraph, n)
        path_lengths.extend(list(lengths.values()))
        
    avg_distance = np.mean(path_lengths)
    scale_factor.append(avg_distance)
    
    print(f"    {epoch:<6d} | {current_N+N_add_per_epoch:<6d} | C = {clustering:<14.4f} | a(t) = {avg_distance:<10.4f} | {lambda_decay:.5f}")

print(f"\n[+] Эволюция завершена за {time.time()-start_time:.1f} сек.")

# ВЫЧИСЛЕНИЕ УСКОРЕНИЯ (Тёмная Энергия)
scale_factor = np.array(scale_factor)
expansion_rate = np.gradient(scale_factor) # Скорость v
acceleration = np.gradient(expansion_rate) # Ускорение a

# Сглаживание графика ускорения (чтобы убрать шум случайных графов)
window = 3
accel_smooth = np.convolve(acceleration, np.ones(window)/window, mode='same')
accel_smooth[0] = acceleration[0]
accel_smooth[-1] = acceleration[-1]

# РЕНДЕРИНГ ГРАФИКОВ
print("[+] Рендеринг доказательств...")
fig, axes = plt.subplots(1, 3, figsize=(22, 6))

ax1 = axes[0]
ax1.plot(time_epochs, dark_matter, 's-', color='indigo', lw=2.5, markersize=7)
ax1.set_title('Формирование Галактик\n(Глобальная кластеризация графа)', fontsize=13, pad=10)
ax1.set_xlabel('Космологическое время (Эпохи $t$)', fontsize=12)
ax1.set_ylabel('Топологическая плотность (Тёмная материя)', fontsize=12)
ax1.grid(True, ls='--', alpha=0.5)

ax2 = axes[1]
ax2.plot(time_epochs, scale_factor, 'o-', color='teal', lw=2.5, markersize=7)
ax2.set_title('Расширение Вселенной $a(t)$\n(Среднее топологическое расстояние)', fontsize=13, pad=10)
ax2.set_xlabel('Космологическое время (Эпохи $t$)', fontsize=12)
ax2.set_ylabel('Масштабный фактор $a(t)$', fontsize=12)
ax2.grid(True, ls='--', alpha=0.5)

ax3 = axes[2]
ax3.plot(time_epochs, accel_smooth, '^-', color='crimson', lw=2.5, markersize=7)
ax3.axhline(y=0, color='black', linestyle='-', alpha=0.8)

ax3.fill_between(time_epochs, 0, accel_smooth, where=(accel_smooth > 0), color='crimson', alpha=0.2, label='Тёмная Энергия (Ускорение > 0)')
ax3.fill_between(time_epochs, 0, accel_smooth, where=(accel_smooth < 0), color='blue', alpha=0.2, label='Торможение (Гравитация)')

ax3.set_title(r'Эффект Тёмной Энергии' + '\n' + r'(Ускорение расширения $\ddot{a}$)', fontsize=13, pad=10)
ax3.set_xlabel('Космологическое время (Эпохи $t$)', fontsize=12)
ax3.set_ylabel(r'Ускорение $\ddot{a}$ (сглаженное)', fontsize=12)
ax3.legend(loc='upper left', fontsize=11)
ax3.grid(True, ls='--', alpha=0.5)

plt.tight_layout()
plt.show()
if name == "main":
run_dark_energy_horizon()
