import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings("ignore")

# =====================================================================
# SBE DARK UNIVERSE LABORATORY (Graph Cosmology Model)
# =====================================================================

def generate_dynamic_phases(N_total):
    """Генерация 6D фаз для растущего графа"""
    D = 6
    omega = 2 * np.sin(np.pi * np.array([1,4,5,7,8,11]) / 12)
    phases = np.zeros((N_total, D))
    phases[0] = np.random.uniform(0, 2*np.pi, D)
    for n in range(N_total - 1):
        phases[n+1] = (phases[n] + omega + 0.1 * np.sin(phases[n])) % (2 * np.pi)
    return phases

def run_graph_cosmology_experiment(n_epochs=25, n_simulations=10, N_initial=500, N_add_per_epoch=150):
    """
    Моделирование растущего графа с динамическим затуханием связей.
    lambda_decay зависит от локальной плотности (аналог космологического принципа).
    """
    print("╔" + "═"*65 + "╗")
    print("║  ЭКСПЕРИМЕНТ 3.1: ГРАФОВАЯ МОДЕЛЬ РАСШИРЕНИЯ         ║")
    print("║  Динамическое затухание связей и рост длины пути      ║")
    print("╚" + "═"*65 + "╝")
    
    # Массивы для результатов по всем симуляциям
    all_scale_factor = []
    all_clustering = []
    all_acceleration = []
    
    start_time = time.time()
    
    for sim in range(n_simulations):
        if sim % 5 == 0:
            print(f"[+] Запуск симуляции {sim+1}/{n_simulations}...")
        
        phases = generate_dynamic_phases(N_initial + epochs * N_add_per_epoch)
        
        G = nx.Graph()
        G.add_nodes_from(range(N_initial))
        for i in range(N_initial - 1):
            G.add_edge(i, i + 1)
            
        scale_factor = []
        clustering = []
        
        for epoch in range(n_epochs):
            current_N = N_initial + epoch * N_add_per_epoch
            new_nodes = range(current_N, current_N + N_add_per_epoch)
            
            G.add_nodes_from(new_nodes)
            
            # Хронологическая стрела времени (предшественники)
            for i in new_nodes:
                G.add_edge(i - 1, i)
                
            # === ВЫЧИСЛЕНИЕ LAMBDA_DECAY ИЗ ЛОКАЛЬНОЙ ПЛОТНОСТИ ===
            # Аналог космологического принципа: влияние "темной энергии" (затухание)
            # зависит от локальной структуры, а не от времени напрямую.
            # Здесь: lambda_decay ~ 1 / (средняя степень узла в прошлом)
            degrees = np.array([G.degree(n) for n in range(current_N)])
            avg_degree = np.mean(degrees)
            
            # Lambda decay теперь зависит от "плотности" графа в прошлом
            # Чем плотнее граф (больше связей), тем сильнее затухание (больше lambda)
            # Это имитирует эффект: при высокой плотности "темная энергия" сильнее раздвигает узлы
            lambda_decay = 0.001 + 0.0005 * (epoch ** 1.5) * (100 / (avg_degree + 1))
            
            # Динамический порог вероятности для дальних связей
            # (аналог горизонта событий: чем больше lambda, тем меньше дальних связей)
            prob_threshold = np.exp(-lambda_decay * current_N)
            
            for i in new_nodes:
                # Вероятность связи с прошлым убывает с расстоянием во времени
                delta_t = i - np.arange(current_N)
                scores = (degrees + 1) * np.exp(-lambda_decay * delta_t)
                scores_sum = np.sum(scores)
                
                if scores_sum > 0:
                    weights = scores / scores_sum
                else:
                    weights = np.ones(current_N) / current_N
                    
                # Выбираем кандидатов на связь
                candidates = np.random.choice(range(current_N), size=min(current_N, 80), p=weights, replace=False)
                
                connections_made = 0
                for j in candidates:
                    if connections_made >= 5:
                        break
                        
                    diff = np.abs(phases[i] - phases[j])
                    diff = np.minimum(diff, 2*np.pi - diff)
                    dist = np.linalg.norm(diff)
                    
                    # Если фазы совпали - образуется связь (кротовина)
                    if dist < 1.5:  # epsilon_base
                        G.add_edge(i, j)
                        connections_made += 1
                        
            # Замеряем кластеризацию
            clustering.append(nx.average_clustering(G))
            
            # Замеряем средний кратчайший путь (аналог масштабного фактора)
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            sample_size = min(len(subgraph), 50)
            if sample_size > 0:
                sample_nodes = np.random.choice(list(subgraph.nodes()), sample_size, replace=False)
                path_lengths = []
                for n in sample_nodes:
                    lengths = nx.single_source_shortest_path_length(subgraph, n)
                    path_lengths.extend(list(lengths.values()))
                avg_distance = np.mean(path_lengths)
            else:
                avg_distance = 0
            scale_factor.append(avg_distance)
        
        # Вычисление ускорения для этой симуляции
        scale_factor_arr = np.array(scale_factor)
        expansion_rate = np.gradient(scale_factor_arr)
        acceleration = np.gradient(expansion_rate)
        
        all_scale_factor.append(scale_factor)
        all_clustering.append(clustering)
        all_acceleration.append(acceleration)
    
    # === АГРЕГАЦИЯ РЕЗУЛЬТАТОВ ===
    mean_scale_factor = np.mean(all_scale_factor, axis=0)
    std_scale_factor = np.std(all_scale_factor, axis=0)
    
    mean_clustering = np.mean(all_clustering, axis=0)
    std_clustering = np.std(all_clustering, axis=0)
    
    mean_acceleration = np.mean(all_acceleration, axis=0)
    std_acceleration = np.std(all_acceleration, axis=0)
    
    # Сглаживание ускорения
    window = 3
    accel_smooth = np.convolve(mean_acceleration, np.ones(window)/window, mode='same')
    accel_smooth[0] = mean_acceleration[0]
    accel_smooth[-1] = mean_acceleration[-1]
    
    print(f"\n[+] Эволюция завершена за {time.time()-start_time:.1f} сек.")
    print(f"[+] Среднее время на эпоху: {(time.time()-start_time)/n_simulations:.1f} сек.")
    
    # === РЕНДЕРИНГ ===
    print("[+] Рендеринг результатов...")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    time_epochs = np.arange(n_epochs)
    
    # 1. Кластеризация (аналог темной материи)
    ax1 = axes[0]
    ax1.plot(time_epochs, mean_clustering, 's-', color='indigo', lw=2.5, markersize=7, label='Среднее')
    ax1.fill_between(time_epochs, 
                     mean_clustering - std_clustering, 
                     mean_clustering + std_clustering, 
                     color='indigo', alpha=0.2)
    ax1.set_title('Кластеризация графа (аналог структурообразования)', fontsize=13, pad=10)
    ax1.set_xlabel('Эпоха (шаги времени)', fontsize=12)
    ax1.set_ylabel('Средний кластерный коэффициент', fontsize=12)
    ax1.grid(True, ls='--', alpha=0.5)
    ax1.legend()
    
    # 2. Средний кратчайший путь (аналог масштабного фактора)
    ax2 = axes[1]
    ax2.plot(time_epochs, mean_scale_factor, 'o-', color='teal', lw=2.5, markersize=7, label='Среднее')
    ax2.fill_between(time_epochs, 
                     mean_scale_factor - std_scale_factor, 
                     mean_scale_factor + std_scale_factor, 
                     color='teal', alpha=0.2)
    ax2.set_title('Средний кратчайший путь (аналог расширения)', fontsize=13, pad=10)
    ax2.set_xlabel('Эпоха (шаги времени)', fontsize=12)
    ax2.set_ylabel('Средняя длина пути', fontsize=12)
    ax2.grid(True, ls='--', alpha=0.5)
    ax2.legend()
    
    # 3. Ускорение (аналог темной энергии)
    ax3 = axes[2]
    ax3.plot(time_epochs, accel_smooth, '^-', color='crimson', lw=2.5, markersize=7, label='Среднее')
    ax3.fill_between(time_epochs, 
                     accel_smooth - std_acceleration, 
                     accel_smooth + std_acceleration, 
                     color='crimson', alpha=0.2)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    
    # Зона темной энергии (ускорение > 0)
    positive_mask = accel_smooth > 0
    ax3.fill_between(time_epochs, 0, accel_smooth, where=positive_mask, color='crimson', alpha=0.2, label='Ускорение > 0')
    ax3.fill_between(time_epochs, 0, accel_smooth, where=~positive_mask, color='blue', alpha=0.2, label='Торможение < 0')
    
    ax3.set_title('Ускорение расширения (аналог темной энергии)', fontsize=13, pad=10)
    ax3.set_xlabel('Эпоха (шаги времени)', fontsize=12)
    ax3.set_ylabel('Ускорение (сглаженное)', fontsize=12)
    ax3.legend(loc='upper left', fontsize=11)
    ax3.grid(True, ls='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'mean_scale_factor': mean_scale_factor,
        'mean_clustering': mean_clustering,
        'mean_acceleration': accel_smooth,
        'std_scale_factor': std_scale_factor,
        'std_clustering': std_clustering,
        'std_acceleration': std_acceleration
    }

if __name__ == "__main__":
    # Запуск с уменьшенным количеством эпох для скорости (можно увеличить)
    results = run_graph_cosmology_experiment(n_epochs=25, n_simulations=10, N_initial=500, N_add_per_epoch=150)
