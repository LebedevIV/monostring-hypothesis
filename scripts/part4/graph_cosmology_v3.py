"""
Графовая космология v3: сравнительный эксперимент
==================================================
Три модели сравниваются на идентичных фазовых ландшафтах:
  A) Null:     λ = 0  (нет затухания, все узлы равнодоступны)
  B) Constant: λ = λ₀ (фиксированное экспоненциальное затухание)
  C) Feedback: λ = λ₀ × ⟨d⟩_prev / ⟨d⟩_initial
               (без явного времени; только состояние графа)

Цель: проверить, является ли ускоренный рост среднего пути
      эмерджентным свойством обратной связи (модель C),
      или он присутствует в любом растущем графе (модель A).
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings("ignore")


# =====================================================================
#  УТИЛИТЫ
# =====================================================================

def generate_phases(N_total, seed):
    """
    Генерация 6D-фаз с детерминированной динамикой.
    Использует ЛОКАЛЬНЫЙ генератор — не затрагивает глобальное состояние.
    """
    rng = np.random.RandomState(seed)
    D = 6
    omega = 2.0 * np.sin(np.pi * np.array([1, 4, 5, 7, 8, 11]) / 12)
    phases = np.zeros((N_total, D))
    phases[0] = rng.uniform(0, 2 * np.pi, D)
    for n in range(N_total - 1):
        phases[n + 1] = (phases[n] + omega + 0.1 * np.sin(phases[n])) % (2 * np.pi)
    return phases


def measure_graph(G, n_sample=100):
    """
    Средний кратчайший путь (по подвыборке BFS)
    и средний кластерный коэффициент.
    """
    largest_cc = max(nx.connected_components(G), key=len)
    sub = G.subgraph(largest_cc)
    nodes = list(sub.nodes())
    k = min(len(nodes), n_sample)
    sources = np.random.choice(nodes, k, replace=False)

    all_lengths = []
    for s in sources:
        lengths = nx.single_source_shortest_path_length(sub, s)
        all_lengths.extend(lengths.values())

    avg_path = float(np.mean(all_lengths)) if all_lengths else 0.0
    clustering = nx.average_clustering(G)
    return avg_path, clustering


# =====================================================================
#  ЯДРО МОДЕЛИ
# =====================================================================

def run_single(model_type, phases, n_epochs, n_init, n_add,
               lam0, eps, max_conn, n_cand, path_sample):
    """
    Один прогон растущего графа.

    model_type:
      'null'     — λ = 0
      'constant' — λ = lam0
      'feedback' — λ = lam0 × (prev_path / ref_path)
                   Никакого явного epoch. Только состояние графа.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_init))
    for i in range(n_init - 1):
        G.add_edge(i, i + 1)

    hist = {'path': [], 'clust': [], 'lam': [], 'deg': [], 'edges': []}
    ref_path = None      # ⟨d⟩ после первой эпохи (опорное значение)
    prev_path = None     # ⟨d⟩ предыдущей эпохи

    for ep in range(n_epochs):
        cur_N = n_init + ep * n_add
        new_nodes = range(cur_N, cur_N + n_add)
        G.add_nodes_from(new_nodes)

        # Стрела времени — цепочка
        for i in new_nodes:
            G.add_edge(i - 1, i)

        # ── Определение λ (БЕЗ ЯВНОГО ВРЕМЕНИ) ──
        if model_type == 'null':
            lam = 0.0
        elif model_type == 'constant':
            lam = lam0
        elif model_type == 'feedback':
            if ref_path is None or prev_path is None:
                lam = lam0                          # первая эпоха: начальное значение
            else:
                lam = lam0 * (prev_path / ref_path) # обратная связь от графа
        else:
            raise ValueError(f"Неизвестная модель: {model_type}")

        hist['lam'].append(lam)

        # ── Добавление «кротовин» ──
        degrees = np.array([G.degree(n) for n in range(cur_N)], dtype=np.float64)

        for i in new_nodes:
            delta_t = (i - np.arange(cur_N)).astype(np.float64)

            # Весовая функция: (степень + 1) × экспоненциальное затухание
            if lam > 0:
                scores = (degrees + 1.0) * np.exp(-lam * delta_t)
            else:
                scores = degrees + 1.0  # без затухания

            total = scores.sum()
            if total > 0:
                weights = scores / total
            else:
                weights = np.ones(cur_N) / cur_N

            # Защита от числовых нулей
            weights = np.maximum(weights, 1e-30)
            weights /= weights.sum()

            pool_size = min(cur_N, n_cand)
            candidates = np.random.choice(cur_N, pool_size, p=weights, replace=False)

            added = 0
            for j in candidates:
                if added >= max_conn:
                    break
                diff = np.abs(phases[i] - phases[j])
                diff = np.minimum(diff, 2 * np.pi - diff)
                if np.linalg.norm(diff) < eps:
                    G.add_edge(i, j)
                    added += 1

        # ── Измерения ──
        avg_p, clust = measure_graph(G, path_sample)
        hist['path'].append(avg_p)
        hist['clust'].append(clust)
        hist['deg'].append(2.0 * G.number_of_edges() / G.number_of_nodes())
        hist['edges'].append(G.number_of_edges())

        # Обновление обратной связи
        if ref_path is None:
            ref_path = avg_p       # фиксируем опорное значение
        prev_path = avg_p

    # Производные
    a = np.array(hist['path'])
    hist['vel'] = np.gradient(a)
    hist['acc'] = np.gradient(hist['vel'])
    return hist


# =====================================================================
#  ГЛАВНАЯ ФУНКЦИЯ
# =====================================================================

def main():
    t_global = time.time()

    # ── Конфигурация ──
    CFG = dict(
        n_epochs     = 25,
        n_init       = 500,
        n_add        = 150,
        lam0         = 0.005,    # базовое затухание
        eps          = 1.5,      # порог фазового совпадения
        max_conn     = 5,        # бюджет связей на узел
        n_cand       = 80,       # пул кандидатов
        path_sample  = 120,      # BFS-источников для оценки ⟨d⟩
        n_sim        = 15,       # Monte Carlo реализаций
    )
    N_TOTAL = CFG['n_init'] + CFG['n_epochs'] * CFG['n_add']

    MODELS = ['null', 'constant', 'feedback']
    LABELS = {
        'null':     'A: λ=0 (нет затухания)',
        'constant': f'B: λ={CFG["lam0"]} (постоянное)',
        'feedback': 'C: λ ∝ ⟨d⟩/⟨d⟩₀ (обратная связь)',
    }
    COLORS = {'null': '#2ca02c', 'constant': '#1f77b4', 'feedback': '#d62728'}
    SEED_OFFSETS = {'null': 0, 'constant': 10000, 'feedback': 20000}

    # ── Заголовок ──
    print("=" * 72)
    print("  ГРАФОВАЯ КОСМОЛОГИЯ v3: СРАВНИТЕЛЬНЫЙ ЭКСПЕРИМЕНТ")
    print("  Null (λ=0)  vs  Constant (λ=λ₀)  vs  Feedback (λ ∝ ⟨d⟩)")
    print("=" * 72)
    print("  Параметры:")
    for k, v in CFG.items():
        print(f"    {k:15s} = {v}")
    print(f"    {'N_total':15s} = {N_TOTAL}")
    print(f"    {'модели':15s} = {', '.join(MODELS)}")
    print("=" * 72)

    # ── Фазы (общие для всех моделей) ──
    print(f"\n[1/4] Генерация {CFG['n_sim']} фазовых ландшафтов ({N_TOTAL} × 6D)...")
    all_phases = [generate_phases(N_TOTAL, seed=42 + s) for s in range(CFG['n_sim'])]

    # ── Запуск моделей ──
    results = {m: [] for m in MODELS}

    for m in MODELS:
        print(f"\n[2/4] ── {LABELS[m]} ──")
        for s in range(CFG['n_sim']):
            np.random.seed(SEED_OFFSETS[m] + s)   # воспроизводимость
            t1 = time.time()
            h = run_single(
                model_type  = m,
                phases      = all_phases[s],
                n_epochs    = CFG['n_epochs'],
                n_init      = CFG['n_init'],
                n_add       = CFG['n_add'],
                lam0        = CFG['lam0'],
                eps         = CFG['eps'],
                max_conn    = CFG['max_conn'],
                n_cand      = CFG['n_cand'],
                path_sample = CFG['path_sample'],
            )
            dt = time.time() - t1
            results[m].append(h)

            if (s + 1) % 5 == 0:
                print(f"    sim {s+1:2d}/{CFG['n_sim']}  "
                      f"⟨d⟩_last={h['path'][-1]:6.2f}  "
                      f"λ_last={h['lam'][-1]:.5f}  "
                      f"⟨k⟩_last={h['deg'][-1]:.2f}  "
                      f"({dt:.1f}s)")

    elapsed = time.time() - t_global
    print(f"\n[3/4] Все симуляции завершены за {elapsed:.0f}s ({elapsed/60:.1f} мин)")

    # ── Агрегация ──
    def agg(model_key, field):
        data = np.array([r[field] for r in results[model_key]])
        return data.mean(axis=0), data.std(axis=0), data

    epochs = np.arange(CFG['n_epochs'])

    # ── Графики ──
    print("[4/4] Построение графиков...")
    fig, ax = plt.subplots(2, 3, figsize=(20, 12))

    # (0,0) Средний путь
    for m in MODELS:
        mu, sig, _ = agg(m, 'path')
        ax[0, 0].plot(epochs, mu, 'o-', color=COLORS[m], lw=2, ms=3, label=LABELS[m])
        ax[0, 0].fill_between(epochs, mu - sig, mu + sig, color=COLORS[m], alpha=0.12)
    ax[0, 0].set(title='Средний кратчайший путь ⟨d⟩', xlabel='Эпоха', ylabel='⟨d⟩')
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].grid(True, ls='--', alpha=0.4)

    # (0,1) Кластеризация
    for m in MODELS:
        mu, sig, _ = agg(m, 'clust')
        ax[0, 1].plot(epochs, mu, 's-', color=COLORS[m], lw=2, ms=3, label=LABELS[m])
        ax[0, 1].fill_between(epochs, mu - sig, mu + sig, color=COLORS[m], alpha=0.12)
    ax[0, 1].set(title='Кластеризация C', xlabel='Эпоха', ylabel='C')
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].grid(True, ls='--', alpha=0.4)

    # (0,2) Lambda
    for m in ['constant', 'feedback']:
        mu, sig, _ = agg(m, 'lam')
        ax[0, 2].plot(epochs, mu, 'D-', color=COLORS[m], lw=2, ms=3, label=LABELS[m])
        ax[0, 2].fill_between(epochs, mu - sig, mu + sig, color=COLORS[m], alpha=0.12)
    ax[0, 2].set(title='Параметр затухания λ(t)', xlabel='Эпоха', ylabel='λ')
    ax[0, 2].legend(fontsize=8)
    ax[0, 2].grid(True, ls='--', alpha=0.4)

    # (1,0) Скорость
    for m in MODELS:
        mu, sig, _ = agg(m, 'vel')
        ax[1, 0].plot(epochs, mu, '-', color=COLORS[m], lw=2, label=LABELS[m])
        ax[1, 0].fill_between(epochs, mu - sig, mu + sig, color=COLORS[m], alpha=0.10)
    ax[1, 0].axhline(0, color='black', ls='--', alpha=0.5)
    ax[1, 0].set(title='Скорость d⟨d⟩/dt', xlabel='Эпоха', ylabel='v')
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].grid(True, ls='--', alpha=0.4)

    # (1,1) Ускорение (сглаженное)
    for m in MODELS:
        mu, sig, _ = agg(m, 'acc')
        w = 3
        sm = np.convolve(mu, np.ones(w) / w, mode='same')
        sm[0], sm[-1] = mu[0], mu[-1]
        ax[1, 1].plot(epochs, sm, '^-', color=COLORS[m], lw=2, ms=3, label=LABELS[m])
        ax[1, 1].fill_between(epochs, sm - sig, sm + sig, color=COLORS[m], alpha=0.08)
    ax[1, 1].axhline(0, color='black', ls='--', alpha=0.5)
    ax[1, 1].set(title='Ускорение d²⟨d⟩/dt² (сглаж.)', xlabel='Эпоха', ylabel='a')
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].grid(True, ls='--', alpha=0.4)

    # (1,2) Средняя степень
    for m in MODELS:
        mu, sig, _ = agg(m, 'deg')
        ax[1, 2].plot(epochs, mu, 'v-', color=COLORS[m], lw=2, ms=3, label=LABELS[m])
        ax[1, 2].fill_between(epochs, mu - sig, mu + sig, color=COLORS[m], alpha=0.12)
    ax[1, 2].set(title='Средняя степень ⟨k⟩', xlabel='Эпоха', ylabel='⟨k⟩')
    ax[1, 2].legend(fontsize=8)
    ax[1, 2].grid(True, ls='--', alpha=0.4)

    fig.suptitle(
        f'Графовая космология v3 — сравнение моделей (N_sim={CFG["n_sim"]})',
        fontsize=15, y=1.01
    )
    plt.tight_layout()
    plt.savefig('graph_cosmology_v3.png', dpi=200, bbox_inches='tight')
    plt.show()

    # ═════════════════════════════════════════════════════════════════
    #  ИТОГОВАЯ ТАБЛИЦА
    # ═════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  ИТОГОВАЯ ТАБЛИЦА")
    print("=" * 80)

    header = (f"  {'Модель':<40s}│{'⟨d⟩₀':>6s}│{'⟨d⟩₂₄':>7s}│"
              f"{'Δ⟨d⟩':>7s}│{'⟨a⟩₁₅₋₂₄':>9s}│{'SEM':>7s}│{'sig':>4s}│")
    print(header)
    print("  " + "─" * 76)

    verdicts = {}
    for m in MODELS:
        mu_path, _, _ = agg(m, 'path')
        # Ускорение: по каждой реализации усреднить эпохи 15–24, затем статистика
        late_accs = np.array([r['acc'][15:].mean() for r in results[m]])
        late_mean = late_accs.mean()
        late_sem  = late_accs.std() / np.sqrt(CFG['n_sim'])
        sig_flag  = "**" if abs(late_mean) > 2 * late_sem else ""
        verdicts[m] = (late_mean, late_sem)

        print(f"  {LABELS[m]:<40s}│{mu_path[0]:6.2f}│{mu_path[-1]:7.2f}│"
              f"{mu_path[-1] - mu_path[0]:+7.2f}│{late_mean:+9.4f}│"
              f"{late_sem:7.4f}│{sig_flag:>4s}│")

    print("\n  ** = значимо на уровне 2σ  (|mean| > 2 × SEM)")

    # ═════════════════════════════════════════════════════════════════
    #  АВТОМАТИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ
    # ═════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ")
    print("=" * 80)

    feed_mean, feed_sem = verdicts['feedback']
    null_mean, null_sem = verdicts['null']
    const_mean, const_sem = verdicts['constant']

    feed_accel  = feed_mean > 2 * feed_sem     # ускорение значимо?
    null_accel  = null_mean > 2 * null_sem
    const_accel = const_mean > 2 * const_sem

    if feed_accel and not null_accel:
        print("""
  ► РЕЗУЛЬТАТ: Модель C (обратная связь) даёт статистически значимое
    ускоренное расширение, ОТСУТСТВУЮЩЕЕ в нулевой модели A.

    → Ускорение — эмерджентное свойство положительной обратной связи
      λ ∝ ⟨d⟩. Растущие пути → большее затухание → меньше шорткатов
      → ещё более длинные пути. Это кандидат на графовый аналог
      «тёмной энергии».

    ⚠ ОГРАНИЧЕНИЯ (почему это НЕ доказательство):
    • Это свойство ГРАФА, а не Вселенной
    • Для физического вывода необходимо:
      1) Получить уравнения Фридмана как непрерывный предел модели
      2) Подогнать a(t) под данные сверхновых (Pantheon+)
      3) Предсказать наблюдаемый эффект, отличный от ΛCDM
      4) Объяснить, почему именно линейная обратная связь λ ∝ ⟨d⟩
""")
    elif feed_accel and null_accel:
        print("""
  ► РЕЗУЛЬТАТ: Ускорение наблюдается во ВСЕХ моделях, включая нулевую (A).

    → Это свойство ЛЮБОГО растущего графа с хронологической цепочкой,
      а НЕ специфическое следствие механизма затухания.
    → Модель НЕ объясняет тёмную энергию.
    → Необходимо пересмотреть базовую конструкцию графа.
""")
    elif not feed_accel and not null_accel:
        print("""
  ► РЕЗУЛЬТАТ: НИ ОДНА модель не даёт значимого ускоренного расширения.

    → Линейная обратная связь λ ∝ ⟨d⟩/⟨d⟩₀ недостаточна для
      воспроизведения аналога тёмной энергии при данных параметрах.
    → Возможные направления:
      a) Увеличить λ₀ (усилить базовое затухание)
      b) Использовать нелинейную связь: λ ∝ (⟨d⟩/⟨d⟩₀)²
      c) Изменить механизм связывания (каузальный горизонт)
""")
    elif not feed_accel and null_accel:
        print("""
  ► РЕЗУЛЬТАТ: Нулевая модель (A) ускоряется, а модель C — нет.

    → Обратная связь подавляет ускорение, а не усиливает его.
    → Это контр-интуитивно: λ ∝ ⟨d⟩ приводит к стабилизации пути.
    → Возможное объяснение: при большом пути λ растёт, что
      уменьшает число шорткатов, но также уменьшает и средний путь.
""")
    else:
        print("""
  ► РЕЗУЛЬТАТ: Неопределённый — требуется анализ вручную.
""")

    print("\n" + "=" * 80)
    print("  ВРЕМЯ ВЫПОЛНЕНИЯ")
    print("=" * 80)
    total_time = time.time() - t_global
    print(f"  Общее время: {total_time:.1f}s ({total_time/60:.1f} мин)")
    print("=" * 80)


if __name__ == "__main__":
    main()
