"""
Простая модель: БЭ как последовательный автомат
на дискретной решётке, цикл = t_P

Проверяем: воспроизводит ли усреднённое поле
статистику квантового вакуума?
"""

import numpy as np
import matplotlib.pyplot as plt

# Параметры
N_sites  = 1000        # точек пространства
N_cycles = 100000      # циклов (= t_P каждый)
noise    = 0.1         # мера неопределённости БЭ

# БЭ как последовательный обход + шум
field = np.zeros(N_sites)
field_history = []

rng = np.random.RandomState(42)

for cycle in range(N_cycles):
    # БЭ обходит все точки за один цикл
    order = rng.permutation(N_sites)  # случайный порядок

    for site in order:
        # "Посещение" сайта: обновляем поле
        field[site] = (field[site] * 0.99 +
                       noise * rng.randn())

    if cycle % 1000 == 0:
        field_history.append(field.copy())

field_history = np.array(field_history)

# Статистика
print("Статистика поля (усреднение за много циклов):")
print(f"  ⟨φ⟩ = {field.mean():.4f}  (ожидаем ≈ 0)")
print(f"  ⟨φ²⟩ = {(field**2).mean():.4f}")
print(f"  σ(φ) = {field.std():.4f}")

# Пространственная корреляция
corr = np.correlate(field - field.mean(),
                    field - field.mean(),
                    mode='full')
corr /= corr.max()

print(f"\nКорреляционная длина:")
half = np.where(corr[N_sites:] < 0.5)[0]
if len(half) > 0:
    print(f"  l_corr ≈ {half[0]} планковских единиц")

# Спектр мощности
fft = np.fft.rfft(field)
power = np.abs(fft)**2
freqs = np.fft.rfftfreq(N_sites)

# Наклон спектра
mask = freqs > 0
log_f = np.log(freqs[mask])
log_P = np.log(power[mask])
slope = np.polyfit(log_f, log_P, 1)[0]
print(f"\nСпектр мощности: P(k) ~ k^{slope:.2f}")
print(f"  (вакуум КТП: P(k) ~ k для 1D)")
