import random
import math
import time as tm
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import numpy as np
import seaborn as sns  # Импортируем seaborn для палитры, если потребуется

# Ввод данных
m = int(input("Введите количество заказов: "))
speed = int(input("Введите скорость всех курьеров (км/ч): "))
work_time = int(input("Введите время работы всех курьеров (часы): "))
print("Введите матрицу расстояний (m+1 строк по m+1 чисел, где строка 0 — склад):")
dist_matrix = []
for i in range(m + 1):
    row = list(map(float, input(f"Введите строку {i+1}: ").split()))
    dist_matrix.append(row)

dist_matrix = np.array(dist_matrix)

# Восстановление координат с помощью MDS
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
coords = mds.fit_transform(dist_matrix)

warehouse_index = 0
orders_indices = list(range(1, m + 1))
max_orders_per_courier = m  # Максимальное количество заказов на курьера ( = кол-ву заказов )
print("Введите начальное количество курьеров: АВТОМАТИЧЕСКИ ПРИСВОЕНО ЧИСЛО, РАВНОЕ КОЛИЧЕСТВУ ЗАКАЗОВ")
n = m

# Создание графа с помощью networkx
G = nx.DiGraph()
for i in range(m + 1):
    G.add_node(i, label='Склад' if i == 0 else f'{i}')
    for j in range(m + 1):
        if i != j:
            G.add_edge(i, j, weight=dist_matrix[i][j])

# Позиционирование вершин на основе восстановленных координат
pos = {i: (coords[i, 0], coords[i, 1]) for i in range(m + 1)}

start_time = tm.time()

# Жадный алгоритм для построения маршрута
def greedy_route(courier_order_indices, warehouse_index, dist_matrix):
    if not courier_order_indices:
        return [], 0
    route = []
    current = warehouse_index
    total_distance = 0
    unvisited = list(courier_order_indices)
    while unvisited:
        next_order = min(unvisited, key=lambda idx: dist_matrix[current][idx])
        route.append(next_order)
        total_distance += dist_matrix[current][next_order]
        current = next_order
        unvisited.remove(next_order)
    total_distance += dist_matrix[current][warehouse_index]  # Возвращение на склад
    return route, total_distance

# Время доставки для курьера
def calculate_time(courier_order_indices, warehouse_index, dist_matrix, speed):
    route, total_distance = greedy_route(courier_order_indices, warehouse_index, dist_matrix)
    delivery_time = total_distance / speed
    return delivery_time, route

# Функция приспособленности
def fitness(chromosome, n, orders_indices, warehouse_index, dist_matrix, speed, work_time, max_orders_per_courier):
    courier_assignments = [[] for _ in range(n)]
    for order_idx, courier in enumerate(chromosome):
        courier_assignments[courier].append(orders_indices[order_idx])
    
    total_time_excess = 0
    used_courier_count = sum(1 for ca in courier_assignments if ca)  # Количество занятых курьеров
    total_time = 0
    penalty = 0
    
    for ca in courier_assignments:
        if len(ca) > max_orders_per_courier:  # Штраф за превышение количества заказов
            penalty += 1000
        if ca:
            delivery_time, _ = calculate_time(ca, warehouse_index, dist_matrix, speed)
            total_time += delivery_time  # Учитываем общее время
            if delivery_time > work_time:
                total_time_excess += (delivery_time - work_time)  # Штраф за превышение времени
    
    if total_time_excess > 0 or penalty > 0:
        return 10000 + 100 * total_time_excess + penalty + used_courier_count
    
    # Минимизируем количество курьеров и общее время
    return used_courier_count + 0.1 * total_time

# Генетический алгоритм с элитизмом
def genetic_algorithm(n, orders_indices, warehouse_index, dist_matrix, speed, work_time, max_orders_per_courier):
    pop_size = 200  # Размер популяции
    generations = 5000  # Количество поколений
    mutation_rate = 0.02  # Вероятность мутации
    elite_size = int(pop_size * 0.1)  # Сохранение 10% лучших решений
    
    population = [[random.randint(0, n-1) for _ in range(len(orders_indices))] for _ in range(pop_size)]
    for generation in range(generations):
        fitness_scores = [fitness(chrom, n, orders_indices, warehouse_index, dist_matrix, speed, work_time, max_orders_per_courier) for chrom in population]
        best_score = min(fitness_scores)
        
        if best_score < 1000:  # Решение найдено
            best_chrom = population[fitness_scores.index(best_score)]
            ca = [[] for _ in range(n)]
            for idx, c in enumerate(best_chrom):
                ca[c].append(orders_indices[idx])
            used = sum(1 for x in ca if x)
            print(f"НАЙДЕНО РЕШЕНИЕ: Поколение {generation}, курьеров: {used}, лучшее приспособление: {best_score}")
            return True, best_chrom
        
        # Элитизм: сохраняем лучшие решения
        sorted_population = sorted(zip(population, fitness_scores), key=lambda x: x[1])
        new_population = [x[0] for x in sorted_population[:elite_size]]
        
        # Турнирная селекция для остальной популяции
        while len(new_population) < pop_size:
            tournament = random.sample(list(zip(population, fitness_scores)), 3)
            winner = min(tournament, key=lambda x: x[1])[0]
            new_population.append(winner[:])
        
        # Кроссовер
        for i in range(0, pop_size, 2):
            if i + 1 < pop_size and random.random() < 0.8:
                point1 = random.randint(1, len(orders_indices) - 2)
                point2 = random.randint(point1 + 1, len(orders_indices) - 1)
                parent1, parent2 = new_population[i], new_population[i + 1]
                new_population[i] = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
                new_population[i + 1] = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        
        # Мутация
        for chrom in new_population:
            for j in range(len(chrom)):
                if random.random() < mutation_rate:
                    chrom[j] = random.randint(0, n-1)
        
        population = new_population
        
        # Вывод прогресса
        if generation % 50 == 0 or generation < 100:
            best_chrom = population[fitness_scores.index(best_score)]
            ca = [[] for _ in range(n)]
            for idx, c in enumerate(best_chrom):
                ca[c].append(orders_indices[idx])
            used = sum(1 for x in ca if x)
            print(f"Поколение {generation}, курьеров: {used}, лучшее приспособление: {best_score}")
    
    return False, None

low = 1
high = m  # Максимальное количество курьеров равно количеству заказов
min_couriers = None
best_chrom = None

while low <= high:
    mid = (low + high) // 2
    print(f"\nПроверка с {mid} курьерами")
    success, chrom = genetic_algorithm(mid, orders_indices, warehouse_index, dist_matrix, speed, work_time, max_orders_per_courier)
    if success:
        min_couriers = mid
        best_chrom = chrom
        high = mid - 1
    else:
        low = mid + 1

# Обработка результата
if min_couriers is not None:
    # Распределение заказов
    courier_assignments = [[] for _ in range(min_couriers)]
    for order_idx, courier in enumerate(best_chrom):
        courier_assignments[courier].append(orders_indices[order_idx])
    
    # Подсчитываем реальное количество занятых курьеров
    actual_used_couriers = sum(1 for ca in courier_assignments if ca)
    print(f"\nМинимальное количество курьеров: {actual_used_couriers}")
    
    # Оптимальные маршруты для каждого курьера
    best_routes = []
    for ca in courier_assignments:
        if ca:
            _, route = calculate_time(ca, warehouse_index, dist_matrix, speed)
            best_routes.append(route)
        else:
            best_routes.append([])
    
    # Вывод маршрутов
    courier_number = 1
    for ca, route in zip(courier_assignments, best_routes):
        if ca:
            delivery_time, _ = calculate_time(ca, warehouse_index, dist_matrix, speed)
            route_str = ' -> '.join([f'Заказ {idx}' for idx in route])
            print(f"Курьер {courier_number}: склад -> {route_str} -> склад, время: {delivery_time:.2f} ч")
            courier_number += 1
    
    end_time = tm.time()
    execution_time = end_time - start_time
    print(f"Время выполнения программы: {execution_time:.2f} секунд")
    
    # Отрисовка маршрутов курьеров
    used_routes = [route for route in best_routes if route]
    num_used = len(used_routes)
    
    plt.figure(figsize=(18, 15))
    if num_used <= 10:
        colors = plt.cm.tab10(range(num_used))
    elif num_used <= 20:
        colors = plt.cm.tab20(range(num_used))
    else:
        colors = sns.color_palette("husl", num_used)
    
    for idx, route in enumerate(used_routes):
        full_route = [warehouse_index] + route + [warehouse_index]
        route_edges = [(full_route[j], full_route[j+1]) for j in range(len(full_route) - 1)]
        edge_colors = [colors[idx]] * len(route_edges)
        nx.draw_networkx_edges(G, pos, edgelist=route_edges, edge_color=edge_colors,
                               width=2, label=f'Курьер {idx+1}', arrows=True)
        route_edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in route_edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=route_edge_labels, font_color='black', font_size=6)
    
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=100)
    nx.draw_networkx_labels(G, pos, labels={i: ('Склад' if i == 0 else str(i)) for i in range(m+1)}, font_size=10)
    
    handles = [plt.Line2D([0], [0], color=colors[idx], lw=2, label=f'Курьер {idx+1}') 
               for idx in range(num_used)]
    plt.legend(handles=handles)
    plt.title("Маршруты курьеров (с учетом масштаба)")
    plt.show()
    
else:
    end_time = tm.time()
    execution_time = end_time - start_time
    print(f"Время выполнения программы: {execution_time:.2f} секунд")
    print("Невозможно доставить все заказы с заданными параметрами.")
