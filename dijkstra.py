import heapq

def dijkstra(graph, start_node):
    distances = {node: float('infinity') for node in graph} # расстояния до всех вершин
    distances[start_node] = 0 # расстояние до стартовой вершины равно 0

    previous_nodes = {node: None for node in graph} # предшественники вершин
    
    priority_queue = [(0, start_node)] # очередь с приоритетом (расстояние, вершина)
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue) # выбираем вершину с минимальным расстоянием

        # Пропускаем, если расстояние до вершины уже известно и меньше current_distance
        if current_distance > distances[current_node]:
            continue

        # Обходим соседей
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight # новое расстояние до соседа

            # Если найден более короткий путь
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor)) # добавляем соседа в очередь

    return distances, previous_nodes

# Пример использования
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}

distances, previous_nodes = dijkstra(graph, 'A')

print("Расстояния:", distances)
print("Предшественники:", previous_nodes)

def get_path(previous_nodes, start_node, end_node):
    path = []
    current_node = end_node
    while current_node is not None:
        path.append(current_node)
        current_node = previous_nodes[current_node]
    return path[::-1]

print("Кратчайший путь от A до D:", get_path(previous_nodes, 'A', 'D'))