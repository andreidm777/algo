 # Пусть \inline n городов, между ними существуют пути, заданные массивом edges[i] = [city_a, city_b, distance_ab], а также дано целое число mileage.


# Требуется найти такой город из данных, из которого можно добраться до наименьшего числа городов не превысив mileage.
def least_reachable_city(n, edges, mileage):
        """
        входные параметры:
            n --- количество городов,
            edges --- тройки (a, b, distance_ab),
            mileage --- максимально допустимое расстояние между городами 
            для соседства
        """
        # заполняем список смежности (adjacency list), в нашем случае это 
        # словарь, в котором ключи это города, а значения --- пары 
        # (<другой_город>, <расстояние_до_него>)

        graph = {}
        for u, v, w in edges:
            if graph.get(u, None) is None:
                graph[u] = [(v, w)]
            else:
                graph[u].append((v, w))
            if graph.get(v, None) is None:
                graph[v] = [(u, w)]
            else:
                graph[v].append((u, w))
        
        # локально объявим функцию, которая будет считать кратчайшие пути в 
        # графе от вершины, до всех вершин, удовлетворяющих условию
        def num_reachable_neighbors(city):
            # создаем кучу, из одного элемента с парой, задающей нулевую 
            # длину пути до самого исходного города
            heap = [(0, city)]
            # и массив, содержащий города и кратчайшие 
            # расстояния до них от исходного
            distances = {}
            # затем, пока куча не пуста, извлекаем ближайший 
            # от посещенных городов город
            while heap:
                currDist, neighb = heapq.heappop(heap)
                # если кратчайшее ребро ведет к городу, где мы уже знаем 
                # оптимальный маршрут, то завершаем итерацию
                if neighb in distances:
                    continue
                # в остальных случаях, и если сосед не является отправным 
                # городом, мы добавляем новую запись в массив кратчайших расстояний
                if neighb != city:    
                    distances[neighb] = currDist
                # обрабатываем всех смежных городов с соседом, добавляя их в кучу 
                # но только если: а) до них еще не известен кратчайший маршрут и б) путь до них через neighb не выходит за пределы mileage
                for node, d in graph[neighb]:
                    if node in distances:
                        continue
                    if currDist + d <= mileage:
                        heapq.heappush(heap, (currDist + d, node))
            # возвращаем количество городов, прошедших проверку
            return len(distances)
        
        # выполним поиск соседей для каждого из городов
        cities_neighbors = {num_reachable_neighbors(city): city for city in range(n)}
        # вернём номер города, у которого наименьшее число соседей
        # в пределах досигаемости
        return cities_neighbors[min(cities_neighbors)]
