from collections import deque

def main():
    X, Y = map(int, input().split())
    
    # Если X уже равен Y, то операций не нужно
    if X == Y:
        print(0)
        return
        
    # Ограничим диапазон значений: от 0 до 2 * Y (или больше, но чтобы избежать бесконечности)
    # Установим верхнюю границу как Y * 10 + 1000, а нижнюю как 0 (так как отрицательные могут быть не нужны)
    max_val = max(Y * 10 + 1000, 100000)
    min_val = 0
    
    visited = set()
    queue = deque()
    queue.append((X, 0))
    visited.add(X)
    
    while queue:
        current, steps = queue.popleft()
        
        # Перебираем все цифры от 0 до 9
        for c in range(0, 10):
            # Операция сложения
            next_val = current + c
            if next_val == Y:
                print(steps + 1)
                return
            if min_val <= next_val <= max_val and next_val not in visited:
                visited.add(next_val)
                queue.append((next_val, steps + 1))
                
            # Операция вычитания
            next_val = current - c
            if next_val == Y:
                print(steps + 1)
                return
            if min_val <= next_val <= max_val and next_val not in visited:
                visited.add(next_val)
                queue.append((next_val, steps + 1))
                
            # Операция умножения
            next_val = current * c
            if next_val == Y:
                print(steps + 1)
                return
            if min_val <= next_val <= max_val and next_val not in visited:
                visited.add(next_val)
                queue.append((next_val, steps + 1))
                
    # Если не нашли (хотя по условию должно быть решение)
    print(-1)

if __name__ == "__main__":
    main()