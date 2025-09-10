
Обход в глубину/ширину (DFS/BFS)

Алгоритмы поиска кратчайшего пути (Дейкстра, Беллмана-Форда, Флойда-Уоршелла)

Топологическая сортировка

Обнаружение циклов

Связные компоненты

Минимальное остовное дерево (MST)

Union-Find (Объединение-поиск, DSU)

Задачи на графах на основе сетки

Раскраска графа

Сильно связные компоненты (SCC)

Эйлеровы и Гамильтоновы пути


## 1. Обход в глубину/ширину (DFS/BFS)

```go
package main

import (
    "fmt"
)


DFS(graph, v, used):
    stack q
    q.push(v)
    used[v] = 1
    while(!q.empty())
      v = q.popfront()
      for (var to : graph[v]):
        if (!used[to]):
          used[to] = true
          q.push(to)

BFS(graph, v, used):
    queue q
    q.push(v)
    used[v] = 1
    while(!q.empty())
      v = q.front()
      for (var to : graph[v]):
        if (!used[to]):
          used[to] = true
          q.push(to)

// BFS - обход в ширину
func BFS(graph map[int][]int, start int) {
    visited := make(map[int]bool)
    queue := []int{start}
    visited[start] = true
    
    for len(queue) > 0 {
        current := queue[0]
        queue = queue[1:]
        fmt.Printf("%d ", current)
        
        for _, neighbor := range graph[current] {
            if !visited[neighbor] {
                visited[neighbor] = true
                queue = append(queue, neighbor)
            }
        }
    }
}

func main() {
    graph := map[int][]int{
        0: {1, 2},
        1: {0, 3, 4},
        2: {0, 5},
        3: {1},
        4: {1},
        5: {2},
    }
    
    fmt.Println("DFS:")
    DFS(graph, 0, make(map[int]bool))
    
    fmt.Println("\nBFS:")
    BFS(graph, 0)
}
```

## 2. Алгоритм Дейкстры

```go
package main

import (
    "container/heap"
    "fmt"
    "math"
)

// Item для приоритетной очереди
type Item struct {
    node     int
    priority int
    index    int
}

// PriorityQueue реализация
type PriorityQueue []*Item

func (pq PriorityQueue) Len() int { return len(pq) }

func (pq PriorityQueue) Less(i, j int) bool {
    return pq[i].priority < pq[j].priority
}

func (pq PriorityQueue) Swap(i, j int) {
    pq[i], pq[j] = pq[j], pq[i]
    pq[i].index = i
    pq[j].index = j
}

func (pq *PriorityQueue) Push(x interface{}) {
    n := len(*pq)
    item := x.(*Item)
    item.index = n
    *pq = append(*pq, item)
}

func (pq *PriorityQueue) Pop() interface{} {
    old := *pq
    n := len(old)
    item := old[n-1]
    old[n-1] = nil
    item.index = -1
    *pq = old[0 : n-1]
    return item
}

func dijkstra(graph map[int]map[int]int, start int) map[int]int {
    dist := make(map[int]int)
    for node := range graph {
        dist[node] = math.MaxInt32
    }
    dist[start] = 0
    
    pq := make(PriorityQueue, 0)
    heap.Push(&pq, &Item{node: start, priority: 0})
    
    for pq.Len() > 0 {
        item := heap.Pop(&pq).(*Item)
        u := item.node
        
        for v, weight := range graph[u] {
            if dist[u]+weight < dist[v] {
                dist[v] = dist[u] + weight
                heap.Push(&pq, &Item{node: v, priority: dist[v]})
            }
        }
    }
    
    return dist
}

func main() {
    graph := map[int]map[int]int{
        0: {1: 4, 2: 1},
        1: {3: 1},
        2: {1: 2, 3: 5},
        3: {},
    }
    
    dist := dijkstra(graph, 0)
    fmt.Println("Кратчайшие расстояния:", dist)
}
```

# поиск пути в лабиринте с bfs

```python
import sys

from collections import deque

n, m = map(int, input().split())

arr = [[0 for _ in range(m)] for _ in range(n)]

startX = 0
startY = 0

endX = 0
endY = 0

for i in range(n):
    str = input()
    for j in range(m):
        if str[j] == '#'
            arr[i][j] =  0
        if  str[i] == 'S'
            startX = i
            startY = j
        if  str[i] == 'F'
            endX = i
            endY = j

visited = [[False for _ in range(m)] for _ in range(n)]

queue = deque[(startX, startY, '')]

visisted[startX][startY] = True

result = ""

def find_visited():

    while queue:
        sX, sY, step = queue.popleft()

        if sX == endX && sY == endY:
            return step
        if sX + 1 < n and arr[sX+1] == '.' && not visited[sX+1][sY]:
            # R
            queue.push((sX+1,sY, step + "R"))
            visited[sX+1][sY] = True
        if sY + 1 < m and arr[sX][sY+1] == '.' && not visited[sX][sY+1]:
            # D
            queue.push((sX,sY+1,step + "D"))
            visited[sX][sY+1] = True
        if sY - 1 >= 0 and arr[sX][sY-1] == '.' && not visited[sX][sY-1]:
            # U
            queue.push((sX,sY-1,step + "U"))
            visited[sX][sY-1] = True
        if sX - 1 >= n and arr[sX-1] == '.' && not visited[sX-1][sY]:
            # R
            queue.push((sX-1,sY,step + "L"))
            visited[sX-1][sY] = True

steps = find_visited()

if steps != "":
    print(len(steps))
    print(steps)
else:
    print(-1)
```

# поиск медианы двух массивов

```python

def findMedianSortedArrays(nums1, nums2):
    total = len(nums1) + len(nums2)
    half = total // 2
    
    # Указатели для обхода
    i = j = 0
    prev = current = 0
    
    # Проходим до середины объединенного массива
    for _ in range(half + 1):
        prev = current
        
        # Выбираем меньший элемент
        if i < len(nums1) and (j >= len(nums2) or nums1[i] < nums2[j]):
            current = nums1[i]
            i += 1
        else:
            current = nums2[j]
            j += 1
    
    if total % 2 == 0:
        return (prev + current) / 2
    else:
        return current
```

# скобочные последовательности

```go
func generate(current string, open, close, n int, result *[]string) {
	if len(current) == 2*n {
		*result = append(*result, current)
		return
	}
	if open < n {
		// пока есть открытые скобки, можно добавить еще одну
		generate(current+"(", open+1, close, n, result)
	}
	if close < open {
		generate(current+")", open, close+1, n, result)
	}
}
```

# является ли степенью двойки

```go
def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0
```

# вероятность исчерпания супов

You have two soups, A and B, each starting with n mL. On every turn, one of the following four serving operations is chosen at random, each with probability 0.25 independent of all previous turns:

pour 100 mL from type A and 0 mL from type B
pour 75 mL from type A and 25 mL from type B
pour 50 mL from type A and 50 mL from type B
pour 25 mL from type A and 75 mL from type B
Note:

There is no operation that pours 0 mL from A and 100 mL from B.
The amounts from A and B are poured simultaneously during the turn.
If an operation asks you to pour more than you have left of a soup, pour all that remains of that soup.
The process stops immediately after any turn in which one of the soups is used up.

Return the probability that A is used up before B, plus half the probability that both soups are used up in the same turn. Answers within 10-5 of the actual answer will be accepted.

```go
package main

import (
	"fmt"
)

func soupServings(n int) float64 {
    if n >= 5000 {
        return 1.0
    }
    memo := make(map[[2]int]float64)
    var dp func(int, int) float64
    dp = func(a, b int) float64 {
        if a <= 0 && b <= 0 {
            return 0.5
        }
        if a <= 0 {
            return 1.0
        }
        if b <= 0 {
            return 0.0
        }
        key := [2]int{a, b}
        if val, ok := memo[key]; ok {
            return val
        }
        res := 0.25 * (dp(a-100, b) + dp(a-75, b-25) + dp(a-50, b-50) + dp(a-25, b-75))
        memo[key] = res
        return res
    }
    return dp(n, n)
}

func main() {
    fmt.Println(soupServings(50))   // Пример вывода
    fmt.Println(soupServings(100))  // Пример вывода
}
```

# степень тройки
```python
def is_power_of_three(n):
    if n < 1:
        return False
    while n % 3 == 0:
        n //= 3
    return n == 1
```

# найти позицию для вставки

```go
func abs(x int) int {
  if x < 0 {
    return -x
  }

  return x
}

func neghbor(goods []uint, need int) int {

  left := 0
  right := len(goods)

  for {
    mid := (right + left) / 2
    switch {
    case left == right:
      return left
    case need > int(goods[mid]):
      left = mid + 1
    default:
      right = mid
    }
  }
}
```

# heap
```go
// Определяем тип для приоритетной очереди (max-heap)
type Class struct {
  pass  int
  total int
}

// Определяем кучу (heap) для классов
type MaxHeap []Class

// Реализуем методы интерфейса heap.Interface
func (h MaxHeap) Len() int { return len(h) }
func (h MaxHeap) Less(i, j int) bool {
  // Сравниваем по marginal gain (прирост процента сдачи)
  gainI := gain(h[i].pass, h[i].total)
  gainJ := gain(h[j].pass, h[j].total)
  return gainI > gainJ // Max-heap: больший gain имеет высший приоритет
}
func (h MaxHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }

func (h *MaxHeap) Push(x interface{}) {
  *h = append(*h, x.(Class))
}

func (h *MaxHeap) Pop() interface{} {
  old := *h
  n := len(old)
  x := old[n-1]
  *h = old[0 : n-1]
  return x
}
```


# обход дерева

```go
package main

import (
    "fmt"
)

// Узел бинарного дерева
type TreeNode struct {
    Value int
    Left  *TreeNode
    Right *TreeNode
}

// Структура для хранения результата
type Result struct {
    Node     *TreeNode
    Branches int
}

// Функция для поиска узла с максимальным количеством веток (итеративная версия)
func findNodeWithMaxBranches(root *TreeNode) *Result {
    if root == nil {
        return nil
    }
    
    result := &Result{Node: nil, Branches: -1}
    stack := []*TreeNode{root}
    
    for len(stack) > 0 {
        // Извлекаем узел из стека
        node := stack[len(stack)-1]
        stack = stack[:len(stack)-1]
        
        // Считаем количество веток у текущего узла
        branches := 0
        if node.Left != nil {
            branches++
            stack = append(stack, node.Left)
        }
        if node.Right != nil {
            branches++
            stack = append(stack, node.Right)
        }
        
        // Обновляем результат если нашли узел с большим количеством веток
        if branches > result.Branches {
            result.Node = node
            result.Branches = branches
        }
    }
    
    return result
}

// Альтернативная версия с обходом в ширину (BFS)
func findNodeWithMaxBranchesBFS(root *TreeNode) *Result {
    if root == nil {
        return nil
    }
    
    result := &Result{Node: nil, Branches: -1}
    queue := []*TreeNode{root}
    
    for len(queue) > 0 {
        // Извлекаем узел из очереди
        node := queue[0]
        queue = queue[1:]
        
        // Считаем количество веток у текущего узла
        branches := 0
        if node.Left != nil {
            branches++
            queue = append(queue, node.Left)
        }
        if node.Right != nil {
            branches++
            queue = append(queue, node.Right)
        }
        
        // Обновляем результат если нашли узел с большим количеством веток
        if branches > result.Branches {
            result.Node = node
            result.Branches = branches
        }
    }
    
    return result
}

// Вспомогательная функция для создания тестового дерева
func createTree() *TreeNode {
    return &TreeNode{
        Value: 1,
        Left: &TreeNode{
            Value: 2,
            Left:  &TreeNode{Value: 4},
            Right: &TreeNode{Value: 5},
        },
        Right: &TreeNode{
            Value: 3,
            Left: &TreeNode{
                Value: 6,
                Right: &TreeNode{Value: 7},
            },
            Right: &TreeNode{Value: 8},
        },
    }
}

// Вспомогательная функция для печати дерева (для наглядности)
func printTree(node *TreeNode, prefix string, isTail bool) {
    if node == nil {
        return
    }
    
    fmt.Println(prefix + "├── " + fmt.Sprintf("%d", node.Value))
    
    newPrefix := prefix
    if isTail {
        newPrefix += "    "
    } else {
        newPrefix += "│   "
    }
    
    if node.Left != nil && node.Right != nil {
        printTree(node.Left, newPrefix, false)
        printTree(node.Right, newPrefix, true)
    } else if node.Left != nil {
        printTree(node.Left, newPrefix, true)
    } else if node.Right != nil {
        printTree(node.Right, newPrefix, true)
    }
}

func main() {
    tree := createTree()
    
    fmt.Println("Бинарное дерево:")
    printTree(tree, "", true)
    fmt.Println()
    
    // DFS версия
    resultDFS := findNodeWithMaxBranches(tree)
    fmt.Printf("DFS: Узел с максимальным количеством веток: %d (веток: %d)\n", 
        resultDFS.Node.Value, resultDFS.Branches)
    
    // BFS версия
    resultBFS := findNodeWithMaxBranchesBFS(tree)
    fmt.Printf("BFS: Узел с максимальным количеством веток: %d (веток: %d)\n", 
        resultBFS.Node.Value, resultBFS.Branches)
    
    // Тест с деревом из одного узла
    singleNode := &TreeNode{Value: 42}
    resultSingle := findNodeWithMaxBranches(singleNode)
    fmt.Printf("Один узел: значение %d (веток: %d)\n", 
        resultSingle.Node.Value, resultSingle.Branches)
    
    // Тест с пустым деревом
    var emptyTree *TreeNode = nil
    resultEmpty := findNodeWithMaxBranches(emptyTree)
    if resultEmpty == nil {
        fmt.Println("Пустое дерево: результат nil")
    }
}
```

# поиск максимальной подстроки с невстречающимися символами

```go
func lengthOfLongestSubstring(s string) int {
	maxLen := 0
	left := 0
	charIndex := make(map[byte]int)

	for right := 0; right < len(s); right++ {
		if idx, exists := charIndex[s[right]]; exists && idx >= left {
			left = idx + 1
		}
		charIndex[s[right]] = right
		maxLen = max(maxLen, right-left+1)
	}

	return maxLen
}
```
