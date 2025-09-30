
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

# обход дерева в ширину

```python
class Solution:
    class TreeNode:
        def __init__(self, val=0, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right

    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []

        result = []
        queue = [root]

        while queue:
            level_size = len(queue)
            level_nodes = []

            for i in range(level_size):
                node = queue.pop(0)
                level_nodes.append(node.val)

                if node.left:
                    queue.append(node.left)

                if node.right:
                    queue.append(node.right)

            result.append(level_nodes)

        return result
```

# минимальный путь
 ```python
 class Solution:
    def minDepth(self, root):
        if not root:
            return 0   # пусто
        
        stack = [(root, 1)] 
        minDepth = float("inf") № пока максимум
        
        while stack:
            node, depth = stack.pop()   # dfs
            
            # leaf
            if not node.left and not node.right:
                minDepth = min(minDepth, depth)
            
            # 
            if node.left:
                stack.append((node.left, depth + 1))
            
            # 
            if node.right:
                stack.append((node.right, depth + 1))
        
        return minDepth
#    import atexit; atexit.register(lambda: open("display_runtime.txt", "w").write("0"))
```

# перестановка гласных

Given a 0-indexed string s, permute s to get a new string t such that:

All consonants remain in their original places. More formally, if there is an index i with 0 <= i < s.length such that s[i] is a consonant, then t[i] = s[i].
The vowels must be sorted in the nondecreasing order of their ASCII values. More formally, for pairs of indices i, j with 0 <= i < j < s.length such that s[i] and s[j] are vowels, then t[i] must not have a higher ASCII value than t[j].
Return the resulting string.

The vowels are 'a', 'e', 'i', 'o', and 'u', and they can appear in lowercase or uppercase. Consonants comprise all letters that are not vowels.

```go
import "container/heap"

type Alpha struct {
    c   rune
    idx int
}

type MyHeap []Alpha

func (h MyHeap) Len() int {
    return len(h)
}

func (h MyHeap) Less(i, j int) bool {
    return h[i].c > h[j].c
}

func (h MyHeap) Swap(i, j int) {
    h[i], h[j] = h[j], h[i]
    x := h[i].idx
    h[i].idx = h[j].idx
    h[j].idx = x
}

func (h *MyHeap) Push(v any) {
    *h = append(*h, v.(Alpha))
}

func (h *MyHeap) Pop() any {
    old := *h
    x := old[len(old) - 1]
    *h = old[:len(old) - 1]
    return x
}

func isVowel(a rune) bool {
    all := "euioaEUIOA"
    return strings.ContainsRune(all, a)
}

func sortVowels(s string) string {
    h := make(MyHeap,0)
    for i, v := range s {
        if isVowel(v) {
            heap.Push(&h, Alpha{c:v,idx:i})
        }
    }

    res := []rune(s)

    for len(h) > 0 {
        v := heap.Pop(&h).(Alpha)
        res[v.idx] = v.c
    }
    
    return string(res)
}
```

2-й вариант

```go
func isVowel(r rune) bool {
    if r == 'A' || r == 'E' || r == 'O' || r == 'I' || r == 'U' || r == 'a' || r == 'e' || r == 'o' || r == 'i' || r == 'u' {
        return true
    }
    return false
}
func sortVowels(s string) string {
    vowels := make([]int, 123)

    for _, r := range s {
        if isVowel(r) {
            vowels[r]++
        }
    }
    
    var sb strings.Builder

    var p rune = 0
    for _, v := range s {
        if isVowel(v) {
            for vowels[p] == 0 {
                p++
            }
            sb.WriteRune(p)
            vowels[p]--
        } else {
            sb.WriteRune(v)      
        }
    }
    return sb.String()
}
```

# проверка BST на валидность

```go
// стеком
// Итеративное решение с использованием стека
func isValidBSTIterative(root *TreeNode) bool {
	if root == nil {
		return true
	}
	
	stack := []*TreeNode{}
	var prev *TreeNode = nil
	current := root
	
	for current != nil || len(stack) > 0 {
		// Добираемся до самого левого узла
		for current != nil {
			stack = append(stack, current)
			current = current.Left
		}
		
		// Извлекаем узел из стека
		current = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		
		// Проверяем порядок значений (должны идти в возрастающем порядке)
		if prev != nil && current.Val <= prev.Val {
			return false
		}
		
		prev = current
		current = current.Right
	}
	
	return true
}

// Рекурсивное решение с проверкой диапазона значений
func isValidBSTRecursive(root *TreeNode) bool {
	return validate(root, nil, nil)
}

func validate(node *TreeNode, min *int, max *int) bool {
	if node == nil {
		return true
	}
	
	// Проверяем, что значение узла находится в допустимом диапазоне
	if (min != nil && node.Val <= *min) || (max != nil && node.Val >= *max) {
		return false
	}
	
	// Рекурсивно проверяем левое и правое поддеревья
	return validate(node.Left, min, &node.Val) && 
	       validate(node.Right, &node.Val, max)
}

func isValid(node *TreeNode, low, high int64) bool {
    if node == nil {
        return true
    }
    // если текущая вершина не удволетворяет условиям BST то и все
    // дерево не является правильным BST и возвращаем false
    nodeValue := int64(node.Val)
    if low >= nodeValue || nodeValue >= high) {
        return false;
    }
    // обновляем минимальное и максимальное значение для поддеревьев
    return isValid(node.Left, low, nodeValue) &&
            isValid(node.Right, nodeValue, high)
}

func isValidBST(root *TreeNode) bool {
    return isValid(root, math.MinInt64, math.MaxInt64)
}
```

# из массива в дерево

```go
func buldBST(nums []int, l, r int) *TreeNode {
    if l > r {
        return nil
    }

    // вычисляем средний элемент между l и r
    mid := (l + r) / 2

    return &TreeNode{
        Val: nums[mid],
        // рекурсивно строим левое поддерево,
        // которое будет сбалансированно по высоте
        Left: buldBST(nums, l, mid - 1),
        // рекурсивно строим правое поддерево,
        // которое будет сбалансированно по высоте
        Right: buldBST(nums, mid + 1, r),
    }
}

func sortedArrayToBST(nums []int) *TreeNode {
    return buldBST(nums, 0, len(nums) - 1)
}
```

# К-ый наименьшый элемент в BST

```go
func inorder(node *TreeNode, k *int) *TreeNode {
    if node == nil {
        return nil
    }

    // идем влево
    result := inorder(node.Left, k)
    if result != nil {
        // если уже нашлий k-ую наименьшую
        return result
    }
    
    // проверяем k
    *k -= 1
    if (*k == 0) {
        // нашлий k-ую наименьшую
        return node
    }

    // идем вправо
    return inorder(node.Right, k)
}

func kthSmallest(root *TreeNode, k int) int {
    result := inorder(root, &k)
    return result.Val
}
```

# Дано два узла бинарного дерева p и q. Нужно вернуть их наименьшего общего предка (LCA). При этом нужно решить за O(1) по дополнительной памяти
Каждый узел помимо детей хранит ссылку на родительский узел
class Node {
    public int val;
    public Node left;
    public Node right;
    public Node parent;
}

```go
func depth(node *Node) int {
    result := 0
    for node != nil {
        result += 1
        node = node.Parent
    }
    return result
}

func lowestCommonAncestor(p *Node, q *Node) *Node {
    // находим глубину каждой вершины
    pDepth := depth(p)
    qDepth := depth(q)
    // в вершине p храним самую глубокую вершину
    if qDepth > pDepth {
        p, q = q, p
        pDepth, qDepth = qDepth, pDepth
    }
    // делаем "pDepth - qDepth" шагов вверх, чтобы
    // вершины имели одну глубину, ведь вершины могут совпать
    // только имея одинаковую глубину
    for i := 0; i < (pDepth - qDepth); i++ {
        p = p.Parent
    }
    // пока вершины не совпадут будем уменьшать глубину
    for p != q {
        p = p.Parent
        q = q.Parent
    }
    return p
}
```

# задача восстановление дерева где два узла поменены местами in-order обход

```go
package main

import "fmt"

// Определение структуры узла бинарного дерева
type TreeNode struct {
    Val   int
    Left  *TreeNode
    Right *TreeNode
}

func recoverTree(root *TreeNode) {
    // Переменные для хранения ошибочных узлов
    var first, second, prev *TreeNode
    
    // Вспомогательная функция для обхода в порядке in-order
    var traverse func(node *TreeNode)
    traverse = func(node *TreeNode) {
        if node == nil {
            return
        }
        
        // Рекурсивный обход левого поддерева
        traverse(node.Left)
        
        // Проверка нарушения порядка в in-order обходе
        if prev != nil && prev.Val > node.Val {
            // Если first еще не найден, сохраняем предыдущий узел
            if first == nil {
                first = prev
            }
            // Всегда сохраняем текущий узел как второй ошибочный
            second = node
        }
        
        // Обновляем предыдущий узел
        prev = node
        
        // Рекурсивный обход правого поддерева
        traverse(node.Right)
    }
    
    // Запускаем in-order обход
    traverse(root)
    
    // Меняем значения ошибочных узлов местами
    if first != nil && second != nil {
        first.Val, second.Val = second.Val, first.Val
    }
}

// Вспомогательная функция для создания дерева
func createTree(values []interface{}) *TreeNode {
    if len(values) == 0 || values[0] == nil {
        return nil
    }
    
    root := &TreeNode{Val: values[0].(int)}
    queue := []*TreeNode{root}
    i := 1
    
    for i < len(values) && len(queue) > 0 {
        current := queue[0]
        queue = queue[1:]
        
        // Левый потомок
        if i < len(values) && values[i] != nil {
            current.Left = &TreeNode{Val: values[i].(int)}
            queue = append(queue, current.Left)
        }
        i++
        
        // Правый потомок
        if i < len(values) && values[i] != nil {
            current.Right = &TreeNode{Val: values[i].(int)}
            queue = append(queue, current.Right)
        }
        i++
    }
    
    return root
}

// Функция для печати дерева (in-order)
func printInOrder(root *TreeNode) {
    if root == nil {
        return
    }
    printInOrder(root.Left)
    fmt.Printf("%d ", root.Val)
    printInOrder(root.Right)
}

func main() {
    // Пример 1: [1,3,null,null,2] -> [3,1,null,null,2]
    fmt.Println("Пример 1:")
    root1 := createTree([]interface{}{1, 3, nil, nil, 2})
    fmt.Print("До: ")
    printInOrder(root1)
    fmt.Println()
    
    recoverTree(root1)
    fmt.Print("После: ")
    printInOrder(root1)
    fmt.Println("\n")
    
    // Пример 2: [3,1,4,null,null,2] -> [2,1,4,null,null,3]
    fmt.Println("Пример 2:")
    root2 := createTree([]interface{}{3, 1, 4, nil, nil, 2})
    fmt.Print("До: ")
    printInOrder(root2)
    fmt.Println()
    
    recoverTree(root2)
    fmt.Print("После: ")
    printInOrder(root2)
    fmt.Println()
}
```

# найти ближайшего соседа в ВST дереве и написать решение на golang c сложностью алгоритма

```go
package main

import (
    "fmt"
    "math"
)

// Узел бинарного дерева
type TreeNode struct {
    Val   int
    Left  *TreeNode
    Right *TreeNode
}

// Функция для поиска ближайшего значения к target
func closestValue(root *TreeNode, target float64) int {
    closest := root.Val
    current := root
    
    for current != nil {
        // Обновляем ближайшее значение, если нашли лучшее
        if math.Abs(float64(current.Val)-target) < math.Abs(float64(closest)-target) {
            closest = current.Val
        }
        
        // Двигаемся по дереву
        if target < float64(current.Val) {
            current = current.Left
        } else if target > float64(current.Val) {
            current = current.Right
        } else {
            // Если нашли точное совпадение, возвращаем его
            return current.Val
        }
    }
    
    return closest
}

// Вспомогательная функция для создания BST
func insert(root *TreeNode, val int) *TreeNode {
    if root == nil {
        return &TreeNode{Val: val}
    }
    
    if val < root.Val {
        root.Left = insert(root.Left, val)
    } else if val > root.Val {
        root.Right = insert(root.Right, val)
    }
    
    return root
}

// Функция для демонстрации работы
func main() {
    // Создаем BST: [4,2,5,1,3]
    var root *TreeNode
    values := []int{4, 2, 5, 1, 3}
    
    for _, val := range values {
        root = insert(root, val)
    }
    
    // Тестовые случаи
    testCases := []float64{3.4, 3.6, 0.5, 5.5, 2.5}
    
    for _, target := range testCases {
        result := closestValue(root, target)
        fmt.Printf("Ближайшее к %.1f: %d\n", target, result)
    }
}
```

# дополнительно запоминаем путь

```go
package main

import (
    "fmt"
    "math"
)

type TreeNode struct {
    Val   int
    Left  *TreeNode
    Right *TreeNode
}

// Расширенная версия с подробной информацией
func closestValueDetailed(root *TreeNode, target float64) (int, []int) {
    closest := root.Val
    path := []int{}
    current := root
    
    for current != nil {
        path = append(path, current.Val)
        
        // Обновляем ближайшее значение
        currentDiff := math.Abs(float64(current.Val) - target)
        closestDiff := math.Abs(float64(closest) - target)
        
        if currentDiff < closestDiff {
            closest = current.Val
        }
        
        // Сохраняем направление движения
        if target < float64(current.Val) {
            current = current.Left
        } else if target > float64(current.Val) {
            current = current.Right
        } else {
            break // Точное совпадение
        }
    }
    
    return closest, path
}

// Рекурсивная версия (альтернативная реализация)
func closestValueRecursive(root *TreeNode, target float64) int {
    return helper(root, target, root.Val)
}

func helper(node *TreeNode, target float64, closest int) int {
    if node == nil {
        return closest
    }
    
    // Обновляем ближайшее значение
    if math.Abs(float64(node.Val)-target) < math.Abs(float64(closest)-target) {
        closest = node.Val
    }
    
    // Выбираем направление для продолжения поиска
    if target < float64(node.Val) {
        return helper(node.Left, target, closest)
    } else if target > float64(node.Val) {
        return helper(node.Right, target, closest)
    }
    
    return closest
}

func main() {
    // Создаем более сложное BST
    var root *TreeNode
    values := []int{8, 3, 10, 1, 6, 14, 4, 7, 13}
    
    for _, val := range values {
        root = insert(root, val)
    }
    
    targets := []float64{5.2, 9.8, 12.1, 0.5, 15.0}
    
    fmt.Println("Поиск ближайших значений:")
    for _, target := range targets {
        result, path := closestValueDetailed(root, target)
        fmt.Printf("Target: %.1f -> Ближайшее: %d, Путь: %v\n", 
                   target, result, path)
        
        // Альтернативный вызов
        result2 := closestValueRecursive(root, target)
        fmt.Printf("  Рекурсивный результат: %d\n", result2)
    }
}

func insert(root *TreeNode, val int) *TreeNode {
    if root == nil {
        return &TreeNode{Val: val}
    }
    
    if val < root.Val {
        root.Left = insert(root.Left, val)
    } else if val > root.Val {
        root.Right = insert(root.Right, val)
    }
    
    return root
}
```
```python
def insert_into_bst(root, val):
    """
    Вставляет значение в BST и возвращает корень дерева
    """
    if not root:
        return TreeNode(val)
    
    if val < root.val:
        root.left = insert_into_bst(root.left, val)
    else:
        root.right = insert_into_bst(root.right, val)
    
    return root
```
```go
package main

import (
    "fmt"
    "math"
)

type TreeNode struct {
    Val   int
    Left  *TreeNode
    Right *TreeNode
}

func isBalanced(root *TreeNode) bool {
    return checkBalance(root) != -1
}

func checkBalance(node *TreeNode) int {
    if node == nil {
        return 0
    }
    
    leftHeight := checkBalance(node.Left)
    if leftHeight == -1 {
        return -1
    }
    
    rightHeight := checkBalance(node.Right)
    if rightHeight == -1 {
        return -1
    }
    
    if math.Abs(float64(leftHeight-rightHeight)) > 1 {
        return -1
    }
    
    return max(leftHeight, rightHeight) + 1
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}

// Пример использования
func main() {
    // Сбалансированное дерево
    balanced := &TreeNode{
        Val: 1,
        Left: &TreeNode{Val: 2},
        Right: &TreeNode{Val: 3},
    }
    
    // Несбалансированное дерево
    unbalanced := &TreeNode{
        Val: 1,
        Left: &TreeNode{
            Val: 2,
            Left: &TreeNode{Val: 3},
        },
    }
    
    fmt.Println("Balanced tree:", isBalanced(balanced))     // true
    fmt.Println("Unbalanced tree:", isBalanced(unbalanced)) // false
}
```

# максимальная дистанция

```go
func maxDistToClosest(seats []int) int {
    sum, result, first := 0, 0, true

    for _, seat := range seats {
        if seat == 1 {
            if first {
                result = sum
                first = false
            } else {
                result = max((sum+1)/2, result)
            }
            sum = 0
        } else {
            sum++
        }
    }

    return max(sum, result)
}
```

```go
func checkInclusion(s1 string, s2 string) bool {

    if len(s1) > len(s2) {
        return false
    }

    window := [123]int{}
    pattern := [123]int{}

    for i := range s1 {
        pattern[s1[i]]++
    }

    for i:=0; i< len(s1); i++ {
        window[s2[i]]++
    }

    if window == pattern {
        return true
    }

    for i:=1; i < len(s2) - len(s1) + 1; i++ {
        window[s2[i-1]]--
        window[s2[i + len(s1) - 1]]++
        if window == pattern {
            return true
        }
    }

    return false
}
```

```go
func merge(intervals [][]int) [][]int {

    if len(intervals) == 1 {
        return intervals
    }
    
    sort.Slice(intervals, func(i, j int) bool {
        return intervals[i][0] < intervals[j][0]
    })
    
    result := intervals[:1]

    for i := 1; i < len(intervals); i++ {
        last := result[len(result) - 1]
        current := intervals[i]
        if current[0] <= last[1] {
            last[1] = max(last[1], current[1])
        } else {
            result = append(result, current)
        }
    }

    return result
}
```

```go
type RandomizedSet struct {
    r map[int]int
    data []int
}


func Constructor() RandomizedSet {
    return RandomizedSet{
        r: make(map[int]int),
        data: make([]int,0),
    }
}


func (this *RandomizedSet) Insert(val int) bool {
    if _, ok := this.r[val]; ok {
        return false
    }

    this.data = append(this.data, val)

    this.r[val] = len(this.data) - 1

    return true
}


func (this *RandomizedSet) Remove(val int) bool {
    if idx, ok := this.r[val]; ok {
        delete(this.r, val)
        if idx != len(this.data) - 1 {
            this.data[idx] = this.data[len(this.data) - 1]
            this.r[this.data[idx]] = idx
        }
        this.data = this.data[:len(this.data) - 1]
        return true
    }
    return false
}


func (this *RandomizedSet) GetRandom() int {
    return this.data[rand.Intn(len(this.data))]
}


/**
 * Your RandomizedSet object will be instantiated and called as such:
 * obj := Constructor();
 * param_1 := obj.Insert(val);
 * param_2 := obj.Remove(val);
 * param_3 := obj.GetRandom();
 */
 ```

 ```go
 func abs(a int) int {
    if a < 0 {
        return -a
    }
    return a
}

func findClosestElements(arr []int, k int, x int) []int {
    left := 0
    right := len(arr) - 1
    
    for right - left >= k {
        if abs(arr[left] - x) > abs(arr[right] - x) {
            left++
        } else {
            right--
        }
    }

    return arr[left:right+1]
}
```

```go
type MinStack struct {
    stack    []int
    minStack []int
}

func Constructor() MinStack {
    return MinStack{
        stack:    make([]int, 0),
        minStack: make([]int, 0),
    }
}

func (this *MinStack) Push(val int) {
    this.stack = append(this.stack, val)
    
    // Добавляем в minStack только если это новый минимум
    if len(this.minStack) == 0 || val <= this.minStack[len(this.minStack)-1] {
        this.minStack = append(this.minStack, val)
    }
}

func (this *MinStack) Pop() {
    if len(this.stack) == 0 {
        return
    }
    
    popped := this.stack[len(this.stack)-1]
    this.stack = this.stack[:len(this.stack)-1]
    
    // Удаляем из minStack только если удаляем текущий минимум
    if popped == this.minStack[len(this.minStack)-1] {
        this.minStack = this.minStack[:len(this.minStack)-1]
    }
}

func (this *MinStack) Top() int {
    return this.stack[len(this.stack)-1]
}

func (this *MinStack) GetMin() int {
    return this.minStack[len(this.minStack)-1]
}
```

```python
#2. Неэффективное сравнение строк с использованием срезов
#Проблема: хотя текущее решение верно, использование срезов строк ( s[i+1:]и t[i+1:]) создаёт новые строковые #объекты в Python, что может быть неэффективно для очень длинных строк. Это добавляет ненужную сложность, #связанную с пространством.

Более эффективное решение: вместо создания подстрок сравнивайте символы напрямую:

def isOneEditDistance(self, s: str, t: str) -> bool:
    if len(s) < len(t):
        return self.isOneEditDistance(t, s)

    len_s, len_t = len(s), len(t)
    if len_s - len_t > 1:
        return False

    for i in range(len_t):
        if s[i] != t[i]:
            if len_s == len_t:
                # Compare remaining characters one by one
                return all(s[j] == t[j] for j in range(i + 1, len_t))
            else:
                # Compare s[i+1:] with t[i:] without creating substrings
                return all(s[j + 1] == t[j] for j in range(i, len_t))

    return len_s == len_t + 1
```

```go
func validPalindrome(s string) bool {
   left := 0
   right := len(s) - 1
   
   if len(s) <= 2 {
    return true
  }

   for left < right {
        if s[left] != s[right] {
            return isPalindome(s, left+1, right) || isPalindome(s, left, right - 1)
        }
        left++
        right--
   }
   return true
}

func isPalindome(s string, l, r int) bool {
    for l < r {
        if s[l] != s[r] {
            return false
        }
        l++
        r--
    }
    return true
}
```

```go
func findAnagrams(s string, p string) []int {
    result := make([]int, 0)

    window := [123]int{}
    patt := [123]int{}

    if len(s) < len(p) {
        return result
    }

    for i := range p {
        window[p[i]]++
        patt[s[i]]++
    }

    if window == patt {
        result = append(result, 0)
    }

    for i := 1; i <= len(s) - len(p); i++ {
        patt[s[i-1]]--
        patt[s[i + len(p) - 1]]++
        if window == patt {
            result = append(result, i)
        }
        
    }

    return result
    
}
```
```go
package main

import "fmt"

func countSubstringsWithoutRepeats(s string) int {
    n := len(s)
    if n == 0 {
        return 0
    }
    
    count := 0
    // Используем два указателя (sliding window)
    left := 0
    charSet := make(map[byte]bool)
    
    for right := 0; right < n; right++ {
        // Если текущий символ уже есть в окне, двигаем левый указатель
        for charSet[s[right]] {
            delete(charSet, s[left])
            left++
        }
        
        // Добавляем текущий символ в множество
        charSet[s[right]] = true
        
        // Все подстроки от left до right не имеют повторяющихся символов
        // Количество таких подстрок = (right - left + 1)
        count += (right - left + 1)
    }
    
    return count
}

// Альтернативное решение с массивом вместо map (более эффективное для латинских символов)
func countSubstringsWithoutRepeatsOptimized(s string) int {
    n := len(s)
    if n == 0 {
        return 0
    }
    
    count := 0
    left := 0
    // Используем массив для отслеживания последних позиций символов
    lastSeen := make([]int, 128) // ASCII символы
    for i := range lastSeen {
        lastSeen[i] = -1
    }
    
    for right := 0; right < n; right++ {
        // Если символ уже встречался и его позиция >= left, двигаем left
        if lastSeen[s[right]] >= left {
            left = lastSeen[s[right]] + 1
        }
        
        lastSeen[s[right]] = right
        count += (right - left + 1)
    }
    
    return count
}

func main() {
    testCases := []string{
        "abc",
        "aaa",
        "abac",
        "abcabcbb",
        "pwwkew",
        "",
        "a",
    }
    
    for _, test := range testCases {
        result1 := countSubstringsWithoutRepeats(test)
        result2 := countSubstringsWithoutRepeatsOptimized(test)
        fmt.Printf("Строка: \"%s\"\n", test)
        fmt.Printf("  Количество подстрок без повторений: %d\n", result1)
        fmt.Printf("  Оптимизированная версия: %d\n", result2)
        fmt.Println()
    }
}
```

```go
package main

// Дан массив точек целочисленными координатами (x, y).

// Определить, существует ли вертикальная прямая, делящая все точки, не лежащие на ней,
// на 2 симметричных относительно этой прямой набора точек.

func isVertSym(arr [][2]int) bool {
    if len(arr) == 0 {
        return true
    }

    minX, maxY := arr[0][0], arr[0][0]
    p := make(map[[2]int]int)

    for _, v := range arr {
        minX = min(minX, v[0])
        maxX = max(maxY, v[0])
        p[v]++
    }

    for k, v := range p {
        xSem := maxX + minX - k[0]
        if v != p[[2]int{xSem,k[1]}] {
            return false
        }
    }

    return true
}

func main() {
    println(isVertSym([][2]int{{0, 0}, {0, 1}, {1, 1}, {2, 2}, {3, 1}, {4, 1}, {4, 0}})) // true
    println(isVertSym([][2]int{{0, 0}, {0, 0}, {1, 1}, {2, 2}, {3, 1}, {4, 0}, {4, 0}})) // true
    println(isVertSym([][2]int{{0, 0}, {0, 0}, {1, 1}, {2, 2}, {3, 1}, {4, 0}})) // false
    println(isVertSym([][2]int{})) // true
    println(isVertSym([][2]int{{0, 0}})) // true
    println(isVertSym([][2]int{{0, 0}, {10, 0}})) // true
    println(isVertSym([][2]int{{0, 0}, {11, 1}})) // false
    println(isVertSym([][2]int{{0, 0}, {1, 0}, {3, 0}})) // false
}
```

```go
func sortedSquares(nums []int) []int {
    n := len(nums)
    result := make([]int, n)
    
    left, right := 0, n-1
    pos := n-1 // fill from the end
    
    for left <= right {
        leftSquare := nums[left] * nums[left]
        rightSquare := nums[right] * nums[right]
        
        if leftSquare > rightSquare {
            result[pos] = leftSquare
            left++
        } else {
            result[pos] = rightSquare
            right--
        }
        pos--
    }
    
    return result
}
```
```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func findDuplicateSubtrees(root *TreeNode) []*TreeNode {

    res := map[string]int{}

    result := make([]*TreeNode,0)

    var treverse func(n *TreeNode) string

    treverse = func(n *TreeNode) string {
        if n == nil {
            return "#"
        }

        left := treverse(n.Left)
        right := treverse(n.Right)

        subtree := left + "," + right + "," + strconv.Itoa(n.Val)

        if v, ok := res[subtree]; ok && v == 1 {
            result = append(result, n)
        }

        res[subtree]++

        return subtree
    }

    treverse(root)

    return result
}
```
