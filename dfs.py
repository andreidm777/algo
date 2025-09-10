import sys

from collections import deque 

arr = sys.stdin.readline().strip()

n, m = arr.Split()

players = {}

graph = [[] for _ in range(n)]

for i in range(m):
    arr = sys.stdin.readline().strip()
    u, v, t = arr.Split()
    if u == t:
        graph[u] = v
    else:
        graph[v] = u

def dfs(start):
    visited = [False]*n
    queue = deque([start])
    players = set()

    while not queue.empty():
        val = queue.pop()

        for v in graph[val]:
            if not visited[v]:
                visited[v] = True
                queue.append(v)
                players.add(v)
    
    return players


myList = [set() for _ in range(n)]

for i in range(n):
    myList[i] = dfs(i)

for i in range(n):
    for j in range(i+1,n):
        if j not in myList[i] and i not in myList[j]:
            print("NO")

print("YES")