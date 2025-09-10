import sys

from collections import deque

n, m = map(int, input().split())

arr = [[1 for _ in range(m)] for _ in range(n)]

startX = 0
startY = 0

endX = 0
endY = 0

for i in range(n):
    str = input()
    for j in range(m):
        if str[j] == '#':
            arr[i][j] =  0
        if  str[i] == 'S':
            startX = i
            startY = j
        if  str[i] == 'F':
            endX = i
            endY = j

visited = [[False for _ in range(m)] for _ in range(n)]

queue = deque([(startX, startY, "")])

visited[startX][startY] = True

result = ""

def find_visited():

    while queue:
        sX, sY, step = queue.popleft()

        if sX == endX and sY == endY:
            return step
        if sX + 1 < n and arr[sX+1][sY] == 1 and not visited[sX+1][sY]:
            # D
            queue.append((sX+1,sY, step + "D"))
            visited[sX+1][sY] = True
        if sY + 1 < m and arr[sX][sY+1] == 1 and not visited[sX][sY+1]:
            # D
            queue.append((sX,sY+1,step + "R"))
            visited[sX][sY+1] = True
        if sY - 1 >= 0 and arr[sX][sY-1] == 1 and not visited[sX][sY-1]:
            # U
            queue.append((sX,sY-1,step + "L"))
            visited[sX][sY-1] = True
        if sX - 1 >= 0 and arr[sX-1][sY] == 1 and not visited[sX-1][sY]:
            # R
            queue.append((sX-1,sY,step + "U"))
            visited[sX-1][sY] = True
    return ""

steps = find_visited()

if steps != "":
    print(len(steps))
    print(steps)
else:
    print(-1)

