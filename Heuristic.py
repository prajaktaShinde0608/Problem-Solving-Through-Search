from collections import deque
import heapq
import time


# Graph used for search algorithms
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': ['G'],
    'E': ['G', 'H'],
    'F': ['H'],
    'G': [],
    'H': []
}

# Heuristic values for informed search
heuristic = {
    'A': 3,
    'B': 2,
    'C': 4,
    'D': 1,
    'E': 1,
    'F': 3,
    'G': 0,
    'H': 2
}

# Breadth First Search
def bfs(start, goal):
    queue = deque([(start, [start])])
    visited = set()
    nodes = 0

    while queue:
        node, path = queue.popleft()

        if node in visited:
            continue

        visited.add(node)
        nodes += 1

        if node == goal:
            return path, nodes

        for neighbour in graph[node]:
            if neighbour not in visited:
                queue.append((neighbour, path + [neighbour]))

    return None, nodes

# Depth First Search
def dfs(start, goal):
    stack = [(start, [start])]
    visited = set()
    nodes = 0

    while stack:
        node, path = stack.pop()

        if node in visited:
            continue

        visited.add(node)
        nodes += 1

        if node == goal:
            return path, nodes

        for neighbour in reversed(graph[node]):
            if neighbour not in visited:
                stack.append((neighbour, path + [neighbour]))

    return None, nodes

# Greedy Best First Search
def greedy(start, goal):
    priority_queue = [(heuristic[start], start, [start])]
    visited = set()
    nodes = 0

    while priority_queue:
        _, node, path = heapq.heappop(priority_queue)

        if node in visited:
            continue

        visited.add(node)
        nodes += 1

        if node == goal:
            return path, nodes

        for neighbour in graph[node]:
            if neighbour not in visited:
                heapq.heappush(
                    priority_queue,
                    (heuristic[neighbour], neighbour, path + [neighbour])
                )

    return None, nodes

# A* Search
def astar(start, goal):
    priority_queue = [(heuristic[start], 0, start, [start])]
    cost = {start: 0}
    visited = set()
    nodes = 0

    while priority_queue:
        f, g, node, path = heapq.heappop(priority_queue)

        if node in visited:
            continue

        visited.add(node)
        nodes += 1

        if node == goal:
            return path, nodes

        for neighbour in graph[node]:
            new_cost = g + 1

            if neighbour not in cost or new_cost < cost[neighbour]:
                cost[neighbour] = new_cost
                new_f = new_cost + heuristic[neighbour]

                heapq.heappush(
                    priority_queue,
                    (new_f, new_cost, neighbour, path + [neighbour])
                )

    return None, nodes

# Hill Climbing
def hill_climbing(start, goal):
    current = start
    path = [current]
    nodes = 0

    while True:
        nodes += 1

        if current == goal:
            return path, nodes

        neighbours = graph[current]

        if not neighbours:
            return path, nodes

        best = min(neighbours, key=lambda x: heuristic[x])

        if heuristic[best] >= heuristic[current]:
            return path, nodes

        current = best
        path.append(current)

# Backtracking for N-Queens
def is_safe(board, row, col):
    for previous_row in range(row):
        previous_col = board[previous_row]

        if previous_col == col:
            return False

        if abs(previous_col - col) == abs(previous_row - row):
            return False

    return True


def solve_n_queens(n):
    board = [-1] * n
    nodes = 0

    def backtrack(row):
        nonlocal nodes
        nodes += 1

        if row == n:
            return True

        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col

                if backtrack(row + 1):
                    return True

                board[row] = -1

        return False

    if backtrack(0):
        return board, nodes

    return None, nodes

# Performance evaluation
def evaluate(name, function, start, goal):
    begin = time.perf_counter()
    path, nodes = function(start, goal)
    end = time.perf_counter()

    print(f"{name}:")
    print("Path:", path)
    print("Nodes Explored:", nodes)
    print("Execution Time:", round(end - begin, 8), "seconds")
    print()


# Execute search algorithms
print("SEARCH ALGORITHM PERFORMANCE")
print("-" * 40)

evaluate("BFS", bfs, 'A', 'G')
evaluate("DFS", dfs, 'A', 'G')
evaluate("Greedy Best-First", greedy, 'A', 'G')
evaluate("A*", astar, 'A', 'G')
evaluate("Hill Climbing", hill_climbing, 'A', 'G')


# Execute N-Queens
print("N-QUEENS BACKTRACKING")
print("-" * 40)

n = 4
solution, nodes = solve_n_queens(n)

print("N =", n)
print("Solution:", solution)
print("Nodes Explored:", nodes)

if solution:
    for row in solution:
        print(" ".join("Q" if col == row else "." for col in range(n)))
