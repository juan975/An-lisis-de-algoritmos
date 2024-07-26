import heapq
import sys

def cargar_grafo_desde_archivo(nombre_archivo, dirigido=False):
    with open(nombre_archivo, 'r') as f:
        lines = f.readlines()
    edges = []
    vertices = set()
    graph = {}
    for line in lines:
        u, v, w = map(int, line.strip().split(','))
        edges.append((u, v, w))
        vertices.update([u, v])
        if u not in graph:
            graph[u] = {}
        if not dirigido:
            if v not in graph:
                graph[v] = {}
        graph[u][v] = w
        if not dirigido:
            graph[v][u] = w
    return edges, max(vertices), graph

def guardar_en_archivo(nombre_archivo, resultado):
    with open(nombre_archivo, 'w') as f:
        for u, v, w in resultado:
            f.write(f"{u},{v},{w}\n")

class Kruskal:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []

    def add_edge(self, u, v, w):
        self.graph.append([u, v, w])

    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)

        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    def kruskal_mst(self):
        result = []
        i, e = 0, 0

        # Ordenar las aristas por su peso
        self.graph = sorted(self.graph, key=lambda item: item[2])

        # Inicializar los padres y rangos de cada vértice
        parent = [i for i in range(self.V + 1)]
        rank = [0] * (self.V + 1)

        while e < self.V - 1:
            u, v, w = self.graph[i]
            i += 1
            x = self.find(parent, u)
            y = self.find(parent, v)

            if x != y:
                e += 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)

        return result

def prim(graph, start=1):
    mst = []
    visited = set([start])
    edges = [(weight, start, v) for v, weight in graph[start].items()]
    heapq.heapify(edges)

    while edges:
        weight, u, v = heapq.heappop(edges)
        if v not in visited:
            visited.add(v)
            mst.append((u, v, weight))
            for next_v, next_weight in graph[v].items():
                if next_v not in visited:
                    heapq.heappush(edges, (next_weight, v, next_v))

    return mst

class Dijkstra:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[] for _ in range(vertices + 1)]

    def add_edge(self, u, v, w):
        self.graph[u].append((w, v))

    def dijkstra(self, src):
        dist = [float('inf')] * (self.V + 1)
        dist[src] = 0
        pq = [(0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for w, v in self.graph[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    heapq.heappush(pq, (dist[v], v))
        return dist

def main():
    grafos = [
        ('kruskal', 'Grafo50.txt', 'Resultado_kruskal.txt', False),
        ('prim', 'Grafo50.txt', 'Resultado_prim.txt', False),
        ('dijkstra', 'Grafo30.txt', 'Resultado_dijkstra.txt', True)
    ]

    for algoritmo, archivo_entrada, archivo_salida, dirigido in grafos:
        edges, vertices, graph = cargar_grafo_desde_archivo(archivo_entrada, dirigido)

        if algoritmo == 'kruskal':
            g = Kruskal(vertices)
            for u, v, w in edges:
                g.add_edge(u, v, w)
            resultado = g.kruskal_mst()
        elif algoritmo == 'prim':
            resultado = prim(graph)
        elif algoritmo == 'dijkstra':
            nodo_origen = 1  # Cambiar al nodo de origen deseado
            g = Dijkstra(vertices)
            for u, v, w in edges:
                g.add_edge(u, v, w)
            distancias = g.dijkstra(nodo_origen)
            resultado = [(nodo_origen, i, distancias[i]) for i in range(1, vertices + 1) if distancias[i] != float('inf')]
        else:
            print("Algoritmo no reconocido. Use 'kruskal', 'prim' o 'dijkstra'.")
            sys.exit(1)

        guardar_en_archivo(archivo_salida, resultado)

if __name__ == "__main__":
    main()
