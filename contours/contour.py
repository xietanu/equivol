from __future__ import annotations

import numpy as np

import contours


class Contour:
    def __init__(self, point_on_contour: contours.Vertex):
        self.root = point_on_contour
        while self.root.before is not None:
            self.root = self.root.before

        self.vertices = [self.root]
        while self.tail.after is not None:
            self.vertices.append(self.tail.after)

    @property
    def tail(self) -> contours.Vertex:
        return self.vertices[-1]

    def __len__(self):
        return len(self.vertices)

    def __iter__(self):
        return iter(self.vertices)

    def __getitem__(self, index):
        return self.vertices[index]

    @classmethod
    def from_border(cls, border: np.ndarray):
        points = np.argwhere(border)
        root = contours.Vertex(tuple(points[0].tolist()))
        visited = set()
        cur_point = root
        while cur_point:
            visited.add(cur_point.coord)
            cur_point = cur_point.extend_chain(visited, border)
        cur_point = root
        while cur_point:
            visited.add(cur_point.coord)
            cur_point = cur_point.extend_chain(visited, border)

        return cls(root)

    def flip(self):
        for vertex in self.vertices:
            vertex.before, vertex.after = vertex.after, vertex.before

        self.root = self.tail
        self.vertices = list(reversed(self.vertices))

    def smooth(self, degree: int):
        current = self.root.after
        while current.after is not None:
            points = [current.coord]
            start = current
            for _ in range(degree):
                start = start.before
                if start is None:
                    break
                points.append(start.coord)
            end = current
            for _ in range(degree):
                end = end.after
                if end is None:
                    break
                points.append(end.coord)
            current.coord = np.mean(np.array(points), axis=0).astype(int)
            current = current.after

    def draw(self, canvas, color=255):
        for vertex in self.vertices:
            canvas[vertex.row, vertex.col] = color
        return canvas

    def copy(self):
        vertices = []
        for i, vertex in enumerate(self.vertices):
            vertices.append(contours.Vertex(vertex.coord))
            if i > 0:
                vertices[i - 1].after = vertices[i]
                vertices[i].before = vertices[i - 1]

        return Contour(vertices[0])
