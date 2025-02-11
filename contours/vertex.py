from __future__ import annotations
import numpy as np


class Vertex:
    def __init__(self, coord, before=None, after=None):
        self.coord = coord
        self.before = before
        self.after = after
        self.linked_vertex = None

    @property
    def row(self):
        return self.coord[0]

    @property
    def col(self):
        return self.coord[1]

    def __repr__(self):
        return f"({self.coord[0]}, {self.coord[1]})"

    def can_calc_perp(self):
        v = self
        w = self
        for _ in range(9):
            v = v.after
            w = w.before
            if v is None or w is None:
                return False
        return True

    def is_above_perp_line(self, other: Vertex) -> bool:
        assert self.before is not None and self.after is not None
        v = self
        w = self
        for _ in range(9):
            v = v.after
            w = w.before
        vector1 = np.array(v.coord) - np.array(w.coord)
        vector2 = np.array(other.coord) - np.array(self.coord)
        return np.dot(vector1, vector2) < 0

    def is_convex(self, mask) -> bool:
        start = self
        end = self
        for i in range(20):
            if start.before is not None:
                start = start.before
            if end.after is not None:
                end = end.after
        avg_point = (
            int(np.mean([start.coord[0], end.coord[0]])),
            int(np.mean([start.coord[1], end.coord[1]])),
        )
        return mask[avg_point[0], [avg_point[1]]] != 1

    def extend_chain(self, visited: set, border: np.ndarray) -> Vertex | None:
        if self.before is not None and self.after is not None:
            return None

        for offset in [
            (-1, 0),
            (0, -1),
            (1, 0),
            (0, 1),
            (1, -1),
            (-1, -1),
            (1, 1),
            (-1, 1),
        ]:
            new_coord = (self.coord[0] + offset[0], self.coord[1] + offset[1])
            if (
                out_of_bounds(new_coord, border.shape)
                or new_coord in visited
                or not border[new_coord]
            ):
                continue

            if self.after is None:
                new_vertex = Vertex(new_coord, self, None)
                self.after = new_vertex
                return new_vertex
            elif self.before is None:
                new_vertex = Vertex(new_coord, None, self)
                self.before = new_vertex
                return new_vertex

        if len(visited) < np.sum(border) * 0.95:
            all_border_points = np.argwhere(
                border[
                    max(0, self.coord[0] - 50) : min(
                        border.shape[0], self.coord[0] + 50
                    ),
                    max(0, self.coord[1] - 50) : min(
                        border.shape[1], self.coord[1] + 50
                    ),
                ]
                == 1
            )
            min_dist = np.inf
            best_next_pnt = None
            remaining_border_points = set(
                (
                    pnt[0] + max(0, self.coord[0] - 50),
                    pnt[1] + max(0, self.coord[1] - 50),
                )
                for pnt in all_border_points
            )
            remaining_border_points = remaining_border_points - visited
            for pnt in remaining_border_points:
                dist = np.linalg.norm(np.array(self.coord) - np.array(pnt))
                if dist < min_dist:
                    min_dist = dist
                    best_next_pnt = pnt

            if best_next_pnt is not None:
                if self.after is None:
                    new_vertex = Vertex(best_next_pnt, self, None)
                    self.after = new_vertex
                    return new_vertex
                elif self.before is None:
                    new_vertex = Vertex(best_next_pnt, None, self)
                    self.before = new_vertex
                    return new_vertex
        return None


def out_of_bounds(coord, shape):
    return coord[0] < 0 or coord[0] >= shape[0] or coord[1] < 0 or coord[1] >= shape[1]
