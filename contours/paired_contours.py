from __future__ import annotations

import itertools

import cv2
import numpy as np

import contours
import masks


class PairedContours:
    def __init__(
        self, outer_contour: contours.Contour, inner_contour: contours.Contour
    ):
        self.outer_contour = outer_contour
        self.inner_contour = inner_contour
        self.connections = [
            Connector(vertex, vertex.linked_vertex)
            for vertex in outer_contour
            if vertex.linked_vertex is not None
        ]

    @classmethod
    def from_contours(
        cls,
        outer_contour: contours.Contour,
        inner_contour: contours.Contour,
        mask: np.ndarray,
        min_step_size: int = 3,
        decross_links: bool = True,
        trim: bool = True,
    ) -> PairedContours:
        assert masks.is_valid_mask(mask)

        outer_contour = outer_contour.copy()
        inner_contour = inner_contour.copy()

        vote_flip = 0
        vote_not_flip = 0
        for a, b, c in (
            (outer_contour.root, inner_contour.root, inner_contour.tail),
            (outer_contour.tail, inner_contour.tail, inner_contour.root),
            (inner_contour.root, outer_contour.root, outer_contour.tail),
            (inner_contour.tail, outer_contour.tail, outer_contour.root),
        ):
            if np.linalg.norm(np.array(a.coord) - np.array(b.coord)) > np.linalg.norm(
                np.array(a.coord) - np.array(c.coord)
            ):
                vote_flip += 1
            else:
                vote_not_flip += 1

        if vote_flip > vote_not_flip:
            outer_contour.flip()

        if len(outer_contour) > len(inner_contour):
            long_contour = outer_contour
            short_contour = inner_contour

        else:
            long_contour = inner_contour
            short_contour = outer_contour

        cur_step_size = min_step_size
        cur_short_vertex = short_contour.root
        cur_long_vertex = long_contour.root
        last_short_linked = 0
        last_long_linked = 0
        while cur_short_vertex is not None:
            cur_step_size += 1
            if cur_step_size < min_step_size:
                cur_short_vertex = cur_short_vertex.after
                continue

            linked = False

            if not cur_short_vertex.can_calc_perp():
                cur_short_vertex = cur_short_vertex.after
                continue

            while cur_long_vertex.after is not None and not linked:
                if cur_short_vertex.is_above_perp_line(
                    cur_long_vertex
                ) and not cur_short_vertex.is_above_perp_line(cur_long_vertex.after):
                    if cur_long_vertex.linked_vertex is None:
                        cur_long_vertex.linked_vertex = cur_short_vertex
                        cur_short_vertex.linked_vertex = cur_long_vertex
                        linked = True
                    elif cur_long_vertex.after.linked_vertex is None:
                        cur_long_vertex.after.linked_vertex = cur_short_vertex
                        cur_short_vertex.linked_vertex = cur_long_vertex.after
                        linked = True
                cur_long_vertex = cur_long_vertex.after

            if linked:
                cur_step_size = 0
                if not cur_short_vertex.is_convex(mask):
                    short_contour, long_contour = long_contour, short_contour
                    last_short_linked, last_long_linked = (
                        last_long_linked,
                        last_short_linked,
                    )
                    cur_short_vertex, cur_long_vertex = (
                        cur_long_vertex,
                        cur_short_vertex,
                    )

                last_short_linked = max(
                    short_contour.vertices.index(cur_short_vertex), last_short_linked
                )
                last_long_linked = max(
                    long_contour.vertices.index(cur_long_vertex), last_long_linked
                )
                cur_short_vertex = short_contour[last_short_linked].after
                cur_long_vertex = long_contour[last_long_linked]
            else:
                cur_short_vertex = cur_short_vertex.after
                cur_long_vertex = long_contour[last_long_linked]

        paired_contours = cls(outer_contour, inner_contour)

        if decross_links:
            paired_contours.decross_connections()
        if trim:
            paired_contours.trim()

        return paired_contours

    def trim(self):
        nt = 0
        cur_vertex = self.outer_contour.root
        while cur_vertex.after is not None and cur_vertex.linked_vertex is None:
            cur_vertex.after.before = None
            self.outer_contour.vertices.remove(cur_vertex)
            self.outer_contour.root = self.outer_contour.vertices[0]
            cur_vertex = cur_vertex.after
            nt += 1

        cur_vertex = self.inner_contour.root
        while cur_vertex.after is not None and cur_vertex.linked_vertex is None:
            cur_vertex.after.before = None
            self.inner_contour.vertices.remove(cur_vertex)
            self.inner_contour.root = self.inner_contour.vertices[0]
            cur_vertex = cur_vertex.after
            nt += 1

        cur_vertex = self.inner_contour.tail
        while cur_vertex.before is not None and cur_vertex.linked_vertex is None:
            cur_vertex.before.after = None
            self.inner_contour.vertices.remove(cur_vertex)
            cur_vertex = cur_vertex.before
            nt += 1

        cur_vertex = self.outer_contour.tail
        while cur_vertex.before is not None and cur_vertex.linked_vertex is None:
            cur_vertex.before.after = None
            self.outer_contour.vertices.remove(cur_vertex)
            cur_vertex = cur_vertex.before
            nt += 1

    def create_segment_mask(self, shape) -> np.ndarray:
        n_segments = len(self.connections) - 1

        segment_masks = np.zeros((shape[0], shape[1], n_segments), dtype=bool)

        for i, (start_line, end_line) in enumerate(
            itertools.pairwise(self.connections)
        ):
            segment_vertices = []
            cur_vertex = start_line.start
            while cur_vertex != end_line.start:
                segment_vertices.append((cur_vertex.coord[1], cur_vertex.coord[0]))
                cur_vertex = cur_vertex.after
            segment_vertices.append((cur_vertex.coord[1], cur_vertex.coord[0]))
            failed = False
            cur_vertex = end_line.end
            new_vertices = []
            while cur_vertex != start_line.end:
                new_vertices.append((cur_vertex.coord[1], cur_vertex.coord[0]))
                cur_vertex = cur_vertex.after
                if cur_vertex is None:
                    failed = True
                    break
            if failed:
                cur_vertex = end_line.end
                while cur_vertex != start_line.end:
                    segment_vertices.append((cur_vertex.coord[1], cur_vertex.coord[0]))
                    cur_vertex = cur_vertex.before
            else:
                segment_vertices.extend(new_vertices)
            segment_vertices.append((cur_vertex.coord[1], cur_vertex.coord[0]))

            layer = np.zeros(shape, dtype=np.uint8)

            cv2.fillPoly(layer, [np.array(segment_vertices)], 1)

            segment_masks[:, :, i] = layer == 1
            if i > 0:
                segment_masks[
                    np.logical_and(segment_masks[:, :, i - 1] == 1, layer == 1), i - 1
                ] = False

        return segment_masks

    def decross_connections(self):
        n_crossings = [
            sum(line.crosses_line(other) for other in self.connections if other != line)
            for line in self.connections
        ]

        while max(n_crossings) > 0:
            max_index = n_crossings.index(max(n_crossings))
            line_to_remove = self.connections[max_index]
            line_to_remove.start.linked_vertex = None
            line_to_remove.end.linked_vertex = None
            self.connections.remove(line_to_remove)
            n_crossings = [
                sum(
                    line.crosses_line(other)
                    for other in self.connections
                    if other != line
                )
                for line in self.connections
            ]

    def draw(self, canvas):
        self.outer_contour.draw(canvas)
        self.inner_contour.draw(canvas)

        for connection in self.connections:
            cv2.line(
                canvas,
                (connection.start.col, connection.start.row),
                (connection.end.col, connection.end.row),
                255,
                1,
            )

        return canvas


class Connector:
    def __init__(self, start_vertex: contours.Vertex, end_vertex: contours.Vertex):
        self.start = start_vertex
        self.end = end_vertex

    def crosses_line(self, other: Connector) -> bool:
        def ccw(A, B, C):
            return (C.row - A.row) * (B.col - A.col) > (B.row - A.row) * (C.col - A.col)

        #    Return true if line segments AB and CD intersect
        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        return intersect(self.start, self.end, other.start, other.end)
