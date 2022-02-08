from OpenGL.GL import *
from OpenGL.GLU import *

from typing import final

import math
import copy
import importlib

class Vector:
    """
    Class representing a 3D Vector
    """

    def __init__(self, x: float, y: float, z: float):
        self.__x = x
        self.__y = y
        self.__z = z

    def to_list(self) -> list[float]:
        """
        Method that return a Vector as an iterable
        """
        return [self.__x, self.__y, self.__z]

    def translate(self, vector):
        """
        Method that translate a Vector according to an other vector
        """
        if isinstance(vector, Vector):
            self.__x += vector.get_x()
            self.__y += vector.get_y()
            self.__z += vector.get_z()

    def get_x(self) -> float:
        return self.__x

    def set_x(self, x: float):
        self.__x = x

    def get_y(self) -> float:
        return self.__y

    def set_y(self, y: float):
        self.__y = y

    def get_z(self) -> float:
        return self.__z

    def set_z(self, z: float):
        self.__z = z

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(
                self.__x + other.get_x(),
                self.__y + other.get_y(),
                self.__z + other.get_z(),
            )

    def __radd__(self, other):
        if isinstance(other, Vector):
            return Vector(
                self.__x + other.get_x(),
                self.__y + other.get_y(),
                self.__z + other.get_z(),
            )

    def __mul__(self, other):
        if isinstance(other, float):
            return Vector(self.__x * other, self.__y * other, self.__z * other)

    def __rmul__(self, other):
        if isinstance(other, float):
            return Vector(self.__x * other, self.__y * other, self.__z * other)

    def __eq__(self, other):
        if isinstance(other, Vector):
            return (self.__x == other.get_x()) and (self.__y == other.get_y()) and (self.__z == other.get_z())

    def __repr__(self):
        return (
            "(x: "
            + str(self.__x)
            + ",y: "
            + str(self.__y)
            + ",z: "
            + str(self.__z)
            + ")"
        )

class Object:
    """
    Super class representing a 3D Object
    """

    def __init__(
        self,
        vertices: list[Vector],
        edges: list[tuple[int, int]],
        surfaces: list[tuple[int]],
        offset: Vector,
        size: float,
    ):
        self.offset = offset
        self.size = size
        self.vertices = vertices
        self.edges = edges
        self.surfaces = surfaces
        self.color = (1, 1, 1, 1)

    def invert_color(self):
        """
        Method that compute the reverse of the object's color
        """
        new_color = []
        for i in range(3):
            new_color.append(1 - self.color[i])
        new_color.append(1)
        self.color = tuple(new_color)

    def move(self):
        """
        Method that recompute object coordinates with the actual offset
        """
        for i in range(len(self.vertices)):
            self.vertices[i] += self.offset

    def set_offset(self, offset: Vector):
        """
        Method that change the offset of a 3D object to the origin of space
        """
        self.offset = offset
        self.move()

    def resize(self):
        """
        Method that resize the object according to the given size factor
        """
        for i in range(len(self.vertices)):
            self.vertices[i] *= self.size

    def set_size(self, size: float):
        """
        Method that set the object size
        """
        self.size = size
        self.resize()

    def get_vertices_number(self) -> int:
        """
        Method to get the number of vertices of a 3D object
        """
        return len(self.vertices)

    def get_vertices(self) -> list[Vector]:
        """
        Method that return the object vertices
        """
        return self.vertices

    def get_edges(self) -> list[tuple[int,int]]:
        """
        Method that return the object edges"
        """
        return self.edges

    def get_surfaces(self) -> list[tuple[int]]:
        """
        Method that return the object surfaces
        """
        return self.surfaces
        
    def set_color(self, color):
        """
        Method that set the object color
        """
        self.color = color

    def draw(self):
        """
        Method to draw completely (wireframe and surfaces) a 3D Object using OpenGl primitives
        """
        for surface in self.surfaces:
            glBegin(GL_TRIANGLE_STRIP)
            glColor4fv(self.color)
            for vertex in surface:
                glVertex3f(
                    self.vertices[vertex].get_x(),
                    self.vertices[vertex].get_y(),
                    self.vertices[vertex].get_z(),
                )
            glEnd()

            self.invert_color()
            glBegin(GL_LINES)
            glColor4fv(self.color)
            for edge in self.edges:
                glColor4fv(self.color)
                for vertex in edge:
                    glVertex3f(
                        self.vertices[vertex].get_x(),
                        self.vertices[vertex].get_y(),
                        self.vertices[vertex].get_z(),
                    )
            glEnd()

            self.invert_color()

    def draw_wireframe(self):
        """
        Method to draw the wireframe of a 3D Object using OpenGl primitives
        """
        glBegin(GL_LINES)
        for edge in self.edges:
            glColor4fv(self.color)
            for vertex in edge:
                glVertex3f(
                    self.vertices[vertex].get_x(),
                    self.vertices[vertex].get_y(),
                    self.vertices[vertex].get_z(),
                )

        glEnd()


    def to_obj(self, type: str, offset: int) -> str:
        """
        Method to convert a 3D Object in a obj parsable string
        """
        repr_str: str = []
        # Appending vertices coordinates
        for vertex in self.vertices:
            repr_str.append(
                "\nv "
                + str(vertex.get_x())
                + " "
                + str(vertex.get_y())
                + " "
                + str(vertex.get_z())
            )
        repr_str.append("\n")
        # Appending surfaces
        if type == "Cube":
            for face in self.surfaces:
                repr_str.append(
                    "\nf "
                    + str(face[0] + offset + 1)
                    + " "
                    + str(face[1] + offset + 1)
                    + " "
                    + str(face[2] + offset + 1)
                )
                repr_str.append(
                    "\nf "
                    + str(face[3] + offset + 1)
                    + " "
                    + str(face[2] + offset + 1)
                    + " "
                    + str(face[1] + offset + 1)
                )
        elif type == "Tetrahedron":
            for face in self.surfaces:
                repr_str.append(
                    "\nf "
                    + str(face[0] + offset + 1)
                    + " "
                    + str(face[1] + offset + 1)
                    + " "
                    + str(face[2] + offset + 1)
                )
        elif type == "Octahedron":
            face1 = self.surfaces[0]
            repr_str.append(
                "\nf "
                + str(face1[3] + offset + 1)
                + " "
                + str(face1[1] + offset + 1)
                + " "
                + str(face1[1] + offset + 1)
                + " "
                + str(face1[2] + offset + 1)
            )
            for face in self.surfaces:
                repr_str.append(
                    "\nf "
                    + str(face[0] + offset + 1)
                    + " "
                    + str(face[1] + offset + 1)
                    + " "
                    + str(face[2] + offset + 1)
                )
        elif type == "Dodecahedron":
            for face in self.surfaces:
                repr_str.append(
                    "\nf "
                    + str(face[0] + offset + 1)
                    + " "
                    + str(face[1] + offset + 1)
                    + " "
                    + str(face[2] + offset + 1)
                )
                repr_str.append(
                    "\nf "
                    + str(face[0] + offset + 1)
                    + " "
                    + str(face[2] + offset + 1)
                    + " "
                    + str(face[3] + offset + 1)
                )
                repr_str.append(
                    "\nf "
                    + str(face[0] + offset + 1)
                    + " "
                    + str(face[3] + offset + 1)
                    + " "
                    + str(face[4] + offset + 1)
                )
                repr_str.append(
                    "\nf "
                    + str(face[0] + offset + 1)
                    + " "
                    + str(face[4] + offset + 1)
                    + " "
                    + str(face[5] + offset + 1)
                )
                repr_str.append(
                    "\nf "
                    + str(face[0] + offset + 1)
                    + " "
                    + str(face[1] + offset + 1)
                    + " "
                    + str(face[5] + offset + 1)
                )
        elif type == "Icosahedron":
            for face in self.surfaces:
                repr_str.append(
                    "\nf "
                    + str(face[0] + offset + 1)
                    + " "
                    + str(face[1] + offset + 1)
                    + " "
                    + str(face[2] + offset + 1)
                )

        # Return the concaneted string
        separator = ""
        return separator.join(repr_str)

    def copy(self, class_name:str):
        """
        Method that return a deep copy of the current object
        """
        # Instantiate a new class based on what class name is given in arg (i.e Cube, Pyramid....)
        Obj = getattr(importlib.import_module("objects"), class_name) 
        # Deepcopying the actual object
        return Obj(copy.deepcopy(self.offset), copy.deepcopy(self.size))


class Cube(Object):
    """
    3D Object representing a Cube
    """

    def __init__(self, offset: Vector, size: float):
        vertices = [
            Vector(-0.5, -0.5, -0.5),
            Vector(0.5, -0.5, -0.5),
            Vector(-0.5, -0.5, 0.5),
            Vector(-0.5, 0.5, -0.5),
            Vector(0.5, -0.5, 0.5),
            Vector(0.5, 0.5, -0.5),
            Vector(-0.5, 0.5, 0.5),
            Vector(0.5, 0.5, 0.5),
        ]
        edges = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 4),
            (1, 5),
            (2, 4),
            (2, 6),
            (3, 5),
            (3, 6),
            (4, 7),
            (5, 7),
            (6, 7),
        ]
        surfaces = [
            (0, 1, 3, 5),
            (0, 1, 2, 4),
            (0, 2, 3, 6),
            (2, 4, 6, 7),
            (1, 4, 5, 7),
            (3, 5, 6, 7),
        ]
        super().__init__(vertices, edges, surfaces, offset, size)
        self.resize()
        self.move()


class Tetrahedron(Object):
    """
    3D Object representing a Pyramid
    """

    def __init__(self, offset: Vector, size: float):
        alpha = 1/2
        vertices = [
            Vector(-alpha, alpha, -alpha),
            Vector(alpha, -alpha, -alpha),
            Vector(-alpha, -alpha, alpha),
            Vector(-alpha, -alpha, -alpha),
        ]
        edges = [
            (0, 3),
            (0, 1),
            (0, 2),
            (1, 3),
            (1, 2),
            (2, 3),
        ]
        surfaces = [
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 3),
            (1, 3, 2),
        ]
        super().__init__(vertices, edges, surfaces, offset, size)
        self.resize()
        self.move()


class Octahedron(Object):
    """
    3D Object reprenting a Octahedron
    """

    def __init__(self, offset: Vector, size: float):
        vertices = [
            Vector(-0.5, 0.0, -0.5),
            Vector(0.5, 0.0, -0.5),
            Vector(-0.5, 0.0, 0.5),
            Vector(0.5, 0.0, 0.5),
            Vector(0.0, math.sqrt(2 / 3), 0.0),
            Vector(0.0, -math.sqrt(2 / 3), 0.0),
        ]
        edges = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 3),
            (0, 4),
            (1, 4),
            (2, 4),
            (3, 4),
            (0, 5),
            (1, 5),
            (2, 5),
            (3, 5),
        ]
        surfaces = [
            (0, 1, 2, 3),
            (0, 1, 4),
            (0, 2, 4),
            (1, 3, 4),
            (2, 3, 4),
            (0, 1, 5),
            (0, 2, 5),
            (1, 3, 5),
            (2, 3, 5),
        ]
        super().__init__(vertices, edges, surfaces, offset, size)
        self.resize()
        self.move()


class Dodecahedron(Object):
    """
    3D Object representing a Dodecahedron
    """

    def __init__(self, offset: Vector, size: float):
        k = (1 + math.sqrt(5)) / 4
        beta = (3 + math.sqrt(5)) / 4
        vertices = [
            Vector(-k, -k, -k),
            Vector(k, -k, -k),
            Vector(-k, -k, k),
            Vector(-k, k, -k),
            Vector(k, -k, k),
            Vector(k, k, -k),
            Vector(-k, k, k),
            Vector(k, k, k),
            Vector(-0.5, beta, 0.0),
            Vector(0.5, beta, 0.0),
            Vector(-0.5, -beta, 0.0),
            Vector(0.5, -beta, 0.0),
            Vector(-beta, 0.0, 0.5),
            Vector(-beta, 0.0, -0.5),
            Vector(beta, 0.0, 0.5),
            Vector(beta, 0.0, -0.5),
            Vector(0, 0.5, -beta),
            Vector(0, -0.5, -beta),
            Vector(0, 0.5, beta),
            Vector(0, -0.5, beta),
        ]
        edges = [
            (8, 9),
            (10, 11),
            (12, 13),
            (14, 15),
            (16, 17),
            (18, 19),
            (0, 17),
            (0, 13),
            (0, 10),
            (1, 11),
            (1, 17),
            (1, 15),
            (2, 12),
            (2, 10),
            (2, 19),
            (4, 19),
            (4, 14),
            (4, 11),
            (3, 16),
            (3, 13),
            (3, 8),
            (5, 16),
            (5, 9),
            (5, 15),
            (6, 8),
            (6, 12),
            (6, 18),
            (7, 18),
            (7, 9),
            (7, 14),
        ]
        surfaces = [
            (0, 17, 16, 3, 13),
            (17, 1, 15, 5, 16),
            (16, 5, 9, 8, 3),
            (10, 11, 1, 17, 0),
            (15, 14, 7, 9, 5),
            (11, 1, 15, 14, 4),
            (12, 13, 3, 8, 6),
            (10, 0, 13, 12, 2),
            (8, 9, 7, 18, 6),
            (4, 11, 10, 2, 19),
            (19, 18, 7, 14, 4),
            (18, 19, 2, 12, 6),
        ]
        super().__init__(vertices, edges, surfaces, offset, size)
        self.add_center()
        self.resize()
        self.move()
    
    def draw(self):
        """
        Method to draw completely (wireframe and surfaces) a Dodecahedron using OpenGl primitives.
        This method override the Object draw() method because it's a special case that needs an other opengl
        triangles primitives
        """
        for surface in self.surfaces:
            glBegin(GL_TRIANGLE_FAN)
            glColor4fv(self.color)
            for vertex in surface:
                glVertex3f(
                    self.vertices[vertex].get_x(),
                    self.vertices[vertex].get_y(),
                    self.vertices[vertex].get_z(),
                )
            glEnd()

            self.invert_color()
            glBegin(GL_LINES)
            glColor4fv(self.color)
            for i in range(len(self.edges)):
                glColor4fv(self.color)
                for vertex in self.edges[i]:
                    glVertex3f(
                        self.vertices[vertex].get_x(),
                        self.vertices[vertex].get_y(),
                        self.vertices[vertex].get_z(),
                    )
            glEnd()

            self.invert_color()

    def add_center(self):
        """
        Method that compute and add to each surface of the Dodecahedron the
        coordinates of the surface center
        """
        edges = []
        points = []
        # For each pentagon of the the dodecahedron
        for i in range(len(self.surfaces)):
            # For each vertices of the face
            for j in range(5):
                # Getting a notable edge of the pentagon and save it and it's reverse
                # Also saving the opposite point of the known edge
                if j < 4:
                    if (
                        self.surfaces[i][j] + 1 == self.surfaces[i][j + 1]
                        or self.surfaces[i][j] - 1 == self.surfaces[i][j + 1]
                    ):
                        final_edge = (
                            self.surfaces[i][j],
                            self.surfaces[i][j + 1],
                        )
                        if final_edge in edges:
                            if not final_edge[::-1] in edges:
                                edges.append(final_edge[::-1])
                        else:
                            edges.append(final_edge)
                        if j + 3 > 4:
                            points.append(self.surfaces[i][-(5 - (j + 3))])
                        else:
                            points.append(self.surfaces[i][j + 3])
                        break

                elif j == 4:
                    if (
                        self.surfaces[i][j] + 1 == self.surfaces[i][0]
                        or self.surfaces[i][j] - 1 == self.surfaces[i][0]
                    ):
                        final_edge = (
                            self.surfaces[i][j],
                            self.surfaces[i][0],
                        )
                        if final_edge in edges:
                            if not final_edge[::-1] in edges:
                                edges.append(final_edge[::-1])
                        else:
                            edges.append(final_edge)
                        points.append(self.surfaces[i][2])
                        break
        # Then we find the center of the pentagon
        for i in range(len(edges)):
            edge_middle: Vector = compute_middle(
                self.vertices[edges[i][0]], self.vertices[edges[i][1]]
            )  # Computing middle of the edge
            surface_middle: Vector = compute_middle(
                edge_middle, self.vertices[points[i]]
            )  # Computing center of the surface
            self.vertices.append(
                surface_middle
            )  # Appending center to the object vertices
            # Adding the new vertice to the surface vertices index as well as the first point to have a circular drawing
            self.surfaces[i] = (
                (self.vertices.index(self.vertices[-1]),)
                + self.surfaces[i]
                + (self.surfaces[i][0],)
            )


class Icosahedron(Object):
    """
    3D Object representing a Icosahedron
    """

    def __init__(self, offset: Vector, size: float):
        k = (1 + math.sqrt(5)) / 4
        vertices = [
            Vector(-0.5, k, 0.0),
            Vector(0.5, k, 0.0),
            Vector(-0.5, -k, 0.0),
            Vector(0.5, -k, 0.0),
            Vector(-k, 0.0, 0.5),
            Vector(-k, 0.0, -0.5),
            Vector(k, 0.0, 0.5),
            Vector(k, 0.0, -0.5),
            Vector(0.0, 0.5, -k),
            Vector(0.0, 0.5, k),
            Vector(0.0, -0.5, k),
            Vector(0.0, -0.5, -k),
        ]
        edges = [
            (0, 5),
            (0, 4),
            (0, 8),
            (0, 9),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
            (2, 5),
            (2, 4),
            (2, 11),
            (2, 10),
            (3, 6),
            (3, 7),
            (3, 11),
            (3, 10),
            (7, 8),
            (7, 11),
            (5, 8),
            (5, 11),
            (6, 9),
            (6, 10),
            (4, 9),
            (4, 10),
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (8, 11),
            (9, 10),
        ]
        surfaces = [
            (0, 1, 8),
            (0, 1, 9),
            (6, 7, 1),
            (6, 7, 3),
            (8, 11, 7),
            (8, 11, 5),
            (9, 10, 6),
            (9, 10, 4),
            (4, 5, 0),
            (4, 5, 2),
            (2, 3, 11),
            (2, 3, 10),
            (8, 1, 7),
            (1, 6, 9),
            (3, 11, 7),
            (3, 6, 10),
            (3, 6, 10),
            (5, 8, 0),
            (0, 9, 4),
            (5, 11, 2),
            (4, 10, 2),
        ]
        super().__init__(vertices, edges, surfaces, offset, size)
        self.resize()
        self.move()

def compute_middle(start_point: Vector, end_point: Vector) -> Vector:
    """
    Method that compute the middle of a segment
    """
    return Vector(
        (start_point.get_x() + end_point.get_x()) / 2,
        (start_point.get_y() + end_point.get_y()) / 2,
        (start_point.get_z() + end_point.get_z()) / 2,
    )