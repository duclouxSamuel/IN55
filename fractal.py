from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo

from math import sqrt
from objects import *
import importlib

import numpy as np

class Fractal:
    """
    Class that represent a 3D Fractal
    """
    def __init__(self, class_name):
        self.__objects: list[Object] = []
        self.__vertices = []
        self.__vertices_VBO = None
        self.__edges_indices = []
        self.__surfaces_indices = []
        self.__class_name = class_name

    def add_object(self, object):
        """
        Method to add an object to the fractal's objects list
        """
        self.__objects.append(object)

    def get_object(self, indice):
        """
        Method that return the object stored in the objects list at the given index
        """
        return self.__objects[indice]

    def get_objects(self) -> list:
        return self.__objects
        
    def merge_shapes(self):
        """
        Method used to merge all the vertices of the fractal for rendering optimization
        """
        # Checking if the vertices and indices array are numpy array. If yes, we transform them in classic python list
        if isinstance(self.__vertices, np.ndarray):
            self.__vertices = self.__vertices.tolist()
        if isinstance(self.__edges_indices, np.ndarray):
            self.__vertices = self.__edges_indices.tolist()
        if isinstance(self.__surfaces_indices, np.ndarray):
            self.__vertices = self.__surfaces_indices.tolist()
        # We check each surface and each edge of each object in the fractal. 
        for obj in self.__objects:
            surfaces = obj.get_surfaces()
            edges = obj.get_edges()
            vertices = obj.get_vertices()
            for s in surfaces:
                for i in s:
                    # Eliminating double vertices in surfaces and rebuilding correctly surfaces according to new vertices indices
                    if vertices[i] in self.__vertices:
                        self.__surfaces_indices.append(self.__vertices.index(vertices[i]))
                    else:
                        self.__vertices.append(vertices[i])
                        self.__surfaces_indices.append(self.__vertices.index(vertices[i]))
            for e in edges:
                # Rebuilding edges with each new vertices indices
                for i in e:
                    self.__edges_indices.append(self.__vertices.index(vertices[i]))

        # Transforming each vector in a python list, not Vector object
        new_vertices = []
        for v in self.__vertices:
            new_vertices.append(v.to_list())

        # Reshaping list, transforming them to numpy array with appropriate type and binding vertices vbo
        self.__vertices = new_vertices
        self.__vertices = np.array(self.__vertices, dtype=np.float32)
        self.__vertices_VBO = vbo.VBO(np.reshape(self.__vertices,(1, -1)).astype(np.float32))
        self.__vertices_VBO.bind()
        self.__edges_indices = np.array(self.__edges_indices, dtype=np.uint32)
        self.__surfaces_indices = np.array(self.__surfaces_indices, dtype=np.uint32)

    def init_drawing(self, vertex_buffer_size, edges_buffer_size, surfaces_buffer_size):
        """
        Method that init element buffer object for fractal drawing
        """
        # Creating vertex buffer object in gpu
        VBO = glGenBuffers(1)
        # Bind the buffer
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        # Linking buffer to proper element
        glBufferData(GL_ARRAY_BUFFER, vertex_buffer_size, self.__vertices, GL_STATIC_DRAW)

        #Create edge buffer object in gpu
        EBO = glGenBuffers(1)
        # Bind the buffer
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        # Linking buffer to proper element
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, edges_buffer_size, self.__edges_indices, GL_STATIC_DRAW)

        #Create surface buffer object in gpu
        SBO = glGenBuffers(1)
        # Bind the buffer
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, SBO)
        # Linking buffer to proper element
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, surfaces_buffer_size, self.__surfaces_indices, GL_STATIC_DRAW)
    
    def draw_wireframe(self):
        """
        Method that draw the fractal wireframe using modern OpenGl technic
        """
        # Drawing fractal wireframe using OpenGl line primitive
        glDrawElements(GL_LINES,len(self.__edges_indices), GL_UNSIGNED_INT,  self.__edges_indices)

    def draw_wireframe_old(self):
        """
        Method that draw the fractal wirefram using old OpenGL technic
        """
        for o in self.__objects:
            o.draw_wireframe()

    def draw(self):
        """
        Method that draw the entire fractal (surfaces and wireframe) using modern OpenGl technic
        """
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self.__vertices_VBO)
        # Drawing object surfaces. The OpenGl primitives we'll use depends of the object
        if(self.__class_name == "Dodecahedron"):
            glDrawElements(GL_TRIANGLE_FAN,len(self.__surfaces_indices), GL_UNSIGNED_INT,  self.__surfaces_indices)
        else:
            glDrawElements(GL_TRIANGLES,len(self.__surfaces_indices), GL_UNSIGNED_INT,  self.__surfaces_indices)
        # Drawing fractal wireframe using OpenGl line primitive
        glDrawElements(GL_LINES,len(self.__edges_indices), GL_UNSIGNED_INT,  self.__edges_indices)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

    def draw_old(self):
        """
        Method that draw the entire fractal (surfaces and wirefram) using modern OpenGl technic
        """
        for o in self.__objects:
            o.draw()

    def set_color(self, color):
        """
        Method that change the color of each object in the fractal
        """
        for o in self.__objects:
            o.set_color(color)


def generate_fractal(class_name: str, depth: int) -> list[Object]:
    """
    Method that generate a fractal given a specific 3D Object and a recursion depth
    """
    Obj = getattr(importlib.import_module("objects"), class_name)
    obj = Obj(Vector(0, 0, 0), 1.0)
    fractal = Fractal(class_name)
    fractal.add_object(obj)


    if class_name == "Cube":
        fractal.get_object(0).set_size(float(3**(depth)))

        for _ in range (depth):
            fractal = fractal_cube(fractal,Obj)

        return fractal
    elif class_name == "Tetrahedron":
        fractal.get_object(0).set_size(float(2**(depth)))

        for _ in range (depth):
            fractal = fractal_tetrahedron(fractal,Obj)

        return fractal
    elif class_name == "Octahedron":
        fractal.get_object(0).set_size(float(2**(depth)))

        for _ in range (depth):
            fractal = fractal_octahedron(fractal,Obj)

        return fractal
    elif class_name == "Dodecahedron":
        fractal.get_object(0).set_size(float((2+((1+sqrt(5))/2)))**(depth))

        for _ in range (depth):
            fractal = fractal_dodecahedron(fractal,Obj)

        return fractal
    elif class_name == "Icosahedron":
        fractal.get_object(0).set_size(float((1+((1+sqrt(5))/2))**(depth)))

        for _ in range (depth):
            fractal = fractal_icosahedron(fractal,Obj)

        return fractal


def fractal_cube (previous:Fractal,Obj)->Fractal:
    """
    Method that generate a cube fractal in a recursive way
    """
    returned=Fractal(Obj.__class__.__name__)
    for o in previous.get_objects():
        size=o.size/3
        x = o.offset.get_x()
        y = o.offset.get_y()
        z = o.offset.get_z()

        returned.add_object(Obj(Vector(x + size, y + size, z + size), size))
        returned.add_object(Obj(Vector(x + size, y + size, z - size), size))
        returned.add_object(Obj(Vector(x + size, y - size, z + size), size))
        returned.add_object(Obj(Vector(x + size, y - size, z - size), size))
        returned.add_object(Obj(Vector(x - size, y + size, z + size), size))
        returned.add_object(Obj(Vector(x - size, y + size, z - size), size))
        returned.add_object(Obj(Vector(x - size, y - size, z + size), size))
        returned.add_object(Obj(Vector(x - size, y - size, z - size), size))

        returned.add_object(Obj(Vector(x + size, y + size, z), size))
        returned.add_object(Obj(Vector(x + size, y - size, z), size))
        returned.add_object(Obj(Vector(x - size, y + size, z), size))
        returned.add_object(Obj(Vector(x - size, y - size, z), size))

        returned.add_object(Obj(Vector(x + size, y, z + size), size))
        returned.add_object(Obj(Vector(x + size, y, z - size), size))
        returned.add_object(Obj(Vector(x - size, y, z + size), size))
        returned.add_object(Obj(Vector(x - size, y, z - size), size))

        returned.add_object(Obj(Vector(x, y + size, z + size), size))
        returned.add_object(Obj(Vector(x, y + size, z - size), size))
        returned.add_object(Obj(Vector(x, y - size, z + size), size))
        returned.add_object(Obj(Vector(x, y - size, z - size), size))

    return returned

def fractal_tetrahedron (previous:Fractal,Obj)->Fractal:
    """
    Method that generate a tetrahedron fractal in a recursive way
    """
    returned=Fractal(Obj.__class__.__name__)
    previous_size = previous.get_object(0).size
    alpha = 1/2
    for o in previous.get_objects():
        size=o.size/2
        x = o.offset.get_x()
        y = o.offset.get_y()
        z = o.offset.get_z()

        returned.add_object(Obj(Vector(x - ((previous_size - size)*alpha), y + ((previous_size - size)*alpha), z - ((previous_size - size)*alpha)), size))
        returned.add_object(Obj(Vector(x + ((previous_size - size)*alpha), y - ((previous_size - size)*alpha), z - ((previous_size - size)*alpha)), size))
        returned.add_object(Obj(Vector(x - ((previous_size - size)*alpha), y - ((previous_size - size)*alpha), z + ((previous_size - size)*alpha)), size))
        returned.add_object(Obj(Vector(x - ((previous_size - size)*alpha), y - ((previous_size - size)*alpha), z - ((previous_size - size)*alpha)), size))

    return returned

def fractal_octahedron (previous:Fractal,Obj)->Fractal:
    """
    Method that generate a octahedron fractal in a recursive way
    """
    returned=Fractal(Obj.__class__.__name__)
    for o in previous.get_objects():
        size=o.size/2
        x = o.offset.get_x()
        y = o.offset.get_y()
        z = o.offset.get_z()

        returned.add_object(Obj(Vector(x, y + size*sqrt(2 / 3), z), size))
        returned.add_object(Obj(Vector(x, y - size*sqrt(2 / 3), z), size))
        returned.add_object(Obj(Vector(x + size/2, y, z + size/2), size))
        returned.add_object(Obj(Vector(x + size/2, y, z - size/2), size))
        returned.add_object(Obj(Vector(x - size/2, y, z + size/2), size))
        returned.add_object(Obj(Vector(x - size/2, y, z - size/2), size))

    return returned

def fractal_dodecahedron (previous:Fractal,Obj)->Fractal:
    """
    Method that generate a dodecahedron fractal in a recursive way
    """
    returned=Fractal(Obj.__class__.__name__)
    k = ((1 + sqrt(5))/4)
    beta = ((3 + sqrt(5))/4)
    previous_size = previous.get_object(0).size
    for o in previous.get_objects():
        size=o.size/(2+((1+sqrt(5))/2))
        x = o.offset.get_x()
        y = o.offset.get_y()
        z = o.offset.get_z()

        returned.add_object(Obj(Vector(x + ((previous_size - size)*k), y + ((previous_size - size)*k), z + ((previous_size - size)*k)), size))
        returned.add_object(Obj(Vector(x + ((previous_size - size)*k), y + ((previous_size - size)*k), z - ((previous_size - size)*k)), size))
        returned.add_object(Obj(Vector(x + ((previous_size - size)*k), y - ((previous_size - size)*k), z + ((previous_size - size)*k)), size))
        returned.add_object(Obj(Vector(x + ((previous_size - size)*k), y - ((previous_size - size)*k), z - ((previous_size - size)*k)), size))
        returned.add_object(Obj(Vector(x - ((previous_size - size)*k), y + ((previous_size - size)*k), z + ((previous_size - size)*k)), size))
        returned.add_object(Obj(Vector(x - ((previous_size - size)*k), y + ((previous_size - size)*k), z - ((previous_size - size)*k)), size))
        returned.add_object(Obj(Vector(x - ((previous_size - size)*k), y - ((previous_size - size)*k), z + ((previous_size - size)*k)), size))
        returned.add_object(Obj(Vector(x - ((previous_size - size)*k), y - ((previous_size - size)*k), z - ((previous_size - size)*k)), size))

        returned.add_object(Obj(Vector(x + ((previous_size - size)*0.5), y + ((previous_size - size)*beta), z), size))
        returned.add_object(Obj(Vector(x + ((previous_size - size)*0.5), y - ((previous_size - size)*beta), z), size))
        returned.add_object(Obj(Vector(x - ((previous_size - size)*0.5), y + ((previous_size - size)*beta), z), size))
        returned.add_object(Obj(Vector(x - ((previous_size - size)*0.5), y - ((previous_size - size)*beta), z), size))

        returned.add_object(Obj(Vector(x + ((previous_size - size)*beta), y, z + (previous_size - size)*0.5), size))
        returned.add_object(Obj(Vector(x + ((previous_size - size)*beta), y, z - (previous_size - size)*0.5), size))
        returned.add_object(Obj(Vector(x - ((previous_size - size)*beta), y, z + (previous_size - size)*0.5), size))
        returned.add_object(Obj(Vector(x - ((previous_size - size)*beta), y, z - (previous_size - size)*0.5), size))

        returned.add_object(Obj(Vector(x, y + ((previous_size - size)*0.5), z + ((previous_size - size)*beta)), size))
        returned.add_object(Obj(Vector(x, y + ((previous_size - size)*0.5), z - ((previous_size - size)*beta)), size))
        returned.add_object(Obj(Vector(x, y - ((previous_size - size)*0.5), z + ((previous_size - size)*beta)), size))
        returned.add_object(Obj(Vector(x, y - ((previous_size - size)*0.5), z - ((previous_size - size)*beta)), size))

    return returned

def fractal_icosahedron(previous:Fractal,Obj)->Fractal:
    """
    Method that generate a icosahedreon fractal in a recursive way
    """
    returned=Fractal(Obj.__class__.__name__)
    k = (1 + sqrt(5)) / 4
    previous_size = previous.get_object(0).size
    for o in previous.get_objects():
        size=o.size/(1+((1+sqrt(5))/2))
        x = o.offset.get_x()
        y = o.offset.get_y()
        z = o.offset.get_z()

        returned.add_object(Obj(Vector(x + ((previous_size - size)*0.5), y + ((previous_size - size)*k), z), size))
        returned.add_object(Obj(Vector(x + ((previous_size - size)*0.5), y - ((previous_size - size)*k), z), size))
        returned.add_object(Obj(Vector(x - ((previous_size - size)*0.5), y + ((previous_size - size)*k), z), size))
        returned.add_object(Obj(Vector(x - ((previous_size - size)*0.5), y - ((previous_size - size)*k), z), size))

        returned.add_object(Obj(Vector(x + ((previous_size - size)*k), y, z + ((previous_size - size)*0.5)), size))
        returned.add_object(Obj(Vector(x + ((previous_size - size)*k), y, z - ((previous_size - size)*0.5)), size))
        returned.add_object(Obj(Vector(x - ((previous_size - size)*k), y, z + ((previous_size - size)*0.5)), size))
        returned.add_object(Obj(Vector(x - ((previous_size - size)*k), y, z - ((previous_size - size)*0.5)), size))

        returned.add_object(Obj(Vector(x, y + ((previous_size - size)*0.5), z + ((previous_size - size)*k)), size))
        returned.add_object(Obj(Vector(x, y + ((previous_size - size)*0.5), z - ((previous_size - size)*k)), size))
        returned.add_object(Obj(Vector(x, y - ((previous_size - size)*0.5), z + ((previous_size - size)*k)), size))
        returned.add_object(Obj(Vector(x, y - ((previous_size - size)*0.5), z - ((previous_size - size)*k)), size))

    return returned
