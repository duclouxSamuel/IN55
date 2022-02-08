from objects import *
from fractal import Fractal

def export(path: str, filename: str, fractal: Fractal):
    """
    Method that export a generated fractal as a Wavefront .obj file to the given path
    """
    file = open(path + filename + ".obj", "w")

    # Writing header
    file.write("#" + filename + ".obj\n")
    file.write("o " + filename + ".obj\n")

    # Writing body by adding to file each object repr
    offset = 0
    for obj in fractal.get_objects():
        # Writing object header
        # file.write("o " + str(type(obj).__name__) + "." + str(i) + "\n")
        # Writing body by adding to file each object repr str
        file.write(obj.to_obj(str(type(obj).__name__), offset))
        file.write("\n\n")
        offset += obj.get_vertices_number()
