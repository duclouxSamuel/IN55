# C'est quoi un pet froid ?
from PyQt5 import QtCore, QtGui, QtOpenGL, QtWidgets

import OpenGL.GL as gl
from OpenGL import GLU
from PyQt5.QtWidgets import QColorDialog

from objects import *
from obj import export
from fractal import Fractal, generate_fractal

class GLWidget(QtOpenGL.QGLWidget):
    """
    Class that represent an OpenGL render widget
    """
    def __init__(self, parent=None):
        self.parent = parent
        QtOpenGL.QGLWidget.__init__(self, parent)

    def initializeGL(self):
        """
        Method that initialize OpenGL render window
        """
        self.qglClearColor(QtGui.QColor(10, 10, 10)) #Setting screen color to gray
        gl.glEnable(gl.GL_DEPTH_TEST) #Enabling depth testing

        self.rot_x = 0.0
        self.rot_y = 0.0
        self.rot_z = 0.0
        self.zoom = 0.0
        self.fractal_depth = 1
        self.object = Cube(Vector(0.0, 0.0, 0.0), 1.0)
        self.fractal = Fractal(self.object.__class__.__name__)
        self.start_render = False

    def resizeGL(self, width, height):
        """
        Method that update the render window when it's resized
        """
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        aspect = width/ float(height)

        GLU.gluPerspective(45.0, aspect, 1.0, 300.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def paintGL(self):
        """
        Method to render elements on the render window
        """
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glPushMatrix()
        gl.glTranslate(0.0, 0.0, self.zoom)
        gl.glScale(2.0, 2.0, 2.0)
        gl.glRotate(self.rot_x, 1.0, 0.0, 0.0)
        gl.glRotate(self.rot_y, 0.0, 1.0, 0.0)
        gl.glRotate(self.rot_z, 0.0, 0.0, 1.0)
        # Drawing fractal if user clicked on display button
        if self.start_render:
            self.fractal.draw_old()
        gl.glPopMatrix()

    def set_rot_x(self, val):
        """
        Method to set the x rotation
        """
        self.rot_x = val

    def set_rot_y(self, val):
        """
        Method to set the y rotation
        """
        self.rot_y = val

    def set_rot_z(self, val):
        """
        Method to set the z rotation
        """
        self.rot_z = val

    def set_zoom(self, val):
        """
        Method to set window zoom
        """
        self.zoom = val*self.fractal_depth

    def set_object(self, val):
        """
        Method to set the current object
        """
        if val == 0:
            self.object = Cube(Vector(0.0, 0.0, 0.0), 1.0)  
        elif val == 1:
            self.object = Tetrahedron(Vector(0.0, 0.0, 0.0), 1.0)
        elif val == 2:
            self.object = Octahedron(Vector(0.0, 0.0, 0.0), 1.0)
        elif val == 3:
            self.object = Dodecahedron(Vector(0.0, 0.0, 0.0), 1.0)
        elif val == 4:
            self.object = Icosahedron(Vector(0.0, 0.0, 0.0), 1.0)

    def gen_fractal(self, fractal_depth):
        """
        Method that call the specific method to generate a fractal
        """
        if fractal_depth:
            self.fractal_depth = int(fractal_depth)
            self.fractal = generate_fractal(type(self.object).__name__, self.fractal_depth)
            # Merge shapes contained in the generated fractal object to optimize rendering
            #self.fractal.merge_shapes()
            #self.fractal.init_drawing(1000, 1000, 1000) # Init buffers in gpu for drawing

    def export_obj(self):
        """
        Method that call the obj export method to export the currently generated fractal
        """
        export("", "fractal", self.fractal)

    def switch_display(self):
        """
        Method that enable/disable the fractal display
        """
        self.start_render = not self.start_render

    def disable_display(self):
        """
        Method that disable the fractal display
        """
        self.start_render = False

    def enable_display(self):
        """
        Method that enable the fractal display
        """
        self.start_render = True

    def set_object_color(self, red, green, blue, alpha):
        """
        Method that set the object color
        """
        color = (red, green, blue, alpha)
        #print(color)
        self.fractal.set_color(color)

class Window(QtWidgets.QMainWindow):
    """
    Class that represent the software window. It's Inherited from QtMainWindow
    """
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self) # Init super class
        # Defining window parameters
        self.resize(1200, 900)
        self.setWindowTitle('Fractal Generator')
        # Creating specific OpenGl widget
        self.glWidget = GLWidget()
        # Init gui
        self.init_gui()
        # Creating a timer to update render widget
        timer = QtCore.QTimer(self)
        timer.setInterval(20) # Time interval in ms
        timer.timeout.connect(self.glWidget.updateGL)
        timer.start()

    def color_picker(self):
        """
        Method that execute the color picking wizard
        """
        dialog = QColorDialog(self)
        self.glWidget.disable_display()
        color = dialog.getColor()
        if color:
            red = color.red() / 255
            green = color.green() / 255
            blue = color.blue() / 255
            alpha = color.alpha() / 255

            self.glWidget.set_object_color(red, green, blue, alpha)
            self.glWidget.enable_display()

    def init_gui(self):
        combo_options = ["Cube", "Tétraèdre", "Octaèdre", "Dodécaèdre", "Icosaèdre"] # List storing all 3D object user can use to generate fractals

        # Creating central widget
        central_widget = QtWidgets.QWidget()
        # Creating widget that will contain GUI
        gui_widget = QtWidgets.QWidget()
        # Creating a grid layout for the central widget
        window_layout = QtWidgets.QGridLayout()
        # Creating a grid layout for the gui widget
        gui_layout = QtWidgets.QGridLayout()
        # Setting layouts
        central_widget.setLayout(window_layout)
        gui_widget.setLayout(gui_layout)
        # Setting central widget as window central widget
        self.setCentralWidget(central_widget)
        # Adding to the central widget layout the opengl render widget and the gui widget
        window_layout.addWidget(self.glWidget, 0, 0, 1, 3)
        window_layout.addWidget(gui_widget, 0, 3, 1, 1)

        # Creating all the widgets we need for the GUI
        objects_combo = QtWidgets.QComboBox()
        # Adding choices to combobox
        for item in combo_options:
            objects_combo.addItem(item)
        objects_combo.currentIndexChanged.connect(lambda val: self.glWidget.set_object(val)) # Linking combo box to object selection

        objects_label = QtWidgets.QLabel("Objet :")
        iter_label = QtWidgets.QLabel("Profondeur :")
        iter_textbox = QtWidgets.QLineEdit()
        iter_textbox.setMaxLength(3)
        slider_x = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider_x.valueChanged.connect(lambda val: self.glWidget.set_rot_x(360-val)) # Linking slider value to scene rotation
        # Setting min and max values
        slider_x.setMinimum(0.0)
        slider_x.setMaximum(360.0)
        slider_y = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider_y.valueChanged.connect(lambda val: self.glWidget.set_rot_y(360-val)) # Linking slider value to scene rotation
        # Setting min and max values
        slider_y.setMinimum(0.0)
        slider_y.setMaximum(360.0)
        slider_z = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider_z.valueChanged.connect(lambda val: self.glWidget.set_rot_z(360-val)) # Linking slider value to scene rotation
        # Setting min and max values
        slider_z.setMinimum(0.0)
        slider_z.setMaximum(360.0)
        slider_zoom = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider_zoom.valueChanged.connect(lambda val: self.glWidget.set_zoom(val)) # Linking slider value to scene zoom
        # Setting min and max values
        slider_zoom.setMinimum(-40.0)
        slider_zoom.setMaximum(0.0)
        x_label = QtWidgets.QLabel("Rotation X")
        y_label = QtWidgets.QLabel("Rotation Y")
        z_label = QtWidgets.QLabel("Rotation Z")
        zoom_label = QtWidgets.QLabel("Zoom")
        generate_button = QtWidgets.QPushButton("Générer")
        generate_button.clicked.connect(lambda : self.glWidget.gen_fractal(iter_textbox.text())) # Linking generate button to generate function
        display_button = QtWidgets.QPushButton("Afficher")
        display_button.clicked.connect(lambda : self.glWidget.switch_display()) # Linking display button to switch display function
        export_button = QtWidgets.QPushButton("Exporter")
        export_button.clicked.connect(lambda : self.glWidget.export_obj()) # Linking export button to export function
        color_button = QtWidgets.QPushButton("Pick color")
        color_button.clicked.connect(lambda: self.color_picker())


        # Adding all widget to the gui grid layout
        gui_layout.addWidget(objects_label, 0, 0)
        gui_layout.addWidget(objects_combo, 0, 1, 1, 2)
        gui_layout.addWidget(iter_label, 1, 0)
        gui_layout.addWidget(iter_textbox, 1, 1, 1, -1)
        gui_layout.addWidget(display_button, 2,0,1,-1)
        gui_layout.addWidget(generate_button, 3,0,1,-1)
        gui_layout.addWidget(export_button,4,0,1,-1)
        gui_layout.addWidget(x_label, 5, 0)
        gui_layout.addWidget(slider_x, 5,1,1,-1)
        gui_layout.addWidget(y_label, 6, 0)
        gui_layout.addWidget(slider_y,6,1,1,-1)
        gui_layout.addWidget(z_label, 7, 0)
        gui_layout.addWidget(slider_z,7,1,1,-1)
        gui_layout.addWidget(zoom_label, 8, 0)
        gui_layout.addWidget(slider_zoom,8,1,1,-1)
        gui_layout.addWidget(color_button, 9, 1, 1, -1)

# C'est un gaspacho ! (un gaz pas chaud toussa toussa)