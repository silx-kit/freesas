from silx.gui import qt
from silx.gui import plot
import numpy
import silx.test.utils
from silx.gui.plot.utils.axis import SyncAxes
import threading
import time
from silx.gui.plot import Plot1D
from silx.gui.widgets.ThreadPoolPushButton import ThreadPoolPushButton
from silx.gui.widgets.WaitingPushButton import WaitingPushButton
from PyQt4.QtGui import QSlider, QLabel
from PyQt4.QtCore import Qt
from PyQt4.QtCore import pyqtSignal


class IECwindow(qt.QMainWindow):
     
    updateThread = None
    frameslide = None
    runLength = 0
    
    frameSelected = pyqtSignal(int)

    def __init__(self, parent = None):
        qt.QMainWindow.__init__(self)
        self.setWindowTitle("Plot with synchronized axes")
        widget = qt.QWidget(self)
        self.setCentralWidget(widget)
        self.updateThread = None
        layout = qt.QGridLayout()
        widget.setLayout(layout)
        backend = "mpl"
        #self.plot2d = plot.Plot2D(parent=widget, backend=backend)
        #self.plot2d.setInteractiveMode('pan')
        self.plot1d_chromo = plot.Plot1D(parent=widget, backend=backend)
        self.plot1d_log = plot.Plot1D(parent=widget, backend=backend)

        self.plot1d_log.getYAxis().setScale("log")
        
        self.frameSlider = self.createFrameSlider()
        
     
        self.l1 = QLabel( str(self.plot1d_chromo.getXAxis().getLimits()[0]) + "," + str(self.frameSlider.minimum))
        self.l1.setAlignment(Qt.AlignCenter)
        #self.semi = self.plot1d_log.addCurve(x=self.q,y=I)
        #self.plot1d_loglog.addCurve(x=self.q,y=I)
        #self.plot1d_kratky.addCurve(x=self.q, y=I*self.q*self.q)
        #self.plot1d_holtzer.addCurve(x=self.q, y=I*self.q)
              
        #self.constraint1 = SyncAxes([self.plot2d.getXAxis(), self.plot1d_x1.getXAxis(), self.plot1d_x2.getXAxis()])
        #self.constraint2 = SyncAxes([self.plot2d.getYAxis(), self.plot1d_y1.getYAxis(), self.plot1d_y2.getYAxis()])
        #self.constraint3 = SyncAxes([self.plot1d_x1.getYAxis(), self.plot1d_y1.getXAxis()])
        #self.constraint1 = SyncAxes([self.plot1d_log.getXAxis(), self.plot1d_loglog.getXAxis(),self.plot1d_kratky.getXAxis(),self.plot1d_holtzer.getXAxis()], syncScale=False)

        #self.plot1d_kratky.getYAxis().setLimits(0,medfilt(I*self.q*self.q,21).max())
        layout.addWidget(self.plot1d_chromo, 0, 0)
        layout.addWidget(self.plot1d_log, 0, 1)
        
        layout.addWidget(self.frameSlider)
        
      
        layout.addWidget(self.l1)
    
    def createCenteredLabel(self, text):
        label = qt.QLabel(self)
        label.setAlignment(qt.Qt.AlignCenter)
        label.setText(text)
        return label
    
    def createFrameSlider(self):
        self.frameslide  = SyncSlide(self.plot1d_chromo.getXAxis(),  Qt.Horizontal)
        self.frameslide.setMinimum(0)
        self.frameslide.setMaximum(3000)
        self.frameslide.setValue(1200)
        self.frameslide.valueChanged.connect(self.frameSelectedDo)
        self.frameslide.sigLimitsChanged.connect(self.frameRangeChange)
        return self.frameslide

    def frameSelectedDo(self):
        self.frameSelected.emit(self.frameslide.value())
        
    
    def addOneCurve(self,q,I, handle, frameNr):
        color = self.getColor(handle)
        self.plot1d_log.addCurve(x=q,y=I,  legend = handle, color= color)
        self.plot1d_log.setGraphTitle("Frame number " + str(frameNr))
        
    def addChromo(self,intensity,handle,cType):
        color = self.getColor(handle)
        runlength = intensity.shape[0]
        self.plot1d_chromo.addCurve(x=numpy.arange(runlength),y=intensity, legend = handle, color = color)
      
    def getColor(self,handle):
        if "data" in handle:
            return "red"
        if "buffer" in handle:
            return "black"
        if "sub" in handle:
            return "blue"
        
    def frameRangeChange(self):
        self.l1.setText( str(self.plot1d_chromo.getXAxis().getLimits()[0]) + "," + str(self.frameSlider.minimum))

        
import functools
import logging
from contextlib import contextmanager
from silx.utils import weakref

_logger = logging.getLogger(__name__)


#class LabeledSlider(QSlider):
    
    

class SyncSlide(QSlider):
    """Synchronize a slider to an axis
    """
    sigLimitsChanged = pyqtSignal(float, float)
    minimum = 0
    maximum = 0
    
    def __init__(self, axes, syncLimits = None, parent = None, *args, **kwargs):
        """
        Constructor
        :param axes: The axis to synchrnoize to
        :param slider: The slider
        """
        super(SyncSlide, self).__init__(parent=parent, *args, **kwargs)
        self.__axes = [axes]
        self.__locked = True
        self.__syncLimits = syncLimits
        self.__callbacks = []

        self.start()
        
    def setMinimum(self, *args, **kwargs):
        minimum = args[0]
        return QSlider.setMinimum(self, *args, **kwargs)
    
    def setMaximum(self, *args, **kwargs):
        maximum = args[0]
        return QSlider.setMaximum(self, *args, **kwargs)


    def start(self):
        """Start synchronizing axes together.
        The first axis is used as the reference for the first synchronization.
        After that, any changes to any axes will be used to synchronize other
        axes.
        """
        if len(self.__callbacks) != 0:
            raise RuntimeError("Axes already synchronized")

        # register callback for further sync
        axis = self.__axes[0]    
        # the weakref is needed to be able ignore self references
        callback = weakref.WeakMethodProxy(self.__axisLimitsChanged)
        callback = functools.partial(callback, axis)
        sig = axis.sigLimitsChanged
        sig.connect(callback)
        self.__callbacks.append((sig, callback))

        # the weakref is needed to be able ignore self references
        callback = weakref.WeakMethodProxy(self.__axisScaleChanged)
        callback = functools.partial(callback, axis)
        sig = axis.sigScaleChanged
        sig.connect(callback)
        self.__callbacks.append((sig, callback))

        # the weakref is needed to be able ignore self references
        callback = weakref.WeakMethodProxy(self.__axisInvertedChanged)
        callback = functools.partial(callback, axis)
        sig = axis.sigInvertedChanged
        sig.connect(callback)
        self.__callbacks.append((sig, callback))

        # sync the current state
        mainAxis = self.__axes[0]
       
        self.__axisLimitsChanged(mainAxis, *mainAxis.getLimits())
  
        self.__axisScaleChanged(mainAxis, mainAxis.getScale())
   
        self.__axisInvertedChanged(mainAxis, mainAxis.isInverted())

    def stop(self):
        """Stop the synchronization of the axes"""
        if len(self.__callbacks) == 0:
            raise RuntimeError("Axes not synchronized")
        for sig, callback in self.__callbacks:
            sig.disconnect(callback)
        self.__callbacks = []

    def __del__(self):
        """Destructor"""
        # clean up references
        if len(self.__callbacks) != 0:
            self.stop()

    @contextmanager
    def __inhibitSignals(self):
        self.__locked = True
        yield
        self.__locked = False


    def __axisLimitsChanged(self, changedAxis, vmin, vmax):
        if self.__locked:
            return
        with self.__inhibitSignals():
            self.setMinimum(vmin)
            self.setMaximum(vmax)
            self.sigLimitsChanged.emit(vmin,vmax)

    def __axisScaleChanged(self, changedAxis, scale):
        """"
        This does not do anything yet, not sure if it should
        """
        if self.__locked:
            return
        return

    def __axisInvertedChanged(self, changedAxis, isInverted):
        if self.__locked:
            return
        with self.__inhibitSignals():
            self.setInvertedAppearance(isInverted)
            
    
        