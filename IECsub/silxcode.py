from silx.gui import qt
from silx.gui import plot
import numpy
import silx.test.utils
from silx.gui.plot.utils.axis import SyncAxes
import h5py
import threading
import time
from silx.gui.plot import Plot1D
from silx.gui.widgets.ThreadPoolPushButton import ThreadPoolPushButton
from silx.gui.widgets.WaitingPushButton import WaitingPushButton
from PyQt4.QtGui import QSlider
from PyQt4.QtCore import Qt
from scipy.signal import medfilt, savgol_filter


class SyncSAXSPlot(qt.QMainWindow):
     
    updateThread = None
    frameslide = None

    def __init__(self, counter = 1040, parent = None):
        qt.QMainWindow.__init__(self)
        self.setWindowTitle("Plot with synchronized axes")
        widget = qt.QWidget(self)
        self.setCentralWidget(widget)
        self.updateThread = None
        layout = qt.QGridLayout()
        widget.setLayout(layout)
        counter = 1040
        backend = "mpl"
        #self.plot2d = plot.Plot2D(parent=widget, backend=backend)
        #self.plot2d.setInteractiveMode('pan')
        self.plot1d_log = plot.Plot1D(parent=widget, backend=backend)
        self.plot1d_loglog = plot.Plot1D(parent=widget, backend=backend)
        self.plot1d_kratky = plot.Plot1D(parent=widget, backend=backend)
        self.plot1d_holtzer = plot.Plot1D(parent=widget, backend=backend)
        
        self.plot1d_loglog.getXAxis().setScale("log")
        self.plot1d_loglog.getYAxis().setScale("log")
        self.plot1d_log.getYAxis().setScale("log")
        
        print(counter)
        with h5py.File("BSA_047.h5", "r") as data:
            self.q = numpy.asarray(data['q'])[:]
            #I = numpy.asarray(data['subtracted_I'])[counter,:]
            #Ierr = numpy.asarray(data['subtracted_Stdev'])[counter,:]
        
        
        #self.semi = self.plot1d_log.addCurve(x=self.q,y=I)
        #self.plot1d_loglog.addCurve(x=self.q,y=I)
        #self.plot1d_kratky.addCurve(x=self.q, y=I*self.q*self.q)
        #self.plot1d_holtzer.addCurve(x=self.q, y=I*self.q)
        self.updateCurve(counter)
              
        #self.constraint1 = SyncAxes([self.plot2d.getXAxis(), self.plot1d_x1.getXAxis(), self.plot1d_x2.getXAxis()])
        #self.constraint2 = SyncAxes([self.plot2d.getYAxis(), self.plot1d_y1.getYAxis(), self.plot1d_y2.getYAxis()])
        #self.constraint3 = SyncAxes([self.plot1d_x1.getYAxis(), self.plot1d_y1.getXAxis()])
        self.constraint1 = SyncAxes([self.plot1d_log.getXAxis(), self.plot1d_loglog.getXAxis(),self.plot1d_kratky.getXAxis(),self.plot1d_holtzer.getXAxis()], syncScale=False)

        #self.plot1d_kratky.getYAxis().setLimits(0,medfilt(I*self.q*self.q,21).max())
        
        layout.addWidget(self.plot1d_log, 0, 0)
        layout.addWidget(self.plot1d_loglog, 0,1)
        layout.addWidget(self.plot1d_kratky, 1 ,0)
        layout.addWidget(self.plot1d_holtzer, 1, 1)
        
        slider = self.createFrameSlider()
        
        layout.addWidget(slider)
        
        updater = widget.layout().addWidget(self.createStartButton())
        widget.layout().addWidget(self.createStopButton(self.updateThread))

    def createCenteredLabel(self, text):
        label = qt.QLabel(self)
        label.setAlignment(qt.Qt.AlignCenter)
        label.setText(text)
        return label
    
    
    def createFrameSlider(self):
        self.frameslide  = QSlider(Qt.Horizontal)
        self.frameslide.setMinimum(0)
        self.frameslide.setMaximum(100)
        self.frameslide.setValue(20)
        self.frameslide.valueChanged.connect(self.valueChange)
        return self.frameslide
    
    def valueChange(self):
        counter = self.frameslide.value() + 1000
        self.updateCurve(counter)
    
    def createStartButton(self):
        widget = ThreadPoolPushButton(text="Start")
        #widget.clicked.connect(widget.start)
        widget.setCallable(self.update)
        #widget.succeeded.connect()
        #widget.failed.connect()
        return widget
    
    def createStopButton(self,updater):
        widget = ThreadPoolPushButton(text="Stop")
        #widget.clicked.connect(widget.start)
        widget.setCallable(self.stopUpdate)
        return widget

    def stopUpdate(self):
        if self.updateThread:
            self.updateThread.stop()
    
    def update(self):
        self.updateThread = UpdateThreadSAXS(self)
        self.updateThread.start()
        #return self.updateThread
    
    def updateCurve(self,counter):
        with h5py.File("BSA_047.h5", "r") as data:          
            I = numpy.asarray(data['subtracted_I'])[counter,:]
            Ierr = numpy.asarray(data['subtracted_Stdev'])[counter,:]
        self.plot1d_log.addCurve(x=self.q,y=I)
        self.plot1d_loglog.addCurve(x=self.q,y=I)
        self.plot1d_kratky.addCurve(x=self.q, y=I*self.q*self.q, legend = 'data')
        self.plot1d_kratky.addCurve(x=self.q, y = savgol_filter(I*self.q*self.q,len(self.q)//20,1,deriv = 0,delta=1.0), legend = 'fit')
        self.plot1d_holtzer.addCurve(x=self.q, y=I*self.q)
        self.plot1d_kratky.getYAxis().setLimits(0,savgol_filter(I*self.q*self.q,len(self.q)//20,1,deriv = 0,delta=1.0).max()*1.2)
        self.plot1d_holtzer.getYAxis().setLimits(0,savgol_filter(I*self.q,len(self.q)//20,1,deriv = 0,delta=1.0).max()*1.2)

        
        
    
    
class SyncSAXSPlotThread(SyncSAXSPlot):
    
    _sigUpdateCurve = qt.Signal(tuple, dict)
    """Signal used to perform addCurve in the main thread.
    It takes args and kwargs as arguments.
    """
    
    def __updateCurve(self, args, kwargs):
        """Private method calling addCurve from _sigAddCurve"""
        self.updateCurve(*args, **kwargs)

    def updateCurveThreadSafe(self, *args, **kwargs):
        """Thread-safe version of :meth:`silx.gui.plot.Plot.addCurve`
        This method takes the same arguments as Plot.addCurve.
        WARNING: This method does not return a value as opposed to Plot.addCurve
        """
        self._sigUpdateCurve.emit(args, kwargs)

        
    def __init__(self, counter, parent=None):
        super(SyncSAXSPlotThread, self).__init__(parent=parent, counter= 1020)
        # Connect the signal to the method actually calling addCurve
        self._sigUpdateCurve.connect(self.__updateCurve)
        
        
        
        
        
class UpdateThreadSAXS(threading.Thread):
    """Thread updating the curve of a :class:`ThreadSafePlot1D`
    :param plot1d: The ThreadSafePlot1D to update."""

    def __init__(self, plots):
        self.counter = 1000
        self.plots = plots
        self.running = False
        super(UpdateThreadSAXS, self).__init__()

    def start(self):
        """Start the update thread"""
        self.running = True
        super(UpdateThreadSAXS, self).start()
        
    def run(self):
        """Method implementing thread loop that updates the plot"""
        while self.running:
            time.sleep(0.1)
            self.plots.updateCurveThreadSafe(counter = self.counter)
            self.counter += 1
            if self.counter == 1101:
                self.counter = 1000

    def stop(self):
        """Stop the update thread"""
        self.running = False
        self.join(2)
   
   
def main():   
    global app
    app = qt.QApplication([])

    # Create a ThreadSafePlot1D, set its limits and display it
    #plot1d = ThreadSafePlot1D()
    #plot1d.setLimits(0., 1000., 0., 1.)
    #plot1d.show()
    window = SyncSAXSPlotThread(1040)
    window.setVisible(True)

    # Create the thread that calls ThreadSafePlot1D.addCurveThreadSafe
   # updateThread = UpdateThreadSAXS(window)
    #updateThread.start()  # Start updating the plot

    app.exec_()

   # updateThread.stop()  # Stop updating the plot    
    
if __name__ == '__main__':
    main()