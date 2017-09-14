from IECview import IECwindow
from silx.gui import qt
from HPLCrun import HPLCrun
from PyQt4.QtCore import pyqtSlot


#bufferrun = None
#samplerun = None


@pyqtSlot(int)
def frameSelectedDo(frame):
    window.addOneCurve(samplerun.q,samplerun.I[frame,:], handle = "data", frameNr= frame)
    window.addOneCurve(bufferrun.q,bufferrun.I[frame,:],handle = "buffer",frameNr= frame)
    window.addOneCurve(bufferrun.q,samplerun.I[frame,:]-bufferrun.I[frame,:],handle = "sub",frameNr= frame)
def main():   
    global app
    global window
    global samplerun, bufferrun
    app = qt.QApplication([])
    samplerun = HPLCrun("BSA_010.h5")
    bufferrun = HPLCrun("buffer_007.h5")
    # Create a ThreadSafePlot1D, set its limits and display it
    #plot1d = ThreadSafePlot1D()
    #plot1d.setLimits(0., 1000., 0., 1.)
    #plot1d.show()
    
    window = IECwindow()
    window.frameSelected.connect(frameSelectedDo)
    window.addOneCurve(samplerun.q,samplerun.I[1200,:], handle = "data",frameNr= 1200)
    window.addOneCurve(bufferrun.q,bufferrun.I[1200,:],handle = "buffer",frameNr= 1200)
    window.addChromo(samplerun.sum_I, handle = "data", cType= "sum")
    window.addChromo(bufferrun.sum_I, handle = "buffer", cType= "sum")
    #window.addOneCurve(samplerun.q,samplerun.I[1040,:])
    window.setVisible(True)
    
    # Create the thread that calls ThreadSafePlot1D.addCurveThreadSafe
   # updateThread = UpdateThreadSAXS(window)
    #updateThread.start()  # Start updating the plot

    app.exec_()

   # updateThread.stop()  # Stop updating the plot    
    
if __name__ == '__main__':
    
    main()