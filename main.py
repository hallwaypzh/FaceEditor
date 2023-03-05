from PyQt5 import QtGui,QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


import numpy as np
import torch 

import PIL 
from PIL import Image 
from PIL.ImageQt import ImageQt
from torchvision.transforms import ToTensor

from modules.tpsmm.tpsmm import TPSMM 
#import cv2
import time

import argparse



class Canvas(QtWidgets.QWidget):

    DELTA = 16 #for the minimum distance  
    keyPressed = QtCore.pyqtSignal(QtCore.QEvent)      
    # ='/data/RAVDESS/images/01-01-05-02-02-01-02.mp4/00000.png
    # expression /data/RAVDESS/images/01-01-02-01-01-02-24.mp4/00000.png
    def __init__(self, parent, img_path='/data/RAVDESS/images/01-01-05-02-02-01-02.mp4/00000.png'):
        super(Canvas, self).__init__(parent)
        self.draggin_idx = -1        
        self.setGeometry(0,0,512 + 256,512)
        self.points = np.array([[v*5,v*5] for v in range(75)], dtype=np.float) 
        self.tpsmm = TPSMM()
        self.to_tensor = ToTensor()
        self.label = QLabel(self)
        self.DISPLAY_IMG_SIZE = 512
        self.FOMM_INPUTS_SIZE = 256
        self.label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.img0 = Image.open(img_path).convert('RGB').resize((self.FOMM_INPUTS_SIZE, self.FOMM_INPUTS_SIZE))
        self.img_tensor0 = self.to_tensor(self.img0).cuda().unsqueeze(0)
        self.kps0 = (self.tpsmm.get_kps(self.img_tensor0)[0].view(50, 2).detach().cpu().numpy() + 1) / 2 * self.DISPLAY_IMG_SIZE
        self.kps1 = self.tpsmm.get_kps(self.img_tensor0)[0].view(50, 2)
        self.display_kps = True
        

        self.COMP_DISPLAY = 10
        self.values = [0. for i in range(self.COMP_DISPLAY)]
        for i in range(self.COMP_DISPLAY):
            setattr(self, f'slider{i}', QtWidgets.QSlider(self))
            getattr(self, f'slider{i}').setGeometry(QtCore.QRect(self.DISPLAY_IMG_SIZE, i * 40+15, 256, 20))
            getattr(self, f'slider{i}').setOrientation(QtCore.Qt.Horizontal)
            getattr(self, f'slider{i}').setMinimum(-200)
            getattr(self, f'slider{i}').setMaximum(200)
            getattr(self, f'slider{i}').setValue(0)
            getattr(self, f'slider{i}').valueChanged.connect(self.pca_edit(i))


        pca_basis = np.load('pca_basis/RAV_KPS_PCA_30.npy', allow_pickle=True).item()
        self.pca_comp = pca_basis['comp']
        self.pca_stdev = pca_basis['stdev']   

        self.keyPressed.connect(self.on_key)
        # self.vis_flag = [False for i in range(50)]
        # vis_idx = [16, 3, 2, 19, 45, 40, 20, 17, 49]
        # for idx in vis_idx:
        #     self.vis_flag[idx] = True

    def paintEvent(self, e):
        qp = QtGui.QPainter()
        qp.begin(self)
        self.updateCanvas(qp)
        qp.end()
    
    def updateCanvas(self, qp):
        #self.drawPoints(qp)
       
        img1 = np.array(self.img0) / 255. 
        kps = self.kps0 / (self.DISPLAY_IMG_SIZE // 2) - 1.        
        img1 = self.tpsmm.generate(self.img_tensor0, self.kps1, torch.from_numpy(kps).cuda())
        img1 = img1[0].permute(1, 2, 0).detach().cpu().numpy()
        if self.display_kps:
            img1 = self.tpsmm.draw_image_with_kp(img1, kps.reshape(-1, 5, 2))
        #img1 = self.tpsmm.draw_image_with_kp_partial(img1, kps, self.vis_flag)
        img1 = Image.fromarray(np.uint8(img1 * 255.))

        showedImagePixmap = self.pil2pixmap(img1.resize((self.DISPLAY_IMG_SIZE, self.DISPLAY_IMG_SIZE)))

        self.label.setPixmap(showedImagePixmap)
        self.label.resize(showedImagePixmap.width(), showedImagePixmap.height())

    def _get_point(self, evt):
        return np.array([evt.pos().x(),evt.pos().y()])

    #get the click coordinates
    def mousePressEvent(self, evt):
        if evt.button() == QtCore.Qt.LeftButton and self.draggin_idx == -1:
            point = self._get_point(evt)
            #dist will hold the square distance from the click to the points
            dist = self.kps0 - point
            dist = dist[:,0]**2 + dist[:,1]**2
            dist[dist>self.DELTA] = np.inf #obviate the distances above DELTA
            if dist.min() < np.inf:
                self.draggin_idx = dist.argmin()  
                #print(self.draggin_idx)      

    def mouseMoveEvent(self, evt):
        if self.draggin_idx != -1:
            point = self._get_point(evt)
            self.kps0[self.draggin_idx] = point
            self.update()
            
    def mouseReleaseEvent(self, evt):
        if evt.button() == QtCore.Qt.LeftButton and self.draggin_idx != -1:
            point = self._get_point(evt)
            self.kps0[self.draggin_idx] = point
            self.draggin_idx = -1
            self.update()      

    def pil2pixmap(self, im):
        if im.mode == "RGB":
            r, g, b = im.split()
            im = Image.merge("RGB", (b, g, r))
        elif  im.mode == "RGBA":
            r, g, b, a = im.split()
            im = Image.merge("RGBA", (b, g, r, a))
        elif im.mode == "L":
            im = im.convert("RGBA")
        # Bild in RGBA konvertieren, falls nicht bereits passiert
        im2 = im.convert("RGBA")
        data = im2.tobytes("raw", "RGBA")
        qim = QtGui.QImage(data, im.size[0], im.size[1], QtGui.QImage.Format_ARGB32)
        pixmap = QtGui.QPixmap.fromImage(qim)
        return pixmap

    def on_key(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            exit()

        if event.key() == QtCore.Qt.Key_S:
            self.display_kps = not self.display_kps

        if event.key() == QtCore.Qt.Key_T:
            img1 = np.array(self.img0) / 255. 
            kps = self.kps0 / (self.DISPLAY_IMG_SIZE // 2) - 1.        
            img1 = self.tpsmm.generate(self.img_tensor0, self.kps1, torch.from_numpy(kps).cuda())
            Image.fromarray(np.uint8(img1[0].permute(1, 2, 0).detach().cpu().numpy() * 255.)).save('result.png')
            #img1 = self.tpsmm.draw_image_with_kp_partial(img1[0].permute(1, 2, 0).detach().cpu().numpy(), kps, self.vis_flag)
            #Image.fromarray(np.uint8(img1 * 255.)).save('1.png')

    def keyPressEvent(self, event):
        super(Canvas, self).keyPressEvent(event)
        self.keyPressed.emit(event) 

    def real_pca_edit(self, idx, value):
        pca_coeff0 = (value - self.values[idx]) / 200.
        # self.kps0 += (value * self.pca_comp[0].view(50, 2).detach().cpu().numpy() + 1) * (self.DISPLAY_IMG_SIZE // 2)
        kps = self.kps0 / (self.DISPLAY_IMG_SIZE // 2) - 1.
        kps += pca_coeff0 * (self.pca_stdev[idx] * 40) * self.pca_comp[idx].reshape(50, 2)
        self.kps0 = (kps + 1) * (self.DISPLAY_IMG_SIZE // 2)

        img1 = np.array(self.img0) / 255. 
        kps = self.kps0 / (self.DISPLAY_IMG_SIZE // 2) - 1.        
        img1 = self.tpsmm.generate(self.img_tensor0, self.kps1, torch.from_numpy(kps).cuda())
        img1 = img1[0].permute(1, 2, 0).detach().cpu().numpy()
        img1 = self.tpsmm.draw_image_with_kp(img1, kps.reshape(-1, 5, 2))
        #img1 = self.tpsmm.draw_image_with_kp_partial(img1, kps, self.vis_flag)
        img1 = Image.fromarray(np.uint8(img1 * 255.))

        showedImagePixmap = self.pil2pixmap(img1.resize((self.DISPLAY_IMG_SIZE, self.DISPLAY_IMG_SIZE)))

        self.label.setPixmap(showedImagePixmap)
        self.label.resize(showedImagePixmap.width(), showedImagePixmap.height())
        self.values[idx] = value

    def pca_edit(self, idx):
        return lambda value : self.real_pca_edit(idx, value)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/data/RAVDESS/images/01-01-05-02-02-01-02.mp4/00000.png')
    args = parser.parse_args()
    app = QtWidgets.QApplication([])
    c = Canvas(None, img_path=args.input)
    c.show()
    app.exec_() 