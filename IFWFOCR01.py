#LSI_PC PASSWORD "acce551b1e"
import threading
import multiprocessing
from pypylon import pylon
from pypylon import genicam
import cv2
import time
# import pytesseract
# from pytesseract import Output
from PIL import Image, ImageDraw, ImageFilter
import time
import datetime
import pickle
import numpy as np
import argparse
import shutil
import os
import socket
# import os.system
import sys
import RPi.GPIO as GPIO
# import gpiod as GPIO
import glob
# import pytesseract
# import pyautogui
import requests
import json
import base64
import pymssql
import serial
import logging
import subprocess

# from PyQt5 import *==
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
# from PyQt5.QTimer import *

# pyQTfileName = "dialog.ui"
# from settingDialog import Ui_Dialog
# from PyQt5.uic import loadUi
from PyQt5.uic import *
from getch import getche, getch

Ui_MainWindow, QtBaseClass = uic.loadUiType("dialog1.ui")
# Ui_SettingWindow, QtBaseClass = uic.loadUiType('setting.ui')


_Rpi = True 
# exposureTime = 8000
exposureTime = 25000
Resolution = (1280,960) # (width,height)
# Resolution = (640,480) # (width,height)
liveBit=False
liveintervalTime=50
teachBit= False
teachStep = 0
teachComplete =False




LoadData = []
limoff = []   
chipOffset =[]  #[x,y]
sPoint=[]
logoEnable =[False]
markOffset=[]
logoOffset=[]
MarkTemplateOffset=[]
markPoint=[]
sArea=[0,0,0,0]   #search area(x1,y1,x2,y2)
Tpoint=[]	# x1,y1,x2,y2
logoPoint=[]
templatePoint=[]
teachTemplatebit = False
TeachingMsg =[]
TeachIFLV = ['Set template area',
												'Teaching back camera',
												'Draw rectangle chip ',
												'Teaching completed'
												]

TeachIFLR = ['Draw marking area',
												'Draw LOGO         ',
												'Draw iflr2 ',
												'Teaching completed'
												]



imgChrRefer=[]

xPos= 1
portStart = 3
portSetup = 13
portBusy = 5
portCat = 7
mpos = [0,0]

chBox = np.array([])
ocrEnable = True
markingData = 'BU28131029T75'
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
# portCatFR = 15
# portCatBL = 11
# portCatBR = 19
mcType ='IFLR'
mcNo = ''
currentLot=''
WorkCount = [0,0,0,0]  # good,ng,search ng
opNumber =''

leadTosearchOffset = 0
searchShiftOffset =  0
searchLearPoint =0 #lead edge search position

lightOnbit = False
lightingValue =0 
lightBusy = False
threadKill = False
lightDial = False
threadLiveAlive = False
serialLightingEnable=False


sql ={
	'server':'172.16.0.110',
	'user':'system',
	'password':'p@$$w0rd',
	}


			# serial port for lighting initialize
# try:
# 	ser = serial.Serial(port = '/dev/ttyUSB0',baudrate = 38400,parity=serial.PARITY_NONE,stopbits=serial.STOPBITS_ONE,bytesize = serial.EIGHTBITS,timeout=1)
# 	serialLightingEnable=True
# 	print('Serial Port OK')

# except:
# 	print('Serial port device not found\n Serial port disable')



class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):

	def __init__(self):
		global mcType, TeachingMsg,teachStep,templatePoint,mcNo,lightingValue,serialLightingEnable
		super().__init__()
		QtWidgets.QMainWindow.__init__(self)
		Ui_MainWindow.__init__(self)
		QtWidgets.QMainWindow.showMaximized(self)
		logging.basicConfig(filename='log.log',level=logging.INFO,format='%(levelname)s %(asctime)s %(message)s')
		logging.info('Start application')
		self.setupUi(self)
		# self.btn_Fteaching.clicked.connect(self.TeachingMain)

		self.btnLive.clicked.connect(self.live)
		self.btnOCR.clicked.connect(self.OCRread)
		self.btnOCRSend.clicked.connect(self.OCRsending)
		self.btnCapture.clicked.connect(self.capture)
		self.FgraphicsView.mouseMoveEvent = self.mouseMove
		self.FgraphicsView.mousePressEvent = self.mousePress
		self.btnExit.clicked.connect(self.applicationClose)
		# self.btnROI.clicked.connect(self.setROI)
		self.btnOCRCancel.clicked.connect(self.OCRCancel)
		self.OCRframe.setVisible(False)
		self.groupBox2.move(1300,270)
		self.groupBox2.setVisible(False)
		self.groupBox3.setVisible(False)
		self.btnROI.setVisible(False)
		self.buttonGroup1.buttonClicked[int].connect(self.keyboard)


		self.dial.setMinimum(0)
		self.dial.setMaximum(255)
		self.dial.setNotchesVisible(True)
		# self.dial.setWrapping(True)
		self.dial.valueChanged.connect(self.dial_display)  
		self.dial.sliderReleased.connect(self.dial_method)		

		self.btnLon.clicked.connect(self.lightOn)
		self.btnLoff.clicked.connect(self.lightOff)
		self.btnSaveLight.clicked.connect(self.saveLighting)

		self.textOCR.focusInEvent = self.selectMark
		self.txtOpnum.focusInEvent =self.selectOp

		self.livetimer = QTimer()
		self.updateImgtimer = QTimer()



		subprocess.run(["python3","SyncTime.py"])


# get current IP name
		# local_hostname = socket.gethostname()
		# ip_addresses = socket.gethostbyname(local_hostname)

		# self.lblIP.setText('IP:'+ip_addresses)

		# hostname = socket.gethostname()
		# ip_address = socket.gethostbyname(hostname)
		# self.lblIP.setText('IP:',str(ip_address))


		ret = self.checkComPort()
		# print('Return check com port',ret)
		if ret[0] == '_Ok_':
			# cellcon port
			try:
				self.ser = serial.Serial(port = ret[1],baudrate = 38400,parity=serial.PARITY_NONE,stopbits=serial.STOPBITS_ONE,bytesize = serial.EIGHTBITS,timeout=1)	
				print('Cellcon port OK')
			except:
				print('Cellcon port NG')
			
			# lighting port
			try:
				self.ser1 = serial.Serial(port = ret[2],baudrate = 38400,parity=serial.PARITY_NONE,stopbits=serial.STOPBITS_ONE,bytesize = serial.EIGHTBITS,timeout=1)
				serialLightingEnable=True
				print('Lighting port OK')
			except:
				print('Lighting port NG')
			if 	serialLightingEnable:
				print('start lighting thread recive')
				self.t1 = threading.Thread(target = self.readSerialLighting)
				self.t1.start()
			# self.tLive = threading.Thread(target = self.liveTimeUp)
			
			# if serialCellconEnable:
			# 	self.t2 = threading.Thread(target = self.readSerialCellcon)
			# 	self.t2.start()


		print('\n*****System information*****')
		print('Python version ',sys.version)
		print('Opencv Version ',cv2.__version__)
		print('Qt: version', QT_VERSION_STR, "\nPyQt: version", PYQT_VERSION_STR)
		if _Rpi:
			camID=''

			# fname = 'CameraConfig.txt'
			# if os.path.exists(fname) == False:
			# 	print('CameraConfig.txt not found')
			# 	# exit()
			# else:
			# 	f = open(fname,'r')
			# 	while True:
			# 		s = f.readline()
			# 		if not s: 
			# 			break
			# 		line = s.rstrip().split(':');
			# 		# print(line)
			# 		if line[0] == 'Camera ID':
			# 			camID = line[1]
			# 		if line[0] == 'MC type':
			# 			mcType = line[1]
			# 		if line[0] == 'MCNO':
			# 			mcNo = line[1]
			# 			self.lblmcNo.setText(mcNo+' WSOF OCR monitor')
			# 	f.close()

			fname = 'config.json'
			if os.path.exists(fname) ==False:
				print('CameraConfig.txt not found')
				exit()				
			else:
				with open(fname,'r') as file:
					line = json.load(file)
					camID = line['CamID']
					mcType =line['MCType']
					mcNo = line['MCNum']

				print('Camera ID:'+camID)
				print('Machine type:',mcType)
				print('machine no ',mcNo)
				self.lblmcNo.setText(mcNo+' WSOF OCR monitor')
				logging.info('Machine no:'+mcNo)					
				print('*******************\n')        

				self.CameraF = MyVideoCapture(camID) 
				self.GrapImg_front()
				# self.livetimer = QTimer()
				self.livetimer.timeout.connect(self.liveImgFront)
				self.updateImgtimer.timeout.connect(self.updateImg)
				self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)


				if os.path.exists('Capture') == False:
					os.mkdir('Capture')
				if os.path.exists('OCR') == False:
					os.mkdir('OCR')
				if os.path.exists('Teaching') == False:
					os.mkdir('Teaching')
				if os.path.exists('NGPIC') == False:
					os.mkdir('NGPIC')

				envi = os.getcwd()
				self.pathCaptureimage = os.path.join(envi,'Capture')
				self.pathOCR = os.path.join(envi,'OCR')
				self.pathShared = '/home/pi/shared/'
				self.pathTemplatePoint = 'template.dat'
				self.txtFocus='mark'

				# templatePoint.clear()
				# with open(self.pathTemplatePoint,'rb') as fr:
				# 	dat = pickle.load(fr)
				# 	fr.close()
				# templatePoint = dat


				# start serial.read thread
				# if 	serialLightingEnable==True:
				# 	self.t1 = threading.Thread(target = self.readSerialLighting)
				# 	self.t1.start()
				# self.tLive = threading.Thread(target = self.liveTimeUp)

				# val = 100
				# print('lighting Value',val)
				# with open('lighting.dat','wb') as fw:
				# 	pickle.dump(val, fw)
				# 	return

				# Load lighting from file
				with open('lighting.dat','rb') as fr:
					dat = pickle.load(fr)
					fr.close()
				lightingValue = dat
				print('lighting:',lightingValue)	
				self.lightSetup(lightingValue)
				self.dial.setValue(lightingValue)
			
				if lightingValue > 0:
					self.lightOn()
					if serialLightingEnable == True:
						stTime = time.time()
						while(1):
							if lightBusy == False:
								break
							time.sleep(0.01)
							enTime=time.time()
							if enTime-stTime >= 3:
								break
					# cv2.waitKey(500)
				self.GrapImg_front()
				self.GrapImg_front()
				self.GrapImg_front()
				scene=QGraphicsScene()
				pixmap=QPixmap(QImage(self.frameFront,Resolution[0],Resolution[1],QImage.Format_Grayscale8))
				scene.addPixmap(pixmap)
				# scene.addRect(templatePoint[0],templatePoint[1],templatePoint[2],templatePoint[3],pen=QPen(Qt.red))
				self.FgraphicsView.setScene(scene)       
				self.lightOff()
	def selectMark(self,event):
		self.txtFocus='mark'
	def selectOp(self,event):
		self.txtFocus='opnum'

	def mousePress(self,event):
		global logoEnable,teachStep,Tpoint,logoPoint,teachBit,templatePoint
		if teachBit==True: 
			# when left mouse click 
			if event.button() == Qt.LeftButton:
				# print('step1')
				scene=QGraphicsScene()
				if teachStep ==1 :   #IFLR teaching point save
					# print('step2')
					Tpoint.append (event.x())
					Tpoint.append (event.y())
					print('Tpoint',Tpoint)
					
					if Tpoint[0] > Tpoint[2]:
						buk = Tpoint[0]
						Tpoint[0] = Tpoint[2]
						Tpoint[2] = buk
					
					if Tpoint[1] > Tpoint[3]:
						buk = Tpoint[1]
						Tpoint[1] = Tpoint[3]
						Tpoint[3] = buk


					teachTemplatebit = True
					teachBit =False
					scene.addPixmap(self.TeachPixmap)                    
					scene.addRect(Tpoint[0],Tpoint[1],Tpoint[2]-Tpoint[0],Tpoint[3]-Tpoint[1],pen=QPen(Qt.red))
					self.FgraphicsView.setScene(scene)                        
					QCoreApplication.processEvents()   #update screen 

					templatePoint=[Tpoint[0],Tpoint[1],Tpoint[2]-Tpoint[0],Tpoint[3]-Tpoint[1]]
					with open(self.pathTemplatePoint,'wb')as fw:
						pickle.dump(templatePoint, fw)

					return

				# First setting
				if teachStep == 0 :  #  0=  templateTeaching
					Tpoint = []
					Tpoint.append (event.x())
					Tpoint.append (event.y())
					teachStep = teachStep + 1       
					print('Start point',str(event.x())+':'+str(event.y()))               
				
			if event.button() == Qt.RightButton:
				if logoEnable == True and teachStep ==1:
					teachBit = False

	def mouseMove(self,event):
		self.lblPoint.setText(str(event.x())+':'+str(event.y()))
		# print(str(event.x())+':'+str(event.y()))    
		if teachBit==True: 
			scene=QGraphicsScene()
			scene.addPixmap(self.TeachPixmap)
			if teachStep == 0:
				scene.addItem(self.dispText(xPos,10,'Set first point',20,'lime'))    
			# free size teaching
			elif teachStep == 1:    
				scene.addItem(self.dispText(xPos,10,'Set second point',20,'lime'))                  
				x1 = Tpoint[(teachStep-1)]
				y1 = Tpoint[teachStep]
				x2 = event.x()
				y2 = event.y()
				scene.addRect(x1,y1,x2-x1,y2-y1,QPen(Qt.green))
			
			self.FgraphicsView.setScene(scene)
			QCoreApplication.processEvents()   #update screen


	def liveTimeUp(self):
		stTime = time.time()
		while(threadLiveAlive == True):
			time.sleep(1)
			endTime = time.time()
			if endTime-stTime >= 100:   #time up 5 minute then stop live image
				self.live()
				break
		print('auto exit from live')


	def OCRShowTimeUp(self):
		global waitClearScreen
		stTime = time.time()
		while(waitClearScreen == True):
			time.sleep(1)
			endTime = time.time()
			if endTime-stTime >= 10:   #time up 5 minute then stop live image
				waitClearScreen = False
				# print('point1')
				scene=QGraphicsScene()
				pixmap=QPixmap(QImage(self.frameFront,Resolution[0],Resolution[1],QImage.Format_Grayscale8))
				scene.addPixmap(pixmap)
				self.FgraphicsView.setScene(scene)
				print('Exit from OCR scene')
				break


									
	def live(self):
		global liveBit,threadLiveAlive,waitClearScreen
		waitClearScreen = False
		self.groupBox2.setVisible(False)
		if liveBit == False:
			liveBit = True
			logging.info('Live image')
			self.lightOn()
			self.btnLive.setText('Stop')
			self.livetimer.start(liveintervalTime)
			if 	serialLightingEnable==True:
				self.groupBox3.move(1300,90)
				self.groupBox3.setVisible(True)
			threadLiveAlive = True
			self.tLive = threading.Thread(target = self.liveTimeUp)
			self.tLive.start()
		else:
			liveBit = False
			threadLiveAlive = False
			logging.info('Stop live image')
			self.lightOff()
			self.btnLive.setText('Live')
			self.livetimer.stop()
			self.groupBox3.setVisible(False)
		# print('visible check',str(self.OCRframe.isVisible()))
		if(self.OCRframe.isVisible() == True):
			self.OCRframe.setVisible(False)
			self.groupBox2.setVisible(False)


	def capture(self):
		global waitClearScreen
		waitClearScreen =False
		bfOn=True
		if lightOnbit ==False:
			bfOn=False
			self.lightOn()
			cv2.waitKey(100)
		self.GrapImg_front()
		cv2.imwrite("{a}/{b}.jpg".format(a=self.pathCaptureimage,b=time.strftime("%d%m%Y-%H%M%S")), self.frameFront)
		if bfOn == False:
			self.lightOff()
		logging.info('Capture image')				

	def liveImgFront(self):
		self.GrapImg_front()
		pixmap=QPixmap(QImage(self.frameFront,Resolution[0],Resolution[1],QImage.Format_Grayscale8))
		self.FgraphicsView.setScene(self.dispImage(pixmap,'Live'))
		QCoreApplication.processEvents()   #update screen       

	def GrapImg_front(self):
		self.frameFront = self.CameraF.get_frame()
		if self.frameFront is not None:
			# print('grab image ok')
			return(1)
		else:
			print('grab image ng')
			return(0)       

	
	def dispImage(self,pixmap,txt):
		scene=QGraphicsScene()
		scene.addPixmap(pixmap)
		if txt != '':
			scene.addItem(self.dispText(xPos,20,txt,20,'lime'))       
		return(scene)

	def dispText(self,x,y,text,size,color):
		txt = QGraphicsTextItem(text)
		txt.setPos(x,y)
		txt.setFont(QFont('segoe UI',size))
		txt.setDefaultTextColor(QColor(color))
		return(txt)

	def cross(self, scene,xst,yst,size,color):
		scene.addLine(xst-size,yst-size,xst+size,yst+size,pen=QPen(color))
		scene.addLine(xst-size,yst+size,xst+size,yst-size,pen=QPen(color))


	def setROI(self):
		global teachBit,teachStep
		if liveBit :
			self.live()
			cv2.waitKey(200)
		if lightOnbit == False:
			self.lightOn()
			cv2.waitKey(100)
		self.GrapImg_front()
		self.TeachPixmap=QPixmap(QImage(self.frameFront,Resolution[0],Resolution[1],QImage.Format_Grayscale8))   				        
		teachBit = True
		teachStep = 0			
		self.lightOff()	   
		self.groupBox1.setVisible(False)    						
					

	def OCRsending(self):  
		global waitClearScreen
		waitClearScreen = False
		self.OCRframe.setVisible(True)
		OCRString = self.textOCR.toPlainText()
		OCRString = OCRString.replace(" ","")
		logging.info('OCR start')

		opAction = self.txtOpnum.toPlainText()
		print('opnumber length',len(opAction))
		cat=1
		if len(opAction) == 6:
			for c in opAction:
				if c.isnumeric()==False:
					opAction=''
					self.txtOpnum.setText('')
					cat=0
		else:
			opAction=''
			self.txtOpnum.setText('')
			cat=0

		if cat==0:
			QMessageBox.critical(self,'Error','OP number miss ',QMessageBox.Close)
		else:
			if OCRString == "":  		# no character inside text box
				print('No character')
				QMessageBox.question(self,'Textbox character not found   ','Please input character ',QMessageBox.Close)
				# return(0)			
				logging.info('OCR start: character empty')
				self.textOCR.clear()
				self.textOCR.setFocus()
			else:  						# character present
				xpos = 5
				print('OCR text=',OCRString)
				logging.info('OCR start by text:'+ OCRString)	
				# original image size 1280x960
				resizeRetio = [int(Resolution[0]/2),int(Resolution[1]/2)]   #640x480
				# resizeRetio = [int(Resolution[0]/4),int(Resolution[1]/4)]	#320x240
				print('Resize image:',resizeRetio)
				imgBuf = cv2.resize(self.frameFront,resizeRetio,interpolation=cv2.INTER_AREA)
				print('shape',imgBuf.shape)
				# print('shape1',self.frameFront.shape)
				cv2.imwrite('cropimg.jpg',imgBuf)

				scene=QGraphicsScene()
				pixmap=QPixmap(QImage(self.frameFront,Resolution[0],Resolution[1],QImage.Format_Grayscale8))
				scene.addPixmap(pixmap)
				scene.addRect(0,0,630,280,pen=QPen(Qt.red),brush=QBrush(QColor('black')))
				scene.addItem(self.dispText(xpos,100,'Lot no: '+ self.currentLot,16,'lime'))  

				print('1.Get marking from DB')
				url = 'http://webserv.thematrix.net/ROHMApi/api/OCR/ReadMark'
				# myobj = {'username':self.opNumber,'lot_no':self.currentLot}
				myobj = {'username':opAction,'lot_no':self.currentLot}
				RespondBody = requests.post(url, json = myobj)
				print('1.API result:',RespondBody.status_code)
				if RespondBody.status_code ==200:
					res_dict={}
					res_dict = json.loads(RespondBody.text)
					# print(res_dict[0]['mark'])
					# print(res_dict[0]['lot_no'])
					LotDataInfo = [res_dict[0]['lot_no'],res_dict[0]['mark']]
					scene.addItem(self.dispText(xpos,130,'Standard mark: '+ LotDataInfo[1],16,'lime'))  
					OCRCat=0
					if LotDataInfo[1] == OCRString :
						color = 'lime'
						ocrCompareResult = 'Mark comparison correct'
						OCRCat = 1
						logging.info('OCR compare correct:'+ LotDataInfo[1])	
					else:
						color = 'red'
						ocrCompareResult = 'Marking in-correct'
						logging.info('OCR compare in-correct:'+ LotDataInfo[1])
					scene.addItem(self.dispText(xpos,160,'Current mark: '+ OCRString,16,color)) 
					scene.addItem(self.dispText(xpos,190,'By operator: '+ opAction,16,color)) 				
					scene.addItem(self.dispText(xpos,10,ocrCompareResult,40,color))
					# if OCRCat == 1 of OCRCat == 0:

					with open("cropimg.jpg","rb") as f:
						encodeImg = base64.b64encode(f.read())
						# print(encodeImg)
						print('2.Create record')
						url = 'http://webserv.thematrix.net/ROHMApi/api/OCR/CreateRecord'
						myobj = {'username':opAction,
								'lot_no':self.currentLot,
								'mark': OCRString,
								'image': encodeImg,
								'is_pass':OCRCat,
								'recheck_count':0,
								'is_logo_pass':0
								}						
			
						RespondBody = requests.post(url, json = myobj)
						print('2.API result:',RespondBody.status_code);
						if RespondBody.status_code ==200:
							scene.addItem(self.dispText(xpos,220,'OCR Database save Complete',20,'lime'))	
							logging.info('OCR record complete')								
						else:
							scene.addItem(self.dispText(xpos,220,'OCR Database save error',20,'red'))
							logging.info('OCR record error')	
					logging.info('OCR finished')

				else:
					logging.info('OCR start: result error')	
					scene.addItem(self.dispText(xpos,100,'Error: '+ RespondBody.status_code,16,'red'))  
				scene.addItem(self.dispText(xpos,70,str(time.strftime("%d/%m/%Y %H:%M:%S")),16,'lime'))				
				self.FgraphicsView.setScene(scene)   
				QCoreApplication.processEvents()   #update screen		


				# Save OCR result with comment to files
				pixmap2 = QPixmap(self.FgraphicsView.grab())
				pixmap2.save(self.pathOCR +'/'+ self.currentLot+str(time.strftime("_%d%m%Y_%H%M%S"))+'.jpg')	
			

				logging.info('OCR image save')	
				self.OCRframe.setVisible(False)
				self.groupBox2.setVisible(False)
				self.lightOff()
				self.updateImgtimer.start(5000)

	def updateImg(self):
		scene=QGraphicsScene()
		pixmap=QPixmap(QImage(self.frameFront,Resolution[0],Resolution[1],QImage.Format_Grayscale8))
		scene.addPixmap(pixmap)
		self.FgraphicsView.setScene(scene)
		self.updateImgtimer.stop()	


	def OCRCancel(self):
		print('OCR cancel')
		logging.info('OCR cancel')			
		self.OCRframe.setVisible(False)
		self.groupBox2.setVisible(False)
		# self.groupBox1.setEnabled(True)
		self.lightOff()




	def OCRread(self):
		global teachBit,teachStep,waitClearScreen
		waitClearScreen = False
		if liveBit :
			self.live()
			cv2.waitKey(200)
		# cv2.waitKey(500)
		if lightOnbit ==False:
			# bfOn=False
			self.lightOn()
			cv2.waitKey(300)
		# self.GrapImg_front()
		if serialLightingEnable == True:
			stTime = time.time()
			while(1):
				if lightBusy == False:
					break
				time.sleep(0.01)
				enTime=time.time()
				if enTime-stTime >= 3:
					print('time up')
					break
		logging.info('OCR menu select')
		print('OCR menu select')

		self.GrapImg_front()
		self.GrapImg_front()
		self.GrapImg_front()		
		scene=QGraphicsScene()
		pixmap=QPixmap(QImage(self.frameFront,Resolution[0],Resolution[1],QImage.Format_Grayscale8))
		scene.addPixmap(pixmap)
		print('step2')
		# scene.addRect(templatePoint[0],templatePoint[1],templatePoint[2],templatePoint[3],pen=QPen(Qt.red))
		self.FgraphicsView.setScene(scene)   

		# resizeRetio = [int(Resolution[0]/2),int(Resolution[1]/2)]
		# print(resizeRetio)
		# imgBuf = cv2.resize(self.frameFront,resizeRetio,interpolation=cv2.INTER_LINEAR)
		# cv2.imwrite('cropimg.jpg',imgBuf)

		# print('shape',imgBuf.shape)
		# print('shape1',self.frameFront.shape)


		# if (self.getLotNoFromSQL() == 1):
		# 	# self.groupBox1.setEnabled(False)
		# 	self.groupBox2.setVisible(True)
		# 	# self.OCRframe.setVisible(True)
		# 	self.textOCR.clear()
		# 	self.textOCR.setFocus()

		ret = self.getLotNumFromCellcon()
		if ret == 'err':
			QMessageBox.question(self,'Warning','Lot number not found',QMessageBox.Close)
			logging.info('Get lot:Lot not found')
		else:
			self.currentLot = ret[1]
			self.lblLotNo.setText('Lot no: ' + self.currentLot)
			self.opNumber= ' '
			# self.lblOp.setText(' OP no: ' + self.opNumber)
			self.txtOpnum.setText('')
			logging.info('Get lot:'+ self.currentLot)			

			# self.groupBox1.setEnabled(False)
			self.groupBox2.setVisible(True)
			self.btnOCRSend.setVisible(False)
			self.textOCR.clear()
			self.textOCR.setFocus()
					
					

	# def getLotNoFromSQL(self):

	# 	print('Check lot in database')
	# 	conn = pymssql.connect(sql['server'],sql['user'],sql['password'])


	# 	cursor = conn.cursor()
	# 	# cursor.execute("select lots.lot_no,machines.name from APCSProDB.trans.lots as lots \
	# 	# inner join APCSProDB.mc.machines as machines on machines.id = lots.machine_id \
	# 	# where lots.process_state in (2,102) and machines.name =%s",mcNo)

	# 	cursor.execute("select distinct lots.lot_no,machines.name,man.emp_num,lots.process_state,job.short_name from APCSProDB.trans.lots as lots \
	# 		inner join APCSProDB.mc.machines as machines on machines.id = lots.machine_id \
	# 		inner join [APCSProDB].[trans].[lot_process_records] as process_records on process_records.lot_id = lots.id \
	# 		inner join APCSProDB.man.users as man on man.id = process_records.operated_by \
	# 		inner join [APCSProDB].[method].[jobs] as job on job.id = process_records.job_id \
	# 		where lots.process_state in (2,102) and machines.name = %s and short_name in('FL','FLFTTP','FLFT') and process_records.record_class = 1" ,'FL-'+mcNo)

	# 	retRow =[]
	# 	for row in cursor:
	# 		print('row',row)
	# 		retRow.append(row)
	# 	if(len(retRow) ==0):
	# 		QMessageBox.question(self,'Warning','Lot number not found',QMessageBox.Close)
	# 		logging.info('Get lot:Lot not found')
	# 		return(0)
	# 	else:
	# 		self.currentLot = retRow[0][0].replace(' ','')
	# 		self.lblLotNo.setText('Lot no: ' + self.currentLot)
	# 		self.opNumber= retRow[0][2].replace(' ','')
	# 		# self.lblOp.setText(self.opNumber)
	# 		self.txtOpnum.setText('')

	# 		logging.info('Get lot:'+ self.currentLot +'OP no:' + self.opNumber)			
	# 		return(1)

	def getLotNumFromCellcon(self):
		self.ser.write(b'LA\r\n')
		cnt = 0
		cat=0
		while cnt < 5:
			msg = self.ser.readline().decode('utf-8').strip()
			if msg :
				print('recive',msg)
				part = msg.split(',')
				if part[0] == 'LS':
					cat = 1
					break
			else:
				print('.')
				cnt += 1
		if cat == 1:
			return(part)
		else:
			print('no recive data')
			return('err')
			

	def debugStop(self):
		repry = QMessageBox.question(self,'Stop','please mouse click',QMessageBox.Yes,QMessageBox.Yes)


	def keyboard(self,key):
		# print(key)
		msg = self.buttonGroup1.button(key).text()
		match msg:
			case 'BS':
				msg1 = self.textOCR.toPlainText()
				msg1 = msg1[:-1]
				self.textOCR.setText('')
				self.textOCR.insertPlainText(msg1)
			case 'Enter':
				cv2.waitKey(1)
			case 'Shift':
				cv2.waitKey(1)
			case 'Space':
				cv2.waitKey(1)
			case 'Clear All':
				if self.txtFocus=='mark':
					self.textOCR.setText('')
				if self.txtFocus=='opnum':
					self.txtOpnum.setText('')
			case other:
				if self.txtFocus=='mark':
					self.textOCR.insertPlainText(msg)
				if self.txtFocus=='opnum':
					self.txtOpnum.insertPlainText(msg)					
		if self.textOCR.toPlainText() != '' and self.txtOpnum.toPlainText() != '':
			self.btnOCRSend.setVisible(True)

	def lightOn(self):
		global lightOnbit,lightBusy
		if 	serialLightingEnable==False:
			return
		while(1):
			if lightBusy == False:
				print('Light ON.')
				self.ser1.write(b'@00L1007D\r\n')
				lightOnbit = True
				lightBusy = True
				break


	def lightOff(self):
		global lightOnbit,lightBusy
		if 	serialLightingEnable==False:
			return
		while(1):
			if lightBusy == False:
				print('Light OFF.')
				self.ser1.write(b'@00L0007C\r\n')
				lightOnbit = False
				lightBusy = True
				break


	def lightSetup(self,num1):
		global lightOnbit,lightBusy
		if 	serialLightingEnable==False:
			return
		print('***Lighting setup***')
		self.lightSetup1(num1)
		lightOnbit = False
		lightBusy = True

	def lightSetup1(self,num1):
		stnum1 = f"{num1:03}"
		SendMes = '@00F'+ stnum1 +'00'
		x = bytes(SendMes,'ascii')
		y=0
		for byte in x:
			y += byte
			y &= 0xff
			z = hex(y)
			z = z[2:]
		lightVal = (str(SendMes+str(z.upper()))+'\r\n').encode('utf-8')
		print('Set lightValue',lightVal)
		try:
			self.ser1.write(lightVal)
		except:
			cv2.waitKey(1)






    # method called by the dial 
	def dial_method(self):
		global lightBusy
		val = self.dial.value()
		self.label.setText("Light value: " + str(val))
		print('light setup dial',val)
		# lightDial=True		
		self.lightSetup(val)
		cv2.waitKey(100)
		while(1):
			if lightBusy == False:
				cv2.waitKey(50)
				break
		self.lightOn()

	def dial_display(self):
		val = self.dial.value()
		self.label.setText("Light value: " + str(val))	




	def saveLighting(self):
		val = self.dial.value()
		print('lighting Value',val)
		with open('lighting.dat','wb') as fw:
			pickle.dump(val, fw)
			return


	def readSerialLighting(self):
		global lightBusy,lightDial
		x=0
		while 1:
			x=self.ser1.readline()
			if x != '':
				if lightBusy == True:
					lightBusy = False
					print( 'Serial recived:',x)	
			if threadKill == True:
				break;
		print('Kill thread finished')


	def readSerialCellcon(self):
		x=0
		while 1:
			x=self.ser.readline().decode('utf-8').strip()
			if x != '':
				if lightBusy == True:
					lightBusy = False
					print( 'Serial recived:',x)	
			if threadKill == True:
				break;
		print('Kill thread finished')



	def applicationClose(self):
		global threadKill
		if liveBit==True:
			self.live()
			cv2.waitKey(100)
		threadKill = True
		cv2.waitKey(200)
		logging.info('Close application')			
		exit()

	def checkComPort(self):
		print('***** Check serial port ******')
		if os.path.exists('/dev/ttyUSB0'):
			print('ttyUSB0 OK')
			if os.path.exists('/dev/ttyUSB1') == False:
				msg = ['"/dev/ttyUSB1" not found','please check USB to RS232 cable']
				print(msg)	
				cat=0
			else:
				print('ttyUSB1 OK')
				cat=1
		else:
			msg = ['"/dev/ttyUSB0" not found','please check USB to RS232 cable']
			print(msg)
			cat=0
		print('check complete')
		if cat != 1:
			QMessageBox.question(self,'','Alarm: '+msg[0]+'\n'+msg[1],QMessageBox.Close)
			logging.info(msg[0])
			logging.info('Program exit')
			exit()
		else:
			print('found 2 usb')
			# return('_OK_')
		usbName = ['/dev/ttyUSB0','/dev/ttyUSB1']
		cat=0
		for b in usbName:
			serTmp = serial.Serial(port = b,baudrate = 38400,parity=serial.PARITY_NONE,stopbits=serial.STOPBITS_ONE,bytesize = serial.EIGHTBITS,timeout=1)
			serTmp.write(b'LA\r\n')
			cnt = 0
			while cnt < 10:
				msg = serTmp.readline().decode('utf-8').strip()
				if msg :
					print('USBPort:',b,'recive',msg)
					part = msg.split(',')
					# print('part',part[0])
					if part[0]=='LS':
						cat = 1
						break
				else:
					print('.')
					cnt += 1
			serTmp.close()
			if cat==1:
				print('Cellcon=',b)
				if b == '/dev/ttyUSB0':
					return('_Ok_','/dev/ttyUSB0','/dev/ttyUSB1')
				else:
					return('_Ok_','/dev/ttyUSB1','/dev/ttyUSB0')					
				# break
		print('Com port check error ')
		return('_Err_','','')
		# if boudChkOK == True:
		# 	# print('finish',b1)
		# 	return(b1)
		# else:
		# 	# print('port not found')
		# 	return(9600)

class MyVideoCapture:
	def __init__(self,CameraSerialNumber):
		tlFactory = pylon.TlFactory.GetInstance()
		devices = tlFactory.EnumerateDevices()
		try:
			for j in devices:
				if j.GetSerialNumber() == CameraSerialNumber:
					self.camera = pylon.InstantCamera(tlFactory.CreateDevice(j))
					break
			self.camera.Open()
			self.camera.Width.Value = Resolution[0]
			self.camera.Height.Value= Resolution[1]
			self.camera.ExposureTime.SetValue(exposureTime)
			self.camera.PixelFormat.SetValue('Mono8')
			self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly,pylon.GrabLoop_ProvidedByUser)
			self.converter = pylon.ImageFormatConverter()
			self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
		except genicam.GenericException as e:
			print("An exception occurred.", e.GetDescription())



	def get_frame(self):
		try:
			self.grabResult1 = self.camera.RetrieveResult(5000,pylon.TimeoutHandling_ThrowException)  
			if self.grabResult1.GrabSucceeded():
				img = self.grabResult1.GetArray()
			else:
				print("Error",self.grabResult1.ErrorCode)
							
			self.grabResult1.Release()
			return img
						
		except genicam.GenericException as e:
										# Error handling
			print("An exception occurred.", e.GetDescription())
			exitCode = 1

	def __del__(self):
#         if self.Camera.isOpened():
		self.camera.Close()        

if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	window = MyApp()
	window.show()
	sys.exit(app.exec_())
