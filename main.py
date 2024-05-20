import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import random
import math
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from scipy.fftpack import dct
from scipy.fftpack import idct
import cv2 as cv
import matplotlib.pyplot as plt
import random
import math
from PIL import Image

# Define functions as previously provided
# Include functions like psnr, NCC, dct2, idct2, watermark_image, extract_watermark, etc.

img_name = "image1.jpg"
wm_name = "watermark2.jpg"
watermarked_img = "Watermarked_Image.jpg"
watermarked_extracted = "watermarked_extracted.jpg"
key = 50
bs = 8
w1 = 64
w2= 64
fact = 8
indx = 0
indy = 0
b_cut = 50
val1  = []
val2  = []

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
    	return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def NCC(img1, img2):
	return abs(np.mean(np.multiply((img1-np.mean(img1)),(img2-np.mean(img2))))/(np.std(img1)*np.std(img2)))
def dct2(a):
	return cv.dct(a)
    #return dct(dct(a.T, norm="ortho").T, norm="ortho")
   
def idct2(a):
	return cv.idct(a)
	#return idct(idct(a.T, norm='ortho').T, norm='ortho')  

def watermark_image(img, wm):
	
	c1, c2 = np.size(img,0), np.size(img,1)
	c1x = c1
	c2x = c2
	c1-= b_cut*2
	c2-= b_cut*2
	w1, w2 = np.size(wm,0), np.size(wm, 1)
	
	print(c1, c2, w1, w2)

	if(c1*c2//(bs*bs) < w1*w2):
		print("watermark too large.")
		return img

	st = set()
	blocks = (c1//bs)*(c2//bs)
	print("Blocks availaible", blocks)
	blocks_needed = w1*w2
	
	i = 0
	j = 0
	imf = np.float32(img)
	while(i<c1x):
		while(j<c2x):
			#print(i, j)		
			dst = cv.dct(imf[i:i+bs, j:j+bs]/1.0)
			"""
			if(i==896 and j==160):
				print(dst)
				print(cv.idct(dst))
			"""
			imf[i:i+bs, j:j+bs] = cv.idct(dst)
			j+=bs
		j = 0
		i+=bs
	#print(np.size(imf))
	#print(imf[512:520, 512:520])
	final = img
	random.seed(key)
	i = 0
	print("Blocks needed", blocks_needed)
	cnt = 0
	while(i < blocks_needed):				
		to_embed = wm[i//w2][i%w2]
		ch = 0
		if(to_embed >= 127):
			to_embed = 1
			ch = 255
		else:
			to_embed = 0
		
		wm[i//w2][i%w2] = ch
		"""
		ch = 255
		new_img[i//w2][i%w2] = ch
		print(new_img[i//w2][i%w2], ch)
		"""
		#1- odd, 0 - even
		x = random.randint(1, blocks)
		#print("i",i,x)
		if(x in st):
			#print("there")
			continue
		st.add(x)
		n = c1//bs
		m = c2//bs
		#print("nmx",n,m,x)
		ind_i = (x//m)*bs + b_cut
		ind_j = (x%m)*bs + b_cut
		#print(ind_i, ind_j)
		#print(ind_i, ind_j)
		#print(imf[ind_i:ind_i+bs, ind_j:ind_j+bs])
		dct_block = cv.dct(imf[ind_i:ind_i+bs, ind_j:ind_j+bs]/1.0)
		elem = dct_block[indx][indy]
		elem /= fact
		ch = elem
		if(to_embed%2==1):	
			if(math.ceil(elem)%2==1):
				elem = math.ceil(elem)
			else:
				elem = math.ceil(elem)-1
		else:
			if(math.ceil(elem)%2==0):
				elem = math.ceil(elem)
			else:
				elem = math.ceil(elem)-1

		
		dct_block[indx][indy] = elem*fact
		#dct_block[0][0] = elem
		val1.append((elem*fact, to_embed))
		if(cnt < 5):
			#print(x, elem*fact , to_embed)
			#print(dct_block)
			cnt+=1
		
		final[ind_i:ind_i+bs, ind_j:ind_j+bs] = cv.idct(dct_block)
		imf[ind_i:ind_i+bs, ind_j:ind_j+bs] = cv.idct(dct_block)
		"""if(cnt<5):
			print(x)
			print(dct_block)
			cnt += 1
		"""
		i += 1

	#print(wm)
	final = np.uint8(final)
	#print(final[512:520, 512:520])
	#=============PSNR==========
	print("PSNR is:", psnr(imf, img))
	#=========================
	#new_img = np.uint8(new_img)
	
	#print(np.unique(new_img))
	cv.imshow("Final", final)
	cv.imwrite(watermarked_img , final)
	return imf

def extract_watermark(img, ext_name):
	c1x, c2x = np.size(img,0), np.size(img,1)
	
	if(c1x!=1000 or c2x != 1000):
		img = cv.resize(img, (1000, 1000))
		c1x = 1000
		c2x = 1000
	c1 = c1x - b_cut*2
	c2 = c2x-b_cut*2
	blocks = (c1//bs)*(c2//bs)
	blocks_needed = w1*w2

	wm = [[0 for x in range(w1)] for y in range(w2)]
	st = set()
	random.seed(key)
	i = 0
	cnt = 0
	#print("Blocks needed", blocks_needed)
	while(i<blocks_needed):
		curr = 0
		x = random.randint(1, blocks)
		if(x in st):
			#print("there")
			continue
		st.add(x)
		n = c1//bs
		m = c2//bs
		ind_i = (x//m)*bs + b_cut
		ind_j = (x%m)*bs + b_cut
		dct_block = cv.dct(img[ind_i:ind_i+bs, ind_j:ind_j+bs]/1.0)
		
		elem = dct_block[indx][indy]
		elem = math.floor(elem+0.5)
		"""
		if(cnt < 5):
			print(elem )
			cnt+=1
		"""
		elem /= fact
		
		if(elem%2 == 0):
			curr = 0
		else:
			curr = 255
		val2.append( (elem, bool(curr)))
		
		wm[i//w2][i%w2] = curr
		i+=1
		
		

	wm = np.array(wm)
	"""
	for i in range(64):
		print(wm[30][i])
	print(wm)
	"""
	cv.imwrite(ext_name , wm)
	print("Watermark extracted and saved in", ext_name)
	return wm

def open_image():
    global img
    img_path = filedialog.askopenfilename()
    img = cv.imread(img_path, 0)
    
    # Resize the image to 25x250
    img = cv.resize(img, (500, 250))
    
    img_display = ImageTk.PhotoImage(image=Image.fromarray(img))
    img_label.config(image=img_display)
    img_label.image = img_display


def open_watermark():
    global wm
    wm_path = filedialog.askopenfilename()
    wm = cv.imread(wm_path, 0)
    wm = cv.resize(wm, dsize=(64, 64), interpolation=cv.INTER_CUBIC)
    wm_display = ImageTk.PhotoImage(image=Image.fromarray(wm))
    wm_label.config(image=wm_display)
    wm_label.image = wm_display

def insert_watermark():
    global wmed
    wmed = watermark_image(img, wm)
    wmed_display = ImageTk.PhotoImage(image=Image.fromarray(wmed))
    wmed_label.config(image=wmed_display)
    wmed_label.image = wmed_display

def extract_watermark_ui():
    global wx
    wx = extract_watermark(wmed, "extracted_watermark.jpg")
    wx_display = ImageTk.PhotoImage(image=Image.fromarray(wx))
    wx_label.config(image=wx_display)
    wx_label.image = wx_display

def save_watermarked_image():
    save_path = filedialog.asksaveasfilename(defaultextension=".jpg")
    cv.imwrite(save_path, wmed)

def save_extracted_watermark():
    save_path = filedialog.asksaveasfilename(defaultextension=".jpg")
    cv.imwrite(save_path, wx)

# Initialize Tkinter window
root = tk.Tk()
root.title("Watermarking Application")

# Image display labels
img_label = tk.Label(root)
img_label.grid(row=0, column=0, padx=10, pady=10)

wm_label = tk.Label(root)
wm_label.grid(row=0, column=1, padx=10, pady=10)

wmed_label = tk.Label(root)
wmed_label.grid(row=1, column=0, padx=10, pady=10)

wx_label = tk.Label(root)
wx_label.grid(row=1, column=1, padx=10, pady=10)

# Buttons
open_img_button = tk.Button(root, text="Open Image", command=open_image)
open_img_button.grid(row=2, column=0, padx=10, pady=10)

open_wm_button = tk.Button(root, text="Open Watermark", command=open_watermark)
open_wm_button.grid(row=2, column=1, padx=10, pady=10)

insert_wm_button = tk.Button(root, text="Insert Watermark", command=insert_watermark)
insert_wm_button.grid(row=3, column=0, padx=10, pady=10)

extract_wm_button = tk.Button(root, text="Extract Watermark", command=extract_watermark_ui)
extract_wm_button.grid(row=3, column=1, padx=10, pady=10)

save_wmed_button = tk.Button(root, text="Save Watermarked Image", command=save_watermarked_image)
save_wmed_button.grid(row=4, column=0, padx=10, pady=10)

save_wx_button = tk.Button(root, text="Save Extracted Watermark", command=save_extracted_watermark)
save_wx_button.grid(row=4, column=1, padx=10, pady=10)

# Run the Tkinter event loop
root.mainloop()
