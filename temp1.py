# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:03:09 2020

@author: Ultimate-SK
"""


import tkinter as tk
from tkinter import filedialog

def onOpen():
    global photo
    filename = filedialog.askopenfilename()
    photo = tk.PhotoImage(file=filename)
    button.configure(image=photo)

root = tk.Tk()

photo = tk.PhotoImage(file=" ")
button = tk.Button(root, image=photo, command=onOpen)
button.grid()
root.mainloop()


import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk




root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfile()
im = Image.open(file_path)
im.show





