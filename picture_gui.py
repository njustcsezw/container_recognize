from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

import container_recognize.knn_recognize as kr
import container_recognize.picture_process as pp
import container_recognize.picture_to_matrix as ptm

gui = Tk()
gui.title('container_recognition')
path = StringVar()
result = StringVar()


def select_path():
    #动态（可选择）显示图片
    file_name = filedialog.askopenfilename(
        filetypes=[('All Files', '*'), ("JPG", "*.jpg"), ("PNG", "*.png"), ("GIF", "*.gif")])
    path.set(file_name)
    result.set('')
    #print(path.get())
    img_open = Image.open(path.get())
    img = ImageTk.PhotoImage(img_open)
    label1.config(image=img)
    label1.image = img  # keep a reference


def get_reslut():
    #调用knn，获取识别结果
    pp.pro_process(path.get())
    ptm.run_imagetomatrix('test_sets')
    knn_recoginion = kr.handwritingClassTest()
    result.set(knn_recoginion)


#左边图片选择框
frame1 = Frame(gui, height=576, width=768, bg='#CCCCCC')
frame1.grid(row=0, column=0, padx=30, pady=30)
#图片选择按钮
button1 = Button(frame1, text="选择图片", bg='#B2DFEE', command=select_path,
                 width=10, height=2, font=("黑体", 14, "normal"))
button1.place(relx=0.5, rely=0.5, anchor=CENTER)
#显示选择的图片
label1 = Label(frame1)
label1.place(relx=0.5, rely=0.5, anchor=CENTER)


#右边用户操作按钮
frame2 = Frame(gui, height=576, width=200)
frame2.grid(row=0, column=1, padx=20, pady=30)

button2 = Button(frame2, text="开始识别", command=get_reslut, bg='#5CACEE',
                 width=10, height=2, font=("黑体", 14, "normal"))
button2.grid(row=0, column=0, pady=10, sticky='W')

label2 = Label(frame2, text='识别结果是：', font=("黑体", 14, "normal"))
label2.grid(row=1, column=0, pady=10, sticky='W')

entry = Entry(frame2, textvariable=result, width=16, font=("黑体", 14, "normal"))
entry.grid(row=2, column=0, pady=10, sticky='W')

button3 = Button(frame2, text="选择下一张", command=select_path, bg='#EEEE00',
                 width=10, height=2, font=("黑体", 14, "normal"))
button3.grid(row=3, column=0, pady=10, sticky='W')

button4 = Button(frame2, text="退出系统", command=gui.quit, bg='#FF6A6A',
                 width=10, height=2, font=("黑体", 14, "normal"))
button4.grid(row=4, column=0, sticky=W, pady=10)


gui.mainloop()
