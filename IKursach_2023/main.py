from tkinter import *
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import cv2
import matplotlib.pyplot as plt
import os
from shutil import copytree
import shutil
from fpdf import FPDF
import random


root = Tk()
root.title("KeyPoints")
root.geometry("500x500")
key_p = []
image_mani = []
im_gray = []
detect = []
files = []
compute = []
com_res = []
key_p_arr = []
compute_arr = []
matcher = []
matcher_res = []
Im_gray_arr = []
mat = []
dis_arr = []
vibros = []
ry = []
files2 = []
FILE = []
kolvo_tochek = []
l1 = []
l2 = []


class ImageManipulator:
    def __init__(self, file_name):
        self.image = cv2.imread(file_name)
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def get_image(self):
        return self.image

    def get_grayscale(self):
        return self.gray_image


class ORB_D:
    def __init__(self):
        self.orb = cv2.ORB_create()

    # def __init__(self, nfeatures):
    #     self.nfeatures = nfeatures
    #     self.orb = cv2.ORB_create(nfeatures=self.nfeatures)

    def detect_keypoints(self, gray_image):
        keypoints = self.orb.detect(gray_image, None)
        return keypoints


class FAST_D:
    def __init__(self):
        self.fast = cv2.FastFeatureDetector_create()

    def detect_keypoints(self, gray_image):
        keypoints = self.fast.detect(gray_image, None)
        return keypoints


class SIFT_D:
    def __init__(self):
        self.sift = cv2.SIFT_create()

    def detect_keypoints(self, gray_image):
        keypoints = self.sift.detect(gray_image, None)
        return keypoints


class ORB_C:
    def __init__(self, gray_image):
        self.gray_image = gray_image
        self.orb = cv2.ORB_create()

    def compute(self, keypoints):
        descriptors, keypoints = self.orb.compute(self.gray_image, keypoints)
        return keypoints


class SIFT_C:
    def __init__(self, gray_image):
        self.gray_image = gray_image
        self.sift = cv2.SIFT_create()

    def compute(self, keypoints):
        descriptors, keypoints = self.sift.compute(self.gray_image, keypoints)
        return keypoints


class Brute_Force_Mat_ORB:
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def mat(self, d1, d2):
        matches = self.bf.knnMatch(d1, d2, k=2)
        return matches

    def match_descriptors(self, d1, d2):
        matches = self.bf.knnMatch(d1, d2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good.append([m])
        return good


class Brute_Force_Mat_SIFT:
    def __init__(self):
        self.bf = cv2.BFMatcher()

    def mat(self, d1, d2):
        matches = self.bf.knnMatch(d1, d2, k=2)
        return matches

    def match_descriptors(self, d1, d2):
        matches = self.bf.knnMatch(d1, d2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good.append([m])
        return good


class FLANN_SIFT:
    def __init__(self):
        flann_index_kdtree = 1
        index_params = dict(algorithm=flann_index_kdtree, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def mat(self, des1, des2):
        matches = self.flann.knnMatch(des1, des2, k=2)
        return matches

    def match(self, des1, des2):
        matches = self.flann.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good.append([m])
        return good


class FLANN_ORB:
    def __init__(self):
        flann_index_lsh = 6
        index_params = dict(algorithm=flann_index_lsh,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=2)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def mat(self, des1, des2):
        matches = self.flann.knnMatch(des1, des2, k=2)
        return matches

    def match(self, des1, des2):
        matches = self.flann.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good.append([m])
        return good


def browseFiles():
    filename = filedialog.askdirectory(initialdir="/", title="Select a File")

    label_File.configure(text="Выбрана папка: " + filename)
    copytree(filename, 'C:/Users/Anastasia/PycharmProjects/IKursach_2023/12')


def browseFiles1():
    filename = filedialog.askdirectory(initialdir="/", title="Select a File")

    label_File1.configure(text="Выбрана папка: " + filename)
    copytree(filename, 'C:/Users/Anastasia/PycharmProjects/IKursach_2023/13')


def Del_file():
    der = 'C:/Users/Anastasia/PycharmProjects/IKursach_2023/12'
    shutil.rmtree(der)
    der1 = 'C:/Users/Anastasia/PycharmProjects/IKursach_2023/13'
    shutil.rmtree(der1)


def Detect():
    y = combobox_Det.get()
    global key_p
    global image_mani
    global im_gray
    global detect
    global files
    global Im_gray_arr
    global key_p_arr
    global FILE
    directory1 = 'C:/Users/Anastasia/PycharmProjects/IKursach_2023/12'
    directory2 = 'C:/Users/Anastasia/PycharmProjects/IKursach_2023/13'
    for filename in os.listdir(directory1):
        f1 = os.path.join(directory1, filename)
        files.append(f1)
    for filename in os.listdir(directory2):
        f2 = os.path.join(directory2, filename)
        files2.append(f2)

    if len(files) < len(files2):
        k = len(files)
    else:
        k = len(files2)

    files_rand = random.sample(files, k)
    files2_rand = random.sample(files2, k)
    FILE = [files_rand[0], files_rand[1], files2_rand[0], files2_rand[1], files_rand[0]]
    if y == "ORB":
        for i in range(len(FILE)):
            image_mani = ImageManipulator(FILE[i])
            im_gray = image_mani.gray_image
            Im_gray_arr.append(im_gray)
            detect = ORB_D()
            key_p = detect.detect_keypoints(im_gray)
            key_p_arr.append(key_p)
    if y == "FAST":
        for i in range(len(FILE)):
            image_mani = ImageManipulator(FILE[i])
            im_gray = image_mani.gray_image
            Im_gray_arr.append(im_gray)
            detect = FAST_D()
            key_p = detect.detect_keypoints(im_gray)
            key_p_arr.append(key_p)
    if y == "SIFT":
        for i in range(len(FILE)):
            image_mani = ImageManipulator(FILE[i])
            im_gray = image_mani.gray_image
            Im_gray_arr.append(im_gray)
            detect = SIFT_D()
            key_p = detect.detect_keypoints(im_gray)
            key_p_arr.append(key_p)
    return key_p_arr, image_mani, im_gray, detect, key_p, Im_gray_arr


def Compute():
    y = combobox_Des.get()
    global key_p_arr
    global compute
    global com_res
    global compute_arr
    global Im_gray_arr
    if y == "ORB":
        for i in range(len(key_p_arr)):
            el_im = Im_gray_arr[i]
            el_kp = key_p_arr[i]
            compute = ORB_C(el_im)
            com_res = compute.compute(el_kp)
            compute_arr.append(com_res)

    if y == "SIFT":

        for i in range(len(key_p_arr)):
            el_im = Im_gray_arr[i]
            el_kp = key_p_arr[i]
            compute = SIFT_C(el_im)
            com_res = compute.compute(el_kp)
            compute_arr.append(com_res)

    return compute_arr


def MAT():
    y = combobox_Mat.get()
    q = combobox_Des.get()
    global compute_arr
    global files
    global key_p_arr
    global matcher
    global matcher_res
    global mat
    global ry
    if y == "Brute_Force" and q == "ORB":
        for i in range(len(compute_arr) - 1):
            matcher = Brute_Force_Mat_ORB()
            mat = matcher.mat(compute_arr[i], compute_arr[i + 1])
            matcher_res = matcher.match_descriptors(compute_arr[i], compute_arr[i + 1])
            rt = MET()
        print(kolvo_tochek)
        return kolvo_tochek, matcher_res

    if y == "Brute_Force" and q == "SIFT":

        for i in range(len(compute_arr) - 1):
            matcher = Brute_Force_Mat_SIFT()
            mat = matcher.mat(compute_arr[i], compute_arr[i + 1])
            matcher_res = matcher.match_descriptors(compute_arr[i], compute_arr[i + 1])
            rt = MET()
        print(kolvo_tochek)
        return kolvo_tochek, matcher_res

    if y == "FLANN" and q == "SIFT":
        for i in range(len(compute_arr) - 1):
            matcher = FLANN_SIFT()
            mat = matcher.mat(compute_arr[i], compute_arr[i + 1])
            matcher_res = matcher.match(compute_arr[i], compute_arr[i + 1])
            rt = MET()
        print(kolvo_tochek)
        return kolvo_tochek, matcher_res
    if y == "FLANN" and q == "ORB":
        for i in range(len(compute_arr) - 1):
            matcher = FLANN_ORB()
            mat = matcher.mat(compute_arr[i], compute_arr[i + 1])
            matcher_res = matcher.match(compute_arr[i], compute_arr[i + 1])
            rt = MET()
        print(kolvo_tochek)
        return kolvo_tochek, matcher_res


def MET():
    global matcher_res
    global compute_arr
    kolvo_tochek.append(len(matcher_res))
    return kolvo_tochek


def Otchet():
    global kolvo_tochek
    x = combobox_Det.get()
    z = combobox_Mat.get()
    y = combobox_Des.get()
    w = combobox_Met.get()
    sort_mas = []
    i_mas = []
    for i in range(len(kolvo_tochek)):
        if i % 2 == 1:
            sort_mas.append(kolvo_tochek[i])
    for i in range(len(kolvo_tochek)):
        if i % 2 == 0:
            sort_mas.append(kolvo_tochek[i])
    print(sort_mas)
    # graf = plt.plot(sort_mas)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.subplots_adjust(bottom=0.15, left=0.2)
    ax.plot(sort_mas)
    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('Damped oscillation [V]')
    #
    # plt.show()
    ax.set_xlabel('Номер сопоставления')
    ax.set_ylabel('Количество удачных сопоставлений')

    labels = ['different1', 'different2', 'similar1', 'similar2']
    for i in range(len(kolvo_tochek)):
        i_mas.append (i)
    plt.xticks(i_mas, labels, rotation ='vertical')
    # plt.show()
    plt.savefig('./example_chart.png',
                transparent=False,
                facecolor='white',
                bbox_inches="tight")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"REPORT", ln=1, align="C")
    pdf.cell(200, 10, f"", ln=1, align="L")
    pdf.cell(200, 10, f"Detector : {x}", ln=1, align="L")
    pdf.cell(200, 10, f"Deskriptor : {y}", ln=1, align="L")
    pdf.cell(200, 10, f"Matcher : {z}", ln=1, align="L")
    pdf.cell(200, 10, f"", ln=1, align="L")
    pdf.cell(200, 10, f"Graph 'Number of good matches'", ln=1, align="C")

    pdf.image('./example_chart.png',
              x=10, y=None, w=100, h=0, type='PNG')
    pdf.output('C:/Users/Anastasia/Desktop/Otchet.pdf')


# .....................................ЭЛЕМЕНТЫ_ДЕСКА....................................................................


f_top = Frame(root)
f_bot = Frame(root)
f_bot1 = Frame(root)
f_bot2 = Frame(root)
f_bot23 = Frame(root)
f_bot_4 = Frame(root)
f_top1 = Frame(root)
label_File = tk.Label(
    f_top,
    text='Выберете папку',
    fg='black',
    bg='LightCyan2'
)

btn_File = Button(
    f_top,
    text='Выбрать..',
    fg='black',
    bg='LightBlue1',
    command=lambda: [browseFiles()]
)
label_File1 = tk.Label(
    f_top1,
    text='Выберете еще одну папку',
    fg='black',
    bg='LightCyan2'
)

btn_File1 = Button(
    f_top1,
    text='Выбрать..',
    fg='black',
    bg='LightBlue1',
    command=lambda: [browseFiles1()]
)
label_Type = tk.Label(
    f_bot_4,
    text='Тип датасета',
    fg='black',
    bg='LightCyan2'
)
languages = ['Одно место', 'Последовательность']
label = tk.ttk.Label()
combobox_Type = tk.ttk.Combobox(f_bot_4, values=languages, state='readonly')
label_p2 = tk.Label(
    f_bot_4,
    width=20,
    height=2
)
label_p = tk.Label(
    f_bot23,
    width=20,
    height=2
)
label_p1 = tk.Label(
    f_bot1,
    width=20,
    height=2
)
label_Det = tk.Label(
    f_bot,
    text='Выберете детектор',
    fg='black',
    bg='LightCyan2',
    width=20,
    height=1
)
label_Des = tk.Label(
    f_bot,
    text='Выберете дескриптор',
    fg='black',
    bg='LightCyan2',
    width=20,
    height=1
)
label_Mat = tk.Label(
    f_bot,
    text='Выберете матчер',
    fg='black',
    bg='LightCyan2',
    width=20,
    height=1
)
languages = ['ORB', 'FAST', 'SIFT']
label4 = ttk.Label()
combobox_Det = ttk.Combobox(f_bot2, values=languages, state='readonly')

languages = ['ORB', 'SIFT']
label3 = ttk.Label()
combobox_Des = ttk.Combobox(f_bot2, values=languages, state='readonly')

languages = ['Brute_Force', 'FLANN']
label2 = ttk.Label()
combobox_Mat = ttk.Combobox(f_bot2, values=languages, state='readonly')
label_Met = Label(
    f_bot1,
    text='Выберете метрики',
    fg='black',
    bg='LightCyan2'
)

languages = ['Количество хороших совпадений']
label1 = ttk.Label()
combobox_Met = ttk.Combobox(f_bot1, values=languages, state='readonly')
btn_Par = Button(
    f_bot1,
    text='Параметры',
    fg='black',
    bg='LightBlue1',
)

btn_Otchet = Button(
    f_bot1,
    text='Отчет',
    fg='black',
    bg='LightBlue1',
    command=lambda: [Detect(), Compute(), MAT(), Otchet(), Del_file()]

)
# .....................................ВЕРСТКА...........................................................................
f_top.pack()
f_top1.pack()
f_bot_4.pack()
f_bot23.pack()
f_bot.pack()
f_bot2.pack()
f_bot1.pack()

label_File.pack(side=LEFT, padx=5, pady=5)
btn_File.pack(side=RIGHT, padx=5, pady=5)
label_File1.pack(side=LEFT, padx=5, pady=5)
btn_File1.pack(side=RIGHT, padx=5, pady=5)
label_p.pack(side=BOTTOM, padx=5, pady=5)

label_p2.pack(side=TOP, padx=5, pady=5)
label_Type.pack(side=TOP, padx=5, pady=5)
combobox_Type.pack(side=BOTTOM, padx=5, pady=5)

label_Det.pack(side=LEFT, fill=X, padx=5, pady=5)
label_Des.pack(side=LEFT, fill=X, padx=5, pady=5)
label_Mat.pack(side=LEFT, fill=X, padx=5, pady=5)

combobox_Det.pack(side=LEFT, padx=5, pady=5)
combobox_Des.pack(side=LEFT, padx=5, pady=5)
combobox_Mat.pack(side=LEFT, padx=5, pady=5)

label_p1.pack(anchor=NW, fill=X, padx=5, pady=5)
label_Met.pack(anchor=NW, fill=X, padx=5, pady=5)
combobox_Met.pack(anchor=NW, fill=X, padx=5, pady=5)
btn_Par.pack(side=LEFT, fill=X, padx=5, pady=5)
btn_Otchet.pack(side=RIGHT, fill=X, padx=5, pady=5)

root.mainloop()

