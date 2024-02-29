import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time
import os

class CameraApp:
    def __init__(self, root, window_title):
        self.root = root
        self.root.title(window_title)

        # สร้างและตั้งค่า GUI
        self.label = ttk.Label(root)
        self.label.pack(padx=10, pady=10)

        # สร้างปุ่ม "Open Camera" และ "Capture"
        self.open_button = ttk.Button(root, text="Open Camera", command=self.open_camera)
        self.open_button.pack(pady=5)

        self.capture_button = ttk.Button(root, text="Capture", command=self.capture_image)
        self.capture_button.pack(pady=5)

        # เริ่มกล้องแต่ยังไม่เปิด
        self.cap = None

    def open_camera(self):
        # เปิดกล้อง
        self.cap = cv2.VideoCapture(0)

        # อัปเดต GUI เพื่อแสดงภาพจากกล้อง
        self.update_gui()

    def update_gui(self):
        # อ่านภาพจากกล้อง
        ret, frame = self.cap.read()

        # แปลงภาพจาก OpenCV BGR เป็น RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # แปลงภาพเป็น PhotoImage
        img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))

        # อัปเดต Label ด้วยภาพใหม่
        self.label.img_tk = img_tk
        self.label.config(image=img_tk)

        # เรียกใช้ฟังก์ชั่นนี้เองตามรอบเพื่อเป็นการอัปเดตภาพใน GUI
        self.root.after(10, self.update_gui)

    def capture_image(self):
        # ถ้ากล้องถูกเปิด
        if self.cap is not None:
            # หน่วงเวลา 5 วินาที
            time.sleep(5)

            # อ่านภาพจากกล้อง
            ret, frame = self.cap.read()

            # สร้างชื่อไฟล์ภาพที่ไม่ซ้ำกัน
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            image_name = f'captured_image_{timestamp}.png'

            # บันทึกรูปภาพ
            cv2.imwrite(image_name, frame)
            print(f'Image captured and saved as {image_name}')

    def run(self):
        self.root.mainloop()

# สร้างหน้าต่าง GUI
root = tk.Tk()
app = CameraApp(root, "Camera App")

# เรียกใช้งานโปรแกรม
app.run()

# ปิดกล้องเมื่อปิดโปรแกรม
if app.cap is not None:
    app.cap.release()
