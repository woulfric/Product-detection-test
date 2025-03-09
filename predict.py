from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt


model = YOLO("runs/segment/train3/weights/best.pt")


source = "video\IMG_9553.png" 

model.predict(source, show= True, save= True, conf= 0.7, line_width= 2, save_crop= True, save_txt= True, show_labels= True, show_conf= True, classes= [0, 1, 2])


# results = model(test_img) 

# res_plotted = results[0].plot()
# plt.imshow(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
# plt.axis("off")
# plt.show()
