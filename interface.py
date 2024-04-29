import os
import sys

import gradio as gr

from yolov3.detect import prepare_model, predict_image
from yolov3.models.yolo_detector import YOLOV3TorchObjectDetector
from yolov3.main_gradcam import args as gradcam_args
from yolov3.main_gradcam import names as gradcam_names
from yolov3.main_gradcam import main as gradcam_main

from PyTorchYOLOv3.detect import DetectorYolov3
from torchvision.utils import save_image
from style_transfer import load_img

import subprocess
import shutil

css = """
.column2 {
    padding: 100px 10px 0px;
}
.column34{
    padding: 40px 10px 0px;
}
.column2_btn{
    padding: 112px 0px 0px;
}
"""

# 把原图和各种处理的图片单独创建一个gradio文件夹存起来
PREDICTED_IMAGES = 'gradio/predicted_images/'
GRADCAM_IMAGES = 'gradio/gradcam_images/'

STYLE_IMAGE = 'gradio/style_images/tie1.jpg'

MASK_IMAGE = 'gradio/gradcam_images/{}/mask.jpg'
PATCH_IMAGE = 'gradio/attack_images/patch/{}'
ADV_IMAGE = 'gradio/attack_images/adv_img/{}'
DET_IMAGE = 'gradio/attack_images/det_img/{}'
CLEAR_DET_IMAGE = 'gradio/attack_images/yolov3_clear_det/{}'



def delete_files(folder_path):
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


# 原图检测detect
def wrapped_predict_image(image):
    global predict_model
    global PREDICTED_IMAGES
    delete_files(PREDICTED_IMAGES)
    save_dir = predict_image(model=predict_model, source=image, SAVE_DIR=PREDICTED_IMAGES)
    return os.path.join(save_dir, os.path.basename(image))


# 原图检测 使用PyTorchYOLOv3模型
def wrapped_predict_image2(image):
    global yolov3
    global CLEAR_DET_IMAGE
    delete_files(PREDICTED_IMAGES)

    names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
             'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
             'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
             'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
             'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
             'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
             'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
             'cell phone',
             'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
             'hair drier', 'toothbrush']  # class names

    save_path = CLEAR_DET_IMAGE.format(os.path.basename(image))
    loaded_image = load_img(image, (416, 416))
    _, _, det, _, _ = yolov3.detect(input_imgs=loaded_image, cls_id_attacked=11)
    detected_img = yolov3.plot(loaded_image, names, det, 0.5)
    save_image(detected_img, save_path)
    return save_path


# 生成攻击区域gradcam
def wrapped_gradcam_main(image, attack):
    global gradcam_model
    global GRADCAM_IMAGES
    delete_files(GRADCAM_IMAGES)
    _, save_dir = gradcam_main(img_path=image, model=gradcam_model, SAVE_DIR=GRADCAM_IMAGES, attack_range=attack)
    result_img = os.path.join(save_dir, "15_0.jpg")
    mask = os.path.join(save_dir, "mask.jpg")
    return result_img, mask


def wrapped_attack_main(image, style=None):
    global MASK_IMAGE
    global PATCH_IMAGE
    global ADV_IMAGE
    global DET_IMAGE

    basename = os.path.basename(image)

    delete_files('gradio/attack_images/adv_img/')
    delete_files('gradio/attack_images/det_img/')
    delete_files('gradio/attack_images/patch/')
    delete_files('gradio/attack_images/yolov3_clear_det/')

    command = [sys.executable, "4.py",
               "--img", image,
               "--style", "None" if style is None else style,
               "--mask", MASK_IMAGE.format(os.path.splitext(basename)[0]),
               "--patch", PATCH_IMAGE.format(basename),
               "--adv", ADV_IMAGE.format(basename),
               "--det", DET_IMAGE.format(basename),
               "--call", "1"]
    try:
        results = subprocess.check_output(command).decode()
        print(results)
        print("输出结果: ", results)
        left_pos = results.find('[', results.index('return'))
        right_pos = results.find(']', results.index('return'))
        paths = eval(results[left_pos:right_pos + 1])
        return paths[1], paths[2]
    except subprocess.CalledProcessError as e:
        print(e.output.decode())


# 前端页面代码
with gr.Blocks(css=css) as demo:
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="上传图像", type='filepath')
            style = gr.Image(label="上传风格图像", type='filepath')
            attack_param = gr.Slider(minimum=0, maximum=250, step=1, label="攻击区域", value=220)
        with gr.Column(elem_classes=["column2"]):
            output_img = gr.Image(interactive=False, show_label=False, height="30vh", type='filepath')
            with gr.Row(elem_classes=["column2_btn"]):
                detect_btn = gr.Button(value="原图检测")
                detect_btn.click(fn=wrapped_predict_image2, inputs=input_img, outputs=output_img)
        with gr.Column(elem_classes=["column34"]):
            output_img_1 = gr.Image(interactive=False, show_label=False, type="filepath")
            output_img_2 = gr.Image(interactive=False, show_label=False, type="filepath")
            attack_btn = gr.Button(value="生成攻击区域")
            attack_btn.click(fn=wrapped_gradcam_main, inputs=[input_img, attack_param],
                             outputs=[output_img_1, output_img_2])
        with gr.Column(elem_classes=["column34"]):
            result_img_1 = gr.Image(interactive=False, label="对抗样本检测结果", type="filepath")
            result_img_2 = gr.Image(interactive=False, label="生成的补丁", type="filepath")
            result_btn = gr.Button(value="攻击结果")
            result_btn.click(fn=wrapped_attack_main, inputs=[input_img, style], outputs=[result_img_1, result_img_2])

if __name__ == '__main__':
    # 加载原图检测模型
    predict_model_path = './yolov3/weights/yolov3.pt'
    # predict_model = prepare_model(weights=predict_model_path)
    yolov3 = DetectorYolov3(show_detail=False, tiny=True)

    # 加载热力图模型
    input_size = (gradcam_args.img_size, gradcam_args.img_size)
    gradcam_model = YOLOV3TorchObjectDetector(
        predict_model_path,
        gradcam_args.device,
        img_size=input_size,
        names=gradcam_names
    )

    demo.launch()
