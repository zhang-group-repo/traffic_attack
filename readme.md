# Rust-Style Patch

Official implementation of "Rust-Style Patch: A Physical and Naturalistic Camouflage Attacks on Object Detector for Remote Sensing Images".

Deep neural networks (DNNs) can improve the image analysis and interpretation of remote sensing technology by extracting valuable information from images, and has extensive applications such as military affairs, agriculture, environment, transportation, and urban division. The DNNs for object detection can identify and analyze objects in remote sensing images through fruitful features of images, which improves the efficiency of image processing and enables the recognition of large-scale remote sensing images. However, many studies have shown that deep neural networks are vulnerable to adversarial attack. After adding small perturbations, the generated adversarial examples will cause deep neural network to output undesired results, which will threaten the normal recognition and detection of remote sensing systems. According to the application scenarios, attacks can be divided into the digital domain and the physical domain, the digital domain attack is directly modified on the original image, which is mainly used to simulate the attack effect, while the physical domain attack adds perturbation to the actual objects and captures them with device, which is closer to the real situation. Attacks in the physical domain are more threatening, however, existing attack methods generally generate the patch with bright style and a large attack range, which is easy to be observed by human vision. Our goal is to generate a natural patch with a small perturbation area, which can help some remote sensing images used in the military to avoid detection by object detectors and im-perceptible to human eyes. To address the above issues, we generate a rust-style adversarial patch generation framework based on style transfer. The framework takes a heat map-based interpretability method to obtain key areas of target recognition and generate irregular-shaped natural-looking patches to reduce the disturbance area and alleviates suspicion from humans. To make the generated adversarial examples have a higher attack success rate in the physical domain, we further improve the robustness of the adversarial patch through data augmentation methods such as rotation, scaling, and brightness, and finally, make it impossible for the object detector to detect the camouflage patch. We have attacked the YOLOV3 detection network on multiple datasets. The experimental results show that our model has achieved a success rate of 95.7% in the digital domain. We also conduct physical attacks in indoor and outdoor environments and achieve an attack success rate of 70.6% and 65.3%, respectively. The structural similarity index metric shows that the adversarial patches generated are more natural than existing methods.

![frame](./framework.png)


# 热力图生成
yolov3/
- main_gradcam.py
  cls_name 限定交通标识类别
  输出三个检测尺寸的热力图与最后的mask
- models/gradcam.py 生成激活层的函数

# 图像与文件（公用）
- yolov3/data 各种交通标识图片的文件 
- phy/与物理域相关的文件
- style_transfer.py (与风格迁移相关的函数，eg：风格损失和内容损失)
- patch.py（针对补丁相关操作，eg:patchApplier映射函数） 
- PytorchYOLOv3/ 攻击的模型
   - detect.py 主要文件,涉及检测过程中的一些数据处理；DetectorYolov3()返回的数值用于攻击


# 攻击文件（四）
phy_attack.py(物理域攻击)
 - 攻击图像内容路径：content_img_path = f'{yolov3/data/test/}'
 - 攻击图像从211行主函数 main('000000026162.jpg',detectorYolov3)传入
 其他同上
- style_img_path = './yolov3/tie2.jpg' 风格迁移图像
- mask_path = f'-yolov3/outputs/{imgae_name}/mask.jpg'   mask图像
- patch_path = f'{output_adv/patch/{imgae_name}.jpg'   补丁存储路径

