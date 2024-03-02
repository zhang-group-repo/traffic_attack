# 热力图生成
yolov3/
- main_gradcam.py
  cls_name 限定交通标识类别
  输出三个检测尺寸的热力图与最后的mask
- models/gradcam.py 生成激活层的函数

# 图像与文件（公用）
- style_img_path = './yolov3/tie2.jpg' 风格迁移图像
- mask_path = f'-yolov3/outputs/{imgae_name}/mask.jpg'   mask图像
- patch_path = f'{output_adv/patch/{imgae_name}.jpg'   补丁存储路径
- yolov3/data 各种乱七八糟交通标识图片的文件 
- phy/与物理域相关的文件
- style_transfer.py (与风格迁移相关的函数，eg：风格损失和内容损失)
- patch.py（针对补丁相关操作，eg:patchApplier映射函数） 
- PytorchYOLOv3/ 攻击的模型
   - detect.py 主要文件,涉及检测过程中的一些数据处理；DetectorYolov3()返回的数值用于攻击
   - det_un.py 目标检测文件，保存检测结果
- output det_un.py 检测结果保存路径

# 攻击文件（四）
2.py(用来做消融实验，循环遍历文件夹图，输出相应数据)
 - 攻击图像内容：content_img_path = f'{yolov3/data/abl/}'
 - 对抗样本存储路径 save_path = f'output_adv/adv_img/{imgae_name}.jpg'
 - 最后的对抗样本名变量adv_image/补丁变量名patch

4.py(对主函数中输入的单个文件进行攻击)
 - 攻击图像内容：content_img_path = f'{yolov3/data/test/}'

3.py(生成物理域对抗攻击)


