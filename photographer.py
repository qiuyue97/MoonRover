from controller import Camera, RangeFinder, Supervisor
import numpy as np
import cv2


def d_img_saver(img_depth, world_time):
    img_depth = np.reshape(img_depth, (720, 1280))
    # 将深度图像中的 inf 替换为 nan
    img_depth[img_depth == np.inf] = np.nan
    # 计算有效深度的最小值和最大值
    min_valid_depth = np.nanmin(img_depth)
    max_valid_depth = np.nanmax(img_depth)
    # 将深度数据缩放到0-255的范围内
    scaled_depth_image = 255 * (img_depth - min_valid_depth) / (max_valid_depth - min_valid_depth)
    # 将 nan 值设置为0
    scaled_depth_image[np.isnan(scaled_depth_image)] = 0
    # 转换为 uint8 类型
    scaled_depth_image = scaled_depth_image.astype(np.uint8)
    cv2.imwrite(f'./algorithm/src/fig/d_f_img/{world_time}.png', scaled_depth_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return None

def rgb_img_saver(img_rgb, world_time):
    img_rgb = cv2.cvtColor(np.frombuffer(img_rgb, np.uint8).reshape((720, 1280, 4)), cv2.COLOR_BGRA2BGR)
    cv2.imwrite(f'./algorithm/src/fig/rgb_f_img/{world_time}.png', img_rgb, [cv2.IMWRITE_PNG_COMPRESSION, 0])


if __name__ == '__main__':
    sup = Supervisor()
    # sup.simulationReset()
    # sup.step(1000)

    robot = sup.getFromDef('Rover')  # 连接巡视器

    motor_lf = sup.getDevice('lf_motor')  # 连接左前轮驱动
    motor_lb = sup.getDevice('lb_motor')  # 连接左后轮驱动
    motor_rf = sup.getDevice('rf_motor')  # 连接右前轮驱动
    motor_rb = sup.getDevice('rb_motor')  # 连接右后轮驱动

    motor_lf.setPosition(float('inf'))  # 左前轮初始化
    motor_rf.setPosition(float('inf'))
    motor_lb.setPosition(float('inf'))
    motor_rb.setPosition(float('inf'))
    motor_lf.setVelocity(0)  # 给定左前轮转速，不超过1
    motor_rf.setVelocity(0)
    motor_lb.setVelocity(0)
    motor_rb.setVelocity(0)

    cam_f = Camera('cam_f')  # 连接前相机-RGB
    cam_b = Camera('cam_b')
    cam_f.enable(30)  # 初始化
    cam_b.enable(30)

    dep_f = RangeFinder('depth_f')  # 连接前相机D
    dep_b = RangeFinder('depth_b')
    dep_f.enable(30)  # 初始化
    dep_b.enable(30)

    world_time = 0
    while True:
        """获取RGBD数据"""
        rgb_image_f = cam_f.getImage()  # 前相机获取RGB图像
        rgb_image_b = cam_b.getImage()
        d_image_f = dep_f.getRangeImage()  # 前相机获取D图像
        d_image_b = dep_b.getRangeImage()

        """巡视器轮速控制"""
        motor_velocity = [1, 1, 1, 1]
        motor_lf.setVelocity(motor_velocity[0])  # 给定左前轮转速，-1~1
        motor_rf.setVelocity(motor_velocity[1])
        motor_lb.setVelocity(motor_velocity[2])
        motor_rb.setVelocity(motor_velocity[3])

        """仿真环境运行"""
        for _ in range(30):
            if world_time % 640 == 0:
                # d_img_saver(d_image_f, world_time)
                rgb_img_saver(rgb_image_f, world_time)
            sup.step(32)
            world_time += 32  # 32ms

