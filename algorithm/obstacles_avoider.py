import cv2
import numpy as np
from algorithm.route_planer import Dijkstra_main, Baka_main

def roi_finder(depth_image_shape):
    # 指定矩形框的参数
    top_left_y = 120
    height = 260
    width = 430
    center_x = depth_image_shape[1] // 2
    top_left_x = center_x - width // 2
    bottom_right_x = center_x + width // 2
    bottom_right_y = top_left_y + height
    return [top_left_x, top_left_y, bottom_right_x, bottom_right_y]

def obstacles_judger(depth_image):
    o_flag = False
    # 用一个极大值替换nan
    max_val = np.nanmax(depth_image) * 2
    depth_image_nan_replaced = np.nan_to_num(depth_image, nan=max_val)
    roi_shape = roi_finder(depth_image.shape)
    # 从深度图像中提取矩形框内的区域
    roi = depth_image_nan_replaced[roi_shape[1]:roi_shape[3], roi_shape[0]:roi_shape[2]]
    # 计算深度图像中水平方向上的深度差
    depth_diff = np.abs(np.diff(roi, axis=1))
    # 定义一个深度差阈值，当深度差超过这个阈值时，我们认为可能存在障碍物
    depth_diff_threshold = 0.5
    # 找到深度差超过阈值的像素点
    obstacle_points = np.where(depth_diff > depth_diff_threshold)
    # 如果超过阈值的点大于等于200个，计算这些点的中心点
    if obstacle_points[0].size > 199:
        center_y, center_x = np.mean(obstacle_points, axis=1)
        center_x = int(center_x) + roi_shape[0]
        center_y = int(center_y) + roi_shape[1]
        center_depth = depth_image_nan_replaced[center_y, center_x]
        # 如果中心点深度数值小于0.9m，则判断前方有障碍物
        if center_depth < 0.9:
            o_flag = True
    else:  # 如果不存在超过阈值的点，将中心点设置为一个默认值
        center_x, center_y, center_depth = -1, -1, -1
    return o_flag, [center_x, center_y], center_depth, obstacle_points

def o_pic_saver(depth_image, now_step, center, center_depth, obstacle_points):
    # 获取roi框信息
    roi_shape = roi_finder(depth_image.shape)
    # 计算有效深度的最小值和最大值
    min_valid_depth = np.nanmin(depth_image)
    max_valid_depth = np.nanmax(depth_image)
    # 将深度数据缩放到0-255的范围内
    scaled_depth_image = 255 * (depth_image - min_valid_depth) / (max_valid_depth - min_valid_depth)
    # 将 nan 值设置为0
    scaled_depth_image[np.isnan(scaled_depth_image)] = 0
    # 转换为 uint8 类型
    scaled_depth_image = scaled_depth_image.astype(np.uint8)
    # 转换成3通道图像，以便显示彩色
    depth_image_display = cv2.cvtColor(scaled_depth_image, cv2.COLOR_GRAY2BGR)
    # 为了可视化结果，我们在原始深度图像上标记出可能存在障碍物的点
    for y, x in zip(*obstacle_points):
        cv2.circle(depth_image_display, (roi_shape[0] + x, roi_shape[1] + y), 2, (0, 0, 255), -1)
    # 画出矩形框
    cv2.rectangle(depth_image_display, (roi_shape[0], roi_shape[1]), (roi_shape[2], roi_shape[3]), (0, 255, 0), 2)
    # 在图像的右下角以数字显示找到的超过阈值的像素点
    cv2.putText(depth_image_display, "Obstacle points: " + str(len(obstacle_points[0])),
                (depth_image_display.shape[1] - 450, depth_image_display.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    # 在图像上用黄色标记超过阈值的像素点的中心点
    cv2.circle(depth_image_display, (center[0], center[1]), 4, (0, 255, 255), -1)
    # 在图像右下角显示中心点所在像素的深度
    cv2.putText(depth_image_display, "Center depth: " + str(center_depth),
                (depth_image_display.shape[1] - 450, depth_image_display.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    path = f'./algorithm/temp/img/obstacles/{now_step}.png'
    # path = f'./temp/img/obstacles/{now_step}.png' # 测试用
    cv2.imwrite(path, depth_image_display)

def avoider_main(depth_image, now_step, area_B):
    # 重塑Webots传入的深度图片
    depth_image = np.reshape(depth_image, (720, 1280))
    # 将inf值设为nan，便于后续处理
    depth_image[depth_image == np.inf] = np.nan
    # 将图片传入障碍物判断器
    o_flag, center, center_depth, obstacle_points = obstacles_judger(depth_image)
    # 若发现障碍物，储存障碍物图像并启动避障
    if o_flag is True:
        print('前方发现障碍物，正在重新规划路线...')
        o_pic_saver(depth_image, now_step, center, center_depth, obstacle_points)
        actions = np.load('./algorithm/temp/actions.npy')
        actions = actions[:now_step - 1]
        actions = actions.tolist()
        car_status = np.load('./algorithm/temp/car_status.npy', allow_pickle=True)
        # 根据障碍物中心所处位置判断向左或是向右转，旋转之后行走8步避障
        if center[0] < depth_image.shape[1] // 2:
            actions.extend(['right', 'right'])
            car_status[0] -= np.radians(18.95 * 2)
        else:
            actions.extend(['left', 'left'])
            car_status[0] += np.radians(18.95 * 2)
        actions.extend(['ahead', 'ahead', 'ahead', 'ahead', 'ahead', 'ahead', 'ahead', 'ahead'])
        car_status[1] += (8 / 11) * np.array([-np.sin(car_status[0]), np.cos(car_status[0])])
        # 完成避障后重新寻路
        # actions_extend = Baka_main(car_status, area_B) # 八嘎寻路
        # print(f'避障程序传入完成避障后小车与Y轴夹角为{np.degrees(car_status[0]):.2f}度，'
        #               f'位于({car_status[1][0]:.4f}, {car_status[1][1]:.4f})。')
        actions_extend = Dijkstra_main(car_status, area_B) # 迪杰斯特拉寻路
        actions.extend(actions_extend)
        np.save('./algorithm/temp/actions', actions)
    return o_flag
