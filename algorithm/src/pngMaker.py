from PIL import Image
import numpy as np


def flood_fill_alpha(arr, x, y, target_color, new_alpha):
    """使用洪泛填充算法将连通区域的alpha值设为透明"""
    rows, cols = arr.shape[:2]
    orig_color = arr[x, y]
    stack = [(x, y)]
    while stack:
        x, y = stack.pop()
        if arr[x, y][0] == target_color[0] and arr[x, y][1] == target_color[1] and arr[x, y][2] == target_color[2] and \
                arr[x, y][3] != new_alpha:
            arr[x, y][3] = new_alpha
            if x > 0:
                stack.append((x - 1, y))
            if x < rows - 1:
                stack.append((x + 1, y))
            if y > 0:
                stack.append((x, y - 1))
            if y < cols - 1:
                stack.append((x, y + 1))


def remove_black_border(input_image_path, output_image_path):
    img = Image.open(input_image_path)
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    data = np.array(img)
    target_color = [0, 0, 0]  # 黑色
    new_alpha = 0  # 完全透明

    # 对四个边进行洪泛填充
    for x in range(data.shape[0]):
        if np.array_equal(data[x, 0][:3], target_color):
            flood_fill_alpha(data, x, 0, target_color, new_alpha)
        if np.array_equal(data[x, data.shape[1] - 1][:3], target_color):
            flood_fill_alpha(data, x, data.shape[1] - 1, target_color, new_alpha)
    for y in range(data.shape[1]):
        if np.array_equal(data[0, y][:3], target_color):
            flood_fill_alpha(data, 0, y, target_color, new_alpha)
        if np.array_equal(data[data.shape[0] - 1, y][:3], target_color):
            flood_fill_alpha(data, data.shape[0] - 1, y, target_color, new_alpha)

    new_img = Image.fromarray(data)
    new_img.save(output_image_path, 'PNG')


# 调用函数
remove_black_border('./fig/QQ截图20240307202247.png', './fig/output_image.png')
