# # # import matplotlib.pyplot as plt
# # # import numpy as np
# # # from PIL import Image
# # # import cv2

# # # def image_to_array(image_path):
# # #     img = Image.open(image_path)
# # #     img_array = np.array(img)
# # #     return img_array


# # # def array_to_image(array, output_path):
# # #     img = Image.fromarray(array)
# # #     img.save(output_path)


# # # def lz77_compress_image(image_array, window_size=12):
# # #     compressed_data = []
# # #     height, width, _ = image_array.shape
# # #     flat_image = image_array.flatten()

# # #     i = 0
# # #     while i < len(flat_image):
# # #         match = find_longest_match(flat_image, i, window_size)
# # #         if match:
# # #             length, distance = match
# # #             compressed_data.append(
# # #                 (
# # #                     length,
# # #                     distance,
# # #                     flat_image[i + length] if i + length < len(flat_image) else 0,
# # #                 )
# # #             )
# # #             i += length + 1
# # #         else:
# # #             compressed_data.append((0, 0, flat_image[i]))
# # #             i += 1

# # #     return compressed_data, height, width


# # # def find_longest_match(text, current_index, window_size):
# # #     best_match = (0, 0)

# # #     for i in range(max(0, current_index - window_size), current_index):
# # #         length = 0
# # #         while (
# # #             current_index + length < len(text)
# # #             and text[i + length] == text[current_index + length]
# # #         ):
# # #             length += 1

# # #         if length > best_match[0]:
# # #             best_match = (length, current_index - i)

# # #     return best_match if best_match[0] > 0 else None


# # # def lz77_decompress_image(compressed_data, height, width):
# # #     decompressed_data = np.zeros(height * width * 3, dtype=np.uint8)
# # #     index = 0

# # #     for length, distance, value in compressed_data:
# # #         if length > 0:
# # #             start = index - distance
# # #             for i in range(length):
# # #                 decompressed_data[index] = decompressed_data[start + i]
# # #                 index += 1
# # #         decompressed_data[index] = value
# # #         index += 1

# # #     decompressed_data = decompressed_data[:index].reshape((height, width, -1))
# # #     return decompressed_data.astype(np.uint8)


# # # # Example usage:
# # # if __name__ == "__main__":
# # #     # Replace this with the pathpath/to/your/image.png to your image file
# # #     image_path = "/home/shusrith/projects/mml/part2-image1.jpg"

# # #     original_image = image_to_array(image_path)

# # #     # Compression
# # #     compressed_data, height, width = lz77_compress_image(original_image)
# # #     # print("Compressed data:", compressed_data)
# # #     compressed_size = sum(len(str(item)) for item in compressed_data)
# # #     print("Original Size:", original_image.nbytes, "bytes")
# # #     print("Compressed Size:", compressed_size, "bytes")
# # #     compression_ratio = compressed_size / original_image.nbytes
# # #     print("Compression Ratio:", compression_ratio)

# # #     # Decompression
# # #     decompressed_image = lz77_decompress_image(compressed_data, height, width)

# # #     max_pixel_value = 255  # assuming 8-bit image
# # #     mse = np.mean((original_image - decompressed_image) ** 2)
# # #     psnr_value = 10 * np.log10((max_pixel_value ** 2) / mse) if mse != 0 else float('inf')
# # #     print("PSNR:", psnr_value, "dB")

# # #     # Check if compression and decompression were successful
# # #     assert np.array_equal(original_image, decompressed_image)

# # #     # Save the decompressed image
# # #     # array_to_image(decompressed_image, "one.png")

# # #     visualization = np.zeros_like(original_image, dtype=np.uint8)

# # #     index = 0
# # #     for length, distance, value in compressed_data:
# # #         if length > 0:
# # #             start = index - distance
# # #             for i in range(length):
# # #                 visualization[index] = visualization[start + i]
# # #                 index += 1
# # #         visualization[index] = value
# # #         index += 1

# # #     visualization = visualization.reshape((height, width, -1))

# # #     # Display the visualization
# # #     plt.imshow(visualization)
# # #     plt.title("Visualization of Compressed Data")
# # #     plt.show()


# # # Import the necessary libraries
# # import cv2
# # import numpy as np


# # def find_longest_match(input_string, current_position, window_size, buffer_size):
# #     end_of_buffer = min(current_position + buffer_size, len(input_string))
# #     best_match_distance = 0
# #     best_match_length = 0
# #     for j in range(current_position - window_size, current_position):
# #         if j < 0:
# #             continue
# #         substring = input_string[current_position:end_of_buffer]
# #         match_length = 0
# #         while (
# #             match_length < len(substring)
# #             and substring[match_length] == input_string[j + match_length]
# #         ):
# #             match_length += 1
# #         if match_length > best_match_length:
# #             best_match_distance = current_position - j
# #             best_match_length = match_length
# #     if best_match_distance and best_match_length:
# #         return best_match_distance, best_match_length
# #     else:
# #         return None


# # def compress_lz77(input_string, window_size, buffer_size):
# #     compressed_data = []
# #     i = 0
# #     while i < len(input_string):
# #         match = find_longest_match(input_string, i, window_size, buffer_size)
# #         if match:
# #             best_match_distance, best_match_length = match
# #             compressed_data.append(
# #                 (
# #                     best_match_distance,
# #                     best_match_length,
# #                     input_string[i + best_match_length],
# #                 )
# #             )
# #             i += best_match_length + 1
# #         else:
# #             compressed_data.append((0, 0, input_string[i]))
# #             i += 1
# #     return compressed_data


# # # Function to compress an image using LZ77
# # def compress_image_lz77(image, window_size, buffer_size):
# #     # Convert the image to a 1D array
# #     image_array = image.flatten()

# #     # Compress the 1D array using LZ77
# #     compressed_data = compress_lz77(image_array, window_size, buffer_size)

# #     return compressed_data


# # # Function to decompress an image using LZ77
# # def decompress_image_lz77(compressed_data, image_shape):
# #     # Decompress the 1D array using the LZ77 decompression algorithm
# #     decompressed_array = decompress_lz77(compressed_data)

# #     # Reshape the 1D array to the original image shape
# #     decompressed_image = decompressed_array.reshape(image_shape)

# #     return decompressed_image


# # # Example usage
# # # Load the image
# # image = cv2.imread(
# #     "/home/shusrith/projects/mml/part2-image1.jpg", 0
# # )  # Read the image in grayscale

# # # Set the window size and buffer size
# # window_size = 100
# # buffer_size = 50

# # # Compress the image
# # compressed_data = compress_image_lz77(image, window_size, buffer_size)

# # # Decompress the image
# # decompressed_image = decompress_image_lz77(compressed_data, image.shape)


# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# from scipy import ndimage
# from scipy import optimize
# import numpy as np
# import math

# # Manipulate channels


# def get_greyscale_image(img):
#     return np.mean(img[:, :, :2], 2)


# def extract_rgb(img):
#     return img[:, :, 0], img[:, :, 1], img[:, :, 2]


# def assemble_rbg(img_r, img_g, img_b):
#     shape = (img_r.shape[0], img_r.shape[1], 1)
#     return np.concatenate(
#         (np.reshape(img_r, shape), np.reshape(img_g, shape), np.reshape(img_b, shape)),
#         axis=2,
#     )


# # Transformations


# def reduce(img, factor):
#     result = np.zeros((img.shape[0] // factor, img.shape[1] // factor))
#     for i in range(result.shape[0]):
#         for j in range(result.shape[1]):
#             result[i, j] = np.mean(
#                 img[i * factor : (i + 1) * factor, j * factor : (j + 1) * factor]
#             )
#     return result


# def rotate(img, angle):
#     return ndimage.rotate(img, angle, reshape=False)


# def flip(img, direction):
#     return img[::direction, :]


# def apply_transformation(img, direction, angle, contrast=1.0, brightness=0.0):
#     return contrast * rotate(flip(img, direction), angle) + brightness


# # Contrast and brightness


# def find_contrast_and_brightness1(D, S):
#     # Fix the contrast and only fit the brightness
#     contrast = 0.75
#     brightness = (np.sum(D - contrast * S)) / D.size
#     return contrast, brightness


# def find_contrast_and_brightness2(D, S):
#     # Fit the contrast and the brightness
#     A = np.concatenate((np.ones((S.size, 1)), np.reshape(S, (S.size, 1))), axis=1)
#     b = np.reshape(D, (D.size,))
#     x, _, _, _ = np.linalg.lstsq(A, b)
#     # x = optimize.lsq_linear(A, b, [(-np.inf, -2.0), (np.inf, 2.0)]).x
#     return x[1], x[0]


# # Compression for greyscale images


# def generate_all_transformed_blocks(img, source_size, destination_size, step):
#     factor = source_size // destination_size
#     transformed_blocks = []
#     for k in range((img.shape[0] - source_size) // step + 1):
#         for l in range((img.shape[1] - source_size) // step + 1):
#             # Extract the source block and reduce it to the shape of a destination block
#             S = reduce(
#                 img[
#                     k * step : k * step + source_size, l * step : l * step + source_size
#                 ],
#                 factor,
#             )
#             # Generate all possible transformed blocks
#             for direction, angle in candidates:
#                 transformed_blocks.append(
#                     (k, l, direction, angle, apply_transformation(S, direction, angle))
#                 )
#     return transformed_blocks


# def compress(img, source_size, destination_size, step):
#     transformations = []
#     transformed_blocks = generate_all_transformed_blocks(
#         img, source_size, destination_size, step
#     )
#     i_count = img.shape[0] // destination_size
#     j_count = img.shape[1] // destination_size
#     for i in range(i_count):
#         transformations.append([])
#         for j in range(j_count):
#             print("{}/{} ; {}/{}".format(i, i_count, j, j_count))
#             transformations[i].append(None)
#             min_d = float("inf")
#             # Extract the destination block
#             D = img[
#                 i * destination_size : (i + 1) * destination_size,
#                 j * destination_size : (j + 1) * destination_size,
#             ]
#             # Test all possible transformations and take the best one
#             for k, l, direction, angle, S in transformed_blocks:
#                 contrast, brightness = find_contrast_and_brightness2(D, S)
#                 S = contrast * S + brightness
#                 d = np.sum(np.square(D - S))
#                 if d < min_d:
#                     min_d = d
#                     transformations[i][j] = (
#                         k,
#                         l,
#                         direction,
#                         angle,
#                         contrast,
#                         brightness,
#                     )
#     return transformations


# def decompress(transformations, source_size, destination_size, step, nb_iter=8):
#     factor = source_size // destination_size
#     height = len(transformations) * destination_size
#     width = len(transformations[0]) * destination_size
#     iterations = [np.random.randint(0, 256, (height, width))]
#     cur_img = np.zeros((height, width))
#     for i_iter in range(nb_iter):
#         print(i_iter)
#         for i in range(len(transformations)):
#             for j in range(len(transformations[i])):
#                 # Apply transform
#                 k, l, flip, angle, contrast, brightness = transformations[i][j]
#                 S = reduce(
#                     iterations[-1][
#                         k * step : k * step + source_size,
#                         l * step : l * step + source_size,
#                     ],
#                     factor,
#                 )
#                 D = apply_transformation(S, flip, angle, contrast, brightness)
#                 cur_img[
#                     i * destination_size : (i + 1) * destination_size,
#                     j * destination_size : (j + 1) * destination_size,
#                 ] = D
#         iterations.append(cur_img)
#         cur_img = np.zeros((height, width))
#     return iterations


# # Compression for color images


# def reduce_rgb(img, factor):
#     img_r, img_g, img_b = extract_rgb(img)
#     img_r = reduce(img_r, factor)
#     img_g = reduce(img_g, factor)
#     img_b = reduce(img_b, factor)
#     return assemble_rbg(img_r, img_g, img_b)


# def compress_rgb(img, source_size, destination_size, step):
#     img_r, img_g, img_b = extract_rgb(img)
#     return [
#         compress(img_r, source_size, destination_size, step),
#         compress(img_g, source_size, destination_size, step),
#         compress(img_b, source_size, destination_size, step),
#     ]


# def decompress_rgb(transformations, source_size, destination_size, step, nb_iter=8):
#     img_r = decompress(
#         transformations[0], source_size, destination_size, step, nb_iter
#     )[-1]
#     img_g = decompress(
#         transformations[1], source_size, destination_size, step, nb_iter
#     )[-1]
#     img_b = decompress(
#         transformations[2], source_size, destination_size, step, nb_iter
#     )[-1]
#     return assemble_rbg(img_r, img_g, img_b)


# # Plot


# def plot_iterations(iterations, target=None):
#     # Configure plot
#     plt.figure()
#     nb_row = math.ceil(np.sqrt(len(iterations)))
#     nb_cols = nb_row
#     # Plot
#     for i, img in enumerate(iterations):
#         plt.subplot(nb_row, nb_cols, i + 1)
#         plt.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="none")
#         if target is None:
#             plt.title(str(i))
#         else:
#             # Display the RMSE
#             plt.title(
#                 str(i)
#                 + " ("
#                 + "{0:.2f}".format(np.sqrt(np.mean(np.square(target - img))))
#                 + ")"
#             )
#         frame = plt.gca()
#         frame.axes.get_xaxis().set_visible(False)
#         frame.axes.get_yaxis().set_visible(False)
#     plt.tight_layout()


# # Parameters

# directions = [1, -1]
# angles = [0, 90, 180, 270]
# candidates = [[direction, angle] for direction in directions for angle in angles]

# # Tests


# def test_greyscale():
#     img = mpimg.imread("/home/shusrith/projects/mml/part2-image1.jpg")
#     img = get_greyscale_image(img)
#     img = reduce(img, 4)
#     plt.figure()
#     plt.imshow(img, cmap="gray", interpolation="none")
#     transformations = compress(img, 8, 4, 8)
#     iterations = decompress(transformations, 8, 4, 8)
#     plot_iterations(iterations, img)
#     plt.show()


# def test_rgb():
#     img = mpimg.imread("lena.gif")
#     img = reduce_rgb(img, 8)
#     transformations = compress_rgb(img, 8, 4, 8)
#     retrieved_img = decompress_rgb(transformations, 8, 4, 8)
#     plt.figure()
#     plt.subplot(121)
#     plt.imshow(np.array(img).astype(np.uint8), interpolation="none")
#     plt.subplot(122)
#     plt.imshow(retrieved_img.astype(np.uint8), interpolation="none")
#     plt.show()


# if __name__ == "__main__":
#     test_greyscale()
#     # test_rgb()


import sys, getopt
from PIL import Image
from pathlib import Path
import os

# 全局存放压缩的图片信息
g_compress_data = []


def commandHelp():
    """
    命令帮助
    """
    print("\n")
    print(
        "test.py [-i <imgs>] [-q <quality>] [-s <subsampling>] [-j <jpga>] [-d <dir>]"
    )
    print('     -i, --imgs 需要压缩的图片，多个图片以逗号分隔 "a.jpg,b.jpg')
    print("     -q, --quality 默认压缩的图片质量为15，可以调整0-95 ")
    print("     -j, --jpga 为1时设置将图片统计转换成.jpg格式，默认为0 ")
    print("     -d, --dir 设置一个目录，压缩指定目录下的图片 ")
    print("     -s, subsampling 设置编码器的子采样 默认-1 ")
    print("                     -1: equivalent to keep ")
    print("                      0: equivalent to 4:4:4 ")
    print("                      1: equivalent to 4:2:2 ")
    print("                      2: equivalent to 4:2:0 ")
    print("\n")
    print("命令示例：python test.py -i a.jpg,b.jpg -q 20")
    print("\n")


def main(argv):
    """
    命令执行示例：python test.py -i a.jpg,b.jpg -q 20 \n
    imgs：          接收需要压缩的图片路径，a.jpg,b.jpg \n
    quality：       默认压缩的图片质量为15，可以调整 \n
    subsampling     子采样值 默认-1
    dir：           指定要压缩的目录
    jpga：          当为1时统计转换成jpg格式，默认为0不转换
    """
    imgs = ""
    quality = 15
    subsampling = -1
    jpga = 0  # 判断是否全部转换成jpg格式保存
    output = "output"
    dir_files = ""  # 要执行的目录，也就是图片文件存在的目标目录

    try:
        opts, args = getopt.getopt(
            argv, "hi:q:s:j:d:", ["imgs=", "quality=", "subsampling=", "jpga=", "dir"]
        )
    except getopt.GetoptError:
        commandHelp()
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            commandHelp()
            sys.exit()
        elif opt in ("-i", "--imgs"):
            imgs = arg
        elif opt in ("-q", "--quality"):
            quality = int(arg)
        elif opt in ("-s", "--subsampling"):
            subsampling = int(arg)
        elif opt in ("-j", "--jpga"):
            jpga = int(arg)
            print("opt_jpga: ", jpga)
        elif opt in ("-d", "--dir"):
            dir_files = arg

    # print('imgs:', imgs)
    # print('quality:', quality)

    notfound_imgs = []

    if dir_files:
        dirOfImageCompress(dir_files, quality, subsampling, notfound_imgs, jpga, output)
        return

    if len(imgs) > 0:
        # 创建output目录
        output_dir = Path(output)
        if output_dir.exists() == False:
            os.mkdir(output)
        for img_item in imgs.split(","):
            imageCompress(quality, subsampling, img_item, notfound_imgs, jpga, output)

    # TODO 向屏幕输出压缩结果信息
    # print(g_compress_data)
    if notfound_imgs:
        print("找不到的文件：", notfound_imgs)


def dirOfImageCompress(dir, quality, subsampling, notfound_imgs, jpga, output):
    """
    当命令行中-d不为空时，表示要在指定目录里搜索图片文件进行压缩
    """
    for dirpath, dirname, filenames in os.walk(dir):
        # print('目录：', dirpath)

        if dirpath.endswith("/output") == False:
            # print('目录名：', dirname)
            output_dir = Path("{}/{}".format(dirpath, output))
            if output_dir.exists() == False:
                output_dir.mkdir()

            for filename in filenames:
                # print('文件：', filename)
                if (
                    filename.endswith(".jpg")
                    or filename.endswith(".png")
                    or filename.endswith(".JPG")
                    or filename.endswith(".PNG")
                ):
                    imageCompress(
                        quality,
                        subsampling,
                        "{}/{}".format(dirpath, filename),
                        notfound_imgs,
                        jpga,
                        output_dir,
                    )


def imageCompress(quality, subsampling, img_item, notfound_imgs, jpga, output):
    """
    把单个文件传入此方法进行压缩
    """
    img_item_path = ""
    img_item_path = Path(os.path.abspath(img_item))
    img_item_endswith = (
        img_item.endswith(".png")
        or img_item.endswith(".jpg")
        or img_item.endswith(".JPG")
        or img_item.endswith(".PNG")
    )

    if img_item_path.is_file() and img_item_endswith:
        # 文件存在就开始压缩
        # 压缩前的文件名
        img_file_name = img_item_path.name
        img_item_data = {"fileNameBefore": img_file_name}
        img: Image.Image = Image.open(img_item_path)
        # w,h = img.size
        # print('Origin image size: %sx%s' % (w, h))
        shotname = ""  # 文件名
        extension = ""  # 扩展名
        (shotname, extension) = os.path.splitext(img_file_name)
        # 获取压缩前的文件byte
        byteSizeBefore = len(img.fp.read())
        img_item_data["byteSizeBefore"] = byteSizeBefore

        # 只压缩大于300KB图片
        # if byteSizeBefore < 307200:
        #     return

        # 只压缩大于100KB图片
        if byteSizeBefore < 102400:
            return

        # 区别jpg、png
        if img_item.endswith(".png") or img_item.endswith(".PNG"):
            if jpga > 0:
                img = img.convert("RGB")
                extension = ".jpg"
            else:
                img = img.quantize(colors=256)

        save_file = "{}/{}{}".format(output, shotname, extension)
        img_item_data["fileNameAfter"] = save_file
        img.save(save_file, quality=quality, optimize=True, subsampling=subsampling)
        byteSizeAfter = os.path.getsize(save_file)
        img_item_data["byteSizeAfter"] = byteSizeAfter
        g_compress_data.append(img_item_data)
        print(
            img_file_name,
            "压缩前：",
            str(convert_mb_kb(byteSizeBefore)),
            "压缩后：",
            str(convert_mb_kb(byteSizeAfter)),
        )
        print("-" * 70)

    else:
        # 标记不存在的文件
        notfound_imgs.append(img_item)


def convert_mb_kb(bytesize):
    """
    把byte长度转换成KB,MB
    """
    if bytesize > 0:
        bytesize = bytesize / 1024
        if bytesize < 1024:
            return "%.fKB" % bytesize
        else:
            return "%.2fMB" % (bytesize / 1024)


if __name__ == "__main__":
    main(sys.argv[1:])
