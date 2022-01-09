from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QImage, QPixmap, QPalette, qRgb, qGray
import sys
import numpy as np
from typing import Callable
from numbers import Number


def process_image(
        input_image: np.array,
        kernel_size: int,
        kernel_fn: Callable[[np.array], float]) -> np.array:
    padding_width: int = kernel_size // 2
    padding_height: int = kernel_size // 2
    padding = ((padding_height, padding_height), (padding_width, padding_width))
    input_image_padding: np.array = np.pad(
        array=input_image,
        pad_width=padding,
        mode='edge')
    result_image: np.array = np.zeros(input_image.shape, dtype='float')
    image_height, image_width = result_image.shape
    for image_x in range(image_width):
        for image_y in range(image_height):
            x_pos_begin = image_x
            x_pos_end = image_x + kernel_size
            y_pos_begin = image_y
            y_pos_end = image_y + kernel_size
            image_segment: np.array = input_image_padding[y_pos_begin:y_pos_end, x_pos_begin:x_pos_end]
            result_image[image_y][image_x] = kernel_fn(image_segment)
    return result_image


def mean_fn(
        image_segment: np.array) -> float:
    return float(np.mean(image_segment))


def std_fn(
        image_segment: np.array) -> float:
    return float(np.std(image_segment))


def convert_to_binary(
        input_image: np.array,
        threshold: int = 127) -> np.array:
    max_val: int = 255
    min_val: int = 0
    initial_conv: np.array = np.where((input_image <= threshold), input_image, max_val)
    final_conv: np.array = np.where((initial_conv > threshold), initial_conv, min_val)
    return final_conv


def normalize_image(
        input_image: np.array) -> np.array:
    result_image: np.array = np.zeros(input_image.shape)
    input_max = input_image.max()
    input_min = input_image.min()
    input_range = input_max - input_min
    height, width = input_image.shape
    for y in range(height):
        for x in range(width):
            input_value = input_image[y][x]
            scaled_input_value = (input_value - input_min) / input_range if input_range != 0 else 0
            result_image[y][x] = scaled_input_value * 255.0
    return result_image


def fill_image(
        input_image: np.array,
        value: Number,
        replace_value: Number):
    height, width = input_image.shape
    for y in range(height):
        for x in range(width):
            if input_image[y, x] == value:
                input_image[y, x] = replace_value


def mark_objects(
        input_image: np.array) -> np.array:
    result_image: np.array = np.copy(input_image)
    current_object_id = 1
    height, width = input_image.shape
    for y in range(height):
        for x in range(width):
            if y == 0:
                c = 0
            else:
                c = result_image[y - 1, x]
            if x == 0:
                b = 0
            else:
                b = result_image[y, x - 1]
            a = result_image[y, x]
            if a == 0:
                pass
            elif b == 0 and c == 0:
                current_object_id += 1
                result_image[y, x] = current_object_id
            elif b != 0 and c == 0:
                result_image[y, x] = b
            elif b == 0 and c != 0:
                result_image[y, x] = c
            elif b != 0 and c != 0:
                if b == c:
                    result_image[y, x] = b
                else:
                    result_image[y, x] = b
                    fill_image(
                        input_image=result_image,
                        value=c,
                        replace_value=b)
    return result_image


def delete_objects(
        input_image: np.array,
        object_size: int):
    unique_mask, hist = np.unique(input_image, return_counts=True)
    for i in range(1, len(unique_mask)):
        if hist[i] < object_size:
            for (y, x), _ in np.ndenumerate(input_image):
                if input_image[y, x] == unique_mask[i]:
                    input_image[y, x] = 0


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('Main.ui', self)

        self.action_open = self.findChild(QtWidgets.QAction, 'actionOpen')
        self.action_open.triggered.connect(self.action_open_triggered)

        self.action_exit = self.findChild(QtWidgets.QAction, 'actionExit')
        self.action_exit.triggered.connect(self.action_exit_triggered)

        self.bt_apply = self.findChild(QtWidgets.QPushButton, 'btApply')
        self.bt_apply.clicked.connect(self.bt_apply_pressed)

        self.input_image_canvas = QtWidgets.QLabel()
        self.input_image_canvas.setBackgroundRole(QPalette.Base)
        self.input_image_canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Ignored)
        self.input_image_canvas.setScaledContents(True)
        self.sa_input_image = self.findChild(QtWidgets.QScrollArea, 'saInputImage')
        self.sa_input_image.setWidget(self.input_image_canvas)
        self.sa_input_image.setWidgetResizable(False)

        self.processed_image_canvas = QtWidgets.QLabel()
        self.processed_image_canvas.setBackgroundRole(QPalette.Base)
        self.processed_image_canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Ignored)
        self.processed_image_canvas.setScaledContents(True)
        self.sa_processed_image = self.findChild(QtWidgets.QScrollArea, 'saProcessedImage')
        self.sa_processed_image.setWidget(self.processed_image_canvas)
        self.sa_processed_image.setWidgetResizable(False)

        self.mask_image_canvas = QtWidgets.QLabel()
        self.mask_image_canvas.setBackgroundRole(QPalette.Base)
        self.mask_image_canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Ignored)
        self.mask_image_canvas.setScaledContents(True)
        self.sa_mask_image = self.findChild(QtWidgets.QScrollArea, 'saMask')
        self.sa_mask_image.setWidget(self.mask_image_canvas)
        self.sa_mask_image.setWidgetResizable(False)

        self.segmented_image_canvas = QtWidgets.QLabel()
        self.segmented_image_canvas.setBackgroundRole(QPalette.Base)
        self.segmented_image_canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Ignored)
        self.segmented_image_canvas.setScaledContents(True)
        self.sa_segmented_image = self.findChild(QtWidgets.QScrollArea, 'saSegmentedImage')
        self.sa_segmented_image.setWidget(self.segmented_image_canvas)
        self.sa_segmented_image.setWidgetResizable(False)

        self.cb_method = self.findChild(QtWidgets.QComboBox, 'cbMethod')
        self.cb_method.addItems(['Mean', 'Std'])

        self.le_kernel_size = self.findChild(QtWidgets.QLineEdit, 'leKernelSize')

        self.le_threshold = self.findChild(QtWidgets.QLineEdit, 'leThreshold')

        self.le_delete_objects = self.findChild(QtWidgets.QLineEdit, 'leDeleteObjects')

        self.show()

    def action_open_triggered(self):
        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QtWidgets.QFileDialog.\
            getOpenFileName(self,
                            'QFileDialog.getOpenFileName()',
                            '',
                            'Images (*.png *.jpeg *.jpg *.bmp *.gif)',
                            options=options)
        if file_name:
            image = QImage(file_name).convertToFormat(QImage.Format_Grayscale8)
            if image.isNull():
                QtWidgets.QMessageBox.\
                    information(self,
                                "Texture segmentation",
                                "Cannot load %s." % file_name)
                return

            self.input_image_canvas.setPixmap(QPixmap.fromImage(image))
            self.input_image_canvas.adjustSize()

    def action_exit_triggered(self):
        self.close()

    def bt_apply_pressed(self):
        method = self.cb_method.currentIndex()
        kernel_size = int(self.le_kernel_size.text())
        threshold = int(self.le_threshold.text())
        object_size = int(self.le_delete_objects.text())

        input_q_image = self.input_image_canvas.pixmap().toImage().convertToFormat(QImage.Format_Grayscale8)
        input_image = np.zeros((input_q_image.height(), input_q_image.width()), dtype='float')
        for (y, x), _ in np.ndenumerate(input_image):
            input_image[y, x] = qGray(input_q_image.pixel(x, y))

        if method == 0:
            kernel_fn = mean_fn
        elif method == 1:
            kernel_fn = std_fn
        else:
            return
        processed_image: np.array = process_image(
            input_image=input_image,
            kernel_size=kernel_size,
            kernel_fn=kernel_fn)
        normalized_image: np.array = normalize_image(input_image=processed_image)
        binarized_image: np.array = convert_to_binary(input_image=normalized_image, threshold=threshold)
        marked_image = mark_objects(input_image=binarized_image)
        delete_objects(
            input_image=marked_image,
            object_size=object_size)
        segmented_image = np.copy(input_image)
        for (y, x), _ in np.ndenumerate(segmented_image):
            if marked_image[y, x] == 0:
                segmented_image[y, x] = 0
        self.set_image(
            input_image=normalized_image,
            canvas=self.processed_image_canvas)
        self.set_image(
            input_image=normalize_image(
                input_image=marked_image),
            canvas=self.mask_image_canvas)
        self.set_image(
            input_image=segmented_image,
            canvas=self.segmented_image_canvas)

    @staticmethod
    def set_image(input_image: np.array, canvas: QtWidgets.QLineEdit):
        height, width = input_image.shape
        q_image = QImage(width, height, QImage.Format_RGB32)
        for y in range(height):
            for x in range(width):
                pixel = int(input_image[y, x])
                q_image.setPixel(x, y, qRgb(pixel, pixel, pixel))
        canvas.setPixmap(QPixmap.fromImage(q_image))
        canvas.adjustSize()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    app.exec_()
