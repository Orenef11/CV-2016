# /usr/bin/env python
"""
==================================================================================================================
Welcome to the program that detects cars without the help of special algorithm!.
In addition, the program also attempts to count the number of cars that pass the "line"
Running the program is simple and easy, but it is important to use only MP4 format video files.
If not, we've seen many warnings due to the format (x264), but still the program will work properly and smoothly.

The code supports Python version 2.7 & 3.5
==================================================================================================================
"""

import cv2
import numpy as np
from argparse import ArgumentParser
from math import sqrt, pow
from copy import copy
from random import randrange
from sys import version

""" Default argument values for argument_parser function"""
VIDEO_PATH = r"dataset\1.mp4"
DIRECTION = 0
SPEED = 1

""" Color codes I use them a lot """
WHITE_COLOR = (255, 255, 255)
GREEN_COLOR = (0, 255, 0)
RED_COLOR = (0, 0, 255)

""" CV & Python version by which the code is written """
MY_PYTHON_VERSION = r"2.7.11 (v2.7.11:6d1b6a68f775, Dec  5 2015, 20:32:19) [MSC v.1500 32 bit (Intel)]"
MY_CV_VERSION = r"2.4.11"


class Blob(object):
    """ Class which identifies the vehicles (cars and also motorcycles)
    In addition, with the functions that we eliminate noises and objects that not vehicles! """

    def __init__(self, contour):
        self.contour = contour
        x, y, w, h = cv2.boundingRect(contour)
        self.bounding_rect = cv2.boundingRect(contour)
        self.center_positions = [(int((2 * x + w) / 2.0), int((2 * y + h) / 2.0))]
        self.diagonal_size = sqrt(pow(w, 2) + pow(h, 2))
        self.aspect_ratio = w / float(h)
        self.rect_area = w * h
        self.still_being_tracked = True
        self.current_match_found_or_new_blob = True
        self.num_of_consecutive_frames_without_a_match = 0
        self.color = None
        """ Variable used in predict's function """
        self.predicted_next_position = (0, 0)

    def checking_that_vehicles(self):
        ratio_area = cv2.contourArea(self.contour) / self.rect_area

        if self.rect_area > 400 and 0.2 < self.aspect_ratio < 4.0 and self.bounding_rect[2] > 30 \
                and self.bounding_rect[3] > 30 and self.diagonal_size > 60.0 and ratio_area > 0.5:
            return True
        return False

    """ Predict the next step, as the Kalman algorithm """

    def predict(self):
        num_positions = self.center_positions.__len__()
        if num_positions == 1:
            self.predicted_next_position = (self.center_positions[0][0], self.center_positions[0][1])
        elif num_positions == 2:
            idx = num_positions
            delta_x = (self.center_positions[1][0] - self.center_positions[0][0])
            delta_y = (self.center_positions[1][1] - self.center_positions[0][1])

            self.predicted_next_position = (self.center_positions[idx - 1][0] + delta_x,
                                            self.center_positions[idx - 1][1] + delta_y)

        elif num_positions == 3:
            idx = num_positions
            sum_of_changes = [int((((self.center_positions[2][0] - self.center_positions[1][0]) * 2) +
                                   (self.center_positions[1][0] - self.center_positions[0][0]) * 1) / 3.0),
                              int((((self.center_positions[2][1] - self.center_positions[1][1]) * 2) +
                                   (self.center_positions[1][1] - self.center_positions[0][1]) * 1) / 3.0)]

            delta_x, delta_y = sum_of_changes
            self.predicted_next_position = (self.center_positions[idx - 1][0] + delta_x,
                                            self.center_positions[idx - 1][1] + delta_y)
        elif num_positions == 4:
            idx = num_positions
            sum_of_changes = (int((((self.center_positions[3][0] - self.center_positions[2][0]) * 3) +
                                   ((self.center_positions[2][0] - self.center_positions[1][0]) * 2) +
                                   (self.center_positions[1][0] - self.center_positions[0][0]) * 1) / 6.0),
                              int((((self.center_positions[3][1] - self.center_positions[2][1]) * 3) +
                                   ((self.center_positions[2][1] - self.center_positions[1][1]) * 2) +
                                   (self.center_positions[1][1] - self.center_positions[0][1]) * 1) / 6.0))
            delta_x, delta_y = sum_of_changes
            self.predicted_next_position = (self.center_positions[idx - 1][0] + delta_x,
                                            self.center_positions[idx - 1][1] + delta_y)

        elif num_positions >= 5:
            idx = num_positions
            sum_of_changes = (int(((((self.center_positions[idx - 1][0] - self.center_positions[idx - 2][0]) * 4) +
                                    (self.center_positions[idx - 2][0] - self.center_positions[idx - 3][0]) * 3) +
                                   ((self.center_positions[idx - 3][0] - self.center_positions[idx - 4][0]) * 2) +
                                   (self.center_positions[idx - 4][0] - self.center_positions[idx - 5][0]) * 1) / 10.0),
                              int(((((self.center_positions[idx - 1][1] - self.center_positions[idx - 2][1]) * 4) +
                                    (self.center_positions[idx - 2][1] - self.center_positions[idx - 3][1]) * 3) +
                                   ((self.center_positions[idx - 3][1] - self.center_positions[idx - 4][1]) * 2) +
                                   (self.center_positions[idx - 4][1] - self.center_positions[idx - 5][1]) * 1) / 10.0))

            delta_x, delta_y = sum_of_changes
            self.predicted_next_position = (self.center_positions[idx - 1][0] + delta_x,
                                            self.center_positions[idx - 1][1] + delta_y)

    def update(self, blob):
        self.contour = blob.contour
        self.bounding_rect = blob.bounding_rect
        self.center_positions.append(blob.center_positions[blob.center_positions.__len__() - 1])
        self.diagonal_size = blob.diagonal_size
        self.aspect_ratio = blob.aspect_ratio
        self.still_being_tracked = True
        self.current_match_found_or_new_blob = True

    """ For debug, print all the parameters of the object.
        Created for QA and debug """

    def to_string(self):
        # print("contour - ", self.contour)
        print("bounding_rect - ", self.bounding_rect)
        # print("center_positions - ", self.center_positions)
        print("diagonal_size - ", self.diagonal_size)
        print("aspect_ratio - ", self.aspect_ratio)
        print("rect_area - ", self.rect_area)
        print("still_being_tracked - ", self.still_being_tracked)
        print("current_match_found_or_new_blob - ", self.current_match_found_or_new_blob)
        print("num_of_consecutive_frames_without_a_match - ", self.num_of_consecutive_frames_without_a_match)
        print("predicted_next_position - ", self.predicted_next_position)


def check_version_of_python_and_cv():
    python_version = version
    cv_version = cv2.__version__
    flag_change = 0

    cv_msg = ""
    python_msg = ""
    if cv_version != MY_CV_VERSION and python_version != MY_PYTHON_VERSION:
        cv_msg = "Your ~~CV version~~ ({0}) modified version with which the code is written ({1})" \
            .format(cv_version, MY_CV_VERSION)
        python_msg = "Your ~~Python version~~ ({0}) modified version with which the code is written ({1})" \
            .format(python_version, MY_PYTHON_VERSION)

        flag_change = 1
    elif cv_version != MY_CV_VERSION:
        cv_msg = "Your ~~CV version~~ ({0}) modified version with which the code is written ({1})" \
            .format(cv_version, MY_CV_VERSION)

        flag_change = 1
    elif python_version != MY_PYTHON_VERSION:
        python_msg = "Your ~~Python version~~ ({0}) modified version with which the code is written ({1})" \
            .format(python_version, MY_PYTHON_VERSION)

        flag_change = 1

    """ Checking the version of Python to know what print function to use """
    digit_python_version = python_version.split(".")[0]
    if digit_python_version == "2":
        if flag_change == 1:
            print("Warnings: Please note, could be a program does not work optimally because of"
                  " different versions you use")
        if cv_msg != "":
            print(cv_msg)
        if python_msg != "":
            print(python_msg)
    elif digit_python_version == "3":
        if flag_change == 1:
            print("Warnings: Please note, could be a program does not work optimally because of"
                  " different versions you use")
        if cv_msg != "":
            print(cv_msg)
        if python_msg != "":
            print(python_msg)


def argument_parser():
    parser = ArgumentParser(description='')
    parser.add_argument('-v', '--video_path', type=str, help='The video file path', default=VIDEO_PATH)
    parser.add_argument('-d', '--direction', type=int,
                        help='Directing the traffic of vehicle. 1 marks movement'
                        ' from top to bottom, 0 marks a movement from bottom to top.', default=DIRECTION)
    parser.add_argument('-s', '--speed', type=int, help='The running speed of the frames.'
                        'Number 1 mark a high speed, the Number 0 marks a slow speed.', default=SPEED)
    parser.add_argument('-p', '--line_position', type=int, help='Line location on the photo, to count the vehicles.',
                        required=True)
    args = parser.parse_args()

    return args


def create_and_initialization_of_variables(args):
    color_value = 50
    colors_code_list = []
    """ In total created 75 color codes
        Automatically generated color values for predict vehicles """
    for i in range(1, 6):
        temp = color_value * i
        colors_code_list.append((temp, 0, 0))
        colors_code_list.append((0, temp, 0))
        colors_code_list.append((0, 0, temp))

        colors_code_list.append((temp, 0, temp))
        colors_code_list.append((0, temp, temp))
        colors_code_list.append((temp, temp, 0))

        colors_code_list.append((temp, temp / 2, 0))
        colors_code_list.append((temp, 0, temp / 2))

        colors_code_list.append((temp / 2, temp, 0))
        colors_code_list.append((0, temp, temp / 2))

        colors_code_list.append((temp / 2, 0, temp))
        colors_code_list.append((0, temp / 2, temp))

        colors_code_list.append((temp, temp / 2, temp / 2))
        colors_code_list.append((temp / 2, temp, temp / 2))
        colors_code_list.append((temp / 2, temp / 2, temp))

    capture_video = cv2.VideoCapture(args.video_path)
    if not capture_video.isOpened():
        exit("Video path incorrect Or cann't open this video format!")

    _, img_frame1 = capture_video.read()
    blobs_list = []
    crossing_line = []
    car_count = 0
    horizontal_line_position = int(round(img_frame1.shape[0] * args.line_position / 100, 2))
    crossing_line.append((0, horizontal_line_position))
    crossing_line.append(((img_frame1.shape[1] - 1), horizontal_line_position))

    return colors_code_list, capture_video, img_frame1, blobs_list, crossing_line, car_count, horizontal_line_position


def draw_and_show_contours(image_shape, contours_list, window_name):
    image = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(image, contours_list, -1, WHITE_COLOR, -1)
    cv2.imshow(window_name, image)


def current_frame_blobs_existing_in_all_blobs(all_blobs_list, current_frame_blobs_list):
    for idx, blob in enumerate(all_blobs_list):
        all_blobs_list[idx].current_match_found_or_new_blob = False
        all_blobs_list[idx].predict()

    for frame_blob in current_frame_blobs_list:
        index_of_least_distance = 0
        least_distance = 10000000000
        for i in range(all_blobs_list.__len__()):
            if all_blobs_list[i].still_being_tracked is True:
                idx = frame_blob.center_positions.__len__() - 1
                distance = sqrt(pow(int(frame_blob.center_positions[idx][0] -
                                        all_blobs_list[i].predicted_next_position[0]), 2) +
                                pow(int(frame_blob.center_positions[idx][1] -
                                        all_blobs_list[i].predicted_next_position[1]), 2))
                if distance < least_distance:
                    least_distance = distance
                    index_of_least_distance = i
        if least_distance < frame_blob.diagonal_size * 0.5:
            all_blobs_list[index_of_least_distance].update(frame_blob)
        else:
            frame_blob.current_match_found_or_new_blob = True
            all_blobs_list.append(frame_blob)

    for idx, blob in enumerate(all_blobs_list):
        if blob.current_match_found_or_new_blob is False:
            all_blobs_list[idx].num_of_consecutive_frames_without_a_match += 1

        if blob.num_of_consecutive_frames_without_a_match >= 5:
            all_blobs_list[idx].still_being_tracked = False

    return all_blobs_list, current_frame_blobs_list


def draw_blob_info_on_image(all_blobs_list, frame, colors_code_list):
    for idx in range(all_blobs_list.__len__()):
        if all_blobs_list[idx].still_being_tracked is True:
            if all_blobs_list[idx].color is None:
                color_index = randrange(0, colors_code_list.__len__() - 1)
                all_blobs_list[idx].color = colors_code_list[color_index]
            x, y, w, h = all_blobs_list[idx].bounding_rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), all_blobs_list[idx].color, 2)
            font_face = cv2.FONT_HERSHEY_SIMPLEX

            cv2.putText(frame, str(idx), all_blobs_list[idx].center_positions[
                all_blobs_list[idx].center_positions.__len__() - 1], font_face, 1, GREEN_COLOR, 2)


def check_blob_crossed(all_blob_list, horizontal_line_position, cars_count, direction):
    least_one_blob_crossed_the_line = False

    for blob in all_blob_list:
        index = blob.center_positions.__len__()
        if blob.still_being_tracked is True and index >= 2:
            prev_frame_index = index - 2
            current_frame_index = index - 1

            if blob.center_positions[prev_frame_index][1] > horizontal_line_position >= \
                    blob.center_positions[current_frame_index][1] and direction == 0:
                cars_count += 1
                least_one_blob_crossed_the_line = True
            elif blob.center_positions[prev_frame_index][1] < horizontal_line_position <= \
                    blob.center_positions[current_frame_index][1] and direction == 1:
                cars_count += 1
                least_one_blob_crossed_the_line = True

    return least_one_blob_crossed_the_line, cars_count


def draw_cars_count_on_image(cars_count, frame):
    font_face = cv2.FONT_HERSHEY_SIMPLEX

    bottom_left_text_position = (frame.shape[1] - 1 - 84, 30)
    cv2.putText(frame, str(cars_count), bottom_left_text_position, font_face, 1, RED_COLOR, 2)


def main():
    """ Print the doc on this project """
    print(__doc__)

    """ Get arguments from command-line"""
    args = argument_parser()

    if args.direction != 1 and args.direction != 0:
        exit("Direction error: Entered value does not match the requirements")
    dir = args.direction
    if args.speed != 1 and args.speed != 0:
        exit("Speed error: Entered value does not match the requirements")
    speed = args.speed
    if speed == 0: speed = 50

    """ Checking some versions of Python and CV you use """
    check_version_of_python_and_cv()

    """ Creates all the variables necessary for running the program """
    colors_code_list, capture_video, img_frame1, blobs_list, crossing_line, car_count, horizontal_line_position = \
        create_and_initialization_of_variables(args)

    first_loop_flag = True

    digit_cv_version = cv2.__version__.split(".")[0]
    while capture_video.isOpened():
        ret, img_frame2 = capture_video.read()
        if ret is False:
            print("End of video")
            break

        current_frame_blobs_list = []
        img_frame1_copy = copy(img_frame1)
        img_frame2_copy = copy(img_frame2)

        img_frame1_copy = cv2.cvtColor(img_frame1_copy, cv2.COLOR_BGR2GRAY)
        img_frame2_copy = cv2.cvtColor(img_frame2_copy, cv2.COLOR_BGR2GRAY)
        img_frame1_copy = cv2.GaussianBlur(img_frame1_copy, (5, 5), 0)
        img_frame2_copy = cv2.GaussianBlur(img_frame2_copy, (5, 5), 0, )
        img_difference = cv2.absdiff(img_frame1_copy, img_frame2_copy)

        _, img_thresh = cv2.threshold(img_difference, 30, 255, cv2.THRESH_BINARY)
        cv2.imshow("Threshold", img_thresh)

        structuring_elements5x5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5), (-1, -1))

        for i in range(2):
            img_thresh = cv2.dilate(img_thresh, structuring_elements5x5, None, (-1, -1), 1, cv2.BORDER_DEFAULT,
                                    (0, 0, 0))
            img_thresh = cv2.dilate(img_thresh, structuring_elements5x5, None, (-1, -1), 1, cv2.BORDER_DEFAULT,
                                    (0, 0, 0))
            img_thresh = cv2.erode(img_thresh, structuring_elements5x5, None, (-1, -1), 1, cv2.BORDER_DEFAULT,
                                   (0, 0, 0))

        img_thresh_copy = copy(img_thresh)
        if digit_cv_version == "3":
            _, contours, _ = cv2.findContours(img_thresh_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        elif digit_cv_version == "2":
            contours, _ = cv2.findContours(img_thresh_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        draw_and_show_contours(img_thresh.shape, contours, "Contours")

        convex_hulls_list = [cv2.convexHull(contours[i]) for i in range(contours.__len__())]
        draw_and_show_contours(img_thresh.shape, convex_hulls_list, "Convex Hulls")

        for i in range(convex_hulls_list.__len__()):
            blob = Blob(convex_hulls_list[i])
            if blob.checking_that_vehicles() is True:
                current_frame_blobs_list.append(blob)

        blobs_contours_list = [blob.contour for blob in current_frame_blobs_list if blob.still_being_tracked is True]
        draw_and_show_contours(img_thresh.shape, blobs_contours_list, "Current Frame Blobs")

        if first_loop_flag is True:
            blobs_list = list(current_frame_blobs_list)
            first_loop_flag = False
        else:
            blobs, current_frame_blobs_list = current_frame_blobs_existing_in_all_blobs(blobs_list,
                                                                                        current_frame_blobs_list)

        blobs_contours_list = [blob.contour for blob in blobs_list if blob.still_being_tracked is True]
        draw_and_show_contours(img_thresh.shape, blobs_contours_list, "All Blobs")

        img_frame2_copy = copy(img_frame2)

        draw_blob_info_on_image(blobs_list, img_frame2_copy, colors_code_list)

        least_one_blob_crosses_the_line, car_count = \
            check_blob_crossed(blobs_list, horizontal_line_position, car_count, dir)

        if least_one_blob_crosses_the_line is True:
            cv2.line(img_frame2_copy, crossing_line[0], crossing_line[1], GREEN_COLOR, 2)
        else:
            cv2.line(img_frame2_copy, crossing_line[0], crossing_line[1], RED_COLOR, 2)

        draw_cars_count_on_image(car_count, img_frame2_copy)
        cv2.imshow("Real Frame", img_frame2_copy)
        img_frame1 = copy(img_frame2)

        del current_frame_blobs_list
        """ Do not delete this loop !!!
            It is important to run the capture of the frame """
        k = 0xFF & cv2.waitKey(speed)
        if k == ord('p') or k == ord('P'):
            while 1:
                k = 0xFF & cv2.waitKey(1)

                if k == ord('p') or k == ord('P'):
                    break
                if k == 27 or k == ord('q') or k == ord('Q'):
                    capture_video.release()
                    cv2.destroyAllWindows()
                    exit("End of program!")
        if k == 27 or k == ord('q') or k == ord('Q'):
            capture_video.release()
            cv2.destroyAllWindows()
            exit("End of program!")

    print(car_count)
    capture_video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
