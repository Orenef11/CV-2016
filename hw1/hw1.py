#!/usr/bin/env python
'''
===============================================================================
Interactive Image Segmentation using GrabCut algorithm.
This sample shows interactive image segmentation using grabcut algorithm.
USAGE:
    python grabcut.py <filename>
README FIRST:
    Two windows will show up, one for input and one for output.
    At first, in input window, draw a rectangle around the object using
mouse right button. Then press 'n' to segment the object (once or a few times)
For any finer touch-ups, you can press any of the keys below and draw lines on
the areas you want. Then again press 'n' for updating the output.
Key '0' - To select areas of sure background
Key '1' - To select areas of sure foreground
Key '2' - To select areas of probable background
Key '3' - To select areas of probable foreground
Key 'n' - To update the segmentation
Key 'r' - To reset the setup
Key 's' - To save the results
===============================================================================
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import sys
import time

RED = [0, 0, 255]  # PR BG
GREEN = [0, 255, 0]  # PR FG
BLACK = [0, 0, 0]  # sure BG
WHITE = [255, 255, 255]  # sure FG

DRAW_BLACK = {'color': BLACK, 'val': 4}
DRAW_WHITE = {'color': WHITE, 'val': 1}
DRAW_GREEN = {'color': GREEN, 'val': 3}
DRAW_RED = {'color': RED, 'val': 2}

# setting up flags
drawing = False  # flag for drawing curves
rectangle = False  # flag for drawing rect
rect_over = False  # flag to check if rect drawn
rect_or_mask = 100  # flag for selecting rect or mask mode
value = DRAW_BLACK  # drawing initialized to FG
thickness = 3  # brush thickness


def darw_seg_line_on_image(red_value=0, green_value=0, black_value=0, white_value=0, x_from_onmouse=None,
                           y_from_obmouse=None):
    global mask_red, mask_green, mask_black, mask_white

    cv2.circle(mask_red, (x_from_onmouse, y_from_obmouse), thickness, red_value, -1)
    cv2.circle(mask_green, (x_from_onmouse, y_from_obmouse), thickness, green_value, -1)
    cv2.circle(mask_black, (x_from_onmouse, y_from_obmouse), thickness, black_value, -1)
    cv2.circle(mask_white, (x_from_onmouse, y_from_obmouse), thickness, white_value, -1)


def initial_vriables(filename):
    global input_img, out_img, orginal_img, mask_black, mask_white, mask_green, mask_red

    input_img = cv2.imread(filename)
    out_img = input_img.copy()
    orginal_img = input_img.copy()
    out_img = np.zeros(orginal_img.shape, np.uint8)
    mask_black = np.zeros(orginal_img.shape[:2], dtype=np.uint8)
    mask_white = np.zeros(orginal_img.shape[:2], dtype=np.uint8)
    mask_green = np.zeros(orginal_img.shape[:2], dtype=np.uint8)
    mask_red = np.zeros(orginal_img.shape[:2], dtype=np.uint8)

    # init grabcut
    rect = (0, 1, input_img.shape[1], input_img.shape[0])
    bgdmodel = np.zeros((1, 65), np.float64);
    fgdmodel = np.zeros((1, 65), np.float64)
    cv2.grabCut(orginal_img, mask_black, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
    cv2.grabCut(orginal_img, mask_white, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
    cv2.grabCut(orginal_img, mask_green, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)

    cv2.grabCut(orginal_img, mask_red, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)


def onmouse(event, x, y, flags, param):
    global input_img, value, mask_blue, mask_red, mask_green, mask_black, ix, iy, drawing

    # draw touchup curves
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(input_img, (x, y), thickness, value['color'], -1)
        if value == DRAW_WHITE:
            darw_seg_line_on_image(white_value=1, x_from_onmouse=x, y_from_obmouse=y)
        elif value == DRAW_GREEN:
            darw_seg_line_on_image(green_value=1, x_from_onmouse=x, y_from_obmouse=y)
        elif value == DRAW_BLACK:
            darw_seg_line_on_image(red_value=1, x_from_onmouse=x, y_from_obmouse=y)
        else:
            darw_seg_line_on_image(black_value=1, x_from_onmouse=x, y_from_obmouse=y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(input_img, (x, y), thickness, value['color'], -1)
            if value == DRAW_WHITE:
                darw_seg_line_on_image(white_value=1, x_from_onmouse=x, y_from_obmouse=y)
            elif value == DRAW_GREEN:
                darw_seg_line_on_image(green_value=1, x_from_onmouse=x, y_from_obmouse=y)
            elif value == DRAW_BLACK:
                darw_seg_line_on_image(red_value=1, x_from_onmouse=x, y_from_obmouse=y)
            else:
                darw_seg_line_on_image(black_value=1, x_from_onmouse=x, y_from_obmouse=y)
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing == True:
            drawing = False
            cv2.circle(input_img, (x, y), thickness, value['color'], -1)
            if value == DRAW_WHITE:
                darw_seg_line_on_image(white_value=1, x_from_onmouse=x, y_from_obmouse=y)
            elif value == DRAW_GREEN:
                darw_seg_line_on_image(green_value=1, x_from_onmouse=x, y_from_obmouse=y)
            elif value == DRAW_BLACK:
                darw_seg_line_on_image(red_value=1, x_from_onmouse=x, y_from_obmouse=y)
            else:
                darw_seg_line_on_image(black_value=1, x_from_onmouse=x, y_from_obmouse=y)


if __name__ == '__main__':

    # print documentation
    print(__doc__)

    # Loading images
    if len(sys.argv) == 2:
        filename = sys.argv[1]  # for drawing purposes
    else:
        print("No input image given!!!")
        print("Correct Usage: python grabcut.py <filename> \n")
        filename = 'man.jpg'

    input_img = out_img = orginal_img = mask_black = mask_white = mask_green = mask_red = \
        draw_black_seg = draw_white_seg = draw_green_seg = draw_red_seg = 0

    initial_vriables(filename)

    # input and output windows
    cv2.namedWindow('output')
    cv2.namedWindow('input')
    cv2.setMouseCallback('input', onmouse)

    print(" Instructions: \n")
    print(" Draw a rectangle around the object using right mouse button \n")

    run_loop_flag = True
    first_division_flag = True
    while run_loop_flag:
        cv2.imshow('output', out_img)
        cv2.imshow('input', input_img)
        k = 0xFF & cv2.waitKey(1)

        # key bindings
        if k == ord('q') or k == ord('Q'):  # esc to exit
            run_loop_flag = False
        elif k == ord('0'):  # BG drawing
            print("Black color selected")
            value = DRAW_BLACK
        elif k == ord('1'):  # FG drawing
            print("White color selected")
            value = DRAW_WHITE
        elif k == ord('2'):  # PR_BG drawing
            print("Green color selected")
            value = DRAW_GREEN
        elif k == ord('3'):  # PR_FG drawing
            print("Red color selected")
            value = DRAW_RED
        elif k == ord('s') or k == ord('S'):  # save image
            bar = np.zeros((input_img.shape[0], 5, 3), np.uint8)
            res = np.hstack((input_img, bar, out_img, bar, orginal_img))
            cv2.imwrite('grabcut_output.png', res)
            print("Result saved as image \n")
        elif k == ord('r') or k == ord('R'):  # reset everything
            start = time.clock()
            print("Initialization images, it takes a few seconds")
            input_img = orginal_img.copy()
            initial_vriables(filename)
            value = DRAW_BLACK
            print("End initialization images, it's take ", time.clock() - start, "sec")
        elif k == ord('n') or k == ord('N'):  # segment the image
            start = time.clock()
            print("Segmentation process begins, it takes a few seconds")
            bgdmodel = np.zeros((1, 65), np.float64)
            fgdmodel = np.zeros((1, 65), np.float64)
            cv2.grabCut(orginal_img, mask_black, None, bgdmodel, fgdmodel, thickness, cv2.GC_INIT_WITH_MASK)
            cv2.grabCut(orginal_img, mask_white, None, bgdmodel, fgdmodel, thickness, cv2.GC_INIT_WITH_MASK)
            cv2.grabCut(orginal_img, mask_green, None, bgdmodel, fgdmodel, thickness, cv2.GC_INIT_WITH_MASK)
            cv2.grabCut(orginal_img, mask_red, None, bgdmodel, fgdmodel, thickness, cv2.GC_INIT_WITH_MASK)

            first_seq_part = np.where((mask_black == cv2.GC_FGD) + (mask_black == cv2.GC_PR_FGD), 255, 0).astype(
                'uint8')
            second_mask_red = np.where((mask_white == cv2.GC_FGD) + (mask_white == cv2.GC_PR_FGD), 255, 0).astype(
                'uint8')
            third_mask_green = np.where((mask_green == cv2.GC_FGD) + (mask_green == cv2.GC_PR_FGD), 255, 0).astype(
                'uint8')
            fourth_mask_black = np.where((mask_red == cv2.GC_FGD) + (mask_red == cv2.GC_PR_FGD), 255, 0).astype('uint8')

            out_img[(first_seq_part != 0)] = cv2.mean(orginal_img, first_seq_part)[:3]
            out_img[second_mask_red != 0] = cv2.mean(orginal_img, second_mask_red)[:3]
            out_img[(third_mask_green != 0)] = cv2.mean(orginal_img, third_mask_green)[:3]
            out_img[(fourth_mask_black != 0)] = cv2.mean(orginal_img, fourth_mask_black)[:3]

            print("End segmentation process, it's take ", time.clock() - start, "sec")

    cv2.destroyAllWindows()
    print("End the program!")
