import cv2
import numpy as np



def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]


def word_segmentation(image_path, filename):
    raw_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    scale_percent = 100  # percent of original size
    width = int(raw_img.shape[1] * scale_percent / 100)
    height = int(raw_img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(raw_img, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow('Document image', resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    resize_original_image = resized.copy()

    ret, thresh = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('second', thresh)
    cv2.waitKey(0)
    # adjust the dilation in y and x axis
    kernel = np.ones((4, 18), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    cv2.imshow('dilated', img_dilation)
    cv2.waitKey(0)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img_dilation)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # find contours
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * resize_original_image.shape[1])
    # sorted_ctrs = sorted(ctrs,key=lambda ctr: get_contour_precedence(ctr, resize_original_image.shape[1]))
    minArea = 450

    # for i in range(len(sorted_ctrs)):
    #     img = cv2.putText(resize_original_image, str(i), cv2.boundingRect(sorted_ctrs[i])[:2], cv2.FONT_HERSHEY_COMPLEX, 1, [125])

    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box based of min area
        if cv2.contourArea(ctr) < minArea:
            continue
        x, y, w, h = cv2.boundingRect(ctr)

        # Getting ROI
        roi = resize_original_image[:y + h, x:x + w]

        # show ROI
        # cv2.imshow('segment no:'+str(i),roi)
        # Draw rectangle
        crop_image_word = resize_original_image[y:y + h, x:x + w]
        # increase contrast
        # pxmin = np.min(crop_image_word)
        # pxmax = np.max(crop_image_word)
        # imgContrast = (crop_image_word - pxmin) / (pxmax - pxmin) * 255
        #
        # # increase line width
        # kernel = np.ones((1, 1), np.uint8)
        # imgMorph = cv2.erode(imgContrast, kernel, iterations=1)

        # cv2.imshow('Each word',crop_image_word)
        cv2.imwrite('../output_words/' + filename + '/%d.png' % i, crop_image_word)  # save word
        # cv2.waitKey(0)
        cv2.destroyAllWindows()

        # cv2.imshow('loop show',resize_original_image)
        # write number on it
        cv2.putText(resize_original_image, str(i), cv2.boundingRect(sorted_ctrs[i])[:2], cv2.FONT_HERSHEY_COMPLEX, 1,
                          [125])

        cv2.rectangle(resize_original_image, (x, y), (x + w, y + h), (0, 0, 0), 3)

    scale_percent = 100  # percent of original size
    width = int(resize_original_image.shape[1] * scale_percent / 100)
    height = int(resize_original_image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    final_resize = cv2.resize(resize_original_image, dim, interpolation=cv2.INTER_AREA)

    cv2.namedWindow('Mark', cv2.WINDOW_NORMAL)
    cv2.imshow('Mark', final_resize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
