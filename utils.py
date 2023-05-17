import cv2


def rotateImg(image, angle, center=None, scale=1.0, interp=cv2.INTER_LINEAR):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h), flags=interp)
    return rotated