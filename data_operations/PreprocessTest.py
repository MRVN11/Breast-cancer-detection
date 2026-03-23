import cv2

def preprocess_images():
    image = cv2.imread("../data/MIAS_data/MIAS_images_ori/mdb001.pgm")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.fastNlMeansDenoising(gray, None, 10, 10, 21)
    image = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))
    image = image.astype("float32") / 255.0


    cv2.imshow("image", image)
    cv2.waitKey(0)
preprocess_images()

"""old version incase of problems"""
# def preprocess_images(image_path: str) -> np.ndarray:
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     gray = clahe.apply(gray)
#     image = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, (512, 512))
#     image = image.astype("float32") / 255.0
#     # image = preprocess_input(image)
#     return image


# edges = cv2.Canny(gray, 30, 200)
# contour, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# image = cv2.drawContours(image, contour, -1, (0, 0, 255), 2)