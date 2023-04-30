import cv2
import os


# モザイク処理部
def mosaic(img, scale=0.1):
    h, w = img.shape[:2]  # 画像の大きさ
    # 画像を scale (0 < scale <= 1) 倍に縮小
    dst = cv2.resize(
        img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    # 元の大きさに拡大
    dst = cv2.resize(dst, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
    return dst

# カスケード検出器を作成する
cascade = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))


# 画像を読み込む
photo_list = os.listdir("./input_photo")

for photo in photo_list:
    print(photo)
    img = cv2.imread(f"./input_photo/{photo}")

    # 顔検出する
    faces = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
    for x, y, w, h in faces:
        # roi を抽出する
        roi = img[y : y + h, x : x + w]
        # モザイク処理を行い、結果を roi に代入する
        roi[:] = mosaic(roi)

    # モザイク付き画像を出力
    cv2.imwrite(f"./output_photo/{photo}", img)