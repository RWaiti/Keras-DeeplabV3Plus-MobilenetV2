import cv2
from numpy import argmax, float32, uint8, expand_dims, asarray, stack
from tensorflow.lite.python.interpreter import Interpreter
from PIL.Image import fromarray
from seaborn import color_palette
from time import time
from argparse import ArgumentParser
from os import environ

environ["CUDA_VISIBLE_DEVICES"] = "-1"


def preprocessing(frame):
    frame = cv2.resize(frame, (input_w, input_h), interpolation=cv2.INTER_AREA).astype(float32)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = expand_dims(frame, axis=0)
    return frame / 127.5 - 1

def inference(pre_frame):
    interpreter.set_tensor(input, pre_frame)
    interpreter.invoke()
    return interpreter.get_tensor(output)[0].astype(uint8)

def postprocessing(mask):
    # colorList = color_palette(None, 2)
    # colorListAux = []

    # for i in colorList:
    #     colorListAux.append(int(i[0] * 255))
    #     colorListAux.append(int(i[1] * 255))
    #     colorListAux.append(int(i[2] * 255))
    # colorList = None

    # print(colorListAux)

    mask = cv2.resize(mask, (camera_width, camera_height), interpolation=cv2.INTER_NEAREST)
    mask = stack([mask] * 3, axis=-1)
    mask[mask == 1] = 255
    # mask = fromarray(mask, mode="P")
    # mask.putpalette(colorListAux)
    # mask = mask.convert("RGB")
    return mask
    # return cv2.addWeighted(frame, .75, asarray(mask), .25, 0)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--path", default="1-keras-mblnetv2-deeplabv3.tflite", help="path to model")
    parser.add_argument('--camera_width', type=int, default=640,
                        help='USB Camera resolution (width). (Default=640)')
    parser.add_argument('--camera_height', type=int, default=480,
                        help='USB Camera resolution (height). (Default=480)')
    parser.add_argument('--cam_fps', type=int, default=15,
                        help='FPS (Default=15)')
    parser.add_argument("--thread", type=int, default=4,
                        help="Number of Threads")
    args = parser.parse_args()

    cam_fps = args.cam_fps
    camera_width = args.camera_width
    camera_height = args.camera_height
    interpreter = Interpreter(model_path=args.path, num_threads=args.thread)
    interpreter.allocate_tensors()

    output = interpreter.get_output_details()[0]
    input = interpreter.get_input_details()[0]

    input_w, input_h = input["shape"][1:3]

    output = output['index']
    input = input['index']

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, cam_fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

    fps_count = 0

    while cap.isOpened():
        frame_time = time()

        _, frame = cap.read()

        pre_frame = preprocessing(frame)
        mask = inference(pre_frame)
        # image = postprocessing(frame, mask)

        fps = str(1 / (time() - frame_time))

        if fps_count >= 2:
            print(fps)
            # cv2.putText(image, fps, (5, 16),
            #             cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255), 3, cv2.LINE_AA)
        else:
            fps_count += 1

        # cv2.imshow("segmentation", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
