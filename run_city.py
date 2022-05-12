from argparse import ArgumentParser
from glob import glob
import os

from utils.cityscapesSequence import CitySequence
from modelclass import Model
import models

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--path", default="../cityscapes/",
                        help="Path to where are the gtFine or gtCoarse, and leftImg8bit folders.")
    parser.add_argument("--model_name", default="city_best_model-1.hdf5",
                        help="Model name")
    parser.add_argument("--gtFine", type=int, default=1,
                        help="gtFine? True = 1, False = 0 (default=1)")
    parser.add_argument("--alpha", type=float, default=1.,
                        help="Alpha to MobileNetv2 filters.")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image Size.(Default=256)")
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch Size. (Default=4)')
    parser.add_argument('--lr', type=float, default=7e-6,
                        help='Learning rate. (Default=7e-6)')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Epochs. (Default=15)')
    parser.add_argument('--gpu', type=bool, default=False,
                        help='Turn off/on GPU. (Default=False)')
    args = parser.parse_args()

    if not args.gpu:
        # Turn off GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    gtFine = args.gtFine
    alpha = args.alpha
    batch_size = args.batch_size
    image_size = args.image_size
    lr = args.lr
    epochs = args.epochs
    num_classes = 19

    GENERAL_PATH = os.path.join("model-saved")

    MODEL_NAME = args.model_name
    MODEL_PATH = os.path.join(GENERAL_PATH, MODEL_NAME)

    DATA_DIR = args.path

    if gtFine:
        DATA_TYPE = "gtFine"
        TYPE = "train"
    else:
        DATA_TYPE = "gtCoarse"
        TYPE = "train_extra"

    x_train_dir = sorted(glob(os.path.join(DATA_DIR, DATA_TYPE, "leftImg8bit",
                                           TYPE, "*/*_leftImg8bit.png")))
    y_train_dir = sorted(glob(os.path.join(DATA_DIR, DATA_TYPE, DATA_TYPE,
                                           TYPE, "*/*_labelTrainIds.png")))

    TYPE = "val"
    x_val_dir = sorted(glob(os.path.join(DATA_DIR, DATA_TYPE, "leftImg8bit",
                                         TYPE, "*/*_leftImg8bit.png")))
    y_val_dir = sorted(glob(os.path.join(DATA_DIR, DATA_TYPE, DATA_TYPE,
                                         TYPE, "*/*_labelTrainIds.png")))

    x_train = CitySequence(x_dir=x_train_dir, y_dir=y_train_dir, batch_size=batch_size,
                           blur=3, image_size=image_size, horizontal_flip=True,
                           vertical_flip=False, brightness=0.3, contrast=0.3, crop=True)
    x_val = CitySequence(x_dir=x_val_dir, y_dir=y_val_dir, batch_size=batch_size, blur=0,
                         image_size=image_size, horizontal_flip=True, vertical_flip=False,
                         brightness=0, contrast=0, crop=False)

    print(gtFine)
    print(len(x_train_dir), len(y_train_dir), x_train)
    print(len(x_val_dir), len(y_val_dir), x_val)

    # GENERATE MODEL
    model = Model(image_size=image_size, num_classes=num_classes, alpha=alpha,
                  generate_model=models.generate_model, path=MODEL_PATH)

    model.compile(learning_rate=lr)
    model.summary()

    model.train(x_train=x_train, x_val=x_val, batch_size=batch_size, epochs=epochs,
                SAVE_PATH=MODEL_PATH, monitor="val_loss")
