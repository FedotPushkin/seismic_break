from nn import Unet_NN
from loading import load_db
from utils import gpu_check, parser
from visualisation import show_predicted_images


if __name__ == '__main__':
    '''
    Please view utils.py for parameters values
    '''
    gpu_check()
    args = parser()
    try:
        dataset, val_dataset, train_samples_n, test_samples_n = load_db(args)

    except Exception as e:
        print(f"Error loading images: {e}")
        raise

    Unet = Unet_NN(input_shape=(args.train_shape[0], args.train_shape[1], 1))
    if args.fit:
        Unet.fit_to_data(dataset, val_dataset,
                         batch_size=args.batch_size,
                         epochs=args.epochs,
                         show_perf=True,
                         train_samples=train_samples_n,
                         test_samples=test_samples_n)
    if args.test:
        y_test, y_pred = Unet.make_predictions(val_dataset, args.batch_size, test_samples_n)
        show_predicted_images(y_test, y_pred)






