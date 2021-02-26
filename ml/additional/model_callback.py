
import tensorflow as tf


class SaveInformationCallback(tf.keras.callbacks.Callback):
    db_connection = None

    def __init__(self, db_connection):
        self.db_connection = db_connection

    """
    def on_train_batch_end(self, batch, logs=None):
        print('For batch {}, loss is {:7.2f}.'.format(batch, logs['loss']))

    def on_test_batch_end(self, batch, logs=None):
        print('For batch {}, loss is {:7.2f}.'.format(batch, logs['loss']))
    """

    def on_epoch_end(self, epoch, logs=None):
        history = self.db_connection.get_history()
        history.append({
            "val_loss": float(logs["val_loss"]),
            "loss": float(logs["loss"]),
            "lr": float(logs["lr"]),
        })
        self.db_connection.set_history(history)
        self.db_connection.save_model()

        # print('The average loss for epoch {} is {:7.2f} and mean absolute error is {:7.2f}.'.format(epoch, logs[
        #    'loss'], logs['mae']))

