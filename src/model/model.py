import os

import tensorflow as tf
from util.accuracy import estimated_accuracy
from util.files import load_data
from util.preprocessing import process_data, split_data
from util.statistics import plot_multi_app_consumption, plot_multi_app_consumption_individual


class NilmCNN:
    def __init__(self, args):
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        self.learning_rate = args['lr']
        self.dropout = args['dropout']
        self.validation_split = args['validation_split']
        self.evaluate_split = args['evaluate_split']
        self.batch_size = args['batch_size']
        self.l2 = args['l2']
        self.epochs = args['epochs']
        self.input_size = args['input_size']
        self.features = args['features']
        self.output_size = args['output_size']
        self.bias = args['bias']
        self.save = args['save']
        self.shuffle = args['shuffle']
        self.input_path = args['input_path']
        self.log_path = args['log_path']

    def create_model(self):
        inputs = tf.keras.Input(shape=(self.input_size, self.features), batch_size=None)

        initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.001)

        output = tf.keras.layers.Dense(2048, use_bias=self.bias, kernel_initializer=initializer)(inputs)
        output = tf.keras.layers.Dropout(0.5)(output)

        skip_connections = [output]

        # 1st Convolutional Layer
        signal_out = tf.keras.layers.Conv1D(1024, 3, activation='relu', dilation_rate=2 ** 1, padding='causal', use_bias=self.bias,
                                            kernel_initializer=initializer)(output)
        output = tf.keras.layers.Dropout(self.dropout)(signal_out)
        skip_connections.append(output)

        # 2nd Convolutional Layer
        signal_out = tf.keras.layers.Conv1D(512, 3, activation='relu', dilation_rate=2 ** 2, padding='causal', use_bias=self.bias,
                                            kernel_initializer=initializer)(output)
        output = tf.keras.layers.Dropout(self.dropout)(signal_out)
        skip_connections.append(output)

        # 3rd Convolutional Layer
        signal_out = tf.keras.layers.Conv1D(512, 3, activation='relu', dilation_rate=2 ** 3, padding='causal', use_bias=self.bias,
                                            kernel_initializer=initializer)(output)
        output = tf.keras.layers.Dropout(self.dropout)(signal_out)
        skip_connections.append(output)

        # 4th Convolutional Layer
        signal_out = tf.keras.layers.Conv1D(256, 3, activation='relu', dilation_rate=2 ** 4, padding='causal', use_bias=self.bias,
                                            kernel_initializer=initializer)(output)
        output = tf.keras.layers.Dropout(self.dropout)(signal_out)
        skip_connections.append(output)

        # 5th Convolutional Layer
        signal_out = tf.keras.layers.Conv1D(256, 3, activation='relu', dilation_rate=2 ** 5, padding='causal', use_bias=self.bias,
                                            kernel_initializer=initializer)(output)
        output = tf.keras.layers.Dropout(self.dropout)(signal_out)
        skip_connections.append(output)

        # 6th Convolutional Layer
        signal_out = tf.keras.layers.Conv1D(256, 3, activation='relu', dilation_rate=2 ** 6, padding='causal', use_bias=self.bias,
                                            kernel_initializer=initializer)(output)
        output = tf.keras.layers.Dropout(self.dropout)(signal_out)
        skip_connections.append(output)

        # 7th Convolutional Layer
        signal_out = tf.keras.layers.Conv1D(128, 3, activation='relu', dilation_rate=2 ** 7, padding='causal', use_bias=self.bias,
                                            kernel_initializer=initializer)(output)
        output = tf.keras.layers.Dropout(self.dropout)(signal_out)
        skip_connections.append(output)

        # 8th Convolutional Layer
        signal_out = tf.keras.layers.Conv1D(128, 3, activation='relu', dilation_rate=2 ** 8, padding='causal', use_bias=self.bias,
                                            kernel_initializer=initializer)(output)
        output = tf.keras.layers.Dropout(self.dropout)(signal_out)
        skip_connections.append(output)

        # 9th Convolutional Layer
        signal_out = tf.keras.layers.Conv1D(64, 3, activation='relu', dilation_rate=2 ** 9, padding='causal', use_bias=self.bias,
                                            kernel_initializer=initializer)(output)
        output = tf.keras.layers.Dropout(self.dropout)(signal_out)
        skip_connections.append(output)

        # 10th Convolutional Layer
        signal_out = tf.keras.layers.Conv1D(64, 3, activation='relu', dilation_rate=2 ** 10, padding='causal', use_bias=self.bias,
                                            kernel_initializer=initializer)(output)
        output = tf.keras.layers.Dropout(self.dropout)(signal_out)
        skip_connections.append(output)

        # 11th Convolutional Layer
        signal_out = tf.keras.layers.Conv1D(32, 3, activation='relu', dilation_rate=2 ** 11, padding='causal', use_bias=self.bias,
                                            kernel_initializer=initializer)(output)
        output = tf.keras.layers.Dropout(self.dropout)(signal_out)
        skip_connections.append(output)

        # 12th Convolutional Layer
        signal_out = tf.keras.layers.Conv1D(32, 3, activation='relu', dilation_rate=2 ** 12, padding='causal', use_bias=self.bias,
                                            kernel_initializer=initializer)(output)
        output = tf.keras.layers.Dropout(self.dropout)(signal_out)
        skip_connections.append(output)

        # 13th Convolutional Layer
        signal_out = tf.keras.layers.Conv1D(16, 3, activation='relu', dilation_rate=2 ** 13, padding='causal', use_bias=self.bias,
                                            kernel_initializer=initializer)(output)
        output = tf.keras.layers.Dropout(self.dropout)(signal_out)
        skip_connections.append(output)

        # 14th Convolutional Layer
        signal_out = tf.keras.layers.Conv1D(8, 3, activation='relu', dilation_rate=2 ** 14, padding='causal', use_bias=self.bias,
                                            kernel_initializer=initializer)(output)
        output = tf.keras.layers.Dropout(self.dropout)(signal_out)
        skip_connections.append(output)

        output = tf.keras.layers.Concatenate()(skip_connections)
        output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.output_size, activation='linear'))(output)
        output = tf.keras.layers.ReLU()(output)

        opt = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            amsgrad=True,
            epsilon=1e-6
        )

        model = tf.keras.Model(inputs, output)

        model.compile(optimizer=opt, loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError(), estimated_accuracy])
        model.summary()

        return model

    def train(self):
        callbacks = []

        # Load data from input file
        arr = load_data(self.input_path)

        # Separate aggregate and appliances data
        aggregate, appliances = process_data(arr, self.shuffle)

        # Split data for training and evaluation
        x_train, y_train, x_eval, y_eval = split_data(aggregate, appliances, self.validation_split, self.evaluate_split)

        # Compile model
        model = self.create_model()

        # Declare tensorboard callback
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_path + '/' + os.path.basename(self.input_path)[:-4], histogram_freq=1,
                                                              write_images=True)

        callbacks.append(tensorboard_callback)

        # Train model
        model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, use_multiprocessing=True, validation_split=self.validation_split,
                  callbacks=callbacks, shuffle=True)

        print('\nEvaluate:\n')

        # Evalute model performance
        model.evaluate(x_eval, y_eval)

        eval_predict = model.predict(x_eval)

        plot_multi_app_consumption(y_eval.reshape((y_eval.shape[0]) * y_eval.shape[1], y_eval.shape[2]),
                                   eval_predict.reshape((y_eval.shape[0]) * y_eval.shape[1], y_eval.shape[2]),
                                   os.getenv('HOME') + '/Documents/' + os.path.basename(self.input_path)[:-4])

        plot_multi_app_consumption_individual(y_eval.reshape((y_eval.shape[0]) * y_eval.shape[1], y_eval.shape[2]),
                                              eval_predict.reshape((y_eval.shape[0]) * y_eval.shape[1], y_eval.shape[2]),
                                              os.getenv('HOME') + '/Documents/' + os.path.basename(self.input_path)[:-4])
