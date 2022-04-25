import tensorflow as tf
from tensorflow.keras.regularizers import l2
from util.accuracy import estimated_accuracy
from util.files import load_data
from util.preprocessing import process_data, split_data
from util.statistics import plot_multi_app_consumption


class NilmCNN():
    def __init__(self, args):
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
        self.input_path = args['input_path']
        self.log_path = args['log_path']

    def create_model(self):
        input = tf.keras.Input(shape=(self.input_size, self.features), batch_size=None)

        output = tf.keras.layers.Conv1D(512, 1, padding='same', use_bias=self.bias, kernel_regularizer=l2(self.l2))(input)

        skip_connections = [output]

        # 1st "Layer"
        signal_out = tf.keras.layers.Conv1D(256, 2, activation="relu", dilation_rate=2 ** 1, padding='causal',
                                            use_bias=self.bias, kernel_regularizer=l2(self.l2))(output)
        gate_out = tf.keras.layers.Conv1D(256, 2, activation="sigmoid", dilation_rate=2 ** 1, padding='causal',
                                          use_bias=self.bias, kernel_regularizer=l2(self.l2))(output)
        output = tf.keras.layers.Multiply()([signal_out, gate_out])
        output = tf.keras.layers.Dropout(self.dropout)(output)
        skip_connections.append(output)

        # 2nd "Layer"
        signal_out = tf.keras.layers.Conv1D(512, 2, activation="relu", dilation_rate=2 ** 2, padding='causal',
                                            use_bias=self.bias, kernel_regularizer=l2(self.l2))(output)
        gate_out = tf.keras.layers.Conv1D(512, 2, activation="sigmoid", dilation_rate=2 ** 2, padding='causal',
                                          use_bias=self.bias, kernel_regularizer=l2(self.l2))(output)
        output = tf.keras.layers.Multiply()([signal_out, gate_out])
        output = tf.keras.layers.Dropout(self.dropout)(output)
        skip_connections.append(output)

        # 3rd "Layer"
        signal_out = tf.keras.layers.Conv1D(512, 2, activation="relu", dilation_rate=2 ** 3, padding='causal',
                                            use_bias=self.bias, kernel_regularizer=l2(self.l2))(output)
        gate_out = tf.keras.layers.Conv1D(512, 2, activation="sigmoid", dilation_rate=2 ** 3, padding='causal',
                                          use_bias=self.bias, kernel_regularizer=l2(self.l2))(output)
        output = tf.keras.layers.Multiply()([signal_out, gate_out])
        output = tf.keras.layers.Dropout(self.dropout)(output)
        skip_connections.append(output)

        # 4th "Layer"
        signal_out = tf.keras.layers.Conv1D(256, 2, activation="relu", dilation_rate=2 ** 4, padding='causal',
                                            use_bias=self.bias, kernel_regularizer=l2(self.l2))(output)
        gate_out = tf.keras.layers.Conv1D(256, 2, activation="sigmoid", dilation_rate=2 ** 4, padding='causal',
                                          use_bias=self.bias, kernel_regularizer=l2(self.l2))(output)
        output = tf.keras.layers.Multiply()([signal_out, gate_out])
        output = tf.keras.layers.Dropout(self.dropout)(output)
        skip_connections.append(output)

        # 5th "Layer"
        signal_out = tf.keras.layers.Conv1D(128, 2, activation="relu", dilation_rate=2 ** 5, padding='causal',
                                            use_bias=self.bias, kernel_regularizer=l2(self.l2))(output)
        gate_out = tf.keras.layers.Conv1D(128, 2, activation="sigmoid", dilation_rate=2 ** 5, padding='causal',
                                          use_bias=self.bias, kernel_regularizer=l2(self.l2))(output)
        output = tf.keras.layers.Multiply()([signal_out, gate_out])
        output = tf.keras.layers.Dropout(self.dropout)(output)
        skip_connections.append(output)

        output = tf.keras.layers.Concatenate()(skip_connections)
        output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.output_size, activation='linear'))(output)
        output = tf.keras.layers.ReLU()(output)

        opt = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            decay=0.,
            amsgrad=False,
            epsilon=1e-7,
            clipvalue=0.5
        )

        sgd = tf.keras.optimizers.SGD(
            lr=self.learning_rate,
            decay=1e-6,
            momentum=0.9,
            nesterov=True
        )

        model = tf.keras.Model(input, output)

        model.compile(optimizer=opt, loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError(), estimated_accuracy])
        model.summary()

        return model

    def train(self):
        callbacks = []

        # Compile model
        model = self.create_model()

        # Declare tensorboard callback
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_path)

        callbacks.append(tensorboard_callback)

        # Load data from input file
        arr = load_data(self.input_path)

        # Separate aggregate and appliances data
        aggregate, appliances = process_data(arr)

        # Split data for training and evaluation
        x_train, y_train, x_eval, y_eval = split_data(aggregate, appliances, self.validation_split, self.evaluate_split)

        # Train model
        model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, use_multiprocessing=True,
                  validation_split=self.validation_split, callbacks=callbacks)

        print('\nEvaluate:\n')

        # Evalute model performance
        model.evaluate(x_eval, y_eval)

        eval_predict = model.predict(x_eval)

        plot_multi_app_consumption(y_eval.reshape((y_eval.shape[0]) * 1440),
                                   eval_predict.reshape((y_eval.shape[0]) * 1440))
