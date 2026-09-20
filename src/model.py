from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout


def build_model(vocab_size: int, embedding_dim: int = 128, max_len: int = 200, num_labels: int = 6) -> Model:
    inp = Input(shape=(max_len,), name='input')
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len, name='embedding')(inp)
    x = Bidirectional(LSTM(128, return_sequences=False))(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(num_labels, activation='sigmoid', name='out')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
