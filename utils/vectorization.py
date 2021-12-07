from tensorflow.keras.layers import TextVectorization

def formula_vertorization(vocab_path, sequence_length = 50):
    vectorization = TextVectorization(
        standardize=None,
        output_mode="int",
        output_sequence_length=sequence_length,
    )

    vectorization.set_vocabulary(vocab_path)
    print(len(vectorization.get_vocabulary()))

    return vectorization