import tensorflow as tf
import pandas as pd
import tensorflow_recommenders as tfrs
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
import time

print(f"TensorFlow version: {tf.__version__}")
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
print()

def pandas_dataframe_to_dataset(dataframe):
    return tf.data.Dataset.from_tensor_slices(dict(dataframe.copy()))


def tensor_to_string(x):
    return tf.strings.as_string(x['ISBN']) if type(x) is dict else x


print('Processing data')
start_time = time.time()

books_dir = './BRM-data/BX-Books.csv'
ratings_dir = './BRM-data/BX-Book-Ratings.csv'
# users_dir = '/content/drive/MyDrive/Colab Notebooks/BRM-data/BX-Users.csv' # Users won't be used
encoding = 'latin-1'

with open(books_dir, 'r', encoding=encoding) as file:
    lines = [line.strip() for line in file if len(line.split(';')) == 8]
books_df = pd.read_csv(StringIO('\n'.join(lines)), sep=';', quotechar='"', na_values=['NULL'], encoding='latin-1')
ISBN_df = books_df[['ISBN']]
books_df = books_df[['Book-Title', 'Book-Author', 'Year-Of-Publication', 'ISBN']]

with open(ratings_dir, 'r', encoding=encoding) as file:
    lines = [line.strip() for line in file]
ratings_df = pd.read_csv(StringIO('\n'.join(lines)), sep=';', quotechar='"', na_values=['NULL'], encoding='latin-1')
ratings_df = ratings_df[['User-ID', 'ISBN']]
ratings_df['User-ID'] = ratings_df['User-ID'].astype(str)

data_split = 0.8
msk = np.random.rand(len(ISBN_df)) < data_split
ISBN_data = pandas_dataframe_to_dataset(ISBN_df)
ISBN_train = pandas_dataframe_to_dataset(ISBN_df[msk])

msk = np.random.rand(len(ratings_df)) < data_split
ratings_data = pandas_dataframe_to_dataset(ratings_df)
ratings_train = pandas_dataframe_to_dataset(ratings_df[msk])
ratings_test = ratings_df[~msk]

print(f"Duration: {(time.time() - start_time)} sec\n")

print('Creating vocabularies')
start_time = time.time()
ISBN_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
user_vocabulary = tf.keras.layers.StringLookup(mask_token=None)

ISBN_vocabulary.adapt(ISBN_data.map(lambda x: x["ISBN"]))
user_vocabulary.adapt(ratings_data.map(lambda x: x['User-ID']))
print(f"Duration: {(time.time() - start_time)} sec\n")


class BookRecomenderModel(tfrs.Model):
    def __init__(
        self,
        user_model: tf.keras.Model,
        book_model: tf.keras.Model,
        task: tfrs.tasks.Retrieval
    ):
        super().__init__()

        # Set up user and movie representations.
        self.user_model = user_model
        self.book_model = book_model

        # Set up a retrieval task.
        self.task = task

    def compute_loss(self, features, training=False) -> tf.Tensor:
        # Define how the loss is computed.

        user_embeddings = self.user_model(features["User-ID"])
        book_embeddings = self.book_model(features["ISBN"])

        return self.task(user_embeddings, book_embeddings)


print('Training model')
start_time = time.time()

user_model = tf.keras.Sequential([
    user_vocabulary,
    tf.keras.layers.Embedding(user_vocabulary.vocabulary_size(), 64)
])

book_model = tf.keras.Sequential([
    tf.keras.layers.Lambda(tensor_to_string),
    ISBN_vocabulary,
    tf.keras.layers.Embedding(ISBN_vocabulary.vocabulary_size(), 64)
])

task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
        ISBN_train.batch(128).map(book_model)
    )
)

model = BookRecomenderModel(user_model, book_model, task)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))

history = model.fit(ratings_train.batch(4096), epochs=10)

print(f"Duration: {(time.time() - start_time)} sec\n")

# summarize history for accuracy
plt.plot(history.history['factorized_top_k/top_1_categorical_accuracy'])
plt.plot(history.history['factorized_top_k/top_5_categorical_accuracy'])
plt.plot(history.history['factorized_top_k/top_10_categorical_accuracy'])
plt.plot(history.history['factorized_top_k/top_50_categorical_accuracy'])
plt.plot(history.history['factorized_top_k/top_100_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['top_1_accuracy', 'top_5_accuracy', 'top_10_accuracy', 'top_50_accuracy', 'top_100_accuracy'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc='upper left')
plt.show()


# Generate dataset with ISBN and corresponding embeddings
books_with_embeddings = ISBN_data.map(lambda x: (x["ISBN"], model.book_model(x["ISBN"])))

# Use brute-force search to set up retrieval using the trained representations.
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
index.index_from_dataset(books_with_embeddings.batch(100))


def get_top_recommendations_for_user(user_id, index, num_recommendations):
    # Get top recommendations for the user
    _, top_ISBN = index(np.array([user_id]))

    print(f"Top {num_recommendations} recommendations for user {user_id}:")
    print('-' * 35)
    for ISBN in top_ISBN[0, :num_recommendations]:
        book_info = books_df[books_df['ISBN'] == ISBN].to_dict(orient='records')[0]

        print('Title: ', book_info['Book-Title'])
        print('Author: ', book_info['Book-Author'])
        print('Year of Publication: ', book_info['Year-Of-Publication'])
        print('-' * 35)


def get_acc_for_user(user_id, index):
    _, top_ISBN = index(np.array([user_id]))
    user_ratings = ratings_df[ratings_df['User-ID'] == user_id]

    num_recommendations_list = [1, 5, 10, 50, 100]
    accuracies = {}
    sum_of_correct_recommendations = 0
    for num_recommendations in num_recommendations_list:
        for ISBN in top_ISBN[0, :num_recommendations]:
            sum_of_correct_recommendations += 1 if ISBN in user_ratings['ISBN'].values else 0
        accuracies[num_recommendations] = sum_of_correct_recommendations / num_recommendations

    return accuracies


print('Calculating accuracy')
start_time = time.time()

accuracies = []
for user_id in ratings_test['User-ID'].values:
    accuracies.append(get_acc_for_user(user_id, index))

sums = {}
counts = {}

# Calculate the sum and count for each key
for d in accuracies:
    for key, value in d.items():
        sums[key] = sums.get(key, 0) + value
        counts[key] = counts.get(key, 0) + 1

average_accuracies = {}
for key in sums:
    average_accuracies[key] = sums[key] / counts[key]

print('Accuracy:')
print('-'*35)
for key, value in average_accuracies.items():
    print(f"\t{key}: {value}")

print(f"Duration: {(time.time() - start_time)} sec\n")

print('Checking random sample')
start_time = time.time()

user_id = ratings_test['User-ID'].sample(n=1, random_state=1).values[0]
num_recommendations = 5
get_top_recommendations_for_user(user_id, index, num_recommendations)

print(f"Duration: {(time.time() - start_time)} sec\n")

loc = './BRM-model/checkpoints'
model.save_weights(loc)
