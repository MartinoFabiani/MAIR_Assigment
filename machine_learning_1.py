import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

file_path = 'C:\\Users\\Martino Fabiani\\OneDrive\\Documenti\\Workspace\\dialog_acts.dat'

# Initialize two empty lists to store the first words and the rest of the text
first_words = []
rest_of_text = []

# Open the file and read the lines
with open(file_path, 'r') as file:
    for line in file:
        line = line.strip()  # Remove any white spaces and newline
        words = line.split(' ', 1)  # Split the line into two parts using space as a separator
        if len(words) == 2:
            dialogue_act, rest = words
        else:
            dialogue_act = line
            rest = ''
        first_words.append(dialogue_act)
        rest_of_text.append(rest)

acts_df = pd.DataFrame({'Dialogue_Act': first_words, 'Sentence': rest_of_text})

acts_no_duplicates_df = acts_df.drop_duplicates()
acts_no_duplicates_df.shape

# Shuffle the rows randomly
shuffle_df = acts_no_duplicates_df.sample(frac=1).reset_index(drop=True)

training_df = shuffle_df.iloc[1:4556]
test_df = shuffle_df.iloc[4557:5359]

# Preprocess the sentences in the "Sentence" column
# Tokenization and conversion to lowercase
tokenizer = Tokenizer()
tokenizer.fit_on_texts(training_df['Sentence'])
training_sequences = tokenizer.texts_to_sequences(training_df['Sentence'])
max_seq_length = max(len(seq) for seq in training_sequences)
training_data = pad_sequences(training_sequences, maxlen=max_seq_length)

test_sequences = tokenizer.texts_to_sequences(test_df['Sentence'])
test_data = pad_sequences(test_sequences, maxlen=max_seq_length)

# Preprocess the "Dialogue_Act" column
# Assign numeric identifiers to the classes
label_encoder = LabelEncoder()
training_df['Dialogue_Act_encoded'] = label_encoder.fit_transform(training_df['Dialogue_Act'])
test_df['Dialogue_Act_encoded'] = label_encoder.transform(test_df['Dialogue_Act'])

# Define the maximum number of words in your vocabulary
vocab_size = 10000  # Replace with the appropriate size for your dataset

# Define the embedding dimension (word vector space)
embedding_dim = 20  # Replace with the appropriate dimension

# Define the number of classes for "Dialogue_Act"
num_classes = len(acts_no_duplicates_df['Dialogue_Act'].unique())  # Automatically calculate the number of classes

# Create the model
model = Sequential()

# Add an embedding layer to represent words
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_length))

# Add an LSTM layer to process sequences of words
model.add(LSTM(128))  # You can adjust the number of LSTM units for desired performance

# Add a fully connected layer for classification
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(training_data, training_df['Dialogue_Act_encoded'], epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(test_data, test_df['Dialogue_Act_encoded'])
print("Test loss:", loss)
print("Test accuracy:", accuracy)

# Function to recognize user's intention for a single sentence
def recognize_intention(user_sentence):
    test_sequence = tokenizer.texts_to_sequences([user_sentence])
    test_data = pad_sequences(test_sequence, maxlen=max_seq_length)
    prediction = model.predict(test_data)
    intention = label_encoder.inverse_transform([prediction.argmax()])
    return intention

# Main function of the chatbot
def main():
    print("Welcome! Enter a sentence:")
    
    while True:
        user_sentence = input("You: ").lower()
        
        if user_sentence == 'exit':
            print("Goodbye! Have a great day!")
            break
        
        intention = recognize_intention(user_sentence)
        print("Chatbot: Recognized intention:", intention)

if __name__ == "__main__":
    main()
