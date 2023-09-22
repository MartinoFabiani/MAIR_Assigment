import pandas as pd
from collections import Counter

file_path = 'C:\\Users\\Martino Fabiani\\OneDrive\\Documenti\\Workspace\\dialog_acts.dat'


# Initialize two empty lists to store the first words and the rest of the text
prime_parole = []
resto_del_testo = []

# Open the file and read the lines
with open(file_path, 'r') as file:
    for line in file:
        line = line.strip()  # Remove any white spaces and newline
        parole = line.split(' ', 1)  # Split the line into two parts using space as a separator
        if len(parole) == 2:
            Dialogue_act, resto = parole
        else:
            Dialogue_act = line
            resto = ''
        prime_parole.append(Dialogue_act)
        resto_del_testo.append(resto)


Acts_df = pd.DataFrame({'Dialogue_Act': prime_parole, 'Sentence': resto_del_testo})


Acts_noduplicates_df=Acts_df.drop_duplicates()
Acts_noduplicates_df.shape

# Shuffle the rows randomly
shuffle_df = Acts_noduplicates_df.sample(frac=1).reset_index(drop=True)

training_df = shuffle_df.iloc[1:4556]
training_df

test_df = shuffle_df.iloc[4557:5359]
test_df


def get_top_words_by_dialogue_act(df, n=2):
    result = {}
 
    grouped = df.groupby('Dialogue_Act')
    
    for group_name, group_df in grouped:
        sentences = ' '.join(group_df['Sentence']).split()
        
        # COunt the frequency of the words
        word_counts = Counter(sentences)
        
        # Select the most frequent words
        top_words = word_counts.most_common(n)
        
        result[group_name] = top_words
        
    return result

def find_dialogue_act(sentence, df):
    for index, row in df.iterrows():
        act = row['Dialogue_Act']
        keywords = row['Sentence'].split()
        if any(keyword in sentence for keyword in keywords):
            return act
    return None


#Funzione per leggere l'input dell'utente
def get_user_input():
    return input("You: ").lower()

# Funzione principale del chatbot
def main():
    get_top_words_by_dialogue_act(training_df)
    print("Welcome! Enter the sentence:")

    while True:
        frase_utente = get_user_input()
        
        if frase_utente == 'exit':
            print("Goodbye! Have a great day!")
            break

        result = find_dialogue_act(frase_utente, training_df)
        print("Chatbot: "+ result)

if __name__ == "__main__":
    main()








    

