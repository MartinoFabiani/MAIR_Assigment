import pandas as pd

file_path = 'C:\\Users\\Martino Fabiani\\OneDrive\\Documenti\\Workspace\\dialog_acts.dat'
#file_path = '../data/dialog_acts.dat'
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

# Rimescolare le righe in modo casuale
shuffle_df = Acts_noduplicates_df.sample(frac=1).reset_index(drop=True)

#training_df = shuffle_df.iloc[1:4556]
#training_df

#test_df = shuffle_df.iloc[4557:5359]
#test_df


#Funzione per leggere l'input dell'utente
def get_user_input():
    return input("You: ").lower()

# Funzione per cercare il Dialogue_Act in base alla Sentence
def cerca_dialogue_act(frase):
    # Cerca la frase nella colonna 'Sentence'
    risultato = shuffle_df[shuffle_df['Sentence'] == frase]

    # Se la frase è presente, restituisci il corrispondente Dialogue_Act
    if not risultato.empty:
        return risultato.iloc[0]['Dialogue_Act']
    else:
        return "Frase non trovata"

# Funzione principale del chatbot
def main():
    print("Welcome! Enter the sentence")
    
    while True:
        frase_utente = get_user_input()
        
        if frase_utente == 'exit':
            print("Goodbye! Have a great day!")
            break
        
        #dialogue_act = cerca_dialogue_act(frase_utente)
        #print("Chatbot:", dialogue_act)

        dialogue_act = cerca_dialogue_act(frase_utente)
        print("Chatbot: inform")

if __name__ == "__main__":
    main()