import streamlit as st
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import text, sequence
from sklearn.preprocessing import LabelBinarizer


st.image('1ece.gif')

@st.cache
def prediction(seqs):
    model = keras.models.load_model('model.h5')
    
    max_length = 512
        # initiate Tokenizer 
    tokenizer = Tokenizer(char_level=True) #tokenize in character level 
    tokenizer.fit_on_texts(seqs)


    X = tokenizer.texts_to_sequences([seqs])
    X = sequence.pad_sequences(X, maxlen=max_length)
    
    classes = ['HYDROLASE', 'TRANSFERASE', 'OXIDOREDUCTASE', 'IMMUNE SYSTEM',
       'LYASE', 'HYDROLASE/HYDROLASE INHIBITOR', 'TRANSCRIPTION',
       'VIRAL PROTEIN', 'TRANSPORT PROTEIN', 'VIRUS', 'SIGNALING PROTEIN',
       'ISOMERASE', 'LIGASE', 'MEMBRANE PROTEIN', 'PROTEIN BINDING',
       'STRUCTURAL PROTEIN', 'CHAPERONE',
       'STRUCTURAL GENOMICS, UNKNOWN FUNCTION', 'SUGAR BINDING PROTEIN',
       'DNA BINDING PROTEIN', 'PHOTOSYNTHESIS', 'ELECTRON TRANSPORT',
       'TRANSFERASE/TRANSFERASE INHIBITOR', 'METAL BINDING PROTEIN',
       'CELL ADHESION', 'UNKNOWN FUNCTION', 'PROTEIN TRANSPORT', 'TOXIN',
       'CELL CYCLE', 'RNA BINDING PROTEIN', 'DE NOVO PROTEIN', 'HORMONE',
       'GENE REGULATION', 'OXIDOREDUCTASE/OXIDOREDUCTASE INHIBITOR',
       'APOPTOSIS', 'MOTOR PROTEIN', 'PROTEIN FIBRIL', 'METAL TRANSPORT',
       'VIRAL PROTEIN/IMMUNE SYSTEM', 'CONTRACTILE PROTEIN',
       'FLUORESCENT PROTEIN', 'TRANSLATION', 'BIOSYNTHETIC PROTEIN']
    
    lb= LabelBinarizer()
    Y = lb.fit(classes)
    output = model.predict(X)
    output = lb.inverse_transform(output)
    return(output)  
    
    
def main():
    st.title('Protein Classification Prediction')
    st.write('Welcome to this simple app you can use to input a protein sequence to predict its classification!')
    menu = ['Intro','About the Dataset','Protein Classes','Protein Sequence & Classification']
    choice = st.sidebar.selectbox("Select Activity", menu)
    

    if choice == 'Intro':
        st.subheader("Introduction to Proteins")
        st.write('What are proteins and why are we interested in classifying them?')
        st.write("Proteins are complex molecules that play many critical roles in our body. They are required for the structure, function and reuglation of our body's tissues and organs.")
        st.write("Proteins are made up of small units called amino acids. If proteins are words, amino acids are the alphabets that come together to make up a word. There are 20 differnet types, or 20 'alphabets', of amino acids. The sequence of these amino acids determine the overll 3D structure of the protein. Proteins with different 3D structures end up serving different functions in our body.")
        st.write("This app allows you to input a protein sequence under the 'Protein Sequence & Classification' menu to classify your protein. If you want to learn more about the class of protein, click the 'About' dropdown menu to learn more.")
        
        
    elif choice == 'About the Dataset':
        st.subheader('About the Dataset')
        st.write("The data used in this project was from the Structural Protein Sequences dataset from Kaggle that can be found [here]('https://www.kaggle.com/shahir/protein-data-set'). This dataset was retrieved from the Research Collaboratory for Structural Bioinformatics (RCSB) Protein Data Bank (PDB)")
        st.image('43classes.png')
        st.write('Figure 1. Number of instances for each 43 protein classes.')
        st.write('There were more than 100 different classifications possible. However, classes with more than 1000 instances were taken for model training purposes.')
        st.image('sequencenum.png')
        st.write('Figure 2. Number of Sequences for proteins')
        st.write('To treat protein sequences like a language, "text" preprocessing was required. The maximum length that the model could handle was set to 512 aminoacids.')
        
        
    elif choice == 'Protein Classes':
        st.subheader("What do each of these Proteins do?")
        st.write("Possible 43 classifications:")
        st.write('HYDROLASE : Utilizes water to break larger molecules into smaller molecules. Important for body as it breaks down carbohydrates, fats and proteins that we consume.')
        st.write('TRANSFERASE: Transfers specific functional groups from one molecule to another.')
        st.write('OXIDOREDUCTASE: Catalyze transfer of electrons from one molecule to another.')
        st.write('IMMUNE SYSTEM: Predominantly signaling proteins that act as antibodies.')
        st.write('LYASE: Involved in breaking down molecules, but not through hydrolysis or oxidation.')
        st.write('HYDROLASE INHIBITOR: Inhibits activity of Hydrolases')
        st.write('TRANSCRIPTION: Involved in transcription, a process in making an RNA copy of DNA,')
        st.write('VIRAL PROTEIN: A component, as well as a product of a virus.')
        st.write('TRANSPORT PROTEIN: Takes part in both passive and active transport and moves around molecules across the membrane.')
        st.write('VIRUS: Infectious agents that replicates inside living orgnanisms.')
        st.write('SIGNALING PROTEIN: Interact with target cells and helps activate diffrent messengers that lead to physiological effects.')
        st.write('ISOMERASE: Rearranges structure of a molecule.')
        st.write('LIGASE: Joins molecules through involvement of water.')
        st.write('MEMBRANE PROTEIN: Helps cells maintain their shape.')
        st.write('PROTEIN BINDING: Acts as a binding agent for two or more molecules.')
        st.write('STRUCTURAL PROTEIN: Involved in bones, hair and muscles.')
        st.write('CHAPERONE: Folds mammalian proteins')
        st.write('STRUCTURAL GENOMICS, UNKNOWN FUNCTION: Involved in maintaining structure of a cell, but specific function unknown')
        st.write('SUGAR BINDING PROTEIN: Binds to carbohydrates on the surface of cells')
        st.write('DNA BINDING PROTEIN: Have high affinity towards DNA, specifically the major groove of DNA molecules')
        st.write('PHOTOSYNTHESIS: Involved in Photosynthesis')
        st.write('ELECTRON TRANSPORT: Transfers electrons from donors to acceptors through redox reactions')  
        st.write('TRANSFERASE INHIBITOR: Inhibits the activity of Transferases')
        st.write('METAL BINDING PROTEIN: Plays a role in structural stability, signaling, regulation and homeostasis')
        st.write('CELL ADHESION: Present on the cell surface and involved in binding with other cells')
        st.write('UNKNOWN FUNCTION: Exact function unknown')
        st.write('PROTEIN TRANSPORT: Transportation of molecules')
        st.write('TOXIN: Plays a role in infectious diseases that are self-programmed to reach target organ or cells')
        st.write('CELL CYCLE: Regulation and maintainence of Eukaryotic cell cycles')
        st.write('RNA BINDING PROTEIN: Regulate gene expression in post-transcriptional processes')
        st.write('DE NOVO PROTEIN: Resembles natural proteins that occur in nature')
        st.write('HORMONE: Act as chemical signalling molecules')
        st.write('GENE REGULATION: Ensures that appropriate genes are activated at the proper times')
        st.write('OXIDOREDUCTASE INHIBITOR: Inhibit function of Oxidoreductase')
        st.write('APOPTOSIS: Involved in cell death')
        st.write('MOTOR PROTEIN: Moves cytoskeletal filaments within cell. Involved in muscle movement')
        st.write('PROTEIN FIBRIL: Self-assembled proteins that aggregate at high temperatures')
        st.write('METAL TRANSPORT: Transports specific metal ions in and out of cytosol')
        st.write('VIRAL PROTEIN/IMMUNE SYSTEM: Serve as hormones for the immune system')
        st.write("CONTRACTILE PROTEIN: Mediate contraction of cell's cytoskeleton and muscles")
        st.write('FLUORESCENT PROTEIN: Emits fluorescent light')
        st.write('TRANSLATION: Involved in Protein translation')
        st.write('BIOSYNTHETIC PROTEIN: Involved in synthesis of proteins')
        
    elif choice == 'Protein Sequence & Classification':
        st.subheader("Input Sequence")
        Sequence = st.text_input('Protein Sequence')
        if st.button("Predict"):
            result = prediction(Sequence)
            st.success('The length of your protein is {} amino acids'.format(len(Sequence)))
            st.success('Your protein is in the {} class'.format(result))
            st.write('Click the About menu to learn more about the {} class'.format(result))
            
            
 
    
if __name__=='__main__':
    main()