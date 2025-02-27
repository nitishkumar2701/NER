from flask import Flask, render_template, request, jsonify
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.util import minibatch, compounding
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
import pandas as pd
import os
import json
import time

app = Flask(__name__, template_folder="templates", static_folder="static")

# Define paths
DATA_DIR = "data"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "pubtator_ner_model")
TRAINING_FILE = os.path.join(DATA_DIR, "CDR_TrainingSet.PubTator.txt")
DEV_FILE = os.path.join(DATA_DIR, "CDR_DevelopmentSet.PubTator.txt")
MODEL_INFO_FILE = os.path.join(MODEL_DIR, "model_info.json")

# Create required directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def read_pubtator_file(file_path):
    """
    Read and parse PubTator format files
    """
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return {}
        
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) > 2:  # Title or abstract line
                pmid = parts[0]
                section = parts[1]
                text = parts[2]
                if pmid not in data:
                    data[pmid] = {
                        'text': '',
                        'entities': []
                    }
                if section == 't':  # Title
                    data[pmid]['text'] = text + ' '
                elif section == 'a':  # Abstract
                    data[pmid]['text'] += text
            elif len(parts) == 1:  # Annotation line
                parts = line.split('\t')
                if len(parts) >= 5 and not parts[0].startswith('CID'):
                    pmid, start, end, text, entity_type = parts[:5]
                    if pmid in data:
                        # Ensure start and end are integers
                        try:
                            start_idx = int(start)
                            end_idx = int(end)
                            data[pmid]['entities'].append({
                                'start': start_idx,
                                'end': end_idx,
                                'label': entity_type
                            })
                        except ValueError:
                            print(f"Skipping invalid entity indices: start={start}, end={end}")
                            continue
    return data

def map_entity_type(original_type):
    """
    Map PubTator entity types to standardized entity types for our application
    """
    mapping = {
        # PubTator mappings
        "Chemical": "Chemical",
        "Disease": "Disease",
        "Gene": "Gene",
        "Species": "Species",
        # Common variations
        "CHEMICAL": "Chemical",
        "DISEASE": "Disease",
        "GENE": "Gene",
        "SPECIES": "Species",
        # Abbreviated forms
        "CHEM": "Chemical",
        "DIS": "Disease",
        "GEN": "Gene",
        "SPE": "Species",
        # Other common formats
        "chemical": "Chemical",
        "disease": "Disease",
        "gene": "Gene",
        "species": "Species",
    }
    
    return mapping.get(original_type, original_type)

def convert_spacy_format(nlp, data):
    """
    Creates spaCy Example objects for training
    """
    examples = []
    for pmid, item in data.items():
        text = item['text']
        doc = nlp.make_doc(text)
        ents = []

        # Sort entities by start position
        sorted_entities = sorted(item['entities'], key=lambda x: x['start'])

        for ent in sorted_entities:
            # Map entity types to standardized labels
            entity_type = map_entity_type(ent['label'])
            
            # Create span
            span = doc.char_span(
                ent['start'],
                ent['end'],
                label=entity_type,
                alignment_mode="contract"
            )
            if span is not None:
                ents.append(span)

        # Only create example if we have valid entities
        if ents:
            doc.ents = ents
            # Create the Example object with entity spans
            example = Example.from_dict(
                doc,
                {
                    "entities": [(ent.start_char, ent.end_char, ent.label_) for ent in ents]
                }
            )
            examples.append(example)

    return examples

def train_ner(train_examples, val_examples, output_dir, n_iter=10):
    """
    Train a spaCy NER model using the converted PubTator data
    """
    nlp = spacy.blank("en")

    # Add NER pipe
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    # Collect all unique labels
    entity_labels = set()
    for example in train_examples:
        # Access entities directly from the reference's ents
        for ent in example.reference.ents:
            entity_labels.add(ent.label_)

    # Add labels to the NER pipe
    for label in entity_labels:
        ner.add_label(label)

    # Configure training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    # Begin training
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(train_examples)
            losses = {}
            batches = minibatch(train_examples, size=compounding(4.0, 32.0, 1.001))

            for batch in tqdm(batches, desc=f"Training iteration {itn}"):
                nlp.update(batch, drop=0.5, losses=losses)

            print(f"Losses at iteration {itn}: {losses}")

    # Make sure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    nlp.to_disk(output_dir)
    return nlp

def needs_training():
    """
    Determine if the model needs to be trained by checking:
    1. If the model directory exists
    2. If the model info file exists
    3. If the source data files have been modified after the model was trained
    """
    # Check if model directory and info file exist
    if not os.path.exists(MODEL_PATH) or not os.path.exists(MODEL_INFO_FILE):
        return True
        
    # Check model info
    with open(MODEL_INFO_FILE, 'r') as f:
        model_info = json.load(f)
        
    # Get last modification times of the training files
    training_file_time = os.path.getmtime(TRAINING_FILE) if os.path.exists(TRAINING_FILE) else 0
    dev_file_time = os.path.getmtime(DEV_FILE) if os.path.exists(DEV_FILE) else 0
    
    # If either file was modified after the model was trained, retrain
    return (training_file_time > model_info['trained_at'] or 
            dev_file_time > model_info['trained_at'])

def train_model():
    """
    Main function to train the model if needed
    """
    try:
        start_time = time.time()
        print("Starting model training process...")
        
        # Create a blank spaCy model
        nlp = spacy.blank("en")
        
        # Step 1: Read training and development data
        print(f"Reading training data from {TRAINING_FILE}...")
        training_data = read_pubtator_file(TRAINING_FILE)
        
        print(f"Reading development data from {DEV_FILE}...")
        dev_data = read_pubtator_file(DEV_FILE)
        
        # Check if we have data
        if not training_data:
            print(f"Error: No training data found in {TRAINING_FILE}")
            return False
            
        # Step 2: Combine the data
        print("Combining datasets...")
        combined_data = {**training_data, **dev_data}  # Merge dictionaries
        print(f"Combined dataset contains {len(combined_data)} documents")
        
        # Step 3: Convert to spaCy format
        print("Converting to spaCy format...")
        examples = convert_spacy_format(nlp, combined_data)
        
        if not examples:
            print("Error: No valid examples were created. Please check the data files.")
            return False
            
        # Step 4: Split into training and validation sets
        train_examples, val_examples = train_test_split(examples, test_size=0.2, random_state=42)
        print(f"Training with {len(train_examples)} examples, validating with {len(val_examples)} examples")
        
        # Step 5: Train the model
        print(f"Training NER model and saving to {MODEL_PATH}...")
        trained_nlp = train_ner(train_examples, val_examples, MODEL_PATH)
        
        # Step 6: Save model info
        model_info = {
            'trained_at': time.time(),
            'num_examples': len(examples),
            'training_file': TRAINING_FILE,
            'dev_file': DEV_FILE
        }
        
        with open(MODEL_INFO_FILE, 'w') as f:
            json.dump(model_info, f)
            
        training_time = time.time() - start_time
        print(f"Model training completed in {training_time:.2f} seconds")
        return True
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        return False

# Initialize the model - train if needed or load existing
if needs_training():
    print("Model needs training. Starting training process...")
    success = train_model()
    if not success:
        print("Warning: Model training failed. Will use a minimal model.")
        # Create a minimal model
        nlp = spacy.blank("en")
        ner = nlp.add_pipe("ner")
        for label in ["Chemical", "Disease", "Gene", "Species"]:
            ner.add_label(label)
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        nlp.to_disk(MODEL_PATH)
else:
    print(f"Loading existing model from {MODEL_PATH}")

# Load the model
try:
    nlp = spacy.load(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print("Creating a blank model as fallback")
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    for label in ["Chemical", "Disease", "Gene", "Species"]:
        ner.add_label(label)

# Make sure the model has the entity types we need
if "ner" in nlp.pipe_names:
    ner = nlp.get_pipe("ner")
    # Ensure our expected entity types are in the model
    for label in ["Chemical", "Disease", "Gene", "Species"]:
        if label not in ner.labels:
            ner.add_label(label)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'entities': [],
                'html_text': ''
            }), 400
            
        text = data.get('text', '')
        
        if not text:
            return jsonify({
                'entities': [],
                'html_text': ''
            })
        
        # Process the text with our NER model
        doc = nlp(text)
        entities = []
        
        for ent in doc.ents:
            # Map entity labels if needed
            label = map_entity_type(ent.label_)
            
            entities.append({
                'text': ent.text,
                'label': label,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        # Create a colorized HTML version of the text with highlighted entities
        html_text = ""
        last_end = 0
        
        # Sort entities by start position to avoid overlap issues
        sorted_entities = sorted(entities, key=lambda x: x['start'])
        
        # Filter out overlapping entities
        filtered_entities = []
        for entity in sorted_entities:
            # Check if this entity overlaps with any already processed entity
            overlaps = False
            for processed in filtered_entities:
                if (entity['start'] < processed['end'] and entity['end'] > processed['start']):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_entities.append(entity)
        
        # Now generate the HTML with highlighted entities
        for entity in filtered_entities:
            # Add text before the entity
            html_text += text[last_end:entity['start']]
            
            # Add the entity with color coding based on its type
            entity_class = entity['label'].lower()  # Use label as CSS class
            html_text += f'<mark class="{entity_class}">{text[entity["start"]:entity["end"]]}</mark>'
            
            # Update the last end position
            last_end = entity['end']
        
        # Add any text after the last entity
        html_text += text[last_end:]
        
        # Debug info
        print(f"Processed text: {text[:50]}..." if len(text) > 50 else f"Processed text: {text}")
        print(f"Found {len(filtered_entities)} entities")
        
        return jsonify({
            'entities': filtered_entities,
            'html_text': html_text
        })
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({
            'error': str(e),
            'entities': [],
            'html_text': f'An error occurred: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Check if the necessary directories exist, create if they don't
    for directory in ["templates", "static", "data", "models"]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created {directory} directory")
    
    # Check if the index.html exists in templates directory
    html_path = os.path.join('templates', 'index.html')
    if not os.path.exists(html_path) and os.path.exists('index.html'):
        import shutil
        shutil.copy('index.html', html_path)
        print(f"Copied index.html to {html_path}")
    
    app.run(debug=True)