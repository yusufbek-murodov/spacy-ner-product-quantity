import spacy 
import random
from spacy.util import minibatch
from spacy.training.example import Example 

# Training data 
train_data = [
    ("What is the price of 10 bananas?", {"entities": [(21, 23, "QUANTITY"), (24, 31, "PRODUCT")]}),
    ("I need 2 kg of apples.", {"entities": [(7, 10, "QUANTITY"), (14, 20, "PRODUCT")]}),
    ("Can you deliver 3 liters of milk?", {"entities": [(15, 24, "QUANTITY"), (28, 32, "PRODUCT")]}),
    ("Order 5 packets of sugar.", {"entities": [(6, 15, "QUANTITY"), (19, 24, "PRODUCT")]}),
    ("She bought 1.5 kg of rice yesterday.", {"entities": [(11, 16, "QUANTITY"), (20, 24, "PRODUCT"), (26, 35, "DATE")]}),
    ("John purchased 12 bottles of water.", {"entities": [(15, 26, "QUANTITY"), (30, 35, "PRODUCT"), (0, 4, "PERSON")]}),
    ("Price of 2 dozen eggs?", {"entities": [(9, 17, "QUANTITY"), (18, 22, "PRODUCT")]}),
    ("I want 500 grams of cheese.", {"entities": [(7, 17, "QUANTITY"), (21, 27, "PRODUCT")]}),
    ("Lisa ordered 4 packs of coffee.", {"entities": [(13, 20, "QUANTITY"), (24, 30, "PRODUCT"), (0, 4, "PERSON")]}),
    ("Where can I find 6 kg of tomatoes?", {"entities": [(16, 19, "QUANTITY"), (23, 31, "PRODUCT")]}),
    ("Buy me 2 cans of soda.", {"entities": [(7, 13, "QUANTITY"), (17, 21, "PRODUCT")]}),
    ("We need 3 kg of chicken.", {"entities": [(8, 11, "QUANTITY"), (15, 22, "PRODUCT")]}),
    ("The store has 10 pounds of flour.", {"entities": [(13, 22, "QUANTITY"), (26, 31, "PRODUCT")]}),
    ("Can I get 1 kg of butter?", {"entities": [(10, 13, "QUANTITY"), (17, 23, "PRODUCT")]}),
    ("James bought 3 dozen oranges.", {"entities": [(14, 23, "QUANTITY"), (24, 31, "PRODUCT"), (0, 5, "PERSON")]}),
    ("Give me 5 bags of potatoes.", {"entities": [(8, 14, "QUANTITY"), (18, 26, "PRODUCT")]}),
    ("How much is 7 liters of juice?", {"entities": [(11, 20, "QUANTITY"), (24, 29, "PRODUCT")]}),
    ("I'd like 2 bottles of oil.", {"entities": [(10, 18, "QUANTITY"), (22, 25, "PRODUCT")]}),
    ("Can I get 6 kg of fish?", {"entities": [(10, 13, "QUANTITY"), (17, 21, "PRODUCT")]}),
    ("The restaurant ordered 8 sacks of rice.", {"entities": [(20, 28, "QUANTITY"), (32, 36, "PRODUCT")]}),
    ("I need 2 dozen bananas.", {"entities": [(7, 15, "QUANTITY"), (16, 23, "PRODUCT")]}),
    ("Mark wants 4 bags of flour.", {"entities": [(12, 18, "QUANTITY"), (22, 27, "PRODUCT"), (0, 4, "PERSON")]}),
    ("She bought 15 apples yesterday.", {"entities": [(11, 13, "QUANTITY"), (14, 20, "PRODUCT"), (22, 31, "DATE")]}),
    ("I will buy 3 cartons of milk.", {"entities": [(12, 21, "QUANTITY"), (25, 29, "PRODUCT")]}),
    ("We ordered 9 trays of eggs.", {"entities": [(11, 18, "QUANTITY"), (22, 26, "PRODUCT")]}),
    ("I need 500 ml of olive oil.", {"entities": [(7, 13, "QUANTITY"), (17, 26, "PRODUCT")]}),
    ("Get me 7 pieces of chocolate.", {"entities": [(7, 15, "QUANTITY"), (19, 28, "PRODUCT")]}),
    ("How much for 20 kg of sugar?", {"entities": [(13, 18, "QUANTITY"), (22, 27, "PRODUCT")]}),
    ("They purchased 5 dozen mangoes.", {"entities": [(16, 24, "QUANTITY"), (25, 32, "PRODUCT")]}),
    ("Is 1 liter of honey enough?", {"entities": [(3, 10, "QUANTITY"), (14, 19, "PRODUCT")]}),
]


# Load spacy blank English Model
nlp = spacy.blank("en")

# Add NER pipeline if not present
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

# Add labels to the NER model
for _, annotations in train_data:
    for ent in annotations["entities"]:
        ner.add_label(ent[2])

# Disable other pipes for training 
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):

    # Start training
    optimizer = nlp.begin_training()
    epochs = 50

    for epoch in range(epochs):
        random.shuffle(train_data)
        losses = {}
        batches = minibatch(train_data, size=2)

        for batch in batches:
            examples = []
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)

            nlp.update(examples, drop=0.3, losses=losses)
        
        print(f"Epoch {epoch + 1}, Loss: {losses}")
    
# Save trained model
model_path = "custom_ner_model"
nlp.to_disk(model_path)
print(f"Model saved to {model_path}")

# Load and test the trained model
trained_nlp = spacy.load(model_path)

test_texts = [
    "How much is 3 kg of rice?",
    "I want 15 oranges",
    "Can you give me the price for 6 kg of sugar?",
    "Can you give me the price for 12.5 kg of potato?",
    "How much is 1kg of tomato?"
]

# Run predictions
for text in test_texts:
    doc = trained_nlp(text)
    print(f"\nText: {text}")
    print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])