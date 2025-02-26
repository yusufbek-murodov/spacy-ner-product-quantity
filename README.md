 🏷️ Spacy NER: Product & Quantity Extractor

A **custom Named Entity Recognition (NER) model** built with **spaCy** to extract **product names** and **quantities** from text.

 🚀 Features
✅ Detects **quantities** (e.g., "2 kg", "5 liters")  
✅ Identifies **product names** (e.g., "apples", "milk")  
✅ Trained on **real-world purchase queries**  
✅ Built with **spaCy's NLP pipeline**  

---

📌 Installation

1️⃣ **Clone the repository**  
```bash
git clone https://github.com/YOUR_USERNAME/spacy-ner-product-quantity.git
cd spacy-ner-product-quantity
```

2️⃣ **Create & activate a virtual environment**  
```bash
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
```

3️⃣ **Install dependencies**  
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

---

## 🏋️ Training the Model
Run the following command to train the custom NER model:  
```bash
python train_ner.py
```

---

## 🎯 Testing the Model
Test the trained model with sample inputs:  
```bash
python test_ner.py
```

### ✅ Example Input
```
"I need 5 bags of rice."
```

### 🔍 Expected Output
```
[('5 bags', 'QUANTITY'), ('rice', 'PRODUCT')]
```

---

## 📜 License
This project is licensed under the **MIT License**.
