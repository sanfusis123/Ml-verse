# Comprehensive NLP 

## Table of Contents
1. [Core NLP Concepts](#1-core-nlp-concepts)
2. [Classical NLP Models & Techniques](#2-classical-nlp-models--techniques)
3. [Deep Learning for NLP](#3-deep-learning-for-nlp)
4. [Transformer Architecture & LLMs](#4-transformer-architecture--llms)
5. [Practical System Design](#5-practical-system-design)
6. [Evaluation Metrics & Challenges](#6-evaluation-metrics--challenges)
7. [NLP Tools & Libraries](#7-nlp-tools--libraries)
8. [Questions](#8-questions)

---

## 1. Core NLP Concepts

### Q1.1: What is tokenization and why is it important?

**Answer:** Tokenization is the process of breaking down text into smaller units called tokens (words, subwords, or characters). It's the first and most fundamental step in text preprocessing.

**Types:**
- **Word-level:** Splits text into individual words
  - Example: "NLP is fun" → ["NLP", "is", "fun"]
- **Subword:** Splits words into meaningful subunits (WordPiece, BPE)
  - Example: "unhappiness" → ["un", "happiness"]
- **Character-level:** Splits into individual characters
  - Example: "NLP" → ["N", "L", "P"]

**Why it's important:**
- Enables machines to process text as discrete units
- Required for frequency analysis and feature extraction
- Facilitates tasks like POS tagging and NER
- Handles out-of-vocabulary (OOV) words with subword tokenization

---

### Q1.2: Explain the difference between stemming and lemmatization with examples.

**Answer:**

**Stemming:**
- Rule-based process that removes suffixes to reach root form
- Faster but less accurate
- May produce non-existent words
- Example using Porter Stemmer:
  - "running", "runner", "ran" → "run"
  - "studies", "studying" → "studi" (not a real word)

**Lemmatization:**
- Uses vocabulary and morphological analysis
- Returns actual dictionary words (lemmas)
- Considers context (POS tags)
- Slower but more accurate
- Example:
  - "better" → "good" (with POS="adjective")
  - "running" → "run"
  - "studies" → "study"

**Code Example (Python):**
```python
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import word_tokenize

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

text = "The cats are running faster than dogs"
tokens = word_tokenize(text)

# Stemming
stemmed = [stemmer.stem(word) for word in tokens]
# ['the', 'cat', 'are', 'run', 'faster', 'than', 'dog']

# Lemmatization
lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
# ['The', 'cat', 'are', 'running', 'faster', 'than', 'dog']
```

---

### Q1.3: What is Part-of-Speech (POS) tagging and why is it important?

**Answer:** POS tagging assigns grammatical categories (noun, verb, adjective, etc.) to each word based on context and definition.

**Importance:**
- Disambiguates word meaning based on context
- Essential for Named Entity Recognition
- Required for dependency parsing
- Improves information extraction accuracy

**Example:**
```python
import nltk

text = "They refuse to go" vs "We need the refuse permit"
print(nltk.pos_tag(word_tokenize(text1)))
# [('They', 'PRP'), ('refuse', 'VBP'), ('to', 'TO'), ('go', 'VB')]

print(nltk.pos_tag(word_tokenize(text2)))
# [('We', 'PRP'), ('refuse', 'NN'), ('permit', 'NN')]
```

**Common POS Tags:**
- NN: Noun
- VB: Verb
- JJ: Adjective
- RB: Adverb
- DT: Determiner

---

### Q1.4: Explain Named Entity Recognition (NER) with examples.

**Answer:** NER identifies and classifies named entities in text into predefined categories like person, organization, location, date, etc.

**Common Entity Types:**
- **PERSON:** Names of people
- **ORG:** Organizations, companies
- **GPE:** Geopolitical entities (countries, cities)
- **DATE:** Dates and time expressions
- **MONEY:** Monetary values
- **LOCATION:** Non-GPE locations

**Example:**
```
Input: "Elon Musk founded SpaceX in 2002 in California."

Output:
- "Elon Musk" → PERSON
- "SpaceX" → ORGANIZATION
- "2002" → DATE
- "California" → GPE
```

**Applications:**
- Information extraction
- Question answering systems
- Content classification
- Knowledge graph construction

---

## 2. Classical NLP Models & Techniques

### Q2.1: What are N-grams and how are they used in NLP?

**Answer:** N-grams are contiguous sequences of n items (words, characters) from text.

**Types:**
- **Unigram (1-gram):** Single word ["the", "cat", "sat"]
- **Bigram (2-gram):** Two consecutive words ["the cat", "cat sat"]
- **Trigram (3-gram):** Three consecutive words ["the cat sat"]

**Applications:**
- Language modeling (predicting next word)
- Text generation
- Spell correction
- Machine translation

**Example:**
```python
from nltk import ngrams

text = "Natural Language Processing is amazing"
tokens = text.split()

bigrams = list(ngrams(tokens, 2))
# [('Natural', 'Language'), ('Language', 'Processing'), 
#  ('Processing', 'is'), ('is', 'amazing')]

trigrams = list(ngrams(tokens, 3))
# [('Natural', 'Language', 'Processing'), 
#  ('Language', 'Processing', 'is'), 
#  ('Processing', 'is', 'amazing')]
```

---

### Q2.2: Explain TF-IDF and its importance.

**Answer:** TF-IDF (Term Frequency-Inverse Document Frequency) measures word importance relative to a document and corpus.

**Formula:**
```
TF-IDF(t, d) = TF(t, d) × IDF(t)

Where:
TF(t, d) = (Number of times term t appears in document d) / (Total terms in d)
IDF(t) = log(Total documents / Documents containing term t)
```

**Key Concepts:**
- **High TF-IDF:** Word is frequent in document but rare in corpus (important)
- **Low TF-IDF:** Word is common across all documents (less important)

**Example:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "I love machine learning",
    "I love deep learning",
    "Deep learning is amazing"
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
# ['amazing', 'deep', 'is', 'learning', 'love', 'machine']

print(tfidf_matrix.toarray())
# Higher values = more important words
```

**Applications:**
- Information retrieval
- Document similarity
- Feature extraction for ML models
- Keyword extraction

---

### Q2.3: What is Naive Bayes classifier and how is it used in NLP?

**Answer:** Naive Bayes is a probabilistic classifier based on Bayes' theorem with the "naive" assumption of feature independence.

**Formula:**
```
P(Class|Features) = P(Features|Class) × P(Class) / P(Features)
```

**Types:**
- **Multinomial NB:** For discrete features (word counts)
- **Bernoulli NB:** For binary features (word presence/absence)
- **Gaussian NB:** For continuous features

**Example - Spam Classification:**
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Training data
emails = [
    "win prize money now",  # spam
    "meeting at 3pm tomorrow",  # not spam
    "free lottery winner",  # spam
    "project deadline reminder"  # not spam
]
labels = [1, 0, 1, 0]  # 1=spam, 0=not spam

# Vectorize
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Train
clf = MultinomialNB()
clf.fit(X, labels)

# Predict
test = ["free money prize"]
test_vec = vectorizer.transform(test)
print(clf.predict(test_vec))  # [1] - spam
```

**Advantages:**
- Fast training and prediction
- Works well with small datasets
- Handles high-dimensional data

**Limitations:**
- Assumes feature independence (rarely true)
- Struggles with feature correlations

---

### Q2.4: Explain Hidden Markov Models (HMMs) in NLP.

**Answer:** HMMs are statistical models representing systems with hidden states, commonly used for sequence labeling tasks.

**Components:**
- **States:** Hidden variables (POS tags, entity types)
- **Observations:** Visible data (words)
- **Transition probabilities:** P(state_t | state_t-1)
- **Emission probabilities:** P(observation | state)

**Applications:**
- POS tagging
- Speech recognition
- Named Entity Recognition
- Protein sequence analysis

**Example - POS Tagging:**
```
Sentence: "The cat sat"

Hidden states (POS): DT → NN → VBD
Observations (words): The → cat → sat

Transition: P(NN|DT) × P(VBD|NN)
Emission: P(The|DT) × P(cat|NN) × P(sat|VBD)
```

**Algorithms:**
- **Viterbi:** Find most likely state sequence
- **Forward-Backward:** Compute probabilities
- **Baum-Welch:** Train model parameters

---

### Q2.5: What are Word Embeddings (Word2Vec, GloVe)?

**Answer:** Word embeddings are dense vector representations of words that capture semantic relationships.

**Word2Vec (2013):**
- **Skip-gram:** Predicts context words from target word
- **CBOW:** Predicts target word from context words
- Captures semantic and syntactic relationships

**Example relationships:**
```
king - man + woman ≈ queen
Paris - France + Italy ≈ Rome
```

**GloVe (Global Vectors):**
- Uses global word co-occurrence statistics
- Combines benefits of matrix factorization and local context

**Code Example:**
```python
from gensim.models import Word2Vec

# Training data
sentences = [
    ['natural', 'language', 'processing'],
    ['machine', 'learning', 'algorithms'],
    ['deep', 'learning', 'neural', 'networks']
]

# Train Word2Vec
model = Word2Vec(sentences, vector_size=100, window=5, 
                 min_count=1, workers=4)

# Get vector
vector = model.wv['learning']  # 100-dimensional vector

# Find similar words
similar = model.wv.most_similar('learning', topn=3)
print(similar)
```

**Advantages:**
- Captures semantic meaning
- Reduces dimensionality
- Enables similarity calculations

---

## 3. Deep Learning for NLP

### Q3.1: What are Recurrent Neural Networks (RNNs) and why are they used in NLP?

**Answer:** RNNs are neural networks designed to process sequential data by maintaining hidden states across time steps.

**Architecture:**
```
h_t = f(W_hh × h_(t-1) + W_xh × x_t + b_h)
y_t = W_hy × h_t + b_y

Where:
h_t = hidden state at time t
x_t = input at time t
y_t = output at time t
W = weight matrices
b = bias vectors
```

**Key Features:**
- Sequential processing
- Shared parameters across time steps
- Memory of previous inputs

**Applications:**
- Language modeling
- Machine translation
- Sentiment analysis
- Text generation

**Limitations:**
- **Vanishing gradient problem:** Difficulty learning long-term dependencies
- **Exploding gradient problem:** Gradients become too large
- Cannot be parallelized (slow training)

---

### Q3.2: Explain LSTM and why it's better than vanilla RNN.

**Answer:** LSTM (Long Short-Term Memory) is an RNN variant designed to address vanishing gradient problem.

**Key Components (Gates):**

1. **Forget Gate:** Decides what information to discard
   ```
   f_t = σ(W_f × [h_(t-1), x_t] + b_f)
   ```

2. **Input Gate:** Decides what new information to store
   ```
   i_t = σ(W_i × [h_(t-1), x_t] + b_i)
   C̃_t = tanh(W_C × [h_(t-1), x_t] + b_C)
   ```

3. **Output Gate:** Decides what to output
   ```
   o_t = σ(W_o × [h_(t-1), x_t] + b_o)
   h_t = o_t × tanh(C_t)
   ```

**Advantages over RNN:**
- Solves vanishing gradient problem
- Captures long-term dependencies
- Selective memory retention

**Example:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(128, input_shape=(sequence_length, features)),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

---

### Q3.3: What is GRU and how does it differ from LSTM?

**Answer:** GRU (Gated Recurrent Unit) is a simplified LSTM variant with fewer parameters.

**Differences from LSTM:**

| Aspect | LSTM | GRU |
|--------|------|-----|
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| Memory Cell | Separate cell state | Combined with hidden state |
| Parameters | More parameters | Fewer parameters |
| Speed | Slower | Faster |
| Performance | Slightly better on complex tasks | Comparable on most tasks |

**GRU Gates:**

1. **Update Gate:** Controls how much past information to keep
   ```
   z_t = σ(W_z × [h_(t-1), x_t])
   ```

2. **Reset Gate:** Decides how much past information to forget
   ```
   r_t = σ(W_r × [h_(t-1), x_t])
   ```

**When to use:**
- **LSTM:** Complex tasks requiring precise memory control
- **GRU:** Faster training needed, similar performance acceptable

---

### Q3.4: Can CNNs be used for NLP? How?

**Answer:** Yes! CNNs can effectively process text using 1D convolutions for local pattern extraction.

**How CNNs work for text:**
- Treat text as 1D sequence (word embeddings)
- Apply sliding window (kernel) to capture n-gram features
- Use pooling to extract important features
- Stack multiple layers for hierarchical feature learning

**Architecture:**
```
Input (sentence) → Embedding → Conv1D → MaxPooling → Dense → Output
```

**Example - Text Classification:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    Conv1D(128, 5, activation='relu'),  # 5-gram features
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])
```

**Advantages:**
- Faster than RNNs (parallelizable)
- Captures local patterns (n-grams)
- Good for text classification

**Limitations:**
- Doesn't capture long-range dependencies as well as RNNs
- Position information less explicit

**Applications:**
- Sentiment analysis
- Text classification
- Intent detection
- Sentence classification

---

## 4. Transformer Architecture & LLMs

### Q4.1: Explain the Transformer architecture and why it revolutionized NLP.

**Answer:** Transformers use self-attention mechanisms to process sequences in parallel, eliminating recurrence.

**Key Components:**

1. **Self-Attention Mechanism:**
   - Computes relationships between all words simultaneously
   - Query (Q), Key (K), Value (V) matrices
   
   ```
   Attention(Q, K, V) = softmax(QK^T / √d_k) × V
   ```

2. **Multi-Head Attention:**
   - Multiple attention layers in parallel
   - Captures different relationships

3. **Position Encoding:**
   - Adds positional information (sine/cosine functions)
   - Compensates for lack of sequential processing

4. **Encoder-Decoder Structure:**
   - **Encoder:** Processes input sequence
   - **Decoder:** Generates output sequence

**Why Revolutionary:**
- **Parallelization:** Processes entire sequence simultaneously
- **Long-range dependencies:** Direct connections between any positions
- **Scalability:** Can be scaled to billions of parameters
- **Transfer learning:** Pre-train once, fine-tune for many tasks

**Architecture Diagram:**
```
Input → Embedding + Positional Encoding
       → Multi-Head Attention
       → Add & Normalize
       → Feed Forward
       → Add & Normalize
       → Output
```

---

### Q4.2: What is the Attention Mechanism? Explain with an example.

**Answer:** Attention allows models to focus on relevant parts of input when producing output.

**How it works:**
1. Compute attention scores (similarity between query and keys)
2. Normalize scores with softmax
3. Weight values by attention scores
4. Sum weighted values

**Example - Machine Translation:**
```
English: "The cat is on the mat"
French:  "Le chat est sur le tapis"

When translating "chat" (cat), attention mechanism:
- Focuses heavily on "cat" (high weight)
- Slightly on "the" (medium weight)
- Ignores "mat" (low weight)
```

**Mathematical Steps:**
```
1. Compute scores: score(h_i, s_j) = h_i^T × W × s_j
2. Apply softmax: α_ij = exp(score_ij) / Σ exp(score_ik)
3. Compute context: c_i = Σ (α_ij × h_j)
```

**Types of Attention:**
- **Self-Attention:** Attention within same sequence
- **Cross-Attention:** Attention between encoder and decoder
- **Masked Attention:** Prevents looking at future tokens

---

### Q4.3: Explain BERT architecture and its pre-training objectives.

**Answer:** BERT (Bidirectional Encoder Representations from Transformers) is a bidirectional language model.

**Architecture:**
- **Base:** 12 encoder layers, 768 hidden size, 12 attention heads
- **Large:** 24 encoder layers, 1024 hidden size, 16 attention heads
- Uses only encoder part of Transformer
- Bidirectional context (reads text both directions)

**Pre-training Objectives:**

1. **Masked Language Modeling (MLM):**
   - Randomly mask 15% of tokens
   - Predict masked tokens using bidirectional context
   
   Example:
   ```
   Input:  "The [MASK] is on the mat"
   Output: "cat" (predicted)
   ```

2. **Next Sentence Prediction (NSP):**
   - Predict if sentence B follows sentence A
   - Helps understand sentence relationships
   
   Example:
   ```
   A: "I love NLP"
   B: "It's very interesting"  → IsNext: True
   
   A: "I love NLP"
   B: "The sky is blue"  → IsNext: False
   ```

**Fine-tuning:**
- Add task-specific layers on top
- Fine-tune entire model for downstream tasks

**Code Example:**
```python
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

text = "BERT is amazing for NLP tasks"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
```

---

### Q4.4: How does GPT differ from BERT?

**Answer:** GPT (Generative Pre-trained Transformer) and BERT have different architectures and objectives.

**Key Differences:**

| Aspect | BERT | GPT |
|--------|------|-----|
| **Architecture** | Encoder-only | Decoder-only |
| **Direction** | Bidirectional | Unidirectional (left-to-right) |
| **Attention** | Full attention | Masked (causal) attention |
| **Pre-training** | MLM + NSP | Next token prediction |
| **Best for** | Understanding tasks | Generation tasks |
| **Context** | Reads from both sides | Reads from left only |

**BERT Use Cases:**
- Question answering
- Sentiment analysis
- Named Entity Recognition
- Text classification

**GPT Use Cases:**
- Text generation
- Code generation
- Creative writing
- Conversational AI

**Example:**

BERT processing "The cat is on the mat":
```
[The] ← → [cat] ← → [is] ← → [on] ← → [the] ← → [mat]
(all words see all words)
```

GPT processing same sentence:
```
[The] → [cat] → [is] → [on] → [the] → [mat]
(each word only sees previous words)
```

---

### Q4.5: Explain fine-tuning vs. feature extraction with pre-trained models.

**Answer:** Two approaches to use pre-trained models for downstream tasks.

**Feature Extraction (Feature-based Transfer Learning):**
- Freeze pre-trained model weights
- Use output embeddings as features
- Train only task-specific layers

```python
from transformers import BertModel

# Load pre-trained BERT
bert = BertModel.from_pretrained('bert-base-uncased')
bert.eval()  # Set to evaluation mode

# Freeze parameters
for param in bert.parameters():
    param.requires_grad = False

# Use embeddings as features
with torch.no_grad():
    embeddings = bert(**inputs).last_hidden_state
```

**Fine-tuning:**
- Initialize with pre-trained weights
- Update ALL model parameters
- Adapt model to specific task

```python
from transformers import BertForSequenceClassification

# Load pre-trained BERT with classification head
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

# All parameters trainable
# Fine-tune on task-specific data
```

**Comparison:**

| Aspect | Feature Extraction | Fine-tuning |
|--------|-------------------|-------------|
| Training Speed | Faster | Slower |
| Data Required | Less | More |
| Performance | Good | Better |
| Computational Cost | Lower | Higher |
| When to Use | Small datasets | Sufficient data available |

---

## 5. Practical System Design

### Q5.1: Design a sentiment analysis pipeline for social media data.

**Answer:** A production-ready sentiment analysis system requires multiple components.

**System Architecture:**

```
Data Collection → Preprocessing → Feature Extraction → Model → Post-processing → API
```

**Step-by-Step Implementation:**

**1. Data Collection:**
```python
import tweepy

# Connect to Twitter API
auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
api = tweepy.API(auth)

# Collect tweets
tweets = api.search_tweets(q="#NLP", count=100, lang="en")
```

**2. Preprocessing Pipeline:**
```python
import re
import emoji

def preprocess_social_media_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Convert emojis to text
    text = emoji.demojize(text)
    
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Lowercase
    text = text.lower()
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    return text
```

**3. Feature Engineering:**
```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def extract_features(texts):
    # Tokenize
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    return encoded
```

**4. Model Architecture:**
```python
from transformers import BertForSequenceClassification

class SentimentClassifier:
    def __init__(self):
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=3  # positive, negative, neutral
        )
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def predict(self, text):
        # Preprocess
        clean_text = preprocess_social_media_text(text)
        
        # Tokenize
        inputs = self.tokenizer(
            clean_text,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        
        # Predict
        outputs = self.model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
        
        labels = ['negative', 'neutral', 'positive']
        sentiment = labels[predictions.argmax()]
        confidence = predictions.max().item()
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'scores': {
                label: score.item() 
                for label, score in zip(labels, predictions[0])
            }
        }
```

**5. Handling Challenges:**

```python
# Sarcasm detection
def detect_sarcasm(text):
    sarcasm_indicators = ['yeah right', 'sure', 'totally']
    return any(indicator in text.lower() for indicator in sarcasm_indicators)

# Handle emoji sentiment
def get_emoji_sentiment(text):
    positive_emojis = ['😊', '😃', '❤️']
    negative_emojis = ['😢', '😠', '😞']
    
    pos_count = sum(text.count(e) for e in positive_emojis)
    neg_count = sum(text.count(e) for e in negative_emojis)
    
    return pos_count - neg_count
```

**6. Evaluation Metrics:**
```python
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(y_true, y_pred):
    print(classification_report(y_true, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
```

**7. API Deployment:**
```python
from flask import Flask, request, jsonify

app = Flask(__name__)
classifier = SentimentClassifier()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    
    result = classifier.predict(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Challenges & Solutions:**

| Challenge | Solution |
|-----------|----------|
| Sarcasm | Rule-based detection + context analysis |
| Informal language | Custom vocabulary + subword tokenization |
| Mixed languages | Language detection + multilingual models |
| Class imbalance | Weighted loss functions + oversampling |
| Real-time processing | Batch processing + caching |

---

### Q5.2: Design a chatbot system for customer support.

**Answer:** A production chatbot requires intent classification, entity extraction, and response generation.

**System Architecture:**

```
User Input → Intent Classification → Entity Extraction → Dialog Management → Response Generation → User
```

**1. Intent Classification:**
```python
from transformers import pipeline

# Load intent classifier
intent_classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

intents = {
    "greeting": ["hello", "hi", "hey"],
    "order_status": ["where is my order", "track order"],
    "return": ["return product", "refund"],
    "complaint": ["not working", "broken"],
    "goodbye": ["bye", "goodbye", "see you"]
}

def classify_intent(text):
    # Simple rule-based for common patterns
    text_lower = text.lower()
    for intent, patterns in intents.items():
        if any(pattern in text_lower for pattern in patterns):
            return intent
    
    # Use ML model for complex cases
    result = intent_classifier(text)[0]
    return result['label']
```

**2. Entity Extraction:**
```python
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    entities = {
        'order_id': None,
        'product': None,
        'date': None
    }
    
    # Extract order ID (custom pattern)
    order_pattern = r'#?\d{6,10}'
    import re
    order_match = re.search(order_pattern, text)
    if order_match:
        entities['order_id'] = order_match.group()
    
    # Extract entities using spaCy
    for ent in doc.ents:
        if ent.label_ == 'PRODUCT':
            entities['product'] = ent.text
        elif ent.label_ == 'DATE':
            entities['date'] = ent.text
    
    return entities
```

**3. Dialog Management:**
```python
class DialogManager:
    def __init__(self):
        self.context = {}
        self.conversation_history = []
    
    def process_turn(self, user_input, user_id):
        # Classify intent
        intent = classify_intent(user_input)
        
        # Extract entities
        entities = extract_entities(user_input)
        
        # Update context
        if user_id not in self.context:
            self.context[user_id] = {'state': 'start'}
        
        self.context[user_id].update(entities)
        
        # Store history
        self.conversation_history.append({
            'user_id': user_id,
            'input': user_input,
            'intent': intent,
            'entities': entities
        })
        
        # Generate response
        response = self.generate_response(intent, entities, user_id)
        
        return response
    
    def generate_response(self, intent, entities, user_id):
        if intent == 'greeting':
            return "Hello! How can I help you today?"
        
        elif intent == 'order_status':
            order_id = entities.get('order_id')
            if order_id:
                # Query order status from database
                status = self.get_order_status(order_id)
                return f"Your order {order_id} is currently {status}."
            else:
                return "Please provide your order ID to track your order."
        
        elif intent == 'return':
            return "I can help you with a return. Please provide your order number."
        
        elif intent == 'complaint':
            return "I'm sorry to hear that. Let me connect you with a specialist."
        
        else:
            return "I'm not sure I understand. Could you rephrase that?"
    
    def get_order_status(self, order_id):
        # Simulate database query
        return "shipped and will arrive in 2 days"
```

**4. Response Generation with Templates:**
```python
response_templates = {
    'greeting': [
        "Hello! How can I assist you today?",
        "Hi there! What can I help you with?",
        "Welcome! How may I help you?"
    ],
    'order_status': "Your order {order_id} is {status}.",
    'return_initiate': "To process your return for order {order_id}, I'll need a few details.",
    'goodbye': "Thank you for contacting us. Have a great day!"
}

def generate_from_template(intent, **kwargs):
    template = response_templates.get(intent)
    if isinstance(template, list):
        import random
        template = random.choice(template)
    return template.format(**kwargs)
```

**5. Advanced: Using Retrieval-Augmented Generation:**
```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class RAGChatbot:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_base = self.load_knowledge_base()
        self.index = self.build_index()
    
    def load_knowledge_base(self):
        return [
            "You can track your order using the order ID sent to your email.",
            "Returns are accepted within 30 days of delivery.",
            "Shipping typically takes 3-5 business days.",
            "We accept credit cards, PayPal, and bank transfers."
        ]
    
    def build_index(self):
        # Encode knowledge base
        embeddings = self.encoder.encode(self.knowledge_base)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))
        
        return index
    
    def retrieve_relevant_docs(self, query, k=2):
        # Encode query
        query_embedding = self.encoder.encode([query])
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Return relevant documents
        return [self.knowledge_base[idx] for idx in indices[0]]
    
    def generate_response(self, query):
        # Retrieve relevant information
        relevant_docs = self.retrieve_relevant_docs(query)
        
        # Combine with query for response generation
        context = "\n".join(relevant_docs)
        
        # Here you would use an LLM to generate response
        # For now, return most relevant doc
        return relevant_docs[0]
```

**6. Evaluation Metrics:**
```python
def evaluate_chatbot(test_data):
    metrics = {
        'intent_accuracy': 0,
        'entity_f1': 0,
        'response_time': [],
        'user_satisfaction': []
    }
    
    correct_intents = 0
    total = len(test_data)
    
    for sample in test_data:
        predicted_intent = classify_intent(sample['text'])
        if predicted_intent == sample['true_intent']:
            correct_intents += 1
    
    metrics['intent_accuracy'] = correct_intents / total
    
    return metrics
```

---

### Q5.3: Design a machine translation system.

**Answer:** A neural machine translation (NMT) system using sequence-to-sequence architecture.

**System Architecture:**

```
Source Text → Encoder → Context Vector → Decoder → Target Text
```

**1. Data Preprocessing:**
```python
import sentencepiece as spm

class TranslationPreprocessor:
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.sp_src = None
        self.sp_tgt = None
    
    def train_tokenizer(self, src_file, tgt_file):
        # Train SentencePiece tokenizers
        spm.SentencePieceTrainer.train(
            input=src_file,
            model_prefix='src_tokenizer',
            vocab_size=self.vocab_size,
            model_type='bpe'
        )
        
        spm.SentencePieceTrainer.train(
            input=tgt_file,
            model_prefix='tgt_tokenizer',
            vocab_size=self.vocab_size,
            model_type='bpe'
        )
        
        self.sp_src = spm.SentencePieceProcessor()
        self.sp_src.load('src_tokenizer.model')
        
        self.sp_tgt = spm.SentencePieceProcessor()
        self.sp_tgt.load('tgt_tokenizer.model')
    
    def preprocess_pair(self, src_text, tgt_text):
        # Normalize
        src_text = src_text.lower().strip()
        tgt_text = tgt_text.lower().strip()
        
        # Tokenize
        src_tokens = self.sp_src.encode(src_text, out_type=str)
        tgt_tokens = self.sp_tgt.encode(tgt_text, out_type=str)
        
        return src_tokens, tgt_tokens
```

**2. Transformer-based NMT Model:**
```python
import torch
import torch.nn as nn
from transformers import MarianMTModel, MarianTokenizer

class TranslationSystem:
    def __init__(self, model_name='Helsinki-NLP/opus-mt-en-de'):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
    
    def translate(self, text, max_length=512):
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        # Generate translation
        translated = self.model.generate(**inputs)
        
        # Decode
        translated_text = self.tokenizer.batch_decode(
            translated,
            skip_special_tokens=True
        )[0]
        
        return translated_text
    
    def translate_batch(self, texts, batch_size=32):
        translations = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_translations = [self.translate(text) for text in batch]
            translations.extend(batch_translations)
        
        return translations
```

**3. Beam Search Decoding:**
```python
def beam_search_decode(model, src_tokens, beam_width=5, max_length=50):
    # Initialize beam with start token
    beams = [([], 0)]  # (sequence, score)
    
    for _ in range(max_length):
        all_candidates = []
        
        for seq, score in beams:
            if len(seq) > 0 and seq[-1] == model.eos_token_id:
                all_candidates.append((seq, score))
                continue
            
            # Get next token probabilities
            logits = model.get_logits(src_tokens, seq)
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Get top-k tokens
            top_k_probs, top_k_indices = torch.topk(log_probs, beam_width)
            
            for i in range(beam_width):
                candidate = seq + [top_k_indices[i].item()]
                candidate_score = score + top_k_probs[i].item()
                all_candidates.append((candidate, candidate_score))
        
        # Keep top beam_width sequences
        beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        
        # Check if all beams ended
        if all(seq[-1] == model.eos_token_id for seq, _ in beams):
            break
    
    return beams[0][0]  # Return best sequence
```

**4. Handling Challenges:**

```python
# Handle long sentences
def split_long_sentence(text, max_length=100):
    """Split long sentences at punctuation"""
    import re
    
    sentences = re.split(r'[.!?;]', text)
    chunks = []
    current_chunk = ""
    
    for sent in sentences:
        if len(current_chunk) + len(sent) < max_length:
            current_chunk += sent + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sent + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# Handle rare words
def handle_unknown_tokens(text, tokenizer):
    """Transliterate or copy unknown words"""
    tokens = text.split()
    processed = []
    
    for token in tokens:
        token_id = tokenizer.encode(token, add_special_tokens=False)
        if token_id == tokenizer.unk_token_id:
            # Keep original or transliterate
            processed.append(f"[{token}]")
        else:
            processed.append(token)
    
    return " ".join(processed)
```

**5. Quality Estimation:**
```python
def estimate_translation_quality(source, translation):
    """Simple heuristics for quality estimation"""
    issues = []
    
    # Length ratio check
    length_ratio = len(translation.split()) / len(source.split())
    if length_ratio < 0.5 or length_ratio > 2.0:
        issues.append("Unusual length ratio")
    
    # Repeated words check
    words = translation.split()
    if len(words) != len(set(words)) and len(words) > 5:
        issues.append("Word repetition detected")
    
    # Incomplete translation check
    if translation.endswith(('...', 'and', 'or', ',')):
        issues.append("Possibly incomplete translation")
    
    quality_score = 1.0 - (len(issues) * 0.2)
    
    return {
        'score': max(0, quality_score),
        'issues': issues
    }
```

**6. Complete Pipeline:**
```python
class ProductionMTSystem:
    def __init__(self):
        self.translator = TranslationSystem()
        self.cache = {}
    
    def translate_with_preprocessing(self, text, src_lang='en', tgt_lang='de'):
        # Check cache
        cache_key = f"{src_lang}_{tgt_lang}_{text}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Preprocess
        text = text.strip()
        
        # Split if too long
        if len(text.split()) > 100:
            chunks = split_long_sentence(text)
            translations = [self.translator.translate(chunk) for chunk in chunks]
            result = " ".join(translations)
        else:
            result = self.translator.translate(text)
        
        # Quality check
        quality = estimate_translation_quality(text, result)
        
        # Cache result
        self.cache[cache_key] = {
            'translation': result,
            'quality': quality
        }
        
        return self.cache[cache_key]
```

---

## 6. Evaluation Metrics & Challenges

### Q6.1: Explain precision, recall, and F1 score for NLP tasks.

**Answer:** These metrics evaluate classification performance, crucial for NER, text classification, etc.

**Definitions:**

**Precision:** Of all predicted positives, how many are actually positive?
```
Precision = True Positives / (True Positives + False Positives)
```

**Recall:** Of all actual positives, how many did we predict?
```
Recall = True Positives / (True Positives + False Negatives)
```

**F1 Score:** Harmonic mean of precision and recall
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Example - Named Entity Recognition:**

```
Text: "Apple Inc. was founded in California"

True entities:
- "Apple Inc." → ORG
- "California" → LOC

Predicted entities:
- "Apple" → ORG (partial match)
- "Inc." → ORG (partial match)
- "California" → LOC (correct)

Evaluation:
- True Positives: 1 (California)
- False Positives: 2 (Apple, Inc. as separate)
- False Negatives: 1 (missed "Apple Inc." as single entity)

Precision = 1 / (1 + 2) = 0.33
Recall = 1 / (1 + 1) = 0.50
F1 = 2 × (0.33 × 0.50) / (0.33 + 0.50) = 0.40
```

**Code Implementation:**
```python
from sklearn.metrics import precision_recall_fscore_support, classification_report

y_true = ['ORG', 'LOC', 'PER', 'ORG', 'LOC']
y_pred = ['ORG', 'ORG', 'PER', 'ORG', 'LOC']

# Calculate metrics
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, average='weighted'
)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Detailed report
print(classification_report(y_true, y_pred))
```

**When to optimize for what:**
- **High Precision:** When false positives are costly (spam detection)
- **High Recall:** When false negatives are costly (disease detection)
- **F1 Score:** When you need balance (most NLP tasks)

---

### Q6.2: What is BLEU score and how is it calculated?

**Answer:** BLEU (Bilingual Evaluation Understudy) evaluates machine translation quality by comparing n-gram overlap.

**Formula:**
```
BLEU = BP × exp(Σ w_n × log(p_n))

Where:
- BP = Brevity Penalty (penalizes short translations)
- p_n = n-gram precision
- w_n = weight for n-gram (usually 1/N)
```

**Calculation Steps:**

1. **Compute n-gram precision (typically 1 to 4-grams)**
2. **Apply brevity penalty**
3. **Calculate geometric mean**

**Example:**

```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

# Reference translations (can have multiple references)
reference = [['the', 'cat', 'is', 'on', 'the', 'mat']]

# Candidate translation
candidate = ['the', 'cat', 'is', 'on', 'the', 'mat']

# Calculate BLEU
score = sentence_bleu(reference, candidate)
print(f"BLEU Score: {score:.4f}")  # 1.0 (perfect match)

# With different candidate
candidate2 = ['the', 'cat', 'sat', 'on', 'the', 'mat']
score2 = sentence_bleu(reference, candidate2)
print(f"BLEU Score: {score2:.4f}")  # Lower (one word different)
```

**Manual Calculation Example:**

```
Reference: "The cat is on the mat"
Candidate: "The cat sat on the mat"

1-gram precision: 5/6 = 0.833 (5 out of 6 words match)
2-gram precision: 3/5 = 0.600 (3 out of 5 bigrams match)
3-gram precision: 2/4 = 0.500
4-gram precision: 1/3 = 0.333

BLEU-4 = (0.833 × 0.600 × 0.500 × 0.333)^(1/4) ≈ 0.54
```

**Limitations:**
- Doesn't consider semantic meaning
- Favors shorter translations
- Requires reference translations
- Not suitable for creative tasks

**Variants:**
- **BLEU-1, BLEU-2, BLEU-3, BLEU-4:** Different n-gram levels
- **SacreBLEU:** Standardized implementation for reproducibility

---

### Q6.3: Explain ROUGE metrics for summarization.

**Answer:** ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures overlap between generated and reference summaries.

**Types:**

**1. ROUGE-N:** N-gram overlap
```
ROUGE-1 = (Count of overlapping unigrams) / (Count of unigrams in reference)
```

**2. ROUGE-L:** Longest Common Subsequence
```
ROUGE-L = LCS(candidate, reference) / len(reference)
```

**3. ROUGE-S:** Skip-bigram overlap (allows gaps)

**Example:**

```python
from rouge import Rouge

rouge = Rouge()

# Reference summary
reference = "The quick brown fox jumps over the lazy dog"

# Generated summary
hypothesis = "The fast brown fox jumps over the dog"

# Calculate ROUGE scores
scores = rouge.get_scores(hypothesis, reference)[0]

print(f"ROUGE-1 F1: {scores['rouge-1']['f']:.4f}")
print(f"ROUGE-2 F1: {scores['rouge-2']['f']:.4f}")
print(f"ROUGE-L F1: {scores['rouge-l']['f']:.4f}")
```

**Manual Calculation:**

```
Reference: "The cat sat on the mat"
Candidate: "The cat is on the mat"

ROUGE-1:
- Overlapping unigrams: {The, cat, on, the, mat} = 5
- Total in reference: 6
- Recall = 5/6 = 0.833

ROUGE-2:
- Reference bigrams: {The cat, cat sat, sat on, on the, the mat}
- Candidate bigrams: {The cat, cat is, is on, on the, the mat}
- Overlap: {The cat, on the, the mat} = 3
- Recall = 3/5 = 0.600

ROUGE-L:
- LCS = "The cat on the mat" (length 5)
- Recall = 5/6 = 0.833
```

**Interpretation:**
- **ROUGE-1:** Measures word-level overlap
- **ROUGE-2:** Measures phrase-level overlap
- **ROUGE-L:** Measures fluency and structure

**Comparison with BLEU:**

| Aspect | BLEU | ROUGE |
|--------|------|-------|
| Focus | Precision | Recall |
| Best for | Translation | Summarization |
| N-grams | Fixed (1-4) | Flexible |
| References | Single/Multiple | Usually single |

---

### Q6.4: What is perplexity and how is it used?

**Answer:** Perplexity measures how well a language model predicts a sample. Lower is better.

**Formula:**
```
Perplexity = 2^(Cross-Entropy)
          = 2^(-1/N × Σ log₂ P(w_i|context))

Where:
- N = number of words
- P(w_i|context) = probability of word given context
```

**Intuitive Meaning:**
- Perplexity of 100 = model is as confused as if it had to choose uniformly from 100 words
- Lower perplexity = better prediction = better model

**Example:**

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def calculate_perplexity(text, model, tokenizer):
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt')
    
    # Get model output
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
    
    # Calculate perplexity
    loss = outputs.loss
    perplexity = torch.exp(loss)
    
    return perplexity.item()

# Load model
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Calculate perplexity
text1 = "The cat sat on the mat"
text2 = "Colorless green ideas sleep furiously"

ppl1 = calculate_perplexity(text1, model, tokenizer)
ppl2 = calculate_perplexity(text2, model, tokenizer)

print(f"Perplexity (natural): {ppl1:.2f}")  # Lower
print(f"Perplexity (unnatural): {ppl2:.2f}")  # Higher
```

**Use Cases:**
- Compare language models
- Evaluate generation quality
- Hyperparameter tuning
- Detect anomalous text

**Limitations:**
- Not comparable across different vocabularies
- Doesn't measure semantic quality
- Can be misleading for short texts

---

### Q6.5: What are common challenges in NLP and how to address them?

**Answer:** NLP faces multiple challenges requiring specific solutions.

**1. Data Leakage**

**Problem:** Test data information leaks into training
```python
# WRONG: Vectorizing before splitting
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(all_texts)
X_train, X_test = train_test_split(X)  # Leakage!

# CORRECT: Split first, then vectorize
X_train_text, X_test_text = train_test_split(all_texts)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)  # Only transform!
```

**2. Class Imbalance**

**Solutions:**
```python
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

# Option 1: Oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Option 2: Class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y),
    y=y
)

# Option 3: Focal Loss
def focal_loss(y_true, y_pred, gamma=2):
    ce_loss = -y_true * torch.log(y_pred)
    focal_weight = (1 - y_pred) ** gamma
    return (focal_weight * ce_loss).mean()
```

**3. Bias in Models**

**Detection:**
```python
def detect_gender_bias(model, tokenizer):
    templates = [
        "The {} is a doctor",
        "The {} is a nurse",
        "The {} is an engineer"
    ]
    
    pronouns = ["man", "woman", "person"]
    
    for template in templates:
        for pronoun in pronouns:
            text = template.format(pronoun)
            score = model.predict_probability(text)
            print(f"{text}: {score:.3f}")
```

**Mitigation:**
```python
# Data augmentation for fairness
def debias_training_data(texts, labels):
    debiased_texts = []
    debiased_labels = []
    
    gender_words = {
        'he': 'she', 'him': 'her', 'his': 'her',
        'man': 'woman', 'boy': 'girl'
    }
    
    for text, label in zip(texts, labels):
        # Add original
        debiased_texts.append(text)
        debiased_labels.append(label)
        
        # Add gender-swapped version
        swapped = text
        for male, female in gender_words.items():
            swapped = swapped.replace(male, female)
        
        debiased_texts.append(swapped)
        debiased_labels.append(label)
    
    return debiased_texts, debiased_labels
```

**4. Out-of-Vocabulary (OOV) Words**

**Solutions:**
```python
# Subword tokenization (BPE)
from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer(BPE())

# Character-level fallback
def handle_oov(word, vocab, char_model):
    if word in vocab:
        return vocab[word]
    else:
        # Use character-level representation
        return char_model.encode(word)
```

**5. Long-Range Dependencies**

**Solutions:**
- Use Transformers with attention
- Hierarchical models
- Memory networks
```python
# Hierarchical approach
def hierarchical_encoding(document):
    # Encode sentences
    sentence_encodings = [encode_sentence(sent) 
                         for sent in document.sentences]
    
    # Encode document from sentence encodings
    document_encoding = encode_sentences(sentence_encodings)
    
    return document_encoding
```

**6. Computational Cost**

**Solutions:**
```python
# Model distillation
from transformers import DistilBertModel

# Use distilled version (40% smaller, 60% faster)
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Quantization
import torch.quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Pruning
import torch.nn.utils.prune as prune
prune.l1_unstructured(model.layer, name='weight', amount=0.3)
```

**7. Multilingual Challenges**

**Solutions:**
```python
# Use multilingual models
from transformers import XLMRobertaModel

model = XLMRobertaModel.from_pretrained('xlm-roberta-base')

# Language detection
from langdetect import detect

def process_multilingual(text):
    lang = detect(text)
    model = load_language_specific_model(lang)
    return model.process(text)
```

---

## 7. NLP Tools & Libraries

### Q7.1: Compare NLTK, spaCy, and Hugging Face Transformers.

**Answer:** Three major NLP libraries serving different purposes.

**Comparison Table:**

| Feature | NLTK | spaCy | Hugging Face |
|---------|------|-------|--------------|
| **Purpose** | Educational, research | Production NLP | Transformer models |
| **Speed** | Slow | Fast (Cython) | Varies by model |
| **Ease of Use** | Moderate | Easy | Easy |
| **Best For** | Learning, prototyping | Production pipelines | State-of-the-art models |
| **Tokenization** | Multiple algorithms | Fast, accurate | Subword (BPE, WordPiece) |
| **POS Tagging** | Yes | Yes | Via models |
| **NER** | Basic | Excellent | State-of-the-art |
| **Pretrained Models** | Limited | Yes | Extensive (1000s) |

**NLTK Example:**
```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

text = "Apple Inc. is located in California"

# Tokenize
tokens = word_tokenize(text)

# POS tagging
pos_tags = pos_tag(tokens)

# Named Entity Recognition
entities = ne_chunk(pos_tags)

print(tokens)
print(pos_tags)
print(entities)
```

**spaCy Example:**
```python
import spacy

# Load model
nlp = spacy.load("en_core_web_sm")

text = "Apple Inc. is located in California"
doc = nlp(text)

# Tokenization (automatic)
print([token.text for token in doc])

# POS tagging
print([(token.text, token.pos_) for token in doc])

# Named Entity Recognition
print([(ent.text, ent.label_) for ent in doc.ents])

# Dependency parsing
for token in doc:
    print(f"{token.text} -> {token.dep_} -> {token.head.text}")
```

**Hugging Face Transformers Example:**
```python
from transformers import pipeline

# NER pipeline
ner = pipeline("ner", model="dslim/bert-base-NER")
text = "Apple Inc. is located in California"
entities = ner(text)
print(entities)

# Sentiment analysis
sentiment = pipeline("sentiment-analysis")
result = sentiment("I love NLP!")[0]
print(f"Sentiment: {result['label']}, Score: {result['score']:.3f}")

# Text generation
generator = pipeline("text-generation", model="gpt2")
output = generator("Once upon a time", max_length=50)[0]
print(output['generated_text'])
```

**When to Use What:**

- **NLTK:** 
  - Learning NLP concepts
  - Academic research
  - Custom algorithm development
  
- **spaCy:**
  - Production applications
  - Fast processing needed
  - Complete NLP pipelines
  
- **Hugging Face:**
  - State-of-the-art performance
  - Transfer learning
  - Latest research models

---

### Q7.2: How do you use Hugging Face Transformers for fine-tuning?

**Answer:** Fine-tuning adapts pre-trained models to specific tasks.

**Step-by-Step Fine-tuning:**

**1. Load Pre-trained Model:**
```python
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import torch
from datasets import load_dataset

# Load model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2  # binary classification
)
```

**2. Prepare Dataset:**
```python
# Load dataset
dataset = load_dataset("imdb")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

# Apply tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Prepare for training
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
```

**3. Define Training Arguments:**
```python
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_steps=500,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)
```

**4. Define Metrics:**
```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
```

**5. Create Trainer and Train:**
```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Evaluate
results = trainer.evaluate()
print(results)

# Save model
trainer.save_model("./fine_tuned_model")
```

**6. Inference with Fine-tuned Model:**
```python
from transformers import pipeline

# Load fine-tuned model
classifier = pipeline(
    "text-classification",
    model="./fine_tuned_model",
    tokenizer=tokenizer
)

# Make predictions
texts = [
    "This movie was absolutely fantastic!",
    "Worst film I've ever seen."
]

predictions = classifier(texts)
for text, pred in zip(texts, predictions):
    print(f"Text: {text}")
    print(f"Label: {pred['label']}, Score: {pred['score']:.3f}\n")
```

**Advanced: Custom Training Loop:**
```python
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler

# Prepare dataloaders
train_dataloader = DataLoader(
    tokenized_datasets["train"], 
    shuffle=True, 
    batch_size=16
)
eval_dataloader = DataLoader(
    tokenized_datasets["test"], 
    batch_size=16
)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=num_training_steps
)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    # Evaluation
    model.eval()
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            
            preds = outputs.logits.argmax(dim=-1)
            predictions.extend(preds.cpu().numpy())
            references.extend(batch["labels"].cpu().numpy())
    
    accuracy = accuracy_score(references, predictions)
    print(f"Epoch {epoch+1}, Accuracy: {accuracy:.4f}\n")
```

---

### Q7.3: How to handle custom datasets with Hugging Face?

**Answer:** Process custom data for use with Transformers library.

**Method 1: Using Datasets Library:**
```python
from datasets import Dataset, DatasetDict
import pandas as pd

# Load custom data
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# Create dataset
train_dataset = Dataset.from_pandas(df_train)
test_dataset = Dataset.from_pandas(df_test)

# Combine into DatasetDict
dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

# Tokenize
def preprocess(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=128
    )

tokenized_dataset = dataset.map(preprocess, batched=True)
```

**Method 2: Custom Dataset Class:**
```python
from torch.utils.data import Dataset

class CustomNLPDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Usage
train_dataset = CustomNLPDataset(
    train_texts,
    train_labels,
    tokenizer
)
```

**Method 3: For Multi-label Classification:**
```python
class MultiLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels  # list of lists
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Convert multi-label to binary vector
        label_vector = torch.zeros(num_labels)
        label_vector[labels] = 1
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_vector
        }
```

**Method 4: For Sequence Labeling (NER):**
```python
def tokenize_and_align_labels(examples, label_all_tokens=True):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding='max_length',
        max_length=128
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                # For subword tokens
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Apply to dataset
tokenized_dataset = dataset.map(
    tokenize_and_align_labels,
    batched=True
)
```

---

### Q7.4: Explain spaCy's pipeline architecture.

**Answer:** spaCy uses a modular pipeline architecture for text processing.

**Pipeline Components:**

```
Text → Tokenizer → Tagger → Parser → NER → ... → Doc
```

**Understanding the Pipeline:**
```python
import spacy

# Load model with default pipeline
nlp = spacy.load("en_core_web_sm")

# View pipeline components
print(nlp.pipe_names)
# ['tok2vec', 'tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer']

# Process text
doc = nlp("Apple Inc. is looking for a software engineer in California")

# Each component adds information to Doc object
for token in doc:
    print(f"{token.text:15} {token.pos_:10} {token.dep_:10} {token.head.text}")
```

**Custom Pipeline Component:**
```python
from spacy.language import Language

@Language.component("custom_sentiment")
def custom_sentiment_component(doc):
    """Add custom sentiment scores to doc"""
    positive_words = {'good', 'great', 'excellent', 'amazing'}
    negative_words = {'bad', 'terrible', 'awful', 'horrible'}
    
    pos_count = sum(1 for token in doc if token.text.lower() in positive_words)
    neg_count = sum(1 for token in doc if token.text.lower() in negative_words)
    
    # Add custom attribute
    doc._.sentiment_score = pos_count - neg_count
    
    return doc

# Register custom attribute
from spacy.tokens import Doc
Doc.set_extension("sentiment_score", default=0, force=True)

# Add to pipeline
nlp.add_pipe("custom_sentiment", last=True)

# Use it
doc = nlp("This is a great and amazing product!")
print(f"Sentiment: {doc._.sentiment_score}")
```

**Disabling Pipeline Components:**
```python
# Disable components you don't need for speed
with nlp.select_pipes(disable=["parser", "ner"]):
    doc = nlp("Some text")
    # Only tokenizer, tagger, lemmatizer run

# Disable specific components permanently
nlp.disable_pipes("parser")
```

**Creating Custom Pipeline from Scratch:**
```python
from spacy.lang.en import English

# Create blank English model
nlp = English()

# Add components
nlp.add_pipe("sentencizer")
nlp.add_pipe("custom_sentiment")

# Now use it
doc = nlp("Sentence one. Sentence two.")
print(f"Sentences: {len(list(doc.sents))}")
```

**Training Custom NER Model:**
```python
import random
from spacy.training import Example

# Training data format
TRAIN_DATA = [
    ("Apple is looking at buying U.K. startup", {
        "entities": [(0, 5, "ORG"), (27, 31, "GPE")]
    }),
    ("San Francisco considers banning sidewalk robots", {
        "entities": [(0, 13, "GPE")]
    })
]

# Create blank model or load existing
nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")

# Add labels
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Training loop
optimizer = nlp.begin_training()
for epoch in range(10):
    random.shuffle(TRAIN_DATA)
    losses = {}
    
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], losses=losses, drop=0.5)
    
    print(f"Epoch {epoch}, Losses: {losses}")

# Save model
nlp.to_disk("./custom_ner_model")
```

**Batch Processing for Efficiency:**
```python
# Process multiple texts efficiently
texts = ["Text one", "Text two", "Text three", ...] * 1000

# Use pipe for batch processing (much faster)
for doc in nlp.pipe(texts, batch_size=50):
    # Process doc
    entities = [(ent.text, ent.label_) for ent in doc.ents]
```

---

## 8. Questions

### Q8.1: L1 Questions

**Q: What is the difference between bag-of-words and TF-IDF?**

**Answer:** Both are text vectorization techniques but differ in weighting.

**Bag-of-Words (BoW):**
- Counts word frequency
- Ignores word importance
- Common words get high values

**TF-IDF:**
- Weights words by rarity
- Reduces weight of common words
- Highlights important, distinctive words

**Example:**
```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

corpus = [
    "The cat sat on the mat",
    "The dog sat on the log",
    "Cats and dogs are animals"
]

# Bag of Words
bow = CountVectorizer()
bow_matrix = bow.fit_transform(corpus)
print("BoW:\n", bow_matrix.toarray())
print("Features:", bow.get_feature_names_out())

# TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(corpus)
print("\nTF-IDF:\n", tfidf_matrix.toarray())
```

---

**Q: How do you handle stop words and why?**

**Answer:** Stop words are common words (the, is, at) that often don't add meaning.

**Handling Methods:**
```python
from nltk.corpus import stopwords
import string

# Method 1: Using NLTK
stop_words = set(stopwords.words('english'))
text = "This is a sample sentence with stop words"
filtered = [word for word in text.split() if word.lower() not in stop_words]

# Method 2: Using spaCy
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
filtered_spacy = [token.text for token in doc if not token.is_stop]

# Method 3: Custom stop words
custom_stops = stop_words.union({'sample', 'example'})
```

**When to Remove:**
- Text classification (usually helps)
- Information retrieval
- Keyword extraction

**When to Keep:**
- Sentiment analysis ("not good" vs "good")
- Machine translation
- Question answering

---

**Q: Explain word embeddings in simple terms.**

**Answer:** Word embeddings are dense numerical representations of words that capture meaning.

**Key Concepts:**
- Each word → vector of numbers (e.g., 300 dimensions)
- Similar words have similar vectors
- Can perform arithmetic: king - man + woman ≈ queen

**Simple Example:**
```python
# Conceptual representation (actual embeddings are 100-300 dim)
word_vectors = {
    'king': [0.8, 0.9, 0.1],    # high royalty, high male
    'queen': [0.8, 0.1, 0.9],   # high royalty, high female
    'man': [0.1, 0.9, 0.1],     # low royalty, high male
    'woman': [0.1, 0.1, 0.9]    # low royalty, high female
}

# Using pre-trained embeddings
from gensim.models import KeyedVectors

# Load GloVe or Word2Vec
embeddings = KeyedVectors.load_word2vec_format('path/to/vectors.txt')

# Get vector
vector = embeddings['computer']  # 300-dim vector

# Find similar words
similar = embeddings.most_similar('computer', topn=5)
print(similar)
# [('laptop', 0.82), ('pc', 0.79), ('software', 0.75), ...]
```

---

### Q8.2: Questions

**Q: Explain attention mechanism and why it's important.**

**Answer:** Attention allows models to focus on relevant parts of input when making predictions.

**How It Works:**
1. Compute relevance scores for all input positions
2. Weight inputs by relevance
3. Combine weighted inputs

**Visual Example:**

```
Translating "I love machine learning" to French:

When generating "apprentissage" (learning):
- High attention on "learning" (0.7)
- Medium attention on "machine" (0.2)
- Low attention on "I", "love" (0.05 each)

Output = 0.7 * vec(learning) + 0.2 * vec(machine) + 0.05 * vec(I) + 0.05 * vec(love)
```

**Code Illustration:**
```python
import torch
import torch.nn as nn

class SimpleAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, encoder_outputs):
        # encoder_outputs: (batch, seq_len, hidden_dim)
        
        # Compute attention scores
        scores = self.attention(encoder_outputs)  # (batch, seq_len, 1)
        scores = scores.squeeze(-1)  # (batch, seq_len)
        
        # Apply softmax
        weights = torch.softmax(scores, dim=1)  # (batch, seq_len)
        
        # Compute weighted sum
        context = torch.bmm(
            weights.unsqueeze(1),  # (batch, 1, seq_len)
            encoder_outputs  # (batch, seq_len, hidden_dim)
        )  # (batch, 1, hidden_dim)
        
        return context.squeeze(1), weights

# Usage
attention = SimpleAttention(hidden_dim=256)
encoder_outputs = torch.randn(32, 10, 256)  # batch=32, seq_len=10
context, attention_weights = attention(encoder_outputs)

print(f"Context shape: {context.shape}")  # (32, 256)
print(f"Attention weights shape: {attention_weights.shape}")  # (32, 10)
```

**Why Important:**
- Solves long sequence problems
- Interpretable (can visualize attention)
- Enables parallel processing (Transformers)
- Improves performance on many tasks

---

**Q: How would you handle imbalanced datasets in text classification?**

**Answer:** Multiple strategies depending on imbalance severity.

**Strategy 1: Resampling**
```python
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Oversample minority class
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# SMOTE for text (use with TF-IDF features)
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_tfidf, y)

# Undersample majority class
undersampler = RandomUnderSampler(random_state=42)
X_under, y_under = undersampler.fit_resample(X, y)
```

**Strategy 2: Class Weights**
```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Compute class weights
classes = np.unique(y_train)
class_weights = compute_class_weight(
    'balanced',
    classes=classes,
    y=y_train
)
class_weight_dict = dict(zip(classes, class_weights))

# Use in model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced')

# For neural networks
import torch.nn as nn
criterion = nn.CrossEntropyLoss(
    weight=torch.FloatTensor(class_weights)
)
```

**Strategy 3: Focal Loss**
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
```

**Strategy 4: Data Augmentation**
```python
import nlpaug.augmenter.word as naw

# Synonym replacement
aug = naw.SynonymAug(aug_src='wordnet')

# Augment minority class samples
augmented_texts = []
for text in minority_class_texts:
    # Generate 3 variations
    for _ in range(3):
        augmented = aug.augment(text)
        augmented_texts.append(augmented)
```

**Strategy 5: Ensemble Methods**
```python
# Balance by using multiple models
models = []
for i in range(5):
    # Sample balanced subset
    X_sample, y_sample = resample_balanced(X_train, y_train)
    model = train_model(X_sample, y_sample)
    models.append(model)

# Predict by voting
def ensemble_predict(X):
    predictions = [model.predict(X) for model in models]
    return majority_vote(predictions)
```

**Evaluation with Imbalanced Data:**
```python
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score
)

# Don't just use accuracy!
print(classification_report(y_true, y_pred))

# Use F1 score, especially macro/weighted
f1_macro = f1_score(y_true, y_pred, average='macro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')

# ROC-AUC for binary
auc = roc_auc_score(y_true, y_pred_proba)
```

---

**Q: How do you evaluate a language generation model?**

**Answer:** Combination of automatic metrics and human evaluation.

**Automatic Metrics:**

```python
# 1. BLEU Score (for translation)
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

reference = [['the', 'cat', 'is', 'on', 'the', 'mat']]
candidate = ['the', 'cat', 'is', 'on', 'the', 'mat']
bleu = sentence_bleu(reference, candidate)

# 2. ROUGE Score (for summarization)
from rouge import Rouge

rouge = Rouge()
scores = rouge.get_scores(hypothesis, reference)

# 3. Perplexity (language modeling)
def calculate_perplexity(model, text):
    with torch.no_grad():
        outputs = model(text, labels=text)
        loss = outputs.loss
    return torch.exp(loss).item()

# 4. METEOR (considers synonyms)
from nltk.translate.meteor_score import meteor_score
score = meteor_score([reference], candidate)

# 5. BERTScore (semantic similarity)
from bert_score import score as bert_score

P, R, F1 = bert_score(
    candidates,
    references,
    lang="en",
    verbose=True
)
```

**Quality Dimensions:**
```python
def evaluate_generation_quality(generated_text, reference_text):
    metrics = {}
    
    # 1. Fluency (grammar check)
    import language_tool_python
    tool = language_tool_python.LanguageTool('en-US')
    errors = tool.check(generated_text)
    metrics['fluency_score'] = 1 - (len(errors) / len(generated_text.split()))
    
    # 2. Diversity (unique n-grams)
    from nltk import ngrams
    words = generated_text.split()
    bigrams = list(ngrams(words, 2))
    metrics['diversity'] = len(set(bigrams)) / len(bigrams) if bigrams else 0
    
    # 3. Relevance (cosine similarity)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    emb1 = model.encode([generated_text])
    emb2 = model.encode([reference_text])
    
    from sklearn.metrics.pairwise import cosine_similarity
    metrics['relevance'] = cosine_similarity(emb1, emb2)[0][0]
    
    # 4. Factual consistency (for summaries)
    # Use entailment model
    from transformers import pipeline
    nli = pipeline("text-classification", 
                   model="facebook/bart-large-mnli")
    result = nli(f"{reference_text} {generated_text}")
    metrics['factual_score'] = result[0]['score'] if result[0]['label'] == 'ENTAILMENT' else 0
    
    return metrics
```

**Human Evaluation Framework:**
```python
# Create evaluation interface
evaluation_criteria = {
    'fluency': "Is the text grammatically correct? (1-5)",
    'coherence': "Does the text make logical sense? (1-5)",
    'relevance': "Is the content relevant to the input? (1-5)",
    'informativeness': "Does it contain useful information? (1-5)"
}

def human_evaluation_template(generated_text, reference_text):
    return {
        'input': reference_text,
        'output': generated_text,
        'criteria': evaluation_criteria,
        'ratings': {}  # Filled by human annotator
    }
```

**Combined Evaluation:**
```python
class GenerationEvaluator:
    def __init__(self):
        self.rouge = Rouge()
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def evaluate_comprehensive(self, generated, reference):
        results = {}
        
        # Automatic metrics
        results['rouge'] = self.rouge.get_scores(generated, reference)[0]
        results['bleu'] = sentence_bleu([reference.split()], generated.split())
        
        # Semantic similarity
        emb_gen = self.sentence_model.encode([generated])
        emb_ref = self.sentence_model.encode([reference])
        results['semantic_sim'] = cosine_similarity(emb_gen, emb_ref)[0][0]
        
        # Length analysis
        results['length_ratio'] = len(generated.split()) / len(reference.split())
        
        # Diversity
        words = generated.split()
        results['unique_words'] = len(set(words)) / len(words)
        
        return results
```

---

### Q8.3:  Questions

**Q: Design a scalable NLP system for processing millions of documents daily.**

**Answer:** Architecture for production-scale NLP requires multiple considerations.

**System Architecture:**

```
Ingestion → Queue → Preprocessing → Model Inference → Post-processing → Storage → API
    ↓         ↓           ↓              ↓                ↓              ↓        ↓
 Kafka    RabbitMQ    Spark/Dask    GPU Cluster      Validation      DB     FastAPI
```

**1. Data Ingestion Layer:**
```python
from kafka import KafkaConsumer, KafkaProducer
import json

class DocumentIngestion:
    def __init__(self):
        self.consumer = KafkaConsumer(
            'documents_topic',
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='earliest',
            max_poll_records=1000  # Batch size
        )
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda m: json.dumps(m).encode('utf-8')
        )
    
    def consume_documents(self):
        for message in self.consumer:
            document = message.value
            # Forward to preprocessing
            self.producer.send('preprocessing_topic', document)
```

**2. Distributed Preprocessing:**
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

class DistributedPreprocessing:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("NLP_Preprocessing") \
            .config("spark.executor.memory", "8g") \
            .config("spark.executor.cores", "4") \
            .getOrCreate()
    
    @staticmethod
    def preprocess_text(text):
        # Your preprocessing logic
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def process_batch(self, documents):
        df = self.spark.createDataFrame(documents, ["id", "text"])
        
        # Register UDF
        preprocess_udf = udf(self.preprocess_text, StringType())
        
        # Apply preprocessing
        df_processed = df.withColumn("processed_text", preprocess_udf(df.text))
        
        return df_processed.collect()
```

**3. Model Serving with Load Balancing:**
```python
from fastapi import FastAPI, BackgroundTasks
from transformers import pipeline
import torch
from torch.nn.parallel import DataParallel
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

class ModelServer:
    def __init__(self, num_workers=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
    def load_model(self):
        model = pipeline(
            "text-classification",
            model="bert-base-uncased",
            device=0 if torch.cuda.is_available() else -1
        )
        return model
    
    async def predict_batch(self, texts, batch_size=32):
        """Async batch prediction"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            batch_results = await loop.run_in_executor(
                self.executor,
                self.model,
                batch
            )
            results.extend(batch_results)
        return results

model_server = ModelServer()

@app.post("/predict")
async def predict(texts: list[str]):
    results = await model_server.predict_batch(texts)
    return {"predictions": results}

@app.post("/predict_stream")
async def predict_stream(document_ids: list[str], background_tasks: BackgroundTasks):
    """Non-blocking prediction"""
    background_tasks.add_task(process_documents, document_ids)
    return {"status": "processing", "job_id": "12345"}
```

**4. Caching Layer:**
```python
import redis
import hashlib
import pickle

class PredictionCache:
    def __init__(self):
        self.redis_client = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=False
        )
        self.ttl = 86400  # 24 hours
    
    def _get_cache_key(self, text):
        """Generate consistent cache key"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, text):
        key = self._get_cache_key(text)
        cached = self.redis_client.get(key)
        if cached:
            return pickle.loads(cached)
        return None
    
    def set(self, text, result):
        key = self._get_cache_key(text)
        self.redis_client.setex(
            key,
            self.ttl,
            pickle.dumps(result)
        )
    
    def predict_with_cache(self, text, model):
        # Check cache first
        cached_result = self.get(text)
        if cached_result:
            return cached_result
        
        # Predict if not cached
        result = model(text)
        
        # Cache result
        self.set(text, result)
        
        return result
```

**5. Monitoring and Logging:**
```python
from prometheus_client import Counter, Histogram, Gauge
import time
import logging

class NLPMonitoring:
    def __init__(self):
        # Metrics
        self.request_count = Counter(
            'nlp_requests_total',
            'Total NLP requests'
        )
        self.request_latency = Histogram(
            'nlp_request_latency_seconds',
            'Request latency'
        )
        self.model_accuracy = Gauge(
            'nlp_model_accuracy',
            'Current model accuracy'
        )
        self.queue_size = Gauge(
            'nlp_queue_size',
            'Current queue size'
        )
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def track_prediction(self, func):
        def wrapper(*args, **kwargs):
            self.request_count.inc()
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                latency = time.time() - start_time
                self.request_latency.observe(latency)
                
                self.logger.info(f"Prediction completed in {latency:.2f}s")
                return result
            except Exception as e:
                self.logger.error(f"Prediction failed: {str(e)}")
                raise
        
        return wrapper
```

**6. Auto-scaling Configuration:**
```python
# Kubernetes configuration
"""
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: nlp-model-server
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: nlp-model-server
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
"""

# Python-based scaling decision
class AutoScaler:
    def __init__(self):
        self.min_replicas = 2
        self.max_replicas = 10
        self.target_cpu = 0.7
    
    def should_scale_up(self, current_cpu, current_replicas):
        return (
            current_cpu > self.target_cpu and 
            current_replicas < self.max_replicas
        )
    
    def should_scale_down(self, current_cpu, current_replicas):
        return (
            current_cpu < self.target_cpu * 0.5 and 
            current_replicas > self.min_replicas
        )
```

**7. Complete Pipeline Integration:**
```python
class ScalableNLPPipeline:
    def __init__(self):
        self.ingestion = DocumentIngestion()
        self.preprocessing = DistributedPreprocessing()
        self.model_server = ModelServer()
        self.cache = PredictionCache()
        self.monitoring = NLPMonitoring()
    
    async def process_document(self, document):
        """End-to-end document processing"""
        
        # 1. Preprocessing
        preprocessed = self.preprocessing.preprocess_text(document['text'])
        
        # 2. Check cache
        cached_result = self.cache.get(preprocessed)
        if cached_result:
            self.monitoring.logger.info("Cache hit")
            return cached_result
        
        # 3. Model inference
        @self.monitoring.track_prediction
        async def predict():
            return await self.model_server.predict_batch([preprocessed])
        
        result = await predict()
        
        # 4. Cache result
        self.cache.set(preprocessed, result)
        
        # 5. Store in database
        await self.store_result(document['id'], result)
        
        return result
    
    async def store_result(self, doc_id, result):
        # Store in database (PostgreSQL, MongoDB, etc.)
        pass
```

**Performance Optimization Strategies:**

```python
# 1. Batch Processing
def batch_process(documents, batch_size=32):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        yield process_batch(batch)

# 2. Model Quantization
import torch.quantization as quantization

def quantize_model(model):
    quantized_model = quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    return quantized_model

# 3. Mixed Precision Training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Handling Failures:**
```python
from tenacity import retry, stop_after_attempt, wait_exponential

class ResilientPipeline:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def predict_with_retry(self, text):
        try:
            return self.model(text)
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise
    
    def circuit_breaker(self, func, failure_threshold=5):
        """Prevent cascading failures"""
        failures = 0
        
        def wrapper(*args, **kwargs):
            nonlocal failures
            
            if failures >= failure_threshold:
                raise Exception("Circuit breaker open")
            
            try:
                result = func(*args, **kwargs)
                failures = 0  # Reset on success
                return result
            except Exception as e:
                failures += 1
                raise
        
        return wrapper
```

---

**Q: How do you handle multilingual NLP at scale?**

**Answer:** Multilingual systems require language detection, model selection, and unified processing.

**Architecture:**

```python
from langdetect import detect, DetectorFactory
from transformers import AutoTokenizer, AutoModel

DetectorFactory.seed = 0  # Consistent detection

class MultilingualNLPSystem:
    def __init__(self):
        # Language-specific models
        self.models = {
            'en': AutoModel.from_pretrained('bert-base-uncased'),
            'es': AutoModel.from_pretrained('bert-base-spanish-wwm-uncased'),
            'fr': AutoModel.from_pretrained('camembert-base'),
            'de': AutoModel.from_pretrained('bert-base-german-cased'),
            'multilingual': AutoModel.from_pretrained('xlm-roberta-base')
        }
        
        self.tokenizers = {
            lang: AutoTokenizer.from_pretrained(model_name)
            for lang, model_name in [
                ('en', 'bert-base-uncased'),
                ('es', 'bert-base-spanish-wwm-uncased'),
                ('fr', 'camembert-base'),
                ('de', 'bert-base-german-cased'),
                ('multilingual', 'xlm-roberta-base')
            ]
        }
    
    def detect_language(self, text):
        """Detect language with confidence"""
        try:
            lang = detect(text)
            return lang
        except:
            return 'en'  # Default fallback
    
    def process_text(self, text):
        """Process text in any language"""
        # Detect language
        lang = self.detect_language(text)
        
        # Select appropriate model
        if lang in self.models:
            model = self.models[lang]
            tokenizer = self.tokenizers[lang]
        else:
            # Use multilingual model for unsupported languages
            model = self.models['multilingual']
            tokenizer = self.tokenizers['multilingual']
        
        # Process
        inputs = tokenizer(text, return_tensors='pt', truncation=True)
        outputs = model(**inputs)
        
        return {
            'language': lang,
            'embeddings': outputs.last_hidden_state,
            'model_used': 'specific' if lang in self.models else 'multilingual'
        }
```

**Cross-lingual Transfer Learning:**
```python
class CrossLingualSystem:
    def __init__(self):
        # Use zero-shot cross-lingual model
        self.model = AutoModel.from_pretrained('xlm-roberta-large')
        self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
    
    def train_on_one_language(self, train_data_en):
        """Train on English, apply to other languages"""
        # Train classifier on English
        classifier = self.build_classifier()
        classifier.fit(train_data_en['X'], train_data_en['y'])
        return classifier
    
    def predict_multilingual(self, texts, languages):
        """Predict on multiple languages without language-specific training"""
        results = []
        
        for text, lang in zip(texts, languages):
            # Get language-agnostic embeddings
            inputs = self.tokenizer(text, return_tensors='pt')
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
            
            # Predict using English-trained model
            prediction = self.classifier.predict(embeddings.detach().numpy())
            results.append({
                'text': text,
                'language': lang,
                'prediction': prediction
            })
        
        return results
```

**Handling Language-Specific Challenges:**
```python
class LanguageSpecificHandler:
    def __init__(self):
        self.processors = {
            'zh': self.process_chinese,
            'ja': self.process_japanese,
            'ar': self.process_arabic,
            'default': self.process_default
        }
    
    def process_chinese(self, text):
        """Chinese-specific processing"""
        import jieba  # Chinese word segmentation
        
        # Segment words
        words = jieba.cut(text)
        return ' '.join(words)
    
    def process_japanese(self, text):
        """Japanese-specific processing"""
        import MeCab  # Japanese morphological analyzer
        
        mecab = MeCab.Tagger()
        parsed = mecab.parse(text)
        return parsed
    
    def process_arabic(self, text):
        """Arabic-specific processing"""
        import pyarabic.araby as araby
        
        # Normalize Arabic text
        text = araby.strip_tashkeel(text)  # Remove diacritics
        text = araby.normalize_hamza(text)  # Normalize hamza
        return text
    
    def process_default(self, text):
        """Default processing"""
        return text.lower().strip()
    
    def process(self, text, language):
        processor = self.processors.get(language, self.processors['default'])
        return processor(text)
```

**Translation Pipeline:**
```python
from transformers import MarianMTModel, MarianTokenizer

class TranslationPipeline:
    def __init__(self):
        self.translation_models = {}
        self.supported_pairs = [
            ('en', 'es'), ('en', 'fr'), ('en', 'de'),
            ('es', 'en'), ('fr', 'en'), ('de', 'en')
        ]
        
        # Load models
        for src, tgt in self.supported_pairs:
            model_name = f'Helsinki-NLP/opus-mt-{src}-{tgt}'
            self.translation_models[(src, tgt)] = {
                'model': MarianMTModel.from_pretrained(model_name),
                'tokenizer': MarianTokenizer.from_pretrained(model_name)
            }
    
    def translate(self, text, src_lang, tgt_lang):
        """Translate text between languages"""
        pair = (src_lang, tgt_lang)
        
        if pair not in self.translation_models:
            # Try translating via English as pivot
            if src_lang != 'en' and tgt_lang != 'en':
                text_en = self.translate(text, src_lang, 'en')
                return self.translate(text_en, 'en', tgt_lang)
            else:
                raise ValueError(f"Translation pair {pair} not supported")
        
        model_info = self.translation_models[pair]
        model = model_info['model']
        tokenizer = model_info['tokenizer']
        
        # Translate
        inputs = tokenizer(text, return_tensors='pt', padding=True)
        translated = model.generate(**inputs)
        result = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
        
        return result
    
    def multilingual_search(self, query, documents, target_lang='en'):
        """Search across documents in different languages"""
        # Translate query to target language if needed
        query_lang = detect(query)
        if query_lang != target_lang:
            query = self.translate(query, query_lang, target_lang)
        
        # Translate all documents to target language
        translated_docs = []
        for doc in documents:
            doc_lang = detect(doc)
            if doc_lang != target_lang:
                doc = self.translate(doc, doc_lang, target_lang)
            translated_docs.append(doc)
        
        # Perform search on translated documents
        return self.search_engine.search(query, translated_docs)
```

---

**Q: Explain prompt engineering for LLMs and best practices.**

**Answer:** Prompt engineering optimizes LLM inputs for desired outputs.

**Core Techniques:**

**1. Zero-shot Prompting:**
```python
# Basic zero-shot
prompt = "Classify the sentiment of this review: 'This product is amazing!'"

# Improved zero-shot
prompt = """
Task: Sentiment Classification
Review: "This product is amazing!"
Sentiment (Positive/Negative/Neutral):
"""
```

**2. Few-shot Learning:**
```python
def create_few_shot_prompt(task, examples, test_input):
    prompt = f"Task: {task}\n\n"
    
    # Add examples
    for i, (input_text, output) in enumerate(examples, 1):
        prompt += f"Example {i}:\n"
        prompt += f"Input: {input_text}\n"
        prompt += f"Output: {output}\n\n"
    
    # Add test case
    prompt += f"Now complete this:\n"
    prompt += f"Input: {test_input}\n"
    prompt += f"Output:"
    
    return prompt

# Usage
examples = [
    ("I love this!", "Positive"),
    ("This is terrible.", "Negative"),
    ("It's okay.", "Neutral")
]
prompt = create_few_shot_prompt(
    "Sentiment Analysis",
    examples,
    "Best purchase ever!"
)
```

**3. Chain-of-Thought Prompting:**
```python
def chain_of_thought_prompt(question):
    prompt = f"""
Question: {question}

Let's solve this step by step:
1. First, identify what we need to find
2. Then, determine what information we have
3. Apply the relevant logic or formula
4. Calculate the answer
5. State the final answer

Solution:
"""
    return prompt

# Example
question = "If a train travels 120 miles in 2 hours, what is its average speed?"
prompt = chain_of_thought_prompt(question)
```

**4. Role-based Prompting:**
```python
def role_based_prompt(role, task, input_text):
    prompt = f"""
You are a {role}.

Task: {task}

Input: {input_text}

Response:
"""
    return prompt

# Example
prompt = role_based_prompt(
    role="professional software engineer",
    task="Review this code for bugs and improvements",
    input_text="def add(a,b): return a+b"
)
```

**5. Template-based Prompts:**
```python
class PromptTemplate:
    def __init__(self, template):
        self.template = template
    
    def format(self, **kwargs):
        return self.template.format(**kwargs)

# Define templates
templates = {
    'summarization': """
    Summarize the following text in {num_sentences} sentences:
    
    Text: {text}
    
    Summary:
    """,
    
    'qa': """
    Context: {context}
    
    Question: {question}
    
    Answer based only on the context above:
    """,
    
    'classification': """
    Classify the following text into one of these categories: {categories}
    
    Text: {text}
    
    Category:
    """
}

# Usage
summarization = PromptTemplate(templates['summarization'])
prompt = summarization.format(
    num_sentences=3,
    text="Long text here..."
)
```

**6. Advanced: Self-Consistency:**
```python
def self_consistency_prompting(model, question, num_samples=5):
    """Generate multiple reasoning paths and take majority vote"""
    
    prompt = f"""
{question}

Think through this carefully and show your reasoning:
"""
    
    # Generate multiple responses
    responses = []
    for _ in range(num_samples):
        response = model.generate(prompt, temperature=0.7)
        # Extract answer from response
        answer = extract_answer(response)
        responses.append(answer)
    
    # Take majority vote
    from collections import Counter
    most_common = Counter(responses).most_common(1)[0][0]
    
    return most_common
```

**Best Practices:**

```python
class PromptBestPractices:
    @staticmethod
    def clear_and_specific(task, input_data):
        """Be explicit about what you want"""
        return f"""
Task: {task}
Requirements:
- Be concise
- Use bullet points
- Limit to 3 key points

Input: {input_data}

Output:
"""
    
    @staticmethod
    def provide_format(data):
        """Specify desired output format"""
        return f"""
Analyze this data and return JSON in this exact format:
{{
    "sentiment": "positive/negative/neutral",
    "confidence": 0.0-1.0,
    "key_phrases": ["phrase1", "phrase2"]
}}

Data: {data}

JSON Output:
"""
    
    @staticmethod
    def handle_edge_cases(input_text):
        """Include instructions for edge cases"""
        return f"""
Classify the sentiment. If the text is:
- Unclear or ambiguous: return "neutral"
- Too short (< 3 words): return "insufficient_data"
- Contains mixed sentiments: return "mixed"

Text: {input_text}

Classification:
"""
    
    @staticmethod
    def iterative_refinement(initial_output):
        """Refine output through follow-up prompts"""
        return f"""
Previous output: {initial_output}

Please improve this by:
1. Making it more concise
2. Adding specific examples
3. Checking for accuracy

Improved output:
"""
```

**Prompt Optimization Framework:**
```python
class PromptOptimizer:
    def __init__(self, model, eval_metric):
        self.model = model
        self.eval_metric = eval_metric
        self.prompt_history = []
    
    def optimize(self, base_prompt, test_cases, num_iterations=5):
        """Iteratively optimize prompt"""
        best_prompt = base_prompt
        best_score = self.evaluate_prompt(base_prompt, test_cases)
        
        for i in range(num_iterations):
            # Generate variations
            variations = self.generate_variations(best_prompt)
            
            # Evaluate each
            for variant in variations:
                score = self.evaluate_prompt(variant, test_cases)
                
                if score > best_score:
                    best_score = score
                    best_prompt = variant
                    self.prompt_history.append({
                        'iteration': i,
                        'prompt': best_prompt,
                        'score': best_score
                    })
        
        return best_prompt, best_score
    
    def evaluate_prompt(self, prompt, test_cases):
        """Evaluate prompt on test cases"""
        scores = []
        for input_data, expected_output in test_cases:
            full_prompt = prompt.format(input=input_data)
            output = self.model.generate(full_prompt)
            score = self.eval_metric(output, expected_output)
            scores.append(score)
        
        return sum(scores) / len(scores)
    
    def generate_variations(self, prompt):
        """Generate prompt variations"""
        variations = []
        
        # Add more context
        variations.append(f"Context: You are an expert. \n{prompt}")
        
        # Add examples
        variations.append(f"{prompt}\n\nExample: [add example here]")
        
        # Modify structure
        variations.append(f"Step by step:\n{prompt}")
        
        return variations
```

**Monitoring Prompt Performance:**
```python
import mlflow

class PromptMonitoring:
    def __init__(self):
        mlflow.set_experiment("prompt_engineering")
    
    def log_prompt_performance(self, prompt, metrics):
        with mlflow.start_run():
            mlflow.log_param("prompt", prompt)
            mlflow.log_param("prompt_length", len(prompt))
            
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
    
    def compare_prompts(self, prompts, test_set):
        """A/B testing for prompts"""
        results = {}
        
        for name, prompt in prompts.items():
            metrics = self.evaluate(prompt, test_set)
            results[name] = metrics
            self.log_prompt_performance(prompt, metrics)
        
        # Find best
        best = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"Best prompt: {best[0]} with accuracy {best[1]['accuracy']}")
        
        return results
```

---

## Summary

This comprehensive guide covers NLP  questions across all experience levels, from fundamental concepts to advanced system design. Key areas include:

- **Core Concepts**: Tokenization, stemming, lemmatization, POS tagging, NER
- **Classical Models**: N-grams, TF-IDF, Naive Bayes, HMMs, Word2Vec
- **Deep Learning**: RNNs, LSTMs, CNNs for text
- **Transformers**: Attention, BERT, GPT, fine-tuning strategies
- **System Design**: Scalable architectures, multilingual systems, production deployment
- **Evaluation**: Metrics (BLEU, ROUGE, F1), challenges, and solutions
- **Tools**: NLTK, spaCy, Hugging Face Transformers
- **Advanced Topics**: Prompt engineering, distributed processing, auto-scaling

