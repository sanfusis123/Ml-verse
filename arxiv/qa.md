### **Batch 1: Practical Deep Learning Interview Questions (Real-World + Scenario Style)**

1. Why do we even use Deep Learning when classical ML works well for many tasks?
2. When would you *not* use a deep learning model?
3. Why does deep learning require large datasets to perform well?
4. How do you decide the number of layers in a neural network?
5. Your model is underfitting — what practical steps do you take?
6. Your model is overfitting — how do you fix it?
7. Why is scaling/normalization important before training?
8. When does gradient descent fail in deep learning?
9. Why does Mini-Batch Gradient Descent perform better in practice?
10. What happens practically when the learning rate is too high or too low?
11. In which situations is Adam better than SGD?
12. Why is Adam sometimes worse than SGD in large-scale training?
13. What real-world problem does Momentum solve in training?
14. When should you reduce learning rate during training?
15. What does a bad loss curve look like? How do you debug it?
16. Why is backpropagation the backbone of all deep learning?
17. What situations cause backpropagation to fail?
18. When do vanishing gradients commonly appear in real projects?
19. How does exploding gradient show up during training?
20. Your gradients become NaN — what could be the reasons?
21. Why do ReLU and its variants help deeper networks?
22. When should you avoid using ReLU?
23. What real-world issues does Batch Normalization solve?
24. When should Layer Normalization be preferred over BatchNorm?
25. What problem do Residual Connections solve?

---

### **Batch 2: Deep Learning Theory and Concepts Questions**

26. When should you increase depth vs increase width of a model?
27. What practical issues come with very deep neural networks?
28. Why do we need weight initialization techniques like Xavier/He?
29. How do poor initial weights affect training?
30. Why does Dropout work even though it removes neurons?
31. When is Dropout harmful for model performance?
32. Why do L1/L2 regularization help in real-world models?
33. When to use L1 regularization instead of L2?
34. What does overfitting look like in actual metrics/curves?
35. How does early stopping help in training?
36. Why do deep models need GPUs?
37. What kind of deep learning tasks require TPUs instead of GPUs?
38. How do you know your model capacity is too high?
39. How do you know your model capacity is too low?
40. Why is data augmentation important even if dataset is large?
41. When would you choose transfer learning instead of training from scratch?
42. Why do pretrained models reduce training time?
43. How do you decide which layers to freeze during fine-tuning?
44. When should you replace the last layer of a pretrained network?
45. What is catastrophic forgetting in deep learning?
46. Why monitoring training/validation loss is important?
47. How to detect data leakage in deep learning pipelines?
48. What real-world issues cause a model to predict the same output every time?
49. Why batch size impacts training stability in practice?
50. What are the biggest challenges when deploying deep learning models in production?

---

### **Batch 3: Neural Network Architecture and Design Questions**

1. What is Deep Learning?
2. How does Deep Learning differ from traditional Machine Learning?
3. What is a Neural Network?
4. Explain the concept of a neuron in Deep Learning
5. Explain architecture of Neural Networks in simple way
6. What is an activation function in a Neural Network?
7. Name few popular activation functions and describe them
8. What happens if you do not use any activation functions in a neural network?
9. Describe how training of basic Neural Networks works
10. What is Gradient Descent?
11. What is the function of an optimizer in Deep Learning?
12. What is backpropagation, and why is it important in Deep Learning?
13. How is backpropagation different from gradient descent?
14. Describe what Vanishing Gradient Problem is and its impact on NN
15. Describe what Exploding Gradients Problem is and its impact on NN
16. There is a neuron in the hidden layer that always results in an error. What could be the reason?
17. What do you understand by a computational graph?
18. What is Loss Function and what are various Loss functions used in Deep Learning?
19. What is Cross Entropy loss function and how is it called in industry?
20. Why is Cross-entropy preferred as the cost function for multi-class classification problems?
21. What is SGD and why it’s used in training Neural Networks?
22. Why does stochastic gradient descent oscillate towards local minima?
23. How is GD different from SGD?
24. How can optimization methods like gradient descent be improved? What is the role of the momentum term?
25. Compare batch gradient descent, minibatch gradient descent, and stochastic gradient descent.

---

### **Batch 4: Deep Learning Optimization and Regularization Questions**

26. How to decide batch size in deep learning (considering both too small and too large sizes)?
27. Batch Size vs Model Performance: How does the batch size impact the performance of a deep learning model?
28. What is Hessian, and how can it be used for faster training? What are its disadvantages?
29. What is RMSProp and how does it work?
30. Discuss the concept of an adaptive learning rate. Describe adaptive learning methods
31. What is Adam and why is it used most of the time in NNs?
32. What is AdamW and why it’s preferred over Adam?
33. What is Batch Normalization and why it’s used in NN?
34. What is Layer Normalization, and why it’s used in NN?
35. What are Residual Connections and their function in NN?
36. What is Gradient clipping and its impact on NN?
37. What is Xavier Initialization and why it’s used in NN?
38. What are different ways to solve Vanishing gradients?
39. What are ways to solve Exploding Gradients?
40. What happens if the Neural Network is suffering from Overfitting related to large weights?
41. What is Dropout and how does it work?
42. How does Dropout prevent overfitting in NN?
43. Is Dropout like Random Forest?
44. What is the impact of Dropout on the training vs testing?
45. What are L2/L1 Regularizations and how do they prevent overfitting in NN?
46. What is the difference between L1 and L2 regularisations in NN?
47. How do L1 vs L2 Regularization impact the Weights in a NN?
48. What is the curse of dimensionality in ML or AI?
49. How deep learning models tackle the curse of dimensionality?
50. What are Generative Models? Give examples.



### **Batch 1: CNN Theory and Concepts**

1. What is a Convolutional Neural Network (CNN)?
2. Why are CNNs preferred for image tasks compared to traditional neural networks?
3. What is a convolution operation in CNNs?
4. What is a kernel/filter in CNN?
5. What is stride in CNN and how does it affect output?
6. What is padding and why is it used?
7. What is the difference between valid padding and same padding?
8. What is feature extraction in CNN?
9. What is a feature map?
10. What is pooling in CNN and why is it needed?
11. Compare max pooling vs average pooling.
12. What is a receptive field in CNNs?
13. Why do deeper CNN layers capture high-level features?
14. What is flattening in CNN?
15. What is the purpose of fully connected layers in CNN?
16. What are 1×1 convolutions and why are they used?
17. What happens if kernel size is very large?
18. What is the benefit of using multiple filters in one layer?
19. Explain the concept of parameter sharing in CNN.
20. Why do CNNs require fewer parameters than fully connected networks?
21. What is the role of activation functions in CNN?
22. What is a CNN architecture pipeline from input to output?
23. What is the difference between deep and shallow CNNs?
24. What are vanishing gradients in CNN and when do they appear?
25. Why use Batch Normalization in CNN?

---

### **Batch 2: CNN Practical Application Questions**

26. What is Dropout in CNN? Why is it useful?
27. What is data augmentation and why is it used in CNN training?
28. What is overfitting in CNN and how to reduce it?
29. What is transfer learning in CNN?
30. What is fine-tuning vs feature extraction in CNN transfer learning?
31. What is ImageNet and why is it important?
32. Explain VGG architecture in simple terms.
33. Explain ResNet and skip connections.
34. Explain Inception networks and 1×1 bottleneck layers.
35. What is MobileNet and what is depthwise separable convolution?
36. What is EfficientNet and compound scaling?
37. Explain the role of Global Average Pooling.
38. What is CAM (Class Activation Map)?
39. What is Grad-CAM and why is it used?
40. How do CNNs handle translation, rotation, scaling?
41. What is a deformable convolution?
42. What is dilation in convolution and when is it useful?
43. What is object detection vs image classification?
44. What are bounding boxes and IoU?
45. What is the difference between R-CNN, Fast R-CNN, and Faster R-CNN?
46. What is YOLO and why is it fast?
47. What is SSD (Single Shot Detector)?
48. What is semantic segmentation vs instance segmentation?
49. What is U-Net architecture used for?
50. What are common challenges while training CNNs?

---

### **Batch 3: Practical CNN Interview Questions (Real-World + Scenario-Based)**

1. Why do we even need CNNs when MLPs can also take image pixels?
2. Suppose you reduce image size from 224×224 to 64×64 — what impact will it have?
3. Why do filters slide instead of analyzing the whole image at once?
4. In what scenario would you increase kernel size?
5. What happens if stride is increased to 2 or 3?
6. Suppose your model is losing spatial information — what change can fix it?
7. Why is padding important for small images?
8. Why don't we use large numbers of filters in early layers?
9. A CNN overfits heavily — what practical steps will you take?
10. What is a receptive field and why it matters for object detection tasks?
11. Suppose your CNN is not detecting small objects — what will you modify?
12. When should we prefer average pooling over max pooling?
13. In what scenario would you remove pooling layers entirely?
14. Why do modern architectures use 1×1 convolutions?
15. If your CNN is slow on mobile — what optimizations can you apply?
16. Why are deeper CNNs better but also harder to train?
17. If you double depth, what changes might be needed to keep training stable?
18. What causes vanishing gradients in CNNs?
19. How does BatchNorm practically improve CNN training?
20. Your CNN is learning very slowly — what do you tune first?
21. What is the practical drawback of using too many convolution layers?
22. Your CNN is too large to fit in memory — what reduction techniques can help?
23. Why is dropout rarely used after convolution layers in modern CNNs?
24. What is transfer learning and why is it preferred for small datasets?
25. Should you freeze or unfreeze layers while fine-tuning? When?

---

### **Batch 4: CNN Advanced Practical Interview Questions**

26. Why does VGG perform worse than newer models despite being deep?
27. What problem does ResNet specifically fix?
28. Why do skip connections help in training deep networks?
29. Why does Inception architecture use parallel convolutions?
30. Why do MobileNet models run faster on edge devices?
31. What is the practical use of depthwise separable convolution?
32. When is dilation convolution used in real-world tasks?
33. Why does semantic segmentation need special architectures like U-Net?
34. When should you use Global Average Pooling instead of Fully Connected layers?
35. During detection, what causes overlapping bounding boxes?
36. Explain IoU failure cases in object detection.
37. Why is YOLO faster but sometimes less accurate?
38. Why does SSD give good performance for medium-sized objects?
39. What does non-max suppression solve?
40. A model detects objects but misses them at different scales — what to fix?
41. When do CNNs fail completely in image tasks?
42. Why are CNNs bad at learning long-range dependencies?
43. When would you replace CNN with Vision Transformer (ViT)?
44. How to make CNN robust to rotation or perspective changes?
45. Why do CNNs perform poorly on noisy data?
46. How to debug a CNN that predicts the same class for all images?
47. What causes gradient explosion in deeper CNNs?
48. How to handle class imbalance in CNN image training?
49. Why monitoring training and validation curves is important?
50. What are 3 real-world problems for which CNNs fail and alternatives are used?




### **Batch 1: NLP Theory and Concepts**

1. What is Natural Language Processing (NLP)?
2. What are different steps in an NLP pipeline?
3. What is tokenization in NLP?
4. What is the difference between word-level and subword-level tokenization?
5. What is stemming in NLP?
6. What is lemmatization and how is it different from stemming?
7. What is stop-word removal? Why is it used?
8. What is Bag-of-Words (BoW)?
9. What is TF-IDF and why is it used?
10. What is Count Vectorization?
11. What is word embedding in NLP?
12. Explain Word2Vec in simple terms.
13. What is the difference between CBOW and Skip-gram?
14. What is GloVe? How is it different from Word2Vec?
15. What is OOV (Out-of-Vocabulary) problem in NLP?
16. What is n-gram and why use it?
17. What is language modeling?
18. What is perplexity in NLP?
19. What is a sequence-to-sequence (seq2seq) model?
20. What is attention mechanism and why is it important?
21. What is self-attention?
22. What is the Transformer architecture?
23. What are encoder and decoder in Transformers?
24. What is positional encoding and why needed?
25. What is multi-head attention?

---

### **Batch 2: NLP Models and Techniques**

26. What is beam search decoding?
27. What is greedy decoding?
28. What is BERT and its core idea?
29. What is masked language modeling (MLM)?
30. What is next sentence prediction (NSP)?
31. What is GPT and how is it different from BERT?
32. What is the difference between encoder-only, decoder-only, and encoder-decoder models?
33. What is fine-tuning in NLP?
34. What is transfer learning in NLP?
35. What is token classification? Example tasks?
36. What is named entity recognition (NER)?
37. What is text classification?
38. What is sentiment analysis?
39. What is machine translation?
40. What is question answering in NLP?
41. What is summarization (extractive vs abstractive)?
42. What is text generation in NLP?
43. What is RNN and why was it used earlier in NLP?
44. What are LSTMs and GRUs? Why were they created?
45. Why did Transformers replace RNNs/LSTMs?
46. What is token entropy?
47. What is context window or context length in LLMs?
48. What is prompt engineering?
49. What are hallucinations in LLMs?
50. What are evaluation metrics used in NLP tasks (BLEU, ROUGE, F1, accuracy)?

---

### **Batch 3: Practical NLP Interview Questions (Real-World + Scenario-Based)**

1. Why do we need tokenization before giving text to a model?
2. When is stemming harmful for real-world NLP tasks?
3. Why is lemmatization more expensive but more accurate?
4. Why is TF-IDF still used today when embeddings exist?
5. Your model performs poorly on rare words — what is the fix?
6. How does subword tokenization solve OOV issues?
7. When is Bag-of-Words still better than Transformer models?
8. Why is context important and how did Word2Vec solve it?
9. What is a drawback of Word2Vec embeddings?
10. When would you choose GloVe over Word2Vec?
11. Why do RNNs fail in long-sequence tasks?
12. What real problem does LSTM solve?
13. Why were GRUs introduced?
14. Why Transformers replaced RNNs?
15. Why is attention mechanism better than recurrence?
16. What is the practical use of self-attention?
17. How positional encoding helps Transformers understand order?
18. When should you increase the number of attention heads?
19. What happens if the context length is too small?
20. What is the challenge of long-sequence processing in LLMs?
21. When would you choose encoder-only models like BERT?
22. When should you use decoder-only models like GPT?
23. When to use encoder–decoder models like T5?
24. Why does masked language modeling improve understanding?
25. What real-world task requires next-sentence prediction?

---

### **Batch 4: Advanced NLP Interview Questions**

26. Why is fine-tuning preferred over training from scratch?
27. Why does fine-tuning sometimes destroy pretrained knowledge?
28. What is catastrophic forgetting in NLP?
29. How do you handle imbalanced text classes?
30. Why is semantic search better than keyword search?
31. Why do models hallucinate?
32. What causes bias in NLP models?
33. Why do LLMs fail on factual questions sometimes?
34. How would you reduce hallucinations in a chatbot?
35. What is the practical need of RAG (retrieval augmented generation)?
36. When should you use beam search vs greedy decoding?
37. When do models repeat the same sentence during generation?
38. What is temperature in generation and how to tune it?
39. Why BLEU fails for evaluating modern text models?
40. Why is ROUGE better for summarization evaluation?
41. Why do embeddings work better than TF-IDF?
42. Why is cosine similarity used for NLP tasks?
43. Why do we normalize vectors during similarity search?
44. Why transformer models are expensive to train?
45. Why is quantization used in LLMs?
46. When should you choose 4-bit models vs 8-bit models?
47. How to handle multilingual NLP tasks efficiently?
48. Why does machine translation fail for low-resource languages?
49. What are the challenges of using LLMs in production?
50. What are the limitations of modern NLP models?
