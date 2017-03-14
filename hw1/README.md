# Preprocess
Preprocess does the following jobs:
1. Transform training data:  Remove toxic data. Divide articles to sentences. Conform to questions in testing data.
2. Word vector preparation: Prepare word vectors so that every single word can be mapped to a vector, based on fastText pre-trained model.
3. Provide an interface for models to easily access data.
## Initialization
Just call ```make preprocess```, and it will build essential files.
## Usage
Here is an example:
```python
import preprocess.utils

'''
The following data share the same format.
They are a list with sentences, and each sentence is also stored as a list, containing words.
e.g. [['Hello', 'world'], ['Hello', 'from', 'the', 'other', 'side'], ...]
In testing data, '_____' (5 underscores) represents the blank.
'''
training_data = preprocess.utils.getTrainingData()
val_data = preprocess.utils.getValData()
testing_data = preprocess.utils.getTestingData()

'''
The following is a list of choice list, and each choice list contains 5 words correspondent to a ~ e.
testing_choices[i] is the answer candidate to testing_data[i].
'''
testing_choices = preprocess.utils.getTestingChoiceList()

'''
This is word vector dictionary. It maps a word to a 300-dimentional numpy array.
'''
word_vec_dict = proprocess.utils.getWordVecDict()
hello_vec = word_vec_dict['hello']
```

