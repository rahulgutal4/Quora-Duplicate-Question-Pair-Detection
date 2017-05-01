you will need to run file quora.py

In test.csv you can keep all the questions you would like to test
sample "7,6,5,How to raise money?,What are the sources of making money?" where 7,6,5 are random valuess

here you will need to specify the dimension size and the method to be used, 300 being the recommended value
we have kept both word2vec and doc2vec as the methods, you can choose either of the two.

Graphplotting.py will give you a depiction of the statements with more than 10 words predicted accurately
vs statements with less than 10 words predicted accurately
the general trend suggests 10+ words in a statements have better accuracy

We have created some pre trained models and kept at location: https://drive.google.com/open?id=0B2BWdKNWyHpNY2JpZlBmTlp0ZjQ
you will need to provide either 100,200 as dim and use method word2vec