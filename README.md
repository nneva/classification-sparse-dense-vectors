`program call`

        python PA3.py --input-b data/pa3_B.txt --input-t data/pa3_T.txt --input-text data/pa3_input_text.txt


`window size`

        Window size is 5. Decision was made empirically. I was satisfied with how vectors looked like in both settings, sparse and dense. 
        On the other hand, since Tolstoy tends to have longer sentences, the window size could also be larger, but definitely not smaller than 5.

`preprocessing`

        I have decided to stick with the minimal preprocessing steps with not perfect results. Using a proper tokenizer would be nice, but it would affect execution time a lot, and thus outweigh the benefits of itself.

`comparison`

Accuracy is better in a sparse than in a dense setting for all categories due to the following facts (ordered by importance).

        Association and similarity: For this task it is simply more appropriate to use PMI since it is an association metric. Word2Vec is a similarity metric. Two words that are associated with each other, aren't always similar, when it comes to their meaning. E.g Peter : war

        Rare words: Whereas PMI is favorable towards those, Word2Vec SkipGram is not. Moreover, rare words are deleted when context window is created.

        Closer words: Word2Vec assigns more weight to the closer words, which is not the case with PMI. It treats all co-occurrences the same.

        

`further remarks`

During execution this script is going to produce several files.

        input_features.txt: Since I was asked to print out those, I've decided to write them into the file for better readability. It's not perfect, but definitely more readable than in the terminal, at least in the code editor that I am using.

        batch.txt & rest.txt: These files are necessary as the input for gensim. If I had used option for file iterator, the training would be much slower, so I had to pass .txt files, when doing cross-validation.

        dense_vectors.txt: This file was made somewhere in the middle of developing, I've deleted that line, however since I have already made the file, I submit that one, too. This file containes 80 instead of 83 vectors, due to absence of some names, such as "Fouché" from the input text. In the final matrix with dense vectors, absent words are substituted with zero-vectors.

        word2vec.wordvectors-file: Created by gensim and needed for further processing, since I am loading vectors from this file and deleting the model after the initial training.

Table **results.txt** was made manually.

![Classification-with-sparse-and-dense-vectors's Stats](https://github-readme-stats.vercel.app/api/top-langs/?username=nneva&theme=blue-green)






