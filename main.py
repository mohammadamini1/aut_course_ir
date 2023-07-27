#!/usr/bin/python3


import traceback
from hazm import *
import json
import random
from copy import deepcopy
from math import log10 as log
from math import sqrt
import matplotlib.pyplot as plt


documents_path = './IR_data_news_12k.json'

prebuilt_inverted_index = './inverted_index_6087.json'
prebuilt_inverted_index = None

champion_lists_limit_K = 20

total_doc_count_key = 't'
preprocess_normalizer = Normalizer()
preprocess_tokenizer = WordTokenizer()
preprocess_stop_words = stopwords_list()
preprocess_stemmer = Stemmer()





def calculate_zipf(inverted_index):
    word_frequencies = {}
    for word in inverted_index:
        word_frequencies[word] = inverted_index[word][total_doc_count_key]
    sorted_frequencies = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)

    ranks = [log(x) for x in range(1, len(sorted_frequencies) + 1)]
    frequencies = [log(x[1]) for x in sorted_frequencies]

    plt.scatter(ranks, frequencies)
    plt.xlabel('log(rank)')
    plt.ylabel('log(frequency)')
    plt.title("Zipf's Law (before removing stop words)")
    # plt.title("Zipf's Law (after removing stop words)")
    plt.show()


def plot_heaps_law(inverted_index):
    all_words_count = 0
    tokens_count = len(inverted_index.keys())

    # Iterate through each word in inverted index
    for word in inverted_index:
        all_words_count += inverted_index[word][total_doc_count_key]

    print(f'in all: all words: {all_words_count} token count: {tokens_count}')
    # print(f'in {heaps_max}: all words: {all_words_count} token count: {tokens_count}')


##* preprocess function for cleaning text
def preprocess(text):
    text = text.replace("آ", "ا")

    ##* Tokenize text
    words = preprocess_tokenizer.tokenize(text)

    ##* Normalize text
    text = preprocess_normalizer.normalize(text)

    #* Remove stop words
    words = [word for word in words if word not in preprocess_stop_words]

    ##* Perform stemming
    stemmed_words = [preprocess_stemmer.stem(word) for word in words]

    return stemmed_words


# ! Step 1-1: preproccesing
def preprocessing(docs):
    forcounter = 0
    preprocessed_docs = {}
    ##* preprocess each document content
    for i in range(len(docs)):
        doc_id = str(i)
        preprocessed_content = preprocess(docs[doc_id]['content'])
        preprocessed_docs[doc_id] = {
            'content': preprocessed_content,
        }

        forcounter += 1
        if forcounter % 1000 == 0:
            print(f'last preprocessed doc id: {forcounter}')

    return preprocessed_docs


##* Calculate TF-IDF weight for a word in a document
def calculate_tfidf(ftd, N, nt):
    tf = log(ftd) + 1
    idf = log(N / nt)
    return tf * idf
    d = 1000000
    return int(tf * idf * d) / d


# ! Step 1-2: create_inverted_index
def create_inverted_index(tokenized_docs):
    inverted_index = {}
    total_docs = len(tokenized_docs)
    for doc_id in tokenized_docs:
        doc_word_positions = {}
        ##* Iterate through all words in current document to find positions
        for i, word in enumerate(tokenized_docs[doc_id]['content']):
            ##* If word is not in inverted index, add it with a total count of 0
            if word not in inverted_index:
                inverted_index[word] = {}
            ##* If word is not in current document's word count and position dictionary, add it with count 0 and empty position list
            if word not in doc_word_positions:
                doc_word_positions[word] = []

            doc_word_positions[word].append(i)

        for word in doc_word_positions:
            ##* Add current document's count, positions and weight for current word in inverted index
            inverted_index[word][doc_id] = [
                len(doc_word_positions[word]), doc_word_positions[word], 0]

    ##* Calculate word weight for each word using tf-idf
    doc_vectors = {}
    for word in inverted_index:
        total_doc_count = 0
        for doc_id in inverted_index[word]:
            word_weight = calculate_tfidf(
                inverted_index[word][doc_id][0],
                total_docs,
                len(inverted_index[word]),
            )
            inverted_index[word][doc_id][2] = word_weight
            total_doc_count += inverted_index[word][doc_id][0]

            if doc_id not in doc_vectors:
                doc_vectors[doc_id] = {}
            doc_vectors[doc_id][word] = word_weight

        inverted_index[word][total_doc_count_key] = total_doc_count

    return inverted_index, doc_vectors


##* Tokenize the query and separate query terms
def tokenize_query(query):
    query_terms = []
    in_phrase = False
    phrase = ""
    ##* Loop through all words in the query and check if it's a phrase, not query term or regular query term
    for word in query.split():
        if word == "!":
            query_terms.append(None)
        elif word[0] == '"' and not in_phrase:
            in_phrase = True
            phrase += word[1:] + " "
        elif word[-1] == '"' and in_phrase:
            in_phrase = False
            phrase += word[:-1]
            query_terms.append(phrase)
            phrase = ""
        elif in_phrase:
            phrase += word + " "
        else:
            query_terms.append(word)

    if len(query_terms) == 0:
        return [], []

    ##* Separating the query terms into regular query terms and not query terms
    regular_query_terms = [query_terms[0]]
    not_query_terms = []
    for q in range(1, len(query_terms)):
        if query_terms[q] == None:
            continue
        if query_terms[q - 1] == None:
            not_query_terms.append(query_terms[q])
        else:
            regular_query_terms.append(query_terms[q])

    ##* for each term, preprocess it using preprocess function and join it back to a string
    preprocess_regular_query_terms = []
    for q in range(len(regular_query_terms)):
        p = preprocess(regular_query_terms[q])
        if len(p) > 0:
            preprocess_regular_query_terms.append(' '.join(p))
    preprocess_not_regular_query_terms = []
    for q in range(len(not_query_terms)):
        p = preprocess(not_query_terms[q])
        not_query_terms[q] = ' '.join(p)
        if len(p) > 0:
            preprocess_not_regular_query_terms.append(' '.join(p))

    return preprocess_regular_query_terms, preprocess_not_regular_query_terms


##* Retrieve documents that contain the given token.
def retrieve_documents_with_token(inverted_index, token):
    ##* Return token value in inverted index is exists
    if token in inverted_index:
        return deepcopy(inverted_index[token])

    results = {
        total_doc_count_key: 0,
    }
    words = token.split()
    word_docs = []

    ##* Check if each word is present in the inverted index and store their corresponding data in a list.
    for word in words:
        ##* Copy related documents for each word
        if word in inverted_index:
            word_docs.append(deepcopy(inverted_index[word]))

        ##* return result if any word not in inverted index
        ##! commented -> skip word if not in inverted index. ex: (word1, notFoundWord2, word3) -> (word1, word3) means word3 should be exactly after word1 and not after notFoundWord2
        ##! uncomment this 'else' if you want to return exact match of all words (if any word is not in inverted index, then result is empty)
        # else:
        #     return results

    if len(word_docs) == 0:
        return results

    retrieved_docs = word_docs[0]
    retrieved_docs.pop(total_doc_count_key)
    ##* Retrieve documents that contain all the words in the token
    for doc_id in list(retrieved_docs.keys()):
        for next_word in range(1, len(word_docs)):
            if doc_id not in word_docs[next_word]:
                retrieved_docs.pop(doc_id)
                break

    ##* Check for exact position match if token contains multiple words
    for doc_id in list(retrieved_docs.keys()):
        ##* Create a table of positions for each word in the token
        positions = [word_docs[i][doc_id][1] for i in range(len(word_docs))]

        ##* Filter out positions that don't have an exact one position difference between the words in the token
        for i in range(1, len(positions)):
            positions[i] = [pos for pos in positions[i] if any(
                pos-positions[i-1][j] == 1 for j in range(len(positions[i-1])))]

        if any(not pos for pos in positions):
            continue

        ##* Remove any positions from the first word that don't meet the exact one position difference condition
        if len(positions) > 1:
            positions[0] = [pos for pos in positions[0] if any(
                positions[1][j]-pos == 1 for j in range(len(positions[1])))]

        ##* Save result and recalculate the counts
        results[doc_id] = [len(positions[0]), positions[0], 0]
        results[total_doc_count_key] += len(positions[0])

    ##* Set token weight
    for doc_id in results:
        if doc_id != total_doc_count_key:
            results[doc_id][2] = calculate_tfidf(len(positions[0]), len(documents), results[total_doc_count_key])

    return results


##* Retrieve documents that contain the given query considering operations such as NOT
def retrieve_documents_with_query(inverted_index, not_inverted_index, query):
    ##* Tokenize the query and separate the terms with "not" operators
    query_tokens, not_query_tokens = tokenize_query(query)
    results = {}
    not_results = {}

    ##* Retrieve documents that contain query tokens
    for token in query_tokens:
        results[token] = retrieve_documents_with_token(inverted_index, token)

    ##* Retrieve documents that contain NOT query tokens
    for token in not_query_tokens:
        not_results[token] = retrieve_documents_with_token(not_inverted_index, token)

    ##* Remove documents that contain not_query_tokens from results
    for not_result in not_results:
        for doc_id in not_results[not_result]:
            if doc_id == total_doc_count_key:
                continue
            for result in results:
                if doc_id in results[result].keys():
                    results[result][total_doc_count_key] -= results[result][doc_id][0]
                    results[result].pop(doc_id)

    return results


##* Score and sort documents based on the number of times a query token appears in the document
def doc_repeat_sort(docs):
    scored_docs = {}
    ##* For each token, number of times it appears in the document, is added to document the score
    for token in docs:
        for doc_id in docs[token]:
            if doc_id == total_doc_count_key:
                continue
            if doc_id not in scored_docs:
                scored_docs[doc_id] = docs[token][doc_id][0]
            else:
                scored_docs[doc_id] += docs[token][doc_id][0]

    ##* Sort the documents based on their scores in descending order
    ranked_docs = sorted(scored_docs.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs


# ! Step 1-3: seach query
def basic_search(inverted_index, query):
    ##* Retrieve only related documents
    results = retrieve_documents_with_query(inverted_index, inverted_index, query)

    ##* Scoring and sort
    results = doc_repeat_sort(results)

    return results


##* Create vector for query
def create_query_vector(results, total_docs):
    query_vector = {}
    keys_list = list(results.keys())
    for token in results:
        # tf = results[token][total_doc_count_key]
        # f = keys_list.count(token) / results_length
        f = keys_list.count(token)
        tf = 1 + log(f)
        idf = log(total_docs / (len(results[token])))
        weight = tf * idf
        query_vector[token] = weight
    return query_vector



##* Calculate cosine similarity
def cosine_similarity(query_vector, document_vector):
    dot_product = 0
    ##* Calculate dot products
    for token in query_vector:
        if token in document_vector:
            dot_product += query_vector[token] * document_vector[token]

    ##* Calculate norms
    query_norm = 0
    for token in query_vector:
        query_norm += query_vector[token] ** 2
    document_norm = 0
    for token in document_vector:
        document_norm += document_vector[token] ** 2
    query_norm = sqrt(query_norm)
    document_norm = sqrt(document_norm)

    return dot_product / (query_norm * document_norm)


# ! Step 2-2: vectorized seach query
def vectorize_search(inverted_index, not_inverted_index, query, document_vectors):
    ##* Retrieve only related documents
    results = retrieve_documents_with_query(inverted_index, not_inverted_index, query)

    ##* Create query vector
    query_vector = create_query_vector(results, len(documents))

    ##* Create vector for each related document
    doc_vectors = {}
    for token in results:
        for doc_id in results[token]:
            if doc_id in doc_vectors or doc_id == total_doc_count_key:
                continue
            doc_vectors[doc_id] = document_vectors[doc_id]

    ##* Calculate similarity
    similarity_scores = ()
    for doc_id in doc_vectors:
        similarity_score = cosine_similarity(query_vector, doc_vectors[doc_id])
        similarity_scores += ((doc_id, similarity_score),)

    ##* Sort the documents based on similarity scores
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    return similarity_scores


##* Create champion lists
def create_champion_lists(inverted_index, K):
    champion_lists = {}
    for word in inverted_index:
        champion_lists[word] = {}
        ##* Sort documents based on word weight
        sorted_docs = [x for x in inverted_index[word].items() if x[0] != total_doc_count_key]
        sorted_docs = sorted(sorted_docs, key=lambda x: x[1][2], reverse=True)
        ##* Save best K result
        sorted_docs = sorted_docs[:K]
        for i, (doc_id, doc_info) in enumerate(sorted_docs):
            champion_lists[word][doc_id] = doc_info
        champion_lists[word][total_doc_count_key] = inverted_index[word][total_doc_count_key]
    return champion_lists






if __name__ == '__main__':
    ##* Read documents
    print(f'\nLoading documents ...')
    with open(documents_path) as df:
        documents = json.load(df)
        print(f'Successfully loaded documents from {documents_path}')


    if prebuilt_inverted_index is None:
        ##* Perform preprocessing on the documents
        print('\nPreprocessing documents ...')
        preprocessed_docs = preprocessing(documents)
        print('Preprocessing finished')

        ##* Create inverted index and document vectors for the preprocessed documents
        print('\nCreating inverted index and document vectors ...')
        inverted_index, document_vectors = create_inverted_index(preprocessed_docs)
        print('Inverted index and document vectors created')

        # calculate_zipf(inverted_index)
        # plot_heaps_law(inverted_index)

        ##* Create champion lists
        print('\nCreating champion lists ...')
        champion_lists = create_champion_lists(inverted_index, champion_lists_limit_K)
        print('Champion lists created')

        ##* try to save the inverted index in a file
        try:
            if True:
            # if False:
                iifname = f'./inverted_index_{random.randint(1000, 9999)}.json'
                with open(iifname, 'w') as iif:
                    json.dump({
                        'inverted_index': inverted_index,
                        'document_vectors': document_vectors,
                        'champion_lists': champion_lists,
                    }, iif)
                    print(f'\nInverted index saved in {iifname}')
        except:
            print('\nwarning: failed to dump inverted index')
            pass

    else:
        print(f'\nLoading inverted index from {prebuilt_inverted_index} ...\n(change \'prebuilt_inverted_index = None\' to recreate inverted index)')
        with open(prebuilt_inverted_index) as fff:
            prebuild_data = json.load(fff)
            inverted_index = prebuild_data['inverted_index']
            document_vectors = prebuild_data['document_vectors']
            champion_lists = prebuild_data['champion_lists']
        print(f'Successfully loaded inverted index from {prebuilt_inverted_index}\n')


    ##* Search for queries using the inverted index or champion lists
    search_mode = None
    while True:
        try:
            ##* Choose search mode
            if search_mode is None:
                search_mode = int(input('Choose search mode phase\n1) Basic search (phase 1)\n2) Vectorized search (phase 2)\n3) Vectorized search Champion lists (phase 2)\n> '))

            ##* Basic search (phase 1)
            if search_mode == 1:
                query = input('Enter query: ')
                search_result = basic_search(inverted_index, query)

            ##* Vectorized search (phase 2)
            elif search_mode == 2:
                query = input('Enter query for vectorized search: ')
                search_result = vectorize_search(inverted_index, inverted_index, query, document_vectors)

            ##* Vectorized search with champion lists (phase 2)
            elif search_mode == 3:
                query = input('Enter query for vectorized search (in champion lists): ')
                search_result = vectorize_search(champion_lists, inverted_index, query, document_vectors)
                # search_result = vectorize_search(champion_lists, champion_lists, query, document_vectors)

            else:
                search_mode = None
                continue

            ##* Print results
            showid, k_limit, = 1, 5
            print('-' * 55)
            for result in reversed(search_result[:k_limit]):
                print(f'''\n{'+'*25}\n{showid}) doc id {result[0]} with score {result[1]}:\n{documents[result[0]]['title']}\n{documents[result[0]]['url']}\n{documents[result[0]]['content']}''')
                showid += 1


        ##* Press Ctrl+C to exit
        except KeyboardInterrupt:
            try:
                search_mode = None
                input('\n\nPress Enter to cancel or Ctrl+C to exit ')
            except:
                print('\nExiting ...')
                exit()

        ##* Handle errors
        except:
            print('error:')
            traceback.print_exc()
            print('')
            pass

    exit()


