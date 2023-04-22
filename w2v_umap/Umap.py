from umap import UMAP

path = "word2vec_con.model"

if __name__ == "__main__" :
    #UMAP
    
    model = Word2Vec.load(path)
    #model = Word2Vec.model.load(path)

    reducer = UMAP(n_neighbors =5, min_dist =0.1, n_components = 2, verbose = True)
    
    '''
    #Version 3.8(Gensim) -> updated to 4.0 lots of different functions
    #X = model[model.wv.vocab]
    #list of word
    #X_l = list(model.wv.vocab)
    '''
    
    #X is vector for each vocab words, X_l is vocab list 
    #X and X_l has same order
    
    X = model.wv[model.wv.index_to_key]
    X_l = model.wv.index_to_key

    #Embedding to 2 dimension
    cluster_embedding = reducer.fit_transform(X)
    #get coordination
    df = pd.DataFrame(cluster_embedding)
    

    fig = plt.figure()
    fig.set_size_inches(50,30)
    ax = fig.add_subplot(1,1,1)

    ax.scatter(df[0],df[1])

    for i, txt in enumerate(X_l):
        
        #if i == 171:
        #    print(i, txt)
        
        if txt == 'trump' :
            ax.annotate(txt, (df.loc[i][0],df.loc[i][1]), color = 'red', fontsize=30)
        if txt in sim_list_trump_word :
            ax.annotate(txt, (df.loc[i][0],df.loc[i][1]), color ='red', fontsize=8)
        if txt =='biden' :
            print('biden', i)
            ax.annotate(txt, (df.loc[i][0],df.loc[i][1]),color='green', fontsize=30)
        if txt in sim_list_biden_word :
            ax.annotate(txt, (df.loc[i][0],df.loc[i][1]), color ='green', fontsize=8)  
        '''
        elif txt =='collapse' :
            ax.annotate(txt, (df.loc[i][0],df.loc[i][1]), color = 'blue', fontsize=30)
        elif txt in sim_collapse_word :
            ax.annotate(txt, (df.loc[i][0],df.loc[i][1]), color ='blue', fontsize=8)
        
        elif txt =='building' :
            ax.annotate(txt, (df.loc[i][0],df.loc[i][1]), color ='orange', fontsize=30)
        elif txt in sim_building_word :
            ax.annotate(txt, (df.loc[i][0],df.loc[i][1]), color ='orange', fontsize=8)

        elif txt =='people' :
            ax.annotate(txt, (df.loc[i][0],df.loc[i][1]),color = 'purple', fontsize=30)
        elif txt in sim_people_word :
            ax.annotate(txt, (df.loc[i][0],df.loc[i][1]), color ='purple', fontsize=8)
        else :
            ax.annotate(txt, (df.loc[i][0],df.loc[i][1]), fontsize=8)
        '''

    plt.title ("Word Embdding results with UMAP", fontsize = 20)
    plt.savefig('trump_biden_ex.pdf')